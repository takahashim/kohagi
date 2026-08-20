//! Model sources and their loading helpers.

use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::batch::Pooling;
use crate::errors::UnsupportedRequest;

/// A model source.
pub enum ModelSource {
    /// A Hugging Face repository, downloaded on first use and then cached.
    Hub { repo: String },
    /// Local weights and tokenizer. `config.json` must sit beside the weights.
    Files { model: PathBuf, tokenizer: PathBuf },
    /// A directory of converted CoreML models for `--device coreml`.
    CoreMl { dir: PathBuf },
    /// A Hugging Face repository containing converted CoreML models.
    CoreMlHub { repo: String },
    /// A checkpoint converted to CoreML on first use and then cached.
    CoreMlConvert {
        checkpoint: Box<ModelSource>,
        /// Sequence lengths to convert.
        buckets: Vec<usize>,
        quantize: crate::CoreMlQuantize,
    },
}

/// Files used by the Candle backends.
pub(crate) struct CheckpointFiles {
    pub(crate) weights: PathBuf,
    pub(crate) tokenizer: PathBuf,
    /// Pooling declared in `1_Pooling/config.json`, if present.
    pub(crate) pooling: Option<Pooling>,
}

/// A resolved CoreML bundle.
#[cfg(feature = "coreml")]
pub(crate) struct ResolvedCoreMl {
    pub(crate) dir: PathBuf,
    pub(crate) converted: bool,
}

impl ModelSource {
    /// Resolves this source to the files used by Candle backends.
    pub(crate) fn checkpoint_files(&self) -> Result<CheckpointFiles> {
        match self {
            Self::Files { model, tokenizer } => Ok(CheckpointFiles {
                weights: model.clone(),
                tokenizer: tokenizer.clone(),
                pooling: local_pooling(model),
            }),
            Self::Hub { repo } => {
                let f = fetch_checkpoint(repo)?;
                Ok(CheckpointFiles {
                    weights: f.weights,
                    tokenizer: f.tokenizer,
                    pooling: f.pooling,
                })
            }
            Self::CoreMl { .. } | Self::CoreMlHub { .. } | Self::CoreMlConvert { .. } => {
                Err(UnsupportedRequest::new("a CoreML model source needs `--device coreml`").into())
            }
        }
    }

    /// Resolves this source to a CoreML bundle.
    #[cfg(feature = "coreml")]
    pub(crate) fn resolve_coreml(&self, form: crate::CoreMlForm) -> Result<ResolvedCoreMl> {
        match self {
            Self::CoreMl { dir } => Ok(ResolvedCoreMl {
                dir: dir.clone(),
                converted: false,
            }),
            Self::CoreMlHub { repo } => Ok(ResolvedCoreMl {
                dir: crate::coreml::fetch_from_hub(repo, form)?,
                converted: false,
            }),
            Self::CoreMlConvert {
                checkpoint,
                buckets,
                quantize,
            } => convert_for_coreml(checkpoint, buckets, *quantize),
            Self::Hub { .. } | Self::Files { .. } => Err(UnsupportedRequest::new(
                "`--device coreml` needs a CoreML bundle (`--coreml-dir`), a Hub repo \
                 (`--coreml-model-id`), or a checkpoint to convert (`--model-id`)",
            )
            .into()),
        }
    }
}

/// Converts a checkpoint for the ANE or reuses its cached bundle.
#[cfg(all(feature = "coreml", feature = "coreml-export"))]
fn convert_for_coreml(
    checkpoint: &ModelSource,
    buckets: &[usize],
    quantize: crate::CoreMlQuantize,
) -> Result<ResolvedCoreMl> {
    use crate::coreml::autoconvert;

    let resolved = match checkpoint {
        ModelSource::Hub { repo } => {
            // hf-hub is silent while downloading, so report a cache miss.
            if !hub_checkpoint_is_cached(repo) {
                crate::program::remark!("downloading {repo} (safetensors; first run only) ...");
            }
            let files = fetch_checkpoint(repo)?;
            crate::coreml_export::Checkpoint {
                config: beside(&files.weights, "config.json")?,
                weights: files.weights,
                tokenizer: files.tokenizer,
                pooling: files.pooling_file,
                source: repo.clone(),
            }
        }
        ModelSource::Files { model, tokenizer } => crate::coreml_export::Checkpoint {
            config: beside(model, "config.json")?,
            weights: model.clone(),
            tokenizer: tokenizer.clone(),
            pooling: model
                .parent()
                .map(|d| d.join("1_Pooling").join("config.json"))
                .filter(|p| p.is_file()),
            source: model.display().to_string(),
        },
        // Only checkpoint sources can be converted.
        _ => {
            return Err(UnsupportedRequest::new(
                "`--device coreml` can convert a checkpoint (`--model-id` or \
                 `--model-path`), not another CoreML model",
            )
            .into())
        }
    };
    let provisioned = autoconvert::provision(&resolved, buckets, quantize)?;
    Ok(ResolvedCoreMl {
        dir: provisioned.path().to_path_buf(),
        converted: matches!(provisioned, autoconvert::Provisioned::Converted(_)),
    })
}

#[cfg(all(feature = "coreml", not(feature = "coreml-export")))]
fn convert_for_coreml(
    _checkpoint: &ModelSource,
    _buckets: &[usize],
    _quantize: crate::CoreMlQuantize,
) -> Result<ResolvedCoreMl> {
    Err(UnsupportedRequest::new(
        "this binary cannot convert checkpoints for CoreML; pass an already \
         converted model with `--coreml-dir` or `--coreml-model-id`, or rebuild \
         with `--features coreml,coreml-export`",
    )
    .into())
}

/// Returns whether the Hub cache contains the checkpoint weights.
#[cfg(all(feature = "coreml", feature = "coreml-export"))]
fn hub_checkpoint_is_cached(repo: &str) -> bool {
    hf_hub::Cache::default()
        .model(repo.to_string())
        .get("model.safetensors")
        .is_some()
}

/// Returns a sibling path.
#[cfg(all(feature = "coreml", feature = "coreml-export"))]
fn beside(path: &std::path::Path, name: &str) -> Result<PathBuf> {
    path.parent()
        .map(|d| d.join(name))
        .with_context(|| format!("{} has no parent directory for {name}", path.display()))
}

/// Files fetched from the Hub.
pub(crate) struct Fetched {
    pub(crate) weights: PathBuf,
    pub(crate) tokenizer: PathBuf,
    /// `1_Pooling/config.json`, if present.
    #[cfg_attr(
        not(all(feature = "coreml", feature = "coreml-export")),
        allow(dead_code)
    )]
    pooling_file: Option<PathBuf>,
    pooling: Option<Pooling>,
}

/// Fetches or reuses the files needed to load a model.
fn fetch_checkpoint(repo: &str) -> Result<Fetched> {
    let api = hf_hub::api::sync::Api::new().context("initializing Hugging Face Hub client")?;
    let repo = api.model(repo.to_string());
    let get = |f: &str| {
        repo.get(f)
            .with_context(|| format!("cannot fetch {f} (network down? try local --model-path)"))
    };
    let weights = get("model.safetensors")?;
    get("config.json")?; // Cache it beside the weights.
    let tokenizer = get("tokenizer.json")?;
    // Optional: rerankers and base models may not provide it.
    let pooling_file = repo.get("1_Pooling/config.json").ok();
    let pooling = pooling_file
        .as_ref()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| pooling_from_st_config(&s));
    Ok(Fetched {
        weights,
        tokenizer,
        pooling_file,
        pooling,
    })
}

/// Reads pooling from a checkpoint directory, if present.
pub(crate) fn pooling_in_dir(dir: &std::path::Path) -> Option<Pooling> {
    let text = std::fs::read_to_string(dir.join("1_Pooling").join("config.json")).ok()?;
    pooling_from_st_config(&text)
}

/// Reads pooling beside local weights, if present.
fn local_pooling(model_path: &std::path::Path) -> Option<Pooling> {
    pooling_in_dir(model_path.parent()?)
}

/// Reads `cls` or `mean` pooling from a sentence-transformers config.
///
/// Supports the current `pooling_mode` field and legacy boolean fields. Other
/// or combined modes return `None`.
fn pooling_from_st_config(json: &str) -> Option<Pooling> {
    let v: serde_json::Value = serde_json::from_str(json).ok()?;
    if let Some(mode) = v.get("pooling_mode").and_then(|m| m.as_str()) {
        return match mode {
            "cls" => Some(Pooling::Cls),
            "mean" => Some(Pooling::Mean),
            _ => None,
        };
    }
    if v.get("pooling_mode_cls_token")? == &serde_json::Value::Bool(true) {
        Some(Pooling::Cls)
    } else if v.get("pooling_mode_mean_tokens")? == &serde_json::Value::Bool(true) {
        Some(Pooling::Mean)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_cls_and_mean_from_st_config() {
        let cls = r#"{"pooling_mode_cls_token": true, "pooling_mode_mean_tokens": false}"#;
        let mean = r#"{"pooling_mode_cls_token": false, "pooling_mode_mean_tokens": true}"#;
        assert_eq!(pooling_from_st_config(cls), Some(Pooling::Cls));
        assert_eq!(pooling_from_st_config(mean), Some(Pooling::Mean));
    }

    #[test]
    fn reads_cls_and_mean_from_the_5x_st_config() {
        // sentence-transformers 5 uses a string instead of boolean fields.
        let cls = r#"{"embedding_dimension": 512, "pooling_mode": "cls"}"#;
        let mean = r#"{"embedding_dimension": 512, "pooling_mode": "mean",
                       "include_prompt": true}"#;
        assert_eq!(pooling_from_st_config(cls), Some(Pooling::Cls));
        assert_eq!(pooling_from_st_config(mean), Some(Pooling::Mean));
    }

    #[test]
    fn unsupported_or_absent_pooling_reads_as_none() {
        // Unsupported and invalid values are not guessed.
        let other = r#"{"pooling_mode_max_tokens": true}"#;
        assert_eq!(pooling_from_st_config(other), None);
        assert_eq!(pooling_from_st_config("not json"), None);
    }

    #[test]
    fn unsupported_or_combined_pooling_reads_as_none_in_the_5x_config() {
        let weighted = r#"{"pooling_mode": "weightedmean"}"#;
        // Combined modes are unsupported.
        let combined = r#"{"pooling_mode": ["mean", "cls"]}"#;
        assert_eq!(pooling_from_st_config(weighted), None);
        assert_eq!(pooling_from_st_config(combined), None);
    }
}
