//! Model sources and their loading helpers.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::batch::Pooling;
use crate::errors::UnsupportedRequest;

/// A model source.
pub enum ModelSource {
    /// A Hugging Face repository, downloaded on first use and then cached.
    Hub { repo: String },
    /// Local weights and tokenizer. `config.json` must sit beside the weights.
    Files { model: PathBuf, tokenizer: PathBuf },
    /// A local model directory, laid out as a head's [`Manifest`] describes and
    /// optionally nested under a subfolder. For checkpoints that ship their
    /// encoder config and tokenizer in subdirectories (`encoder/`,
    /// `tokenizer/`), which the flat source cannot name.
    Dir { root: PathBuf },
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

/// What a head needs from a checkpoint, as paths relative to the model root.
///
/// A head declares this and [`ModelSource::resolve`] fetches exactly these, so a
/// checkpoint that nests its files (`encoder/config.json`,
/// `tokenizer/tokenizer.json`) is not a special case for the resolver — only a
/// different manifest. Nothing here names a model.
pub(crate) struct Manifest {
    pub(crate) weights: &'static str,
    pub(crate) tokenizer: &'static str,
    /// Named extra files (a head's or encoder's config), by relative path.
    pub(crate) configs: &'static [(&'static str, &'static str)],
}

/// The flat layout every embedder and reranker here uses: `model.safetensors`
/// beside `config.json` and `tokenizer.json`.
pub(crate) const FLAT: Manifest = Manifest {
    weights: "model.safetensors",
    tokenizer: "tokenizer.json",
    configs: &[("config", "config.json")],
};

/// The files a [`Manifest`] resolved to, all local.
pub(crate) struct Resolved {
    pub(crate) weights: PathBuf,
    pub(crate) tokenizer: PathBuf,
    #[cfg_attr(
        not(all(feature = "coreml", feature = "coreml-export")),
        allow(dead_code)
    )]
    pub(crate) configs: BTreeMap<&'static str, PathBuf>,
    /// `1_Pooling/config.json`, if present.
    #[cfg_attr(
        not(all(feature = "coreml", feature = "coreml-export")),
        allow(dead_code)
    )]
    pub(crate) pooling_file: Option<PathBuf>,
    /// `sentence_bert_config.json`, if present.
    #[cfg_attr(
        not(all(feature = "coreml", feature = "coreml-export")),
        allow(dead_code)
    )]
    pub(crate) sentence_config: Option<PathBuf>,
    pub(crate) pooling: Option<Pooling>,
    pub(crate) declared_max_seq: Option<usize>,
}

/// Files used by the Candle backends.
pub(crate) struct CheckpointFiles {
    pub(crate) weights: PathBuf,
    pub(crate) tokenizer: PathBuf,
    /// Pooling declared in `1_Pooling/config.json`, if present.
    pub(crate) pooling: Option<Pooling>,
    /// Token limit declared in `sentence_bert_config.json`, if present.
    pub(crate) declared_max_seq: Option<usize>,
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
        let resolved = self.resolve(&FLAT, None)?;
        Ok(CheckpointFiles {
            weights: resolved.weights,
            tokenizer: resolved.tokenizer,
            pooling: resolved.pooling,
            declared_max_seq: resolved.declared_max_seq,
        })
    }

    /// Resolves the files `manifest` names, under `subfolder` when the
    /// checkpoint nests them.
    ///
    /// Generic over heads: a flat checkpoint and one with `encoder/` and
    /// `tokenizer/` differ only in their manifest and subfolder, not in the
    /// resolver.
    pub(crate) fn resolve(&self, manifest: &Manifest, subfolder: Option<&str>) -> Result<Resolved> {
        match self {
            Self::Hub { repo } => fetch_manifest(repo, manifest, subfolder),
            Self::Dir { root } => resolve_in_dir(root, manifest, subfolder),
            Self::Files { model, tokenizer } => {
                // Explicit paths win; extra configs are looked for beside the
                // weights, which is where a flat checkpoint keeps them.
                let dir = model.parent().unwrap_or_else(|| Path::new("."));
                let configs = manifest
                    .configs
                    .iter()
                    .map(|(name, rel)| (*name, dir.join(rel)))
                    .collect();
                Ok(Resolved {
                    weights: model.clone(),
                    tokenizer: tokenizer.clone(),
                    configs,
                    pooling_file: existing(dir.join("1_Pooling").join("config.json")),
                    sentence_config: existing(dir.join(SENTENCE_CONFIG)),
                    pooling: pooling_in_dir(dir),
                    declared_max_seq: declared_max_seq_in_dir(dir),
                })
            }
            Self::CoreMl { .. } | Self::CoreMlHub { .. } | Self::CoreMlConvert { .. } => {
                Err(UnsupportedRequest::new(
                    "a CoreML model source is a converted bundle, not a checkpoint",
                )
                .into())
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
            Self::Hub { .. } | Self::Files { .. } | Self::Dir { .. } => {
                Err(UnsupportedRequest::new(
                    "`--device coreml` needs a CoreML bundle (`--coreml-dir`), a Hub repo \
                 (`--coreml-model-id`), or a checkpoint to convert (`--model-id`)",
                )
                .into())
            }
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
            checkpoint.resolve(&FLAT, None)?
        }
        ModelSource::Files { .. } => checkpoint.resolve(&FLAT, None)?,
        // Only checkpoint sources can be converted.
        _ => {
            return Err(UnsupportedRequest::new(
                "`--device coreml` can convert a checkpoint (`--model-id` or \
                 `--model-path`), not another CoreML model",
            )
            .into())
        }
    };
    let config = resolved
        .configs
        .get("config")
        .cloned()
        .context("the checkpoint has no config.json")?;
    let source = match checkpoint {
        ModelSource::Hub { repo } => repo.clone(),
        ModelSource::Files { model, .. } => model.display().to_string(),
        _ => unreachable!("only Hub and Files are converted"),
    };
    let resolved = crate::coreml_export::Checkpoint {
        config,
        weights: resolved.weights,
        tokenizer: resolved.tokenizer,
        pooling: resolved.pooling_file,
        sentence_config: resolved.sentence_config,
        source,
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

/// A path only when the file is there.
fn existing(path: PathBuf) -> Option<PathBuf> {
    path.is_file().then_some(path)
}

/// Resolves a manifest inside a local model directory.
fn resolve_in_dir(root: &Path, manifest: &Manifest, subfolder: Option<&str>) -> Result<Resolved> {
    let base = match subfolder {
        Some(sub) => root.join(sub.trim_matches('/')),
        None => root.to_path_buf(),
    };
    let path = |rel: &str| -> Result<PathBuf> {
        let p = base.join(rel);
        if p.is_file() {
            Ok(p)
        } else {
            Err(anyhow::anyhow!(
                "{} is missing; this checkpoint is not laid out the way this head expects",
                p.display()
            ))
        }
    };
    let mut configs = BTreeMap::new();
    for (name, rel) in manifest.configs {
        configs.insert(*name, path(rel)?);
    }
    Ok(Resolved {
        weights: path(manifest.weights)?,
        tokenizer: path(manifest.tokenizer)?,
        configs,
        pooling_file: existing(base.join("1_Pooling").join("config.json")),
        sentence_config: existing(base.join(SENTENCE_CONFIG)),
        pooling: pooling_in_dir(&base),
        declared_max_seq: declared_max_seq_in_dir(&base),
    })
}

/// Fetches a manifest from a Hugging Face repository, caching each file.
fn fetch_manifest(repo: &str, manifest: &Manifest, subfolder: Option<&str>) -> Result<Resolved> {
    let api = hf_hub::api::sync::Api::new().context("initializing Hugging Face Hub client")?;
    let repo = api.model(repo.to_string());
    let prefix = subfolder
        .map(|s| format!("{}/", s.trim_matches('/')))
        .unwrap_or_default();
    let get = |rel: &str| {
        repo.get(&format!("{prefix}{rel}")).with_context(|| {
            format!("cannot fetch {prefix}{rel} (network down? try local --model-path)")
        })
    };
    let weights = get(manifest.weights)?;
    let tokenizer = get(manifest.tokenizer)?;
    let mut configs = BTreeMap::new();
    for (name, rel) in manifest.configs {
        configs.insert(*name, get(rel)?);
    }
    let pooling_file = repo.get(&format!("{prefix}1_Pooling/config.json")).ok();
    let pooling = pooling_file
        .as_ref()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| pooling_from_st_config(&s));
    let sentence_config = repo.get(&format!("{prefix}{SENTENCE_CONFIG}")).ok();
    let declared_max_seq = sentence_config
        .as_ref()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| max_seq_from_st_config(&s));
    Ok(Resolved {
        weights,
        tokenizer,
        configs,
        pooling_file,
        sentence_config,
        pooling,
        declared_max_seq,
    })
}

/// Where sentence-transformers records the token limit it would use.
pub(crate) const SENTENCE_CONFIG: &str = "sentence_bert_config.json";

/// Reads pooling from a checkpoint directory, if present.
pub(crate) fn pooling_in_dir(dir: &std::path::Path) -> Option<Pooling> {
    let text = std::fs::read_to_string(dir.join("1_Pooling").join("config.json")).ok()?;
    pooling_from_st_config(&text)
}

/// Reads the declared token limit from a checkpoint directory, if present.
pub(crate) fn declared_max_seq_in_dir(dir: &std::path::Path) -> Option<usize> {
    let text = std::fs::read_to_string(dir.join(SENTENCE_CONFIG)).ok()?;
    max_seq_from_st_config(&text)
}

/// Reads `max_seq_length` from a sentence-transformers config.
///
/// This is what the model says it can take, which is not what Kohagi takes:
/// `--max-seq-length` decides that, and its default is 512 whatever the file
/// says. The two disagree often enough to be worth reporting — `ruri-v3-130m`
/// declares 8192 — and sentence-transformers reads this same field, so a
/// checkpoint that behaves differently under the two libraries says so here.
///
/// The newer sentence-transformers format drops the field, so absent is as
/// ordinary as present.
fn max_seq_from_st_config(json: &str) -> Option<usize> {
    let v: serde_json::Value = serde_json::from_str(json).ok()?;
    let n = v.get("max_seq_length")?.as_u64()?;
    (n > 0).then_some(n as usize)
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

    /// What `ruri-v3-130m` ships, and the 8192 that makes this worth reading:
    /// Kohagi's default takes 512 of it.
    #[test]
    fn reads_the_declared_token_limit() {
        let base = r#"{"max_seq_length": 8192, "do_lower_case": false}"#;
        assert_eq!(max_seq_from_st_config(base), Some(8192));
    }

    /// A checkpoint saved by a newer sentence-transformers has no such field,
    /// and a fine-tuned ruri is exactly that. Absent is ordinary, not an error.
    #[test]
    fn a_config_without_the_field_declares_nothing() {
        let tuned = r#"{"transformer_task": "feature-extraction",
                        "module_output_name": "token_embeddings"}"#;
        assert_eq!(max_seq_from_st_config(tuned), None);
        assert_eq!(max_seq_from_st_config("not json"), None);
        // Nothing usable is nothing declared, rather than a limit of zero.
        assert_eq!(max_seq_from_st_config(r#"{"max_seq_length": 0}"#), None);
        assert_eq!(max_seq_from_st_config(r#"{"max_seq_length": "512"}"#), None);
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

    /// A head's manifest is resolved inside a directory, with the subfolder
    /// prefixed and every required file checked. Nothing names a model.
    #[test]
    fn a_manifest_resolves_under_a_subfolder() {
        let root = std::env::temp_dir().join(format!("kohagi-resolve-{}", std::process::id()));
        let model = root.join("multilingual");
        std::fs::create_dir_all(model.join("encoder")).unwrap();
        std::fs::create_dir_all(model.join("tokenizer")).unwrap();
        std::fs::write(model.join("model.safetensors"), b"").unwrap();
        std::fs::write(model.join("tokenizer/tokenizer.json"), b"").unwrap();
        std::fs::write(model.join("encoder/config.json"), b"{}").unwrap();
        std::fs::write(model.join("head.json"), b"{}").unwrap();

        const NESTED: Manifest = Manifest {
            weights: "model.safetensors",
            tokenizer: "tokenizer/tokenizer.json",
            configs: &[("encoder", "encoder/config.json"), ("head", "head.json")],
        };
        let source = ModelSource::Dir { root: root.clone() };
        let resolved = source.resolve(&NESTED, Some("multilingual")).unwrap();
        assert!(resolved.weights.ends_with("multilingual/model.safetensors"));
        assert!(resolved
            .tokenizer
            .ends_with("multilingual/tokenizer/tokenizer.json"));
        assert_eq!(resolved.configs.len(), 2);

        // A missing file is refused, not guessed.
        let missing = source.resolve(&FLAT, Some("multilingual"));
        assert!(missing.is_err());

        std::fs::remove_dir_all(&root).ok();
    }
}
