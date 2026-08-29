//! Cross-encoder reranking: a query and a text in, one score out.
//!
//! An embedding model reads the query and the document apart, and the only
//! thing that ever compares them is a dot product. A **cross-encoder** reads
//! them together, so every layer can attend from one to the other. That costs
//! a forward pass per pair rather than per document and rules it out as a
//! search, which is why it is a *reranker*: retrieve with embeddings, reorder
//! the top few with this.
//!
//! Mechanically it is the encoder Kohagi already has (`crate::encoder`) with a
//! classification head on the CLS token — `ModernBertForSequenceClassification`
//! in Hugging Face terms, one output label. `cl-nagoya/ruri-v3-reranker-310m`
//! and the `hotchpotch/japanese-reranker-*-v2` family are all this shape, so
//! one implementation serves them.
//!
//! The score is the sigmoid of that label's logit by default, which is what
//! `sentence_transformers.CrossEncoder.predict` returns for a one-label model,
//! and therefore what any threshold tuned against that library means.

pub mod stdio;

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use tokenizers::Tokenizer;

use crate::batch::{bucket_encodings, encode_pairs, load_tokenizer, pool_row, Pooling, TokenInfo};
use crate::config::head as names;
use crate::encoder::{Activation, Config};
use crate::errors::UnsupportedRequest;
use crate::fingerprint::Fingerprint;
use crate::info::{ModelInfo, Output};
use crate::model::{
    check_max_seq, check_precision, load_weights, open_device, parse_config, read_config_json,
    run_batches, Backend, Precision, Weights,
};
use crate::program::remark;
use crate::source::ModelSource;

/// Knobs for [`Reranker::load`]. `Default` matches the Ruri v3 reranker.
#[derive(Clone, Copy)]
pub struct Options {
    /// Token-level truncation length, counting both halves of the pair and the
    /// special tokens between them.
    pub max_seq_length: usize,
    /// Rows per padded batch before the memory cap.
    pub batch_size: usize,
    pub precision: Precision,
    pub backend: Backend,
    /// Squash the logit into 0..1. On by default, because that is what
    /// CrossEncoder does for a one-label model and what published thresholds
    /// therefore mean.
    pub sigmoid: bool,
    /// Which form to download from a CoreML Hub repo.
    pub coreml_form: crate::CoreMlForm,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            max_seq_length: 512,
            batch_size: 64,
            precision: Precision::F32,
            backend: Backend::Cpu,
            sigmoid: true,
            coreml_form: crate::CoreMlForm::Compiled,
        }
    }
}

/// What `config.json` says about the classification head, as distinct from the
/// encoder underneath it (which [`Config`] already covers).
#[derive(serde::Deserialize)]
struct HeadConfig {
    architectures: Option<Vec<String>>,
    /// Which token the head reads. `cls` for every reranker measured here.
    classifier_pooling: Option<String>,
    classifier_activation: Option<String>,
    /// ModernBERT's head projection has no bias by default.
    classifier_bias: Option<bool>,
    norm_bias: Option<bool>,
    num_labels: Option<usize>,
    /// How the label count is usually spelled in practice: one entry per label.
    id2label: Option<std::collections::BTreeMap<String, String>>,
}

impl HeadConfig {
    /// Labels this model produces per pair. A reranker has one.
    fn labels(&self) -> usize {
        self.num_labels
            .or_else(|| self.id2label.as_ref().map(std::collections::BTreeMap::len))
            .unwrap_or(1)
    }

    fn pooling(&self) -> Result<Pooling> {
        match self.classifier_pooling.as_deref() {
            // ModernBERT's own default, and what every reranker here declares.
            None | Some("cls") => Ok(Pooling::Cls),
            Some("mean") => Ok(Pooling::Mean),
            Some(other) => Err(anyhow::anyhow!(
                "unsupported classifier_pooling `{other}`; kohagi-rerank implements cls and mean"
            )),
        }
    }

    /// Validates a checkpoint before scoring.
    fn validate(&self) -> Result<()> {
        let labels = self.labels();
        anyhow::ensure!(
            labels == 1,
            "this model produces {labels} labels per pair; kohagi-rerank scores a pair with \
             one number, which only a one-label cross-encoder gives"
        );
        if let Some(names) = &self.architectures {
            if !names
                .iter()
                .any(|n| n.ends_with("ForSequenceClassification"))
            {
                remark!(
                    "warning: this checkpoint declares {names:?} rather than a \
                     sequence-classification model; scoring it assumes a head it may not have"
                );
            }
        }
        Ok(())
    }
}

/// `norm(act(dense(h)))` and then one linear down to a logit — HF's
/// `ModernBertPredictionHead` followed by the classifier.
///
/// Kept on the CPU whatever the encoder runs on: the hidden states come back
/// to host memory anyway, and one 768x768 matrix-vector product per pair is
/// nothing beside a 25-layer forward. That also keeps the head's arithmetic
/// identical across backends.
struct Head {
    dense: Linear,
    act: Activation,
    norm: candle_nn::LayerNorm,
    classifier: Linear,
}

/// One of the head's two optional biases, or `None`.
///
/// `config.json` decides, not the checkpoint: HF builds
/// `ModernBertPredictionHead`'s `Linear` and `LayerNorm` from `classifier_bias`
/// and `norm_bias`, both of which default to false, so a bias tensor a config
/// does not declare is one `from_pretrained` leaves unloaded as well. Following
/// the flag is what keeps the score equal to CrossEncoder's.
///
/// Hence the warning rather than silence: a dropped bias does not fail, it
/// scores slightly wrong. `from_pretrained` reports the same disagreement as
/// an unexpected key.
fn bias(
    vb: &VarBuilder,
    size: usize,
    declared: Option<bool>,
    tensor: &str,
    flag: &str,
) -> Result<Option<Tensor>> {
    if declared == Some(true) {
        return Ok(Some(vb.get(size, "bias").with_context(|| {
            format!("config.json sets `{flag}`, but this checkpoint has no `{tensor}`")
        })?));
    }
    if vb.contains_tensor("bias") {
        remark!(
            "warning: this checkpoint carries `{tensor}` but its config.json \
             does not set `{flag}`; scoring without it, as sentence-transformers would"
        );
    }
    Ok(None)
}

impl Head {
    fn load(vb: &VarBuilder, config: &Config, head: &HeadConfig) -> Result<Self> {
        let size = config.hidden_size;
        let act = Activation::from_name(head.classifier_activation.as_deref())
            .map_err(|e| anyhow::anyhow!("classifier_activation: {e}"))?;

        let dense_vb = vb.pp(names::DENSE);
        let weight = dense_vb.get((size, size), "weight").context(
            "this checkpoint has no `head.dense.weight`; it is an encoder without a \
                      classification head, so there is nothing to score a pair with",
        )?;
        let dense = Linear::new(
            weight,
            bias(
                &dense_vb,
                size,
                head.classifier_bias,
                "head.dense.bias",
                "classifier_bias",
            )?,
        );

        let norm_vb = vb.pp(names::NORM);
        let norm_weight = norm_vb.get(size, "weight")?;
        let norm = match bias(
            &norm_vb,
            size,
            head.norm_bias,
            "head.norm.bias",
            "norm_bias",
        )? {
            Some(b) => candle_nn::LayerNorm::new(norm_weight, b, config.layer_norm_eps),
            None => candle_nn::LayerNorm::new_no_bias(norm_weight, config.layer_norm_eps),
        };

        let cls = vb.pp(names::CLASSIFIER);
        let classifier = Linear::new(cls.get((1, size), "weight")?, Some(cls.get(1, "bias")?));
        Ok(Self {
            dense,
            act,
            norm,
            classifier,
        })
    }

    /// The raw logit for one pooled vector.
    ///
    /// A free-standing step on the head rather than on the [`Reranker`],
    /// because the CPU fan-out hands this to worker threads and a `Reranker`
    /// may hold a CoreML model, which is not shareable between them. The head
    /// is candle tensors and is.
    ///
    /// Sigmoid conversion happens after the backend work, once per run.
    fn logit(&self, pooled: &[f32]) -> Result<f32> {
        let x = Tensor::from_slice(pooled, (1, pooled.len()), &Device::Cpu)?;
        let x = self.dense.forward(&x)?;
        let x = match self.act {
            Activation::Gelu => x.gelu_erf()?,
            Activation::Silu => x.silu()?,
        };
        let x = self.norm.forward(&x)?;
        let x = self.classifier.forward(&x)?;
        Ok(x.flatten_all()?.to_vec1::<f32>()?[0])
    }
}

/// Which engine runs the encoder. The head is the same either way.
enum Engine {
    Candle {
        weights: Weights,
        device: Device,
    },
    #[cfg(feature = "coreml")]
    CoreMl(crate::coreml::CoreMlEncoder),
}

/// A loaded cross-encoder. One instance can score any number of pairs.
pub struct Reranker {
    engine: Engine,
    head: Head,
    tokenizer: Tokenizer,
    opts: Options,
    pooling: Pooling,
    dim: usize,
    /// The checkpoint's digest on the candle path. `None` on the CoreML path,
    /// which reports the bundle's recorded provenance instead.
    fingerprint: Option<Fingerprint>,
    /// What the checkpoint declares it can take. A cross-encoder rarely ships
    /// `sentence_bert_config.json` at all, so this is usually `None`.
    declared_max_seq: Option<usize>,
}

impl Reranker {
    pub fn load(source: &ModelSource, opts: Options) -> Result<Self> {
        if opts.backend == Backend::CoreML {
            return Self::load_coreml(source, opts);
        }
        // Refused rather than quietly served on the CPU. The Vulkan engine
        // exists for the embedder only: a cross-encoder reduces its hidden
        // states differently (`classifier_pooling`, not `1_Pooling`), so it
        // needs its own head on that path rather than the same one.
        anyhow::ensure!(
            opts.backend != Backend::Vulkan,
            "--device vulkan is not implemented for reranking yet; use --device cpu"
        );
        // Cross-encoders use `classifier_pooling`, not `1_Pooling`.
        let files = source.checkpoint_files()?;

        let config_path = files
            .weights
            .parent()
            .map(|d| d.join("config.json"))
            .context("model path has no parent dir for config.json")?;
        let config = resolve_config(&config_path)?;

        check_max_seq(opts.max_seq_length, config.encoder.max_position_embeddings)?;
        check_precision(opts.backend, opts.precision)?;

        let device = open_device(opts.backend)?;
        let fingerprint = Fingerprint::spawn(&files.weights);
        let weights = load_weights(&files.weights, &config.encoder, &device, opts.precision)?;
        fingerprint.confirm(&files.weights);
        // A second view of the same file, on the CPU: the head runs there
        // whatever the encoder runs on.
        let head = Head::load(&cpu_weights(&files.weights)?, &config.encoder, &config.head)?;
        let tokenizer = load_tokenizer(&files.tokenizer, opts.max_seq_length)?;

        Ok(Self {
            engine: Engine::Candle { weights, device },
            head,
            tokenizer,
            opts,
            pooling: config.pooling,
            dim: config.encoder.hidden_size,
            fingerprint: Some(fingerprint),
            declared_max_seq: files.declared_max_seq,
        })
    }

    /// The Neural Engine path: a converted bundle for the encoder, and the head
    /// from the `head.safetensors` the converter wrote beside it.
    ///
    /// The split is the point. A CoreML bundle is fp16 and fixed-shape, which
    /// suits 25 layers of encoder and does not suit four small tensors whose
    /// output is the number being thresholded, so the head stays in f32 on the
    /// CPU — the same code the other backends run.
    #[cfg(feature = "coreml")]
    fn load_coreml(source: &ModelSource, opts: Options) -> Result<Self> {
        let dir = source.resolve_coreml(opts.coreml_form)?.dir;

        let config = resolve_config(&dir.join("config.json"))?;

        // Before loading the encoder, which compiles the bundle on a first run:
        // a bundle that cannot score a pair should say so in a second, not
        // after twenty.
        let head_path = dir.join(crate::config::COREML_HEAD_FILE);
        // An `UnsupportedRequest`, so this exits 3 like every other "the CoreML
        // backend cannot serve this" — the message tells the caller to retry on
        // the CPU, and exit 3 is how a caller learns that without reading it.
        if !head_path.is_file() {
            return Err(UnsupportedRequest::new(format!(
                "{} has no {}, so it holds an encoder with no way to score a pair. Bundles \
                 converted before Kohagi 0.6 do not carry one: reconvert the checkpoint, or \
                 score on --device cpu",
                dir.display(),
                crate::config::COREML_HEAD_FILE
            ))
            .into());
        }
        let head = Head::load(&cpu_weights(&head_path)?, &config.encoder, &config.head)?;

        let encoder = crate::coreml::CoreMlEncoder::load(&dir, config.encoder.hidden_size)?;
        if opts.max_seq_length > encoder.max_bucket() {
            return Err(UnsupportedRequest::new(format!(
                "--max-seq-length {} exceeds the largest converted CoreML bucket ({}); \
                 lower it or convert a longer model",
                opts.max_seq_length,
                encoder.max_bucket()
            ))
            .into());
        }

        let tokenizer = load_tokenizer(&dir.join("tokenizer.json"), opts.max_seq_length)?;

        Ok(Self {
            engine: Engine::CoreMl(encoder),
            head,
            tokenizer,
            opts,
            pooling: config.pooling,
            dim: config.encoder.hidden_size,
            // A bundle has no safetensors of its own; its provenance is what it
            // recorded at conversion, read by `info`.
            fingerprint: None,
            declared_max_seq: crate::source::declared_max_seq_in_dir(&dir),
        })
    }

    #[cfg(not(feature = "coreml"))]
    fn load_coreml(_source: &ModelSource, _opts: Options) -> Result<Self> {
        Err(UnsupportedRequest::new(
            "this binary was built without CoreML support; rebuild with \
             `cargo build --release --features coreml`",
        )
        .into())
    }

    /// What this model is, for a summary line or a results file.
    pub fn info(&self) -> ModelInfo {
        #[allow(unused_mut)]
        let mut info = ModelInfo {
            backend: self.opts.backend.name(),
            precision: self.opts.precision.name(),
            sha256: self.fingerprint.as_ref().and_then(|f| f.get()),
            bundle: None,
            pooling: self.pooling.name(),
            dim: self.dim,
            max_seq_length: self.opts.max_seq_length,
            declared_max_seq_length: self.declared_max_seq,
            // Scores have no output dimension, but must record their scale.
            output: Output::Score {
                score: if self.opts.sigmoid {
                    "sigmoid"
                } else {
                    "logit"
                },
            },
        };
        #[cfg(feature = "coreml")]
        if let Engine::CoreMl(encoder) = &self.engine {
            info.add_bundle(encoder);
        }
        info
    }

    /// Score `(query, text)` pairs, one number per pair, in input order.
    ///
    /// The [`TokenInfo`] alongside is the pair's, not either half's: a
    /// truncated pair lost the tail of whichever side was longer, and the
    /// score was formed without it.
    pub fn score(&self, pairs: &[(&str, &str)]) -> Result<(Vec<f32>, Vec<TokenInfo>)> {
        if pairs.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let (logits, info) = match &self.engine {
            Engine::Candle { weights, device } => self.logits_candle(pairs, weights, device)?,
            #[cfg(feature = "coreml")]
            Engine::CoreMl(encoder) => self.logits_coreml(pairs, encoder)?,
        };
        // Convert logits after backend work so workers only compute logits.
        let scores = if self.opts.sigmoid {
            logits.into_iter().map(sigmoid).collect()
        } else {
            logits
        };
        Ok((scores, info))
    }

    fn logits_candle(
        &self,
        pairs: &[(&str, &str)],
        weights: &Weights,
        device: &Device,
    ) -> Result<(Vec<f32>, Vec<TokenInfo>)> {
        let encodings = encode_pairs(&self.tokenizer, pairs)?;
        let (batches, info) = bucket_encodings(&encodings, self.opts.batch_size);
        let (head, pooling) = (&self.head, self.pooling);
        let logits = run_batches(
            weights,
            device,
            self.opts.backend,
            &batches,
            pairs.len(),
            |hidden, mask, dim| head.logit(&pool_row(hidden, mask, dim, pooling)),
        )?;
        Ok((logits, info))
    }

    /// The ANE path: one fixed-shape, batch=1 forward per pair, routed to the
    /// smallest converted bucket it fits.
    #[cfg(feature = "coreml")]
    fn logits_coreml(
        &self,
        pairs: &[(&str, &str)],
        encoder: &crate::coreml::CoreMlEncoder,
    ) -> Result<(Vec<f32>, Vec<TokenInfo>)> {
        let encodings = encode_pairs(&self.tokenizer, pairs)?;
        let info: Vec<TokenInfo> = encodings.iter().map(crate::batch::token_info).collect();
        let (head, pooling) = (&self.head, self.pooling);
        let logits = encoder.run_rows(&encodings, |hidden, mask, dim| {
            head.logit(&pool_row(hidden, mask, dim, pooling))
        })?;
        Ok((logits, info))
    }
}

/// Converts a logit to the 0..1 score returned by `CrossEncoder.predict`.
///
/// Uses f64 to avoid underflow for small logits.
fn sigmoid(logit: f32) -> f32 {
    (1.0 / (1.0 + f64::from(-logit).exp())) as f32
}

/// Parsed and validated cross-encoder configuration.
struct ScoringConfig {
    /// Encoder configuration.
    encoder: Config,
    /// Classifier head configuration.
    head: HeadConfig,
    /// Classifier pooling.
    pooling: Pooling,
}

/// Reads, parses, and validates a cross-encoder configuration.
fn resolve_config(path: &std::path::Path) -> Result<ScoringConfig> {
    // One read, two views: the encoder's and the head's.
    let json = read_config_json(path)?;
    let encoder: Config = parse_config(json.clone(), path)?;
    let head: HeadConfig = parse_config(json, path)?;

    head.validate()?;
    let pooling = head.pooling()?;
    Ok(ScoringConfig {
        encoder,
        head,
        pooling,
    })
}

/// A CPU f32 view of a safetensors file, for the head.
fn cpu_weights(path: &std::path::Path) -> Result<VarBuilder<'static>> {
    Ok(unsafe {
        VarBuilder::from_mmaped_safetensors(
            std::slice::from_ref(&path.to_path_buf()),
            candle_core::DType::F32,
            &Device::Cpu,
        )?
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(extra: &str) -> HeadConfig {
        serde_json::from_str(&format!("{{\"hidden_size\": 768{extra}}}")).expect("a config")
    }

    /// The label count decides whether this binary can say anything at all, and
    /// checkpoints spell it two ways.
    #[test]
    fn a_reranker_is_recognized_by_its_one_label() {
        assert_eq!(config("").labels(), 1);
        assert_eq!(config(r#", "num_labels": 1"#).labels(), 1);
        // What the Ruri and japanese-reranker checkpoints actually carry.
        assert_eq!(config(r#", "id2label": {"0": "LABEL_0"}"#).labels(), 1);
        assert_eq!(
            config(r#", "id2label": {"0": "yes", "1": "no", "2": "maybe"}"#).labels(),
            3
        );
    }

    #[test]
    fn classifier_pooling_defaults_to_cls_and_refuses_what_it_cannot_do() {
        assert_eq!(config("").pooling().unwrap(), Pooling::Cls);
        assert_eq!(
            config(r#", "classifier_pooling": "cls""#)
                .pooling()
                .unwrap(),
            Pooling::Cls
        );
        assert_eq!(
            config(r#", "classifier_pooling": "mean""#)
                .pooling()
                .unwrap(),
            Pooling::Mean
        );
        assert!(config(r#", "classifier_pooling": "last""#)
            .pooling()
            .is_err());
    }

    fn head_weights(with_bias: bool) -> VarBuilder<'static> {
        let mut tensors = std::collections::HashMap::new();
        let one = |n: usize| Tensor::zeros(n, candle_core::DType::F32, &Device::Cpu).unwrap();
        tensors.insert("head.dense.weight".to_string(), one(4));
        if with_bias {
            tensors.insert("head.dense.bias".to_string(), one(4));
        }
        VarBuilder::from_tensors(tensors, candle_core::DType::F32, &Device::Cpu)
    }

    #[test]
    fn a_head_bias_follows_the_config_and_says_when_the_weights_disagree() {
        let declared = |vb: &VarBuilder, d: Option<bool>| {
            bias(
                &vb.pp("head.dense"),
                4,
                d,
                "head.dense.bias",
                "classifier_bias",
            )
        };

        // Both halves say yes, and both say no.
        assert!(declared(&head_weights(true), Some(true)).unwrap().is_some());
        assert!(declared(&head_weights(false), None).unwrap().is_none());
        assert!(declared(&head_weights(false), Some(false))
            .unwrap()
            .is_none());

        // The tensor is there and the config never asked for it: not loaded,
        // and warned about on stderr.
        assert!(declared(&head_weights(true), None).unwrap().is_none());
        assert!(declared(&head_weights(true), Some(false))
            .unwrap()
            .is_none());

        // The config asked for one that is not there: nothing to load, so this
        // stops.
        let e = declared(&head_weights(false), Some(true)).unwrap_err();
        assert!(
            format!("{e:#}").contains("classifier_bias"),
            "the error should name the flag: {e:#}"
        );
    }
}
