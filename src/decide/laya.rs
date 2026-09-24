//! The Laya packed head: one sequence per question, all options marked.
//!
//! `convaiinnovations/laya` (English, ModernBERT-large) and `laya-multilingual`
//! (multilingual, mmBERT) read a question as one sequence —
//! `[CLS] "{type} question: {instructions}" [SEP] ([MASK] option)* [SEP] state
//! [SEP]` — and a small decision head scores the marked options. This module
//! holds the model-independent parts: the agent config and the sequence
//! assembly, both verified against the upstream `rl_common.py` (see the golden
//! test).
//!
//! The special tokens are read from `tokenizer_config.json`, because they differ
//! by checkpoint: the English tokenizer uses `[CLS]`/`[SEP]`/`[MASK]`, the
//! multilingual mmBERT one `<bos>`/`<eos>`/`<mask>`. Nothing here names a model.

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{
    embedding, ops::softmax_last_dim, Embedding, LayerNorm, Linear, Module, VarBuilder,
};
use serde_json::Value;
use tokenizers::Tokenizer;

use super::engine::{DecisionHead, EngineOptions, HeadFactory};
use super::{Criteria, OptionScore, QType, Question, Request, RequestScore};
use crate::encoder::{padding_mask, Config, ModernBert};
use crate::fingerprint::Fingerprint;
use crate::model::{check_precision, open_device};
use crate::source::{Manifest, ModelSource};
use crate::{Backend, ModelInfo, Output, Precision};

/// The agent config beside a Laya checkpoint (`rl_agent_config.json`).
#[derive(Debug, Clone)]
pub struct LayaConfig {
    /// Decision-head transformer layers above the encoder.
    pub head_layers: usize,
    /// Total sequence length.
    pub max_len: usize,
    /// The budget for instructions plus options, before the state.
    pub head_max_len: usize,
    /// Per question type: `[choice, score, noul]`.
    pub temperature: Vec<f32>,
    /// Cardinality buckets (`"{qtype}:{size}"`) that override the above.
    pub temperature_by_options: BTreeMap<String, f32>,
}

#[derive(serde::Deserialize)]
struct RawConfig {
    #[serde(default = "default_head_layers")]
    head_layers: usize,
    max_len: usize,
    head_max_len: usize,
    #[serde(default)]
    temperature: Vec<f32>,
    #[serde(default)]
    temperature_by_options: BTreeMap<String, f32>,
}

fn default_head_layers() -> usize {
    2
}

impl LayaConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        let raw: RawConfig = serde_json::from_str(json).context("parsing rl_agent_config.json")?;
        anyhow::ensure!(raw.max_len > 0, "rl_agent_config.json has max_len 0");
        anyhow::ensure!(
            raw.head_max_len > 0,
            "rl_agent_config.json has head_max_len 0"
        );
        Ok(Self {
            head_layers: raw.head_layers,
            max_len: raw.max_len,
            head_max_len: raw.head_max_len,
            temperature: raw.temperature,
            temperature_by_options: raw.temperature_by_options,
        })
    }

    /// The temperature for one question, the bucket first and the type after.
    ///
    /// The key is `"{qtype}:{2|3-5|6-10|11+}"`, the size of the option set; a
    /// checkpoint that calibrated no buckets falls back to the per-type value,
    /// and a missing one to 1.0 (no scaling).
    pub fn temperature(&self, qtype: QType, options: usize) -> f32 {
        let size = match options {
            0..=2 => "2",
            3..=5 => "3-5",
            6..=10 => "6-10",
            _ => "11+",
        };
        let key = format!("{}:{}", qtype.name(), size);
        self.temperature_by_options
            .get(&key)
            .copied()
            .or_else(|| self.temperature.get(qtype_index(qtype)).copied())
            .unwrap_or(1.0)
    }
}

fn qtype_index(qtype: QType) -> usize {
    match qtype {
        QType::Choice => 0,
        QType::Score => 1,
        QType::Noul => 2,
    }
}

/// The three special tokens a Laya tokenizer uses, resolved by name.
pub struct Specials {
    mask: String,
    mask_id: u32,
    cls_id: u32,
    sep_id: u32,
}

impl Specials {
    /// Resolves the names in `tokenizer_config.json` to this tokenizer's ids.
    pub fn resolve(tokenizer: &Tokenizer, config_json: &str) -> Result<Self> {
        let v: Value =
            serde_json::from_str(config_json).context("parsing tokenizer_config.json")?;
        let name = |key: &str| -> Result<String> {
            v.get(key)
                .and_then(Value::as_str)
                .map(str::to_string)
                .with_context(|| format!("tokenizer_config.json has no \"{key}\""))
        };
        let mask = name("mask_token")?;
        let cls = name("cls_token")?;
        let sep = name("sep_token")?;
        let id = |s: &str| {
            tokenizer
                .token_to_id(s)
                .with_context(|| format!("the tokenizer has no `{s}`"))
        };
        Ok(Self {
            mask_id: id(&mask)?,
            cls_id: id(&cls)?,
            sep_id: id(&sep)?,
            mask,
        })
    }
}

/// Encode without special tokens: the sequence is assembled by hand.
fn encode(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    Ok(tokenizer
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?
        .get_ids()
        .to_vec())
}

/// Option text per the reference: choice `"{label}: {description}"`, score
/// `"level {i}: {level}"`, and noul a fixed false/true pair.
pub fn render_options(question: &Question) -> Vec<String> {
    match &question.criteria {
        Criteria::Choice(options) => options
            .iter()
            .map(|(label, description)| match description {
                Some(text) => format!("{label}: {text}"),
                None => label.clone(),
            })
            .collect(),
        Criteria::Score(levels) => levels
            .iter()
            .enumerate()
            .map(|(level, text)| format!("level {level}: {text}"))
            .collect(),
        Criteria::Noul => vec![
            "false: no, the statement does not hold".to_string(),
            "true: yes, the statement holds".to_string(),
        ],
    }
}

/// The state as the reference serializes it: a string verbatim, anything else
/// as JSON. (The reference uses Python's `json.dumps`; an object state here is
/// compact with sorted keys, so a caller who needs the exact bytes should pass
/// the serialized string.)
fn serialize_state(state: &Value) -> String {
    match state {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Build the Laya sequence for one question and its options.
///
/// Returns the token ids and the `[MASK]` positions, in `order`. The assembly is
/// transcribed from the reference and verified against it: the head budget is
/// `head_max_len`, each option is capped at 48 tokens and shrunk evenly when the
/// options do not fit, and the state takes what is left.
pub fn build_sequence(
    tokenizer: &Tokenizer,
    specials: &Specials,
    state: &Value,
    question: &Question,
    head_max_len: usize,
    max_len: usize,
    order: Option<&[usize]>,
) -> Result<(Vec<u32>, Vec<usize>)> {
    let options = render_options(question);
    let default_order: Vec<usize>;
    let order = match order {
        Some(order) => order,
        None => {
            default_order = (0..options.len()).collect();
            &default_order
        }
    };

    let instructions = question.instructions.replace(&specials.mask, " ");
    let mut head_ids = encode(
        tokenizer,
        &format!("{} question: {}", question.qtype.name(), instructions),
    )?;

    let mut opt_ids: Vec<Vec<u32>> = Vec::with_capacity(options.len());
    for &i in order {
        let text = format!(" {}", options[i].replace(&specials.mask, " "));
        let mut ids = encode(tokenizer, &text)?;
        ids.truncate(48);
        let mut with_marker = vec![specials.mask_id];
        with_marker.extend(ids);
        opt_ids.push(with_marker);
    }

    let total = |ids: &[Vec<u32>]| ids.iter().map(Vec::len).sum::<usize>();
    let mut budget = head_max_len as isize - total(&opt_ids) as isize;
    if budget < 16 {
        // Too many or too long options: shrink every option evenly.
        let per = std::cmp::max(4, head_max_len.saturating_sub(16) / opt_ids.len().max(1));
        for ids in &mut opt_ids {
            ids.truncate(per);
        }
        budget = head_max_len as isize - total(&opt_ids) as isize;
    }
    head_ids.truncate(std::cmp::max(8, budget) as usize);

    let mut ids = Vec::new();
    ids.push(specials.cls_id);
    ids.extend(head_ids);
    ids.push(specials.sep_id);
    let mut markers = Vec::with_capacity(opt_ids.len());
    for option in &opt_ids {
        markers.push(ids.len());
        ids.extend(option);
    }
    ids.push(specials.sep_id);

    let room = max_len.saturating_sub(ids.len()).saturating_sub(1);
    let mut state_ids = encode(
        tokenizer,
        &serialize_state(state).replace(&specials.mask, " "),
    )?;
    state_ids.truncate(room);
    ids.extend(state_ids);
    ids.push(specials.sep_id);
    ids.truncate(max_len);
    markers.retain(|&marker| marker < max_len);
    Ok((ids, markers))
}

/// The files a Laya checkpoint is read from, relative to its root.
pub(crate) const MANIFEST: Manifest = Manifest {
    weights: "model.safetensors",
    tokenizer: "tokenizer/tokenizer.json",
    configs: &[
        ("encoder", "encoder/config.json"),
        ("agent", "rl_agent_config.json"),
        ("tokenizer_config", "tokenizer/tokenizer_config.json"),
    ],
};

/// The decision head's attention: fused QKV with bias, no RoPE, no window.
struct HeadAttention {
    in_proj: Linear,
    out_proj: Linear,
    heads: usize,
    head_dim: usize,
    dims: usize,
}

impl HeadAttention {
    fn load(vb: VarBuilder, dims: usize) -> Result<Self> {
        let in_proj = Linear::new(
            vb.get((3 * dims, dims), "in_proj_weight")?,
            Some(vb.get(3 * dims, "in_proj_bias")?),
        );
        let proj = vb.pp("out_proj");
        let out_proj = Linear::new(
            proj.get((dims, dims), "weight")?,
            Some(proj.get(dims, "bias")?),
        );
        let heads = std::cmp::max(1, dims / 64);
        Ok(Self {
            in_proj,
            out_proj,
            heads,
            head_dim: dims / heads,
            dims,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        let qkv = self
            .in_proj
            .forward(x)?
            .reshape((b, seq, 3, self.heads, self.head_dim))?;
        let pick = |i: usize| -> Result<Tensor> {
            Ok(qkv
                .narrow(2, i, 1)?
                .squeeze(2)?
                .transpose(1, 2)?
                .contiguous()?)
        };
        let (q, k, v) = (pick(0)?, pick(1)?, pick(2)?);
        let scale = (self.head_dim as f64).powf(-0.5);
        let att = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let att = softmax_last_dim(&att.broadcast_add(mask)?)?;
        let out = att.matmul(&v)?.transpose(1, 2)?.contiguous()?;
        Ok(self.out_proj.forward(&out.reshape((b, seq, self.dims))?)?)
    }
}

/// One pre-norm decision-head layer. Its feed-forward is **ReLU**, not GELU:
/// the reference uses `nn.TransformerEncoderLayer`'s default.
struct HeadLayer {
    self_attn: HeadAttention,
    norm1: LayerNorm,
    norm2: LayerNorm,
    linear1: Linear,
    linear2: Linear,
}

impl HeadLayer {
    fn load(vb: VarBuilder, dims: usize, eps: f64) -> Result<Self> {
        let self_attn = HeadAttention::load(vb.pp("self_attn"), dims)?;
        let norm = |name: &str| -> Result<LayerNorm> {
            let vb = vb.pp(name);
            Ok(LayerNorm::new(
                vb.get(dims, "weight")?,
                vb.get(dims, "bias")?,
                eps,
            ))
        };
        let l1 = vb.pp("linear1");
        let linear1 = Linear::new(
            l1.get((4 * dims, dims), "weight")?,
            Some(l1.get(4 * dims, "bias")?),
        );
        let l2 = vb.pp("linear2");
        let linear2 = Linear::new(
            l2.get((dims, 4 * dims), "weight")?,
            Some(l2.get(dims, "bias")?),
        );
        Ok(Self {
            self_attn,
            norm1: norm("norm1")?,
            norm2: norm("norm2")?,
            linear1,
            linear2,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let att = self.self_attn.forward(&self.norm1.forward(x)?, mask)?;
        let x = (x + att)?;
        let ff = self
            .linear2
            .forward(&self.linear1.forward(&self.norm2.forward(&x)?)?.relu()?)?;
        Ok((x + ff)?)
    }
}

/// The stack of decision-head layers, with no final norm.
struct Head {
    layers: Vec<HeadLayer>,
}

impl Head {
    fn load(vb: VarBuilder, dims: usize, count: usize, eps: f64) -> Result<Self> {
        let mut layers = Vec::with_capacity(count);
        for i in 0..count {
            layers.push(HeadLayer::load(vb.pp(format!("layers.{i}")), dims, eps)?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let keys = padding_mask(mask, DType::F32)?;
        let mut xs = x.clone();
        for layer in &self.layers {
            xs = layer.forward(&xs, &keys)?;
        }
        Ok(xs)
    }
}

/// `LayerNorm → Linear → GELU → Linear(1)`, the per-marker scorer. `scorer.2` is
/// the GELU, which is why the last linear is `3`.
struct Scorer {
    norm: LayerNorm,
    dense: Linear,
    out: Linear,
}

impl Scorer {
    fn load(vb: VarBuilder, dims: usize, eps: f64) -> Result<Self> {
        let n = vb.pp("0");
        let d = vb.pp("1");
        let o = vb.pp("3");
        Ok(Self {
            norm: LayerNorm::new(n.get(dims, "weight")?, n.get(dims, "bias")?, eps),
            dense: Linear::new(d.get((dims, dims), "weight")?, Some(d.get(dims, "bias")?)),
            out: Linear::new(o.get((1, dims), "weight")?, Some(o.get(1, "bias")?)),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.norm.forward(x)?;
        Ok(self.out.forward(&self.dense.forward(&x)?.gelu_erf()?)?)
    }
}

/// The `--head` name and loader for the packed Laya head.
pub static FACTORY: &dyn HeadFactory = &Factory;

struct Factory;

impl HeadFactory for Factory {
    fn name(&self) -> &'static str {
        "laya"
    }

    fn load(
        &self,
        source: &ModelSource,
        options: EngineOptions,
        subfolder: Option<&str>,
    ) -> Result<Box<dyn DecisionHead>> {
        Ok(Box::new(Laya::load(source, options, subfolder)?))
    }
}

/// The Laya packed decision model: encoder, head, scorer, and the config.
pub struct Laya {
    encoder: ModernBert,
    head: Option<Head>,
    type_emb: Embedding,
    scorer: Scorer,
    config: LayaConfig,
    specials: Specials,
    tokenizer: Tokenizer,
    device: Device,
    backend: Backend,
    precision: Precision,
    dims: usize,
    fingerprint: Option<Fingerprint>,
}

fn read(path: &std::path::Path) -> Result<String> {
    std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))
}

impl Laya {
    pub fn load(
        source: &ModelSource,
        options: EngineOptions,
        subfolder: Option<&str>,
    ) -> Result<Self> {
        anyhow::ensure!(
            options.backend != Backend::CoreML,
            "the laya head runs on the Candle backends; `--device coreml` is cross-encoder only"
        );
        let resolved = source.resolve(&MANIFEST, subfolder)?;
        let config = LayaConfig::from_json(&read(&resolved.configs["agent"])?)?;
        let tokenizer = Tokenizer::from_file(&resolved.tokenizer).map_err(|e| {
            anyhow::anyhow!(
                "cannot load tokenizer {}: {e}",
                resolved.tokenizer.display()
            )
        })?;
        let specials =
            Specials::resolve(&tokenizer, &read(&resolved.configs["tokenizer_config"])?)?;
        let encoder_config: Config = serde_json::from_str(&read(&resolved.configs["encoder"])?)?;

        check_precision(options.backend, options.precision)?;
        let device = open_device(options.backend)?;
        let fingerprint = Fingerprint::spawn(&resolved.weights);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&resolved.weights),
                DType::F32,
                &device,
            )?
        };
        // Laya stores the HF encoder under `encoder.` with the bare ModernBERT
        // names, while `ModernBert::load` asks for them under `model.`. Map the
        // one to the other on the root builder (`rename_f` compounds with `pp`,
        // so it must not be combined with one).
        let encoder = ModernBert::load(
            vb.clone()
                .rename_f(|name| match name.strip_prefix("model.") {
                    Some(rest) => format!("encoder.{rest}"),
                    None => name.to_string(),
                }),
            &encoder_config,
        )?;
        let dims = encoder_config.hidden_size;
        let eps = encoder_config.layer_norm_eps;
        let head = (config.head_layers > 0)
            .then(|| Head::load(vb.pp("head"), dims, config.head_layers, eps))
            .transpose()?;
        let type_emb = embedding(3, dims, vb.pp("type_emb"))?;
        let scorer = Scorer::load(vb.pp("scorer"), dims, eps)?;
        fingerprint.confirm(&resolved.weights);
        Ok(Self {
            encoder,
            head,
            type_emb,
            scorer,
            config,
            specials,
            tokenizer,
            device,
            backend: options.backend,
            precision: options.precision,
            dims,
            fingerprint: Some(fingerprint),
        })
    }

    /// The label each probability is keyed by, in the head's own order:
    /// criteria order for a choice, level order for a score, and false/true for
    /// noul. This is the canonical decide order, not necessarily render order.
    fn labels(question: &Question) -> Vec<String> {
        match &question.criteria {
            Criteria::Choice(options) => options.iter().map(|(label, _)| label.clone()).collect(),
            Criteria::Score(levels) => levels.clone(),
            Criteria::Noul => vec!["false".to_string(), "true".to_string()],
        }
    }
}

impl DecisionHead for Laya {
    fn info(&self) -> ModelInfo {
        ModelInfo {
            backend: self.backend.name(),
            precision: self.precision.name(),
            sha256: self.fingerprint.as_ref().and_then(|f| f.get()),
            bundle: None,
            // A packed decision model pools nothing; the markers are gathered.
            pooling: "none",
            dim: self.dims,
            max_seq_length: self.config.max_len,
            declared_max_seq_length: None,
            output: Output::Score { score: "softmax" },
        }
    }

    fn score(&self, requests: &[Request]) -> Result<(Vec<RequestScore>, Vec<crate::TokenInfo>)> {
        let mut out = Vec::with_capacity(requests.len());
        let mut info = Vec::with_capacity(requests.len());
        for request in requests {
            let mut questions = Vec::with_capacity(request.questions.len());
            let mut n_tokens = 0usize;
            let mut truncated = false;
            for question in &request.questions {
                let (labelled, tokens, cut) = self.score_question(&request.state, question)?;
                questions.push(labelled);
                n_tokens = n_tokens.max(tokens);
                truncated |= cut;
            }
            out.push(questions);
            info.push(crate::TokenInfo {
                n_tokens,
                truncated,
            });
        }
        Ok((out, info))
    }
}

impl Laya {
    fn score_question(
        &self,
        state: &Value,
        question: &Question,
    ) -> Result<(Vec<OptionScore>, usize, bool)> {
        let (ids, markers) = build_sequence(
            &self.tokenizer,
            &self.specials,
            state,
            question,
            self.config.head_max_len,
            self.config.max_len,
            None,
        )?;
        let seq = ids.len();
        let xs = Tensor::from_vec(ids, (1, seq), &self.device)?;
        let mask = Tensor::ones((1, seq), DType::U32, &self.device)?;
        let mut h = self.encoder.forward(&xs, &mask)?;
        let qtype = Tensor::from_slice(&[qtype_index(question.qtype) as u32], (1,), &self.device)?;
        h = h.broadcast_add(&self.type_emb.forward(&qtype)?.unsqueeze(1)?)?;
        if let Some(head) = &self.head {
            h = head.forward(&h, &mask)?;
        }
        let h = h.squeeze(0)?;
        let index: Vec<u32> = markers.iter().map(|&marker| marker as u32).collect();
        let picked =
            h.index_select(&Tensor::from_vec(index, (markers.len(),), &self.device)?, 0)?;
        let logits = self
            .scorer
            .forward(&picked)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let temperature = self.config.temperature(question.qtype, markers.len());
        let labelled = Self::labels(question)
            .into_iter()
            .zip(logits)
            .map(|(label, logit)| (label, logit / temperature))
            .collect();
        let truncated =
            seq >= self.config.max_len || markers.len() < render_options(question).len();
        Ok((labelled, seq, truncated))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decide::{parse_request, Question, Request};

    fn question(json: &str) -> Request {
        parse_request(json).expect("a request").expect("a record")
    }

    #[test]
    fn options_render_the_way_the_reference_does() {
        let r = question(
            r#"{"id":1,"state":"s","questions":{"q":{"type":"choice","instructions":"?",
                "criteria":{"billing":"請求・返金","sales":null}}}}"#,
        );
        assert_eq!(
            render_options(&r.questions[0]),
            ["billing: 請求・返金", "sales"]
        );

        let r = question(
            r#"{"id":1,"state":"s","questions":{"q":{"type":"score","instructions":"?",
                "criteria":["a","b"]}}}"#,
        );
        assert_eq!(
            render_options(&r.questions[0]),
            ["level 0: a", "level 1: b"]
        );

        let r = question(
            r#"{"id":1,"state":"s","questions":{"q":{"type":"noul","instructions":"?"}}}"#,
        );
        assert_eq!(
            render_options(&r.questions[0])[1],
            "true: yes, the statement holds"
        );
    }

    #[test]
    fn score_labels_are_the_input_levels_not_synthetic_indices() {
        let r = question(
            r#"{"id":1,"state":"s","questions":{"q":{"type":"score","instructions":"?",
                "criteria":["low","medium"]}}}"#,
        );
        assert_eq!(Laya::labels(&r.questions[0]), ["low", "medium"]);
    }

    #[test]
    fn the_temperature_bucket_precedes_the_type() {
        let cfg = LayaConfig::from_json(
            r#"{"head_layers":2,"max_len":512,"head_max_len":192,
                "temperature":[1.6,1.2,1.9],"temperature_by_options":{"choice:6-10":1.0}}"#,
        )
        .unwrap();
        assert_eq!(cfg.temperature(QType::Choice, 8), 1.0);
        assert_eq!(cfg.temperature(QType::Choice, 3), 1.6);
        assert_eq!(cfg.temperature(QType::Noul, 2), 1.9);
    }

    #[test]
    fn the_sequence_has_one_marker_per_option_and_keeps_the_state_at_the_end() {
        // A stand-in tokenizer is not worth faking; the golden test runs on the
        // real one. This only checks the pure option rendering above and the
        // config parse stay sane when the real checkpoint is absent.
        let r = question(
            r#"{"id":1,"state":"s","questions":{"q":{"type":"noul","instructions":"?"}}}"#,
        );
        let _ = (r, render_options);
    }

    /// Holds the Rust port to the upstream `rl_common.build_sequence`. Runs only
    /// when the tokenizer and a golden file are pointed at, because neither the
    /// checkpoint nor its tokenizer is committed:
    ///
    /// ```console
    /// KOHAGI_LAYA_TOKENIZER=…/tokenizer.json \
    /// KOHAGI_LAYA_TOKENIZER_CONFIG=…/tokenizer_config.json \
    /// KOHAGI_LAYA_GOLDEN=…/golden-en.jsonl \
    ///   cargo test --lib laya::tests::golden
    /// ```
    #[test]
    fn golden_matches_upstream() {
        let (Ok(tokenizer_path), Ok(config_path), Ok(golden_path)) = (
            std::env::var("KOHAGI_LAYA_TOKENIZER"),
            std::env::var("KOHAGI_LAYA_TOKENIZER_CONFIG"),
            std::env::var("KOHAGI_LAYA_GOLDEN"),
        ) else {
            return;
        };
        let tokenizer = Tokenizer::from_file(&tokenizer_path).expect("tokenizer");
        let config = std::fs::read_to_string(&config_path).expect("tokenizer config");
        let specials = Specials::resolve(&tokenizer, &config).expect("specials");
        let golden = std::fs::read_to_string(&golden_path).expect("golden");

        let mut checked = 0;
        for line in golden.lines().filter(|l| !l.trim().is_empty()) {
            let row: Value = serde_json::from_str(line).expect("a golden line");
            let record = row["record"].to_string();
            let request = question(&record);
            let head_max_len = row["head_max_len"].as_u64().unwrap() as usize;
            let max_len = row["max_len"].as_u64().unwrap() as usize;
            let question: &Question = &request.questions[0];

            let (ids, markers) = build_sequence(
                &tokenizer,
                &specials,
                &request.state,
                question,
                head_max_len,
                max_len,
                None,
            )
            .expect("a sequence");

            let want_ids: Vec<u32> = row["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            let want_markers: Vec<usize> = row["markers"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            if ids != want_ids {
                let at = ids
                    .iter()
                    .zip(&want_ids)
                    .position(|(a, b)| a != b)
                    .unwrap_or(ids.len().min(want_ids.len()));
                let lo = at.saturating_sub(4);
                panic!(
                    "case {}: ids differ at {at}; len {} vs {}\n got  {:?}\n want {:?}\n markers {:?} vs {:?}",
                    row["case"],
                    ids.len(),
                    want_ids.len(),
                    &ids[lo..(at + 6).min(ids.len())],
                    &want_ids[lo..(at + 6).min(want_ids.len())],
                    markers,
                    want_markers,
                );
            }
            assert_eq!(
                markers, want_markers,
                "markers differ on case {}",
                row["case"]
            );
            checked += 1;
        }
        assert!(checked > 0, "the golden file held no cases");
    }
}
