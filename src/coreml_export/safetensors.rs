//! Reading a checkpoint's weights for the emitter.
//!
//! Through candle, which is already how the inference path reads the same file —
//! one of the reasons to emit from Rust at all. A tensor is converted to `f32`
//! here and rounded to fp16 by
//! [`super::blob`], which is the one place precision is decided.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

use super::encoder::Weights;

pub struct Checkpoint {
    tensors: HashMap<String, Tensor>,
}

impl Checkpoint {
    /// Load one `model.safetensors`.
    pub fn open(path: &Path) -> Result<Self> {
        let tensors = candle_core::safetensors::load(path, &Device::Cpu)
            .with_context(|| format!("loading {}", path.display()))?;
        Ok(Self { tensors })
    }

    /// Copy the classification head out to its own file, if this checkpoint has
    /// one, and say whether it did.
    ///
    /// A converted bundle holds the encoder alone: the emitter has no place for
    /// `head.*` or `classifier.*`, and reranking needs them. Writing them beside
    /// the bundle keeps it self-contained — `kohagi-rerank --coreml-dir` needs
    /// nothing but the directory — and keeps the head in f32 while the encoder
    /// is fp16, which is the more accurate half of the trade rather than the
    /// less.
    ///
    /// Four small tensors: 2.4 MB for a 768-wide head, against a bundle of
    /// hundreds.
    pub fn write_head(&self, path: &Path) -> Result<bool> {
        // The projection is what says a head is here at all; the norm and the
        // classifier come with it. Biases are optional (ModernBERT's defaults
        // are off), so an absent one is not an error.
        const REQUIRED: [&str; 4] = [
            "head.dense.weight",
            "head.norm.weight",
            "classifier.weight",
            "classifier.bias",
        ];
        const OPTIONAL: [&str; 2] = ["head.dense.bias", "head.norm.bias"];

        if !self.tensors.contains_key(REQUIRED[0]) {
            return Ok(false);
        }
        let mut out = HashMap::new();
        for name in REQUIRED {
            let tensor = self.tensors.get(name).with_context(|| {
                format!(
                    "this checkpoint has `head.dense.weight` but no `{name}`; it is a \
                     classification model this converter does not recognize"
                )
            })?;
            out.insert(name.to_string(), tensor.clone());
        }
        for name in OPTIONAL {
            if let Some(tensor) = self.tensors.get(name) {
                out.insert(name.to_string(), tensor.clone());
            }
        }
        candle_core::safetensors::save(&out, path)
            .with_context(|| format!("writing {}", path.display()))?;
        Ok(true)
    }

    /// The tensor names, sorted — for reporting what a checkpoint actually holds
    /// when a lookup fails.
    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.tensors.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }
}

impl Weights for Checkpoint {
    fn available(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    fn get(&self, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
        // Some checkpoints wrap the encoder under `model.` and some store it at
        // the root. `crate::model` resolves the same two layouts when loading for
        // inference; doing it here keeps the emitter's names to one spelling.
        let tensor = self
            .tensors
            .get(name)
            .or_else(|| self.tensors.get(&format!("model.{name}")))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "the checkpoint has no tensor named `{name}` or `model.{name}`; \
                     it holds {} tensors, the first few being {:?}",
                    self.tensors.len(),
                    &self.names()[..self.names().len().min(4)]
                )
            })?;
        let shape = tensor.dims().to_vec();
        anyhow::ensure!(
            shape == expected,
            "`{name}` is {shape:?} in the checkpoint but the config implies {expected:?}"
        );
        Ok(tensor
            .to_dtype(DType::F32)
            .with_context(|| format!("converting `{name}` to f32"))?
            .flatten_all()?
            .to_vec1()?)
    }
}
