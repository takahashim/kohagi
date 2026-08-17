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

    /// One tensor by its exact name, or `None`. Unlike [`Weights::get`] this
    /// does not fall back to a `model.` prefix: the head sits at the root.
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
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
