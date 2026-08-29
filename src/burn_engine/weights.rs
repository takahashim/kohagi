//! Reading a checkpoint into Burn tensors.
//!
//! The safetensors file is the same one the candle path memory-maps, read
//! through candle's own [`MmapedSafetensors`] rather than a second reader: the
//! dependency is already here for the CPU backend, and mmap is what keeps a
//! half-gigabyte checkpoint off the heap. Only the conversion to Burn's tensor
//! type is new.
//!
//! Two things happen on the host rather than the device:
//!
//! - **Transposition.** Linear weights are stored `[out, in]`; the forward wants
//!   `[in, out]` so the GEMM's right-hand side is contiguous. Doing it once at
//!   load costs a copy per weight; doing it per forward costs a strided read on
//!   every layer of every batch.
//! - **Nothing else.** The cast to f16 is Burn's, decided by the backend's
//!   element type, so a weight arrives in whatever precision the run selected
//!   without this module knowing which.

use anyhow::{Context, Result};
use burn::tensor::{backend::Backend, Tensor, TensorData};
use candle_core::safetensors::MmapedSafetensors;

use crate::encoder::Config;

use super::encoder::{LayerWeights, ModernBert};

/// A memory-mapped checkpoint, with the name prefix it turned out to use.
pub(super) struct Checkpoint {
    tensors: MmapedSafetensors,
    /// `""` or `"model."`, from [`crate::encoder::name_prefix`] — the same
    /// answer the candle loader gets, from the same question.
    prefix: &'static str,
}

impl Checkpoint {
    pub(super) fn open(path: &std::path::Path) -> Result<Self> {
        // Safety: the same contract the candle path accepts — the file must not
        // be mutated underneath the process while it is loaded.
        let tensors = unsafe { MmapedSafetensors::new(path) }
            .with_context(|| format!("cannot memory-map {}", path.display()))?;
        let prefix = crate::encoder::name_prefix(|name| tensors.get(name).is_ok());
        Ok(Checkpoint { tensors, prefix })
    }

    fn raw(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let full = format!("{}{name}", self.prefix);
        let tensor = self
            .tensors
            .load(&full, &candle_core::Device::Cpu)
            .with_context(|| format!("{full} missing from the checkpoint"))?;
        let shape = tensor.dims().to_vec();
        let values = tensor
            .to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Ok((shape, values))
    }

    fn has(&self, name: &str) -> bool {
        self.tensors.get(&format!("{}{name}", self.prefix)).is_ok()
    }

    /// An `[out, in]` Linear weight, transposed to `[in, out]`.
    fn linear<B: Backend>(&self, name: &str, device: &B::Device) -> Result<Tensor<B, 2>> {
        let (shape, values) = self.raw(name)?;
        anyhow::ensure!(shape.len() == 2, "{name} is not a matrix: {shape:?}");
        let (out, input) = (shape[0], shape[1]);
        let mut transposed = vec![0f32; values.len()];
        for r in 0..out {
            for c in 0..input {
                transposed[c * out + r] = values[r * input + c];
            }
        }
        Ok(Tensor::from_data(
            TensorData::new(transposed, [input, out]),
            device,
        ))
    }

    fn vector<B: Backend>(&self, name: &str, device: &B::Device) -> Result<Tensor<B, 1>> {
        let (shape, values) = self.raw(name)?;
        Ok(Tensor::from_data(
            TensorData::new(values, [shape[0]]),
            device,
        ))
    }

    fn matrix<B: Backend>(&self, name: &str, device: &B::Device) -> Result<Tensor<B, 2>> {
        let (shape, values) = self.raw(name)?;
        anyhow::ensure!(shape.len() == 2, "{name} is not a matrix: {shape:?}");
        Ok(Tensor::from_data(
            TensorData::new(values, [shape[0], shape[1]]),
            device,
        ))
    }
}

/// Load every parameter onto the device.
pub(super) fn load<B: Backend>(
    checkpoint: &Checkpoint,
    config: &Config,
    half: bool,
    device: &B::Device,
) -> Result<ModernBert<B>> {
    let layers = (0..config.num_hidden_layers)
        .map(|i| {
            let at = |name: &str| format!("layers.{i}.{name}");
            Ok(LayerWeights {
                // Absent on layer 0, where the reference model applies identity.
                // `crate::encoder` spells the same thing as a `.ok()` on the
                // LayerNorm load.
                attn_norm: checkpoint
                    .has(&at("attn_norm.weight"))
                    .then(|| checkpoint.vector::<B>(&at("attn_norm.weight"), device))
                    .transpose()?,
                wqkv: checkpoint.linear::<B>(&at("attn.Wqkv.weight"), device)?,
                wo: checkpoint.linear::<B>(&at("attn.Wo.weight"), device)?,
                mlp_norm: checkpoint.vector::<B>(&at("mlp_norm.weight"), device)?,
                wi: checkpoint.linear::<B>(&at("mlp.Wi.weight"), device)?,
                mlp_wo: checkpoint.linear::<B>(&at("mlp.Wo.weight"), device)?,
                global: i % config.global_attn_every_n_layers == 0,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(ModernBert {
        // The embedding table is indexed, not multiplied, so it keeps its
        // `[vocab, hidden]` orientation.
        tok_embeddings: checkpoint.matrix::<B>("embeddings.tok_embeddings.weight", device)?,
        embeddings_norm: checkpoint.vector::<B>("embeddings.norm.weight", device)?,
        layers,
        final_norm: checkpoint.vector::<B>("final_norm.weight", device)?,
        config: config.clone(),
        half,
        // Filled by the first forward, which is what knows the padded length.
        tables: Default::default(),
    })
}
