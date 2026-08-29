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

use std::borrow::Cow;

use anyhow::{Context, Result};
use burn::tensor::{backend::Backend, Tensor, TensorData};
use candle_core::safetensors::MmapedSafetensors;

use crate::encoder::Config;

use super::encoder::{LayerWeights, ModernBert, Wi};
use super::FusedOps;

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

    /// One tensor's values as f32, borrowed straight out of the mapping when
    /// the file already holds f32 and the bytes land on an alignment `f32` can
    /// be read at.
    ///
    /// That is the common case and it is worth taking: going through
    /// `MmapedSafetensors::load` builds a candle tensor and `to_vec1` copies it
    /// again, so a checkpoint got walked three times before anything was
    /// transposed. Anything else — f16, bf16, a mapping that starts a tensor
    /// off-alignment — falls back to candle, which knows how to convert.
    fn raw(&self, name: &str) -> Result<(Vec<usize>, Cow<'_, [f32]>)> {
        let full = format!("{}{name}", self.prefix);
        let view = self
            .tensors
            .get(&full)
            .with_context(|| format!("{full} missing from the checkpoint"))?;
        let shape = view.shape().to_vec();

        if view.dtype() == safetensors::Dtype::F32 {
            // SAFETY: every bit pattern is a valid `f32`, and `align_to` only
            // hands back the aligned middle — a mapping that starts this tensor
            // off-alignment leaves a non-empty head and takes the branch below.
            let (head, body, tail) = unsafe { view.data().align_to::<f32>() };
            if head.is_empty() && tail.is_empty() {
                return Ok((shape, Cow::Borrowed(body)));
            }
        }

        let tensor = self.tensors.load(&full, &candle_core::Device::Cpu)?;
        let values = tensor
            .to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Ok((shape, Cow::Owned(values)))
    }

    fn has(&self, name: &str) -> bool {
        self.tensors.get(&format!("{}{name}", self.prefix)).is_ok()
    }

    /// An `[out, in]` Linear weight, transposed to `[in, out]`.
    fn linear<B: Backend>(&self, name: &str, device: &B::Device) -> Result<Tensor<B, 2>> {
        self.linear_rows(name, 0, usize::MAX, device)
    }

    /// Rows `start .. start + len` of an `[out, in]` weight, transposed to
    /// `[in, len]`. `len` past the end takes the rest, which is how
    /// [`Self::linear`] asks for the whole thing.
    fn linear_rows<B: Backend>(
        &self,
        name: &str,
        start: usize,
        len: usize,
        device: &B::Device,
    ) -> Result<Tensor<B, 2>> {
        let (shape, values) = self.raw(name)?;
        anyhow::ensure!(shape.len() == 2, "{name} is not a matrix: {shape:?}");
        let (out, input) = (shape[0], shape[1]);
        let len = len.min(out - start);
        let mut transposed = vec![0f32; len * input];
        for r in 0..len {
            for c in 0..input {
                transposed[c * len + r] = values[(start + r) * input + c];
            }
        }
        Ok(Tensor::from_data(
            TensorData::new(transposed, [input, len]),
            device,
        ))
    }

    fn vector<B: Backend>(&self, name: &str, device: &B::Device) -> Result<Tensor<B, 1>> {
        let (shape, values) = self.raw(name)?;
        Ok(Tensor::from_data(
            TensorData::new(values.into_owned(), [shape[0]]),
            device,
        ))
    }

    fn matrix<B: Backend>(&self, name: &str, device: &B::Device) -> Result<Tensor<B, 2>> {
        let (shape, values) = self.raw(name)?;
        anyhow::ensure!(shape.len() == 2, "{name} is not a matrix: {shape:?}");
        Ok(Tensor::from_data(
            TensorData::new(values.into_owned(), [shape[0], shape[1]]),
            device,
        ))
    }
}

/// Load every parameter onto the device.
pub(super) fn load<B: FusedOps>(
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
                // `Wi` concatenates gate and up along the output axis, so the
                // split is by rows. Which form this device wants is measured;
                // see `FusedOps::SPLIT_WI`.
                wi: if B::SPLIT_WI {
                    let n = &at("mlp.Wi.weight");
                    Wi::Split {
                        gate: checkpoint.linear_rows::<B>(
                            n,
                            0,
                            config.intermediate_size,
                            device,
                        )?,
                        up: checkpoint.linear_rows::<B>(
                            n,
                            config.intermediate_size,
                            config.intermediate_size,
                            device,
                        )?,
                    }
                } else {
                    Wi::Wide(checkpoint.linear::<B>(&at("mlp.Wi.weight"), device)?)
                },
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
