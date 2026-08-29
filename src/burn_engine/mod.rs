//! The Burn engine: ModernBERT on Burn's tensors rather than candle's.
//!
//! Like `crate::coreml`, and unlike the CPU/Metal/CUDA backends, this does not
//! go through candle: Burn has its own tensor type, so there is no `Device` for
//! [`crate::model::open_device`] to return and no way to hand a batch to
//! [`crate::model::run_batches`]. What it *does* share is everything either side
//! of the forward — the tokenizer, the length bucketing and row placement in
//! `crate::batch`, and `embed_row`'s pooling, truncation and normalization — so
//! a vector differs from the CPU path only by the arithmetic that produced the
//! hidden states.
//!
//! [`encoder`] and [`weights`] are generic over `B: Backend` and know nothing
//! about which device runs them; a device module beside [`vulkan`] supplies the
//! element type, the memory budget and whatever setup that device needs.

mod encoder;
pub mod vulkan;
mod weights;

use anyhow::Result;
use burn::tensor::backend::Backend;

pub use vulkan::VulkanEncoder;

/// One loaded model's forward, with the element type erased.
///
/// A device may offer more than one precision, and those are separate Burn
/// backends and so separate Rust types. Erasing the parameter here keeps that
/// from turning into a `match` at every call site whose arms differ in nothing
/// but the type.
trait Forward {
    /// This unit's `[rows, seq, hidden]` states, read back to the host as f32
    /// whatever the graph ran in.
    fn hidden(&self, unit: &crate::batch::Unit) -> Result<Vec<f32>>;
}

impl<B: Backend> Forward for encoder::ModernBert<B> {
    fn hidden(&self, unit: &crate::batch::Unit) -> Result<Vec<f32>> {
        let device = Default::default();
        self.forward(unit.ids(), unit.mask(), unit.rows, unit.batch.seq, &device)
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("cannot read hidden states back from the GPU: {e:?}"))
    }
}
