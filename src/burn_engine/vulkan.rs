//! The Vulkan device for the Burn engine: any Vulkan 1.2 GPU, no vendor runtime.
//!
//! CubeCL compiles the same kernels to SPIR-V, which every Mesa and every vendor
//! driver already speaks. On the Radeon 780M this was developed against, that
//! means no ROCm install, no `HSA_OVERRIDE_GFX_VERSION`, and no dependence on a
//! GPU being on AMD's support list.
//!
//! ## What it costs to start
//!
//! CubeCL autotunes its kernels on first use and caches the result. Cold that is
//! about 12 s; warm it is about 0.4 s on top of the model load, and the cache is
//! roughly 60 KB. [`configure_tune_cache`] decides where it lands.

use anyhow::Result;

use super::{weights, BurnEncoder, Forward, FusedOps, Shape};
use crate::encoder::Config;
use crate::model::Precision;

// Both precisions take the generic kernels. Fusing RoPE by hand here is the
// wrong shape for a GPU — CubeCL already fuses across the graph — and splitting
// `Wi` measured 7% slower than one wider matmul.
impl FusedOps for Exact {}
impl FusedOps for Half {}

/// f32 throughout: the same vectors as the CPU path (worst `1 - cosine` 8.5e-13
/// measured against it), and slower than the CPU's bf16 path. Kept because it is
/// what makes a disagreement diagnosable — if `f16` drifts, this says whether the
/// cause is the precision recipe or the implementation.
type Exact = burn::backend::Vulkan;

/// f16 storage and matmuls with f32 reductions — see `encoder`'s precision note.
type Half = burn::backend::Vulkan<burn::tensor::f16, i32>;

/// Rows allowed in one forward at padded length `seq`, as `budget / seq^2`.
///
/// The attention scores dominate: `[rows, heads, seq, seq]`, materialized in the
/// tensor's dtype and again in f32 for the softmax. Measured at 17.8 MiB per row
/// at seq 473, which is what this bounds.
///
/// Smaller than the candle path's [`crate::model::GPU_ATTN_BUDGET`] on purpose.
/// Throughput here is flat in the row count — 19.3 rows/s at one row against
/// 20.5 at sixty-four, because a single 473-token row already saturates the
/// device — so a tighter bound on memory costs nothing to buy.
const VULKAN_ATTN_BUDGET: usize = 8 * 512 * 512;

/// Where CubeCL should keep its autotune results.
///
/// Its default walks up from the working directory looking for a `Cargo.toml`
/// and writes into that project's `target/`, falling back to the user's cache
/// directory only when it finds none. The fallback is fine; the search is not.
/// Running `kohagi` from inside *any* Rust checkout — this one included — puts
/// the cache in that project's `target/`, so it is per-directory and `cargo
/// clean` throws it away. Naming a path skips the search entirely.
fn tune_cache_dir() -> Option<std::path::PathBuf> {
    // CubeCL appends its own `autotune/<version>/<device>/` below this, so the
    // root is the crate's cache directory rather than a path ending in
    // `autotune` — otherwise the component appears twice.
    dirs::cache_dir().map(|d| d.join("kohagi"))
}

/// Point CubeCL's autotune cache at the user's cache directory, once.
///
/// Set before any device is opened, since the configuration is read when the
/// first runtime client is built and ignored afterwards.
fn configure_tune_cache() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let Some(dir) = tune_cache_dir() else { return };
        use burn::cubecl::config::{cache::CacheConfig, CubeClRuntimeConfig, RuntimeConfig};
        let mut config = CubeClRuntimeConfig::default();
        config.autotune.cache = CacheConfig::File(dir);
        CubeClRuntimeConfig::set(config);
    });
}

/// Refuse a second precision in the same process.
///
/// CubeCL keeps one runtime client per device, and that client's float element
/// type is fixed by whatever built it first. A second [`VulkanEncoder`] asking
/// for the other precision does not get its own client — it silently inherits
/// the first one's, and the vectors that come back are neither what was asked
/// for nor flagged. Worse in the f32-after-f16 direction: the exact path omits
/// the f32 reduction casts precisely because it does not need them, so it runs
/// as *plain* f16 and lands 1-cos 2.5e-2 from the CPU.
///
/// Nothing here can undo that, so this refuses it. One process, one precision;
/// the CLI never hits this, and a library caller that wants both runs two.
fn claim_precision(want: Precision) -> Result<()> {
    use std::sync::OnceLock;
    static CLAIMED: OnceLock<Precision> = OnceLock::new();
    let first = CLAIMED.get_or_init(|| want);
    anyhow::ensure!(
        *first == want,
        "this process already opened the Vulkan device in {}, and CubeCL binds one \
         element type per device: a second encoder in {} would silently run in {} \
         instead. Use one precision per process.",
        first.name(),
        want.name(),
        first.name()
    );
    Ok(())
}

/// Load a checkpoint onto a Vulkan device.
pub fn load(
    weights: &std::path::Path,
    config: &Config,
    precision: Precision,
) -> Result<BurnEncoder> {
    configure_tune_cache();
    let checkpoint = weights::Checkpoint::open(weights)?;
    // Claimed here rather than on entry: `Checkpoint::open` reads the file on
    // the CPU and can fail without the device ever being touched, and a claim
    // made before that would refuse a later, legitimate load in the other
    // precision. The next line is what actually binds the element type.
    claim_precision(precision)?;
    let model: Box<dyn Forward + Send + Sync> = match precision {
        Precision::F16 => {
            let device = Default::default();
            Box::new(weights::load::<Half>(
                &checkpoint,
                config,
                true,
                VULKAN_ATTN_BUDGET,
                &device,
            )?)
        }
        _ => {
            let device = Default::default();
            Box::new(weights::load::<Exact>(
                &checkpoint,
                config,
                false,
                VULKAN_ATTN_BUDGET,
                &device,
            )?)
        }
    };
    Ok(BurnEncoder {
        model,
        dim: config.hidden_size,
        shape: Shape {
            budget: VULKAN_ATTN_BUDGET,
            // One GPU: fanning out would only make threads contend over command
            // submission and multiply scratch memory.
            fan_out: false,
            max_rows: usize::MAX,
        },
    })
}
