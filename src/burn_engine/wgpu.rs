//! What the wgpu-based devices — [`super::vulkan`], [`super::metal`] — share:
//! CubeCL's autotune cache location and the one-precision-per-process rule.
//! Both are properties of the CubeCL runtime underneath, not of either API.

use anyhow::Result;

use crate::model::Precision;

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
pub(super) fn configure_tune_cache() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let Some(dir) = tune_cache_dir() else { return };
        use burn::cubecl::config::{cache::CacheConfig, CubeClRuntimeConfig, RuntimeConfig};
        let mut config = CubeClRuntimeConfig::default();
        config.autotune.cache = CacheConfig::File(dir.clone());
        // The compiled-kernel cache beside it. In CubeCL 0.10 this holds
        // SPIR-V pipelines only, so it is for the Vulkan device and unmeasured
        // there; on Metal the MSL goes to Apple's compiler every process, and
        // nothing here can keep that.
        config.compilation.cache = Some(CacheConfig::File(dir));
        CubeClRuntimeConfig::set(config);
    });
}

/// Refuse a second precision in the same process.
///
/// CubeCL keeps one runtime client per device, and that client's float element
/// type is fixed by whatever built it first. A second encoder asking for the
/// other precision does not get its own client — it silently inherits the
/// first one's, and the vectors that come back are neither what was asked for
/// nor flagged. Worse in the f32-after-f16 direction: the exact path omits
/// the f32 reduction casts precisely because it does not need them, so it runs
/// as *plain* f16 and lands 1-cos 2.5e-2 from the CPU.
///
/// Nothing here can undo that, so this refuses it. One process, one precision;
/// the CLI never hits this, and a library caller that wants both runs two.
pub(super) fn claim_precision(device: &str, want: Precision) -> Result<()> {
    use std::sync::OnceLock;
    static CLAIMED: OnceLock<Precision> = OnceLock::new();
    let first = CLAIMED.get_or_init(|| want);
    anyhow::ensure!(
        *first == want,
        "this process already opened the {} device in {}, and CubeCL binds one \
         element type per device: a second encoder in {} would silently run in {} \
         instead. Use one precision per process.",
        device,
        first.name(),
        want.name(),
        first.name()
    );
    Ok(())
}
