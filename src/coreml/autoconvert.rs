//! Converting a checkpoint to Kohagi's CoreML layout on first use.
//!
//! The Neural Engine needs fixed input shapes, so it runs a converted bundle
//! rather than the safetensors the other devices read. Given the same
//! `--model-id` the CPU path would take, this emits one into a cache directory and
//! hands back a path the ordinary directory loader can open, so that using the ANE
//! costs no separate conversion step.
//!
//! Doing this at runtime is only safe because the emitter is checked rather than
//! trusted: its output for `ruri-v3-130m` is bit-identical to the Python
//! conversion, an unsupported config is refused before anything is written, and
//! [`super::CoreMlEncoder::load`] validates the bundle's inputs and output shape.
//! What it cannot check is whether a checkpoint is
//! *itself* sensitive to fp16, so [`self_check`] measures that once, after the
//! conversion, and says so.
//!
//! The cache is keyed so that a stale bundle cannot be mistaken for a fresh one:
//! the checkpoint revision, the buckets, the quantization, and [`GRAPH_VERSION`].

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::config::CoreMlQuantize as Quantize;
use crate::coreml_export::{encoder, modernbert::Activation, Provenance};

/// The emitted graph's version, part of the cache key.
///
/// Bumped whenever the emitter changes what it builds, so that a new Kohagi never
/// reads a bundle an older one wrote. Deliberately not `CARGO_PKG_VERSION`: a
/// release that leaves the graph alone should not throw away every user's cache.
///
/// The golden-file test in `crate::coreml_export::encoder` pins the emitted bytes;
/// a change that updates those hashes updates this too.
pub const GRAPH_VERSION: u32 = 1;

/// The emitter options a quantization level asks for.
fn options(quantize: Quantize) -> encoder::Options {
    encoder::Options {
        quantize_embeddings: matches!(quantize, Quantize::Embeddings | Quantize::All),
        quantize_projections: quantize == Quantize::All,
    }
}

/// A checkpoint to convert, already resolved to files on disk.
///
/// `crate::model` does the resolving, because a Hub checkpoint is downloaded the
/// same way for every backend and a local one is just paths.
pub struct Checkpoint {
    /// `model.safetensors`.
    pub weights: PathBuf,
    /// `config.json`.
    pub config: PathBuf,
    /// `tokenizer.json`.
    pub tokenizer: PathBuf,
    /// `1_Pooling/config.json`, when the checkpoint ships one. A reranker or a
    /// base LM does not, and the loader falls back to mean pooling with a warning.
    pub pooling: Option<PathBuf>,
    /// How to name this checkpoint in the cache and in the bundle's provenance:
    /// the Hub id, or the path for a local one.
    pub source: String,
}

impl Checkpoint {
    /// What distinguishes one state of this checkpoint from another.
    ///
    /// For a Hub download that is the snapshot commit, which sits in the cache
    /// path — an upstream update lands in a different directory, so the key
    /// changes without hashing 500MB. For a local checkpoint there is no revision
    /// to read, so the weights' size and modification time stand in, as the
    /// compile cache already does for a package.
    fn revision(&self) -> String {
        if let Some(sha) = snapshot_commit(&self.weights) {
            return sha;
        }
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut mix = |bytes: &[u8]| {
            for &b in bytes {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        if let Ok(meta) = std::fs::metadata(&self.weights) {
            mix(&meta.len().to_le_bytes());
            if let Ok(t) = meta.modified() {
                if let Ok(d) = t.duration_since(std::time::UNIX_EPOCH) {
                    mix(&d.as_nanos().to_le_bytes());
                }
            }
        }
        format!("local{h:016x}")
    }
}

/// The commit a Hugging Face cache path belongs to: the component right after
/// `snapshots/`. `None` for any other path shape, including a local checkpoint.
fn snapshot_commit(weights: &Path) -> Option<String> {
    let mut parts = weights.components().rev();
    let dir = parts.nth(1)?.as_os_str().to_str()?;
    (parts.next()?.as_os_str() == "snapshots").then(|| dir.to_string())
}

/// Convert `checkpoint` if the cache does not already hold that exact bundle, and
/// return the directory to load from.
///
/// The returned directory has the layout `--coreml-dir` expects, so the caller
/// carries on through the ordinary loader.
///
/// The second element is whether this call converted, as opposed to finding the
/// bundle already cached — which is what decides whether the caller runs the
/// one-time self-check.
pub fn provision(
    checkpoint: &Checkpoint,
    buckets: &[usize],
    quantize: Quantize,
) -> Result<(PathBuf, bool)> {
    let entry = cache_entry(checkpoint, buckets, quantize);
    if let Some(entry) = &entry {
        if entry.join(bundle_name(buckets)).is_dir() {
            return Ok((entry.clone(), false));
        }
    }

    // Everywhere the conversion writes is inside `staged`, which is only renamed
    // into place once every file is there. A crash, or a second Kohagi converting
    // the same model, therefore cannot leave a half-written bundle looking like a
    // cache hit.
    let staged = match &entry {
        Some(entry) => entry.with_file_name(format!(
            "{}.partial-{}",
            entry.file_name().unwrap_or_default().to_string_lossy(),
            std::process::id()
        )),
        None => std::env::temp_dir().join(format!("kohagi-coreml-{}", std::process::id())),
    };
    let _ = std::fs::remove_dir_all(&staged);
    let result = convert_into(&staged, checkpoint, buckets, quantize);
    if result.is_err() {
        let _ = std::fs::remove_dir_all(&staged);
    }
    result?;

    let Some(entry) = entry else {
        return Ok((staged, true));
    };
    if let Some(parent) = entry.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::remove_dir_all(&entry);
    match std::fs::rename(&staged, &entry) {
        Ok(()) => {
            evict_superseded(&entry);
            Ok((entry, true))
        }
        Err(e) => {
            // The conversion succeeded, so run from the staging copy rather than
            // failing; the next run will try to cache it again.
            eprintln!(
                "kohagi: could not move the converted model into {} ({e}); \
                 this run will use a temporary copy",
                entry.display()
            );
            Ok((staged, true))
        }
    }
}

/// The bundle file name for a set of buckets, matching `coreml-convert`'s so that
/// a cached directory and a hand-converted one are the same thing.
fn bundle_name(buckets: &[usize]) -> String {
    format!(
        "buckets-{}.mlpackage",
        buckets
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("-")
    )
}

/// Where this exact conversion belongs, or `None` if there is no usable cache
/// directory (in which case the caller converts to a temporary directory instead
/// of failing).
fn cache_entry(checkpoint: &Checkpoint, buckets: &[usize], quantize: Quantize) -> Option<PathBuf> {
    let dir = super::provision::cache_root()?
        .join("converted")
        .join(slug(&checkpoint.source));
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir.join(entry_name(&checkpoint.revision(), buckets, quantize)))
}

/// `<revision>-b<buckets>-<quantization>-g<graph version>`.
fn entry_name(revision: &str, buckets: &[usize], quantize: Quantize) -> String {
    format!(
        "{revision}-b{}-{}-g{GRAPH_VERSION}",
        buckets
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("-"),
        quantize.tag()
    )
}

/// A directory name for a Hub id or a path: readable, and unambiguous enough that
/// two different sources cannot collide.
fn slug(source: &str) -> String {
    let cleaned: String = source
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect();
    cleaned.trim_matches('-').to_string()
}

/// Delete this checkpoint's other bundles for the same buckets and quantization,
/// which a new revision or a new graph version has superseded.
///
/// Scoped to one model's directory, and within it to entries that differ only in
/// revision or graph version, so a second bucket set the same user also converted
/// survives. Without this the cache grows by a whole model on every upstream
/// update, and nothing else would clean it up.
fn evict_superseded(keep: &Path) {
    let Some((dir, name)) = keep.parent().zip(keep.file_name().and_then(|n| n.to_str())) else {
        return;
    };
    // Everything from `-b` onwards: the buckets and the quantization, but not the
    // revision that precedes them nor the graph version that ends the name.
    let Some(suffix) = name.find("-b").map(|i| &name[i..]) else {
        return;
    };
    let Some(shape) = suffix.rfind("-g").map(|i| &suffix[..i]) else {
        return;
    };
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(other) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if other != name && other.contains(shape) {
            let _ = std::fs::remove_dir_all(&path);
        }
    }
}

/// Emit the bundle and copy the metadata a loader needs beside it.
fn convert_into(
    dir: &Path,
    checkpoint: &Checkpoint,
    buckets: &[usize],
    quantize: Quantize,
) -> Result<()> {
    let text = std::fs::read_to_string(&checkpoint.config)
        .with_context(|| format!("reading {}", checkpoint.config.display()))?;
    let cfg = encoder::EncoderConfig::from_json(&text)?;

    // Before reading 500MB of weights: a bucket past the trained positions has no
    // RoPE frequencies behind it, and `emit_with` would refuse anyway.
    if let Some(max) = cfg.max_positions {
        if let Some(&over) = buckets.iter().find(|&&b| b > max) {
            anyhow::bail!(
                "bucket {over} is longer than {}'s {max} trained positions; \
                 lower --coreml-buckets",
                checkpoint.source
            );
        }
    }

    eprintln!(
        "kohagi: converting {} for the Neural Engine ({} wide, {} layers, gate {}, \
         buckets {}) — first run only ...",
        checkpoint.source,
        cfg.hidden,
        cfg.layers,
        cfg.activation.name(),
        buckets
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    );

    let weights = crate::coreml_export::safetensors::Checkpoint::open(&checkpoint.weights)?;
    let provenance = Provenance {
        source: checkpoint.source.clone(),
        lengths: buckets.to_vec(),
        quantized_embeddings: quantize != Quantize::None,
        quantized_projections: quantize == Quantize::All,
        activation: (cfg.activation != Activation::default()).then(|| cfg.activation.name()),
    };
    let (model, blob) =
        encoder::emit_with(&cfg, &weights, buckets, &provenance, &options(quantize))?;

    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    crate::coreml_export::write_package(&dir.join(bundle_name(buckets)), &model, &blob)?;

    // The loader reads all three from the directory, so a cached bundle has to be
    // self-contained: the HF cache it came from may be cleared independently.
    std::fs::copy(&checkpoint.config, dir.join("config.json"))
        .with_context(|| format!("copying {}", checkpoint.config.display()))?;
    std::fs::copy(&checkpoint.tokenizer, dir.join("tokenizer.json"))
        .with_context(|| format!("copying {}", checkpoint.tokenizer.display()))?;
    if let Some(pooling) = &checkpoint.pooling {
        let into = dir.join("1_Pooling");
        std::fs::create_dir_all(&into).with_context(|| format!("creating {}", into.display()))?;
        std::fs::copy(pooling, into.join("config.json"))
            .with_context(|| format!("copying {}", pooling.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(revision: &str) -> String {
        entry_name(revision, &[128, 256], Quantize::None)
    }

    #[test]
    fn the_key_covers_revision_buckets_quantization_and_graph_version() {
        assert_eq!(entry("abc"), format!("abc-b128-256-fp16-g{GRAPH_VERSION}"));
        // Each of the four moves the key.
        assert_ne!(entry("abc"), entry("def"));
        assert_ne!(
            entry_name("abc", &[128], Quantize::None),
            entry_name("abc", &[128, 256], Quantize::None)
        );
        assert_ne!(
            entry_name("abc", &[128], Quantize::None),
            entry_name("abc", &[128], Quantize::Embeddings)
        );
        assert_ne!(
            entry_name("abc", &[128], Quantize::Embeddings),
            entry_name("abc", &[128], Quantize::All)
        );
    }

    #[test]
    fn a_hub_path_yields_its_snapshot_commit() {
        let hub = Path::new(
            "/home/u/.cache/huggingface/hub/models--cl-nagoya--ruri-v3-130m\
             /snapshots/deadbeef/model.safetensors",
        );
        assert_eq!(snapshot_commit(hub), Some("deadbeef".to_string()));
        // A local checkpoint has no commit to read, and falls back to a fingerprint.
        assert_eq!(snapshot_commit(Path::new("/tmp/m/model.safetensors")), None);
    }

    #[test]
    fn a_local_checkpoint_gets_a_fingerprint_revision() {
        let dir = std::env::temp_dir().join(format!("kohagi-rev-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let weights = dir.join("model.safetensors");
        std::fs::write(&weights, b"one").unwrap();
        let checkpoint = Checkpoint {
            weights: weights.clone(),
            config: dir.join("config.json"),
            tokenizer: dir.join("tokenizer.json"),
            pooling: None,
            source: dir.display().to_string(),
        };
        let first = checkpoint.revision();
        assert!(first.starts_with("local"), "{first}");
        // A different size gives a different revision, so an edited checkpoint is
        // reconverted rather than served from the cache.
        std::fs::write(&weights, b"another").unwrap();
        assert_ne!(first, checkpoint.revision());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn superseding_spares_other_bucket_sets_and_other_models() {
        let root = std::env::temp_dir().join(format!("kohagi-evict-{}", std::process::id()));
        let dir = root.join("cl-nagoya--ruri-v3-130m");
        let other = root.join("answerdotai--ModernBERT-base");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::create_dir_all(&other).unwrap();

        let keep = dir.join(entry_name("new", &[128, 256], Quantize::None));
        let stale_revision = dir.join(entry_name("old", &[128, 256], Quantize::None));
        let stale_graph = dir.join("old-b128-256-fp16-g0");
        let different_buckets = dir.join(entry_name("old", &[512], Quantize::None));
        let different_quant = dir.join(entry_name("old", &[128, 256], Quantize::All));
        let another_model = other.join(entry_name("old", &[128, 256], Quantize::None));
        for p in [
            &keep,
            &stale_revision,
            &stale_graph,
            &different_buckets,
            &different_quant,
            &another_model,
        ] {
            std::fs::create_dir_all(p).unwrap();
        }

        evict_superseded(&keep);

        assert!(keep.is_dir(), "the new entry survives");
        assert!(!stale_revision.is_dir(), "an older revision is dropped");
        assert!(!stale_graph.is_dir(), "an older graph version is dropped");
        assert!(different_buckets.is_dir(), "another bucket set survives");
        assert!(different_quant.is_dir(), "another quantization survives");
        assert!(another_model.is_dir(), "another model survives");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_slug_is_readable_and_unambiguous() {
        assert_eq!(slug("cl-nagoya/ruri-v3-130m"), "cl-nagoya-ruri-v3-130m");
        assert_eq!(slug("/tmp/my model/"), "tmp-my-model");
        // Two sources that differ only in a separator still differ in the slug's
        // length, so they cannot land in the same directory.
        assert_ne!(slug("a/b"), slug("a//b"));
    }
}
