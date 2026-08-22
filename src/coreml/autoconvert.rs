//! Converting a checkpoint to Kohagi's CoreML layout on first use.
//!
//! The Neural Engine needs fixed input shapes, so it runs a converted bundle
//! rather than the safetensors the other devices read. Given the same
//! `--model-id` the CPU path would take, this emits one into a cache directory and
//! hands back a path the ordinary directory loader can open, so that using the ANE
//! costs no separate conversion step.
//!
//! Doing this at runtime is only safe because the emitter is checked rather than
//! trusted: its vectors for `ruri-v3-130m` were bit-identical to the Python
//! conversion's when both blocked attention with `-inf` — the two now differ in that
//! constant alone (see [`crate::coreml_export::modernbert::BLOCKED`]), which changes
//! only the padded rows pooling drops. An unsupported config is refused before
//! anything is written, and
//! [`super::CoreMlEncoder::load`] validates the bundle's inputs and output shape.
//! What it cannot check is whether a checkpoint is
//! *itself* sensitive to fp16, so [`self_check`] measures that once, after the
//! conversion, and says so.
//!
//! The cache is keyed so that a stale bundle cannot be mistaken for a fresh one:
//! the checkpoint revision, the buckets, the quantization, and [`GRAPH_VERSION`].

use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::config::CoreMlQuantize as Quantize;
use crate::coreml_export::{bundle_name, encoder, Checkpoint};
use crate::program::remark;

/// The emitted graph's version, which is half of this cache's key.
///
/// Defined beside the emitter that moves it, because a bundle records it too — see
/// [`crate::coreml_export::GRAPH_VERSION`].
pub use crate::coreml_export::GRAPH_VERSION;

/// The emitter options a quantization level asks for.
fn options(quantize: Quantize) -> encoder::Options {
    encoder::Options {
        quantize_embeddings: matches!(quantize, Quantize::Embeddings | Quantize::All),
        quantize_projections: quantize == Quantize::All,
    }
}

/// What distinguishes one state of a checkpoint from another, for the cache key.
///
/// For a Hub download that is the snapshot commit, which sits in the cache path —
/// an upstream update lands in a different directory, so the key changes without
/// hashing 500MB. For a local checkpoint there is no revision to read, so the
/// weights' size and modification time stand in, as the compile cache already
/// does for a package.
fn revision(checkpoint: &Checkpoint) -> String {
    if let Some(sha) = snapshot_commit(&checkpoint.weights) {
        return sha;
    }
    let mut h = crate::fnv::Hasher::new();
    if let Ok(meta) = std::fs::metadata(&checkpoint.weights) {
        h.write_metadata(&meta);
    }
    format!("local{:016x}", h.finish())
}

/// The commit a Hugging Face cache path belongs to: the component right after
/// `snapshots/`. `None` for any other path shape, including a local checkpoint.
fn snapshot_commit(weights: &Path) -> Option<String> {
    let mut parts = weights.components().rev();
    let dir = parts.nth(1)?.as_os_str().to_str()?;
    (parts.next()?.as_os_str() == "snapshots").then(|| dir.to_string())
}

/// The outcome of [`provision`].
///
/// Either way the directory has the layout `--coreml-dir` expects, so the caller
/// carries on through the ordinary loader.
pub enum Provisioned {
    /// Found in the cache; nothing was written.
    Cached(PathBuf),
    /// Converted by this call, which is when the one-time self-check is worth its
    /// few seconds.
    Converted(PathBuf),
}

impl Provisioned {
    pub fn path(&self) -> &Path {
        match self {
            Self::Cached(p) | Self::Converted(p) => p,
        }
    }
}

/// Convert `checkpoint` if the cache does not already hold that exact bundle, and
/// return the directory to load from.
///
/// The returned directory has the layout `--coreml-dir` expects, so the caller
/// carries on through the ordinary loader.
pub fn provision(
    checkpoint: &Checkpoint,
    buckets: &[usize],
    quantize: Quantize,
) -> Result<Provisioned> {
    let entry = cache_entry(checkpoint, buckets, quantize);
    if let Some(entry) = &entry {
        if entry.join(bundle_name(buckets)).is_dir() {
            return Ok(Provisioned::Cached(entry.clone()));
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
        return Ok(Provisioned::Converted(staged));
    };
    if let Some(parent) = entry.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::remove_dir_all(&entry);
    match std::fs::rename(&staged, &entry) {
        Ok(()) => {
            evict_superseded(&entry);
            Ok(Provisioned::Converted(entry))
        }
        Err(e) => {
            // The conversion succeeded, so run from the staging copy rather than
            // failing; the next run will try to cache it again.
            remark!(
                "could not move the converted model into {} ({e}); \
                 this run will use a temporary copy",
                entry.display()
            );
            Ok(Provisioned::Converted(staged))
        }
    }
}

/// Where this exact conversion belongs, or `None` if there is no usable cache
/// directory (in which case the caller converts to a temporary directory instead
/// of failing).
fn cache_entry(checkpoint: &Checkpoint, buckets: &[usize], quantize: Quantize) -> Option<PathBuf> {
    let dir = super::provision::cache_root()?
        .join("converted")
        .join(slug(&checkpoint.source));
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir.join(entry_name(&revision(checkpoint), buckets, quantize)))
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

/// Emit the bundle, reporting what is about to take twenty seconds.
///
/// The conversion itself is `coreml_export::convert`, shared with the
/// `coreml-convert` binary so that an automatic bundle and a hand-converted one
/// are the same artifact.
fn convert_into(
    dir: &Path,
    checkpoint: &Checkpoint,
    buckets: &[usize],
    quantize: Quantize,
) -> Result<()> {
    remark!(
        "converting {} for the Neural Engine (buckets {}) — first run only ...",
        checkpoint.source,
        buckets
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    );
    crate::coreml_export::convert(dir, checkpoint, buckets, &options(quantize))?;
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
        let dir = std::env::temp_dir().join(format!("kohagi-converted-rev-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let weights = dir.join("model.safetensors");
        std::fs::write(&weights, b"one").unwrap();
        let checkpoint = Checkpoint {
            weights: weights.clone(),
            config: dir.join("config.json"),
            tokenizer: dir.join("tokenizer.json"),
            pooling: None,
            sentence_config: None,
            source: dir.display().to_string(),
        };
        let first = revision(&checkpoint);
        assert!(first.starts_with("local"), "{first}");
        // A different size gives a different revision, so an edited checkpoint is
        // reconverted rather than served from the cache.
        std::fs::write(&weights, b"another").unwrap();
        assert_ne!(first, revision(&checkpoint));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn superseding_spares_other_bucket_sets_and_other_models() {
        let root =
            std::env::temp_dir().join(format!("kohagi-converted-evict-{}", std::process::id()));
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
