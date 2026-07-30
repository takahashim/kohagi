//! Provisioning: turning a model *source* (a local dir or a Hub repo) into
//! loaded [`MLModel`]s ready for the ANE. Covers three steps that all serve the
//! one goal of "get usable bucket models onto disk and into memory":
//!
//! - **download** the preferred form of each bucket from the Hub ([`fetch_from_hub`]),
//! - **locate** the `seq-<N>` and `buckets-<N>-<N>…` bundles in a directory
//!   ([`collect_buckets`]),
//! - **load** each bucket, compiling a `.mlpackage` when needed ([`load_bucket`]).
//!
//! Running the loaded models lives in the parent module.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use objc2::rc::Retained;
use objc2_core_ml::{MLComputeUnits, MLModel, MLModelConfiguration};
use objc2_foundation::{NSString, NSURL};

use crate::config::CoreMlForm;

/// Where one bucket's model lives: the bundle, and the CoreML function to load
/// from it. `function` is `None` for a single-length `seq-<N>` bundle, whose
/// only function is the default one, and `Some("seq_<N>")` for a multi-function
/// `buckets-<N>-<N>…` bundle, where every length is a function in one file.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Located {
    pub path: PathBuf,
    pub function: Option<String>,
}

/// A bucket's two possible on-disk forms: the compiled `.mlmodelc` and the
/// portable `.mlpackage`.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct BucketForms {
    pub compiled: Option<Located>,
    pub package: Option<Located>,
}

/// Parse a bundle name into the bucket lengths it serves and its form. The
/// single source of truth for both naming schemes:
///
/// - `seq-128.mlpackage` — one length, in the model's default function.
/// - `buckets-128-256-512.mlpackage` — three lengths in one multi-function
///   bundle, as `seq_128` / `seq_256` / `seq_512`. Converting this way shares
///   one copy of the weights between the lengths instead of repeating them per
///   file, which is most of the download for a large-vocabulary model.
fn parse_bundle(name: &str) -> Option<(Vec<usize>, &str)> {
    let (stem, ext) = name.rsplit_once('.')?;
    if ext != "mlmodelc" && ext != "mlpackage" {
        return None;
    }
    if let Some(single) = stem.strip_prefix("seq-") {
        return Some((vec![single.parse().ok()?], ext));
    }
    let seqs: Vec<usize> = stem
        .strip_prefix("buckets-")?
        .split('-')
        .map(|s| s.parse().ok())
        .collect::<Option<_>>()?;
    (!seqs.is_empty()).then_some((seqs, ext))
}

/// The CoreML function a bucket of length `seq` lives in, for a bundle serving
/// `count` lengths. One place so the converter's naming and this side agree.
fn function_name(seq: usize, count: usize) -> Option<String> {
    (count > 1).then(|| format!("seq_{seq}"))
}

/// Parse a repo-relative path (`compiled/seq-128.mlmodelc/...`,
/// `seq-128.mlpackage/...`, `buckets-128-256.mlpackage/...`) into the bucket
/// lengths it serves and its form.
fn bucket_of(rfilename: &str) -> Option<(Vec<usize>, &str)> {
    let rel = rfilename.strip_prefix("compiled/").unwrap_or(rfilename);
    parse_bundle(rel.split('/').next()?)
}

/// Scan one directory for bucket bundles, recording each into `found` keyed by
/// sequence length: `.mlmodelc` in the compiled slot, `.mlpackage` in the
/// package slot. A multi-function bundle records the same path under every
/// length it serves, each with its own function name.
pub(super) fn collect_buckets(
    dir: &Path,
    found: &mut BTreeMap<usize, BucketForms>,
) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        let Some((seqs, ext)) = path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(parse_bundle)
        else {
            continue;
        };
        for &seq in &seqs {
            let found_here = Located {
                path: path.clone(),
                function: function_name(seq, seqs.len()),
            };
            let slot = found.entry(seq).or_default();
            let target = match ext {
                "mlmodelc" => &mut slot.compiled,
                "mlpackage" => &mut slot.package,
                _ => continue,
            };
            // A dedicated `seq-<N>` bundle wins over a multi-function one for
            // the same length, so a directory holding both resolves the same
            // way whatever order the filesystem lists it in.
            if target
                .as_ref()
                .is_none_or(|held| held.function.is_some() && found_here.function.is_none())
            {
                *target = Some(found_here);
            }
        }
    }
    Ok(())
}

/// Load one bucket, preferring the compiled `.mlmodelc` and falling back to the
/// portable `.mlpackage`. At least one of the two is `Some` (the caller only
/// inserts a bucket when it finds a file).
pub(super) fn load_bucket(
    seq: usize,
    compiled: Option<&Located>,
    package: Option<&Located>,
) -> Result<Retained<MLModel>> {
    if let Some(c) = compiled {
        match load_model(c) {
            Ok(model) => return Ok(model),
            Err(e) if package.is_some() => {
                eprintln!(
                    "kohagi: the compiled bundle for seq-{seq} did not load ({e:#}); \
                     compiling its .mlpackage instead"
                );
            }
            Err(e) => return Err(e).with_context(|| format!("loading {}", c.path.display())),
        }
    }
    let package = package.expect("load_bucket called with neither model form");
    load_model(package).with_context(|| format!("loading {}", package.path.display()))
}

/// Load one model, pinned to CPU+ANE. A `.mlpackage` is compiled to a
/// `.mlmodelc` first (cached across runs, see [`compile_cached`]); a `.mlmodelc`
/// is loaded directly.
fn load_model(located: &Located) -> Result<Retained<MLModel>> {
    let path = located.path.as_path();
    if path.extension().and_then(|e| e.to_str()) != Some("mlpackage") {
        return load_at(path, located);
    }
    let compiled = compile_cached(path)?;
    match load_at(&compiled.path, located) {
        Ok(model) => Ok(model),
        // A cached bundle can outlive the OS that compiled it. Rather than
        // carry an OS version in the cache key — which would throw the cache
        // away on every point release, whether or not it mattered — treat a
        // failed load as the signal, drop the entry and compile once more.
        Err(e) if compiled.from_cache => {
            eprintln!(
                "kohagi: the cached compile of {} did not load ({e:#}); recompiling",
                path.display()
            );
            let _ = std::fs::remove_dir_all(&compiled.path);
            let fresh = compile_package(path)?;
            load_at(&fresh, located)
        }
        Err(e) => Err(e),
    }
}

/// Load one already-compiled model, pinned to CPU+ANE.
fn load_at(target: &Path, located: &Located) -> Result<Retained<MLModel>> {
    let path = located.path.as_path();
    unsafe {
        let url = file_url(target)?;
        let config = MLModelConfiguration::new();
        config.setComputeUnits(MLComputeUnits::CPUAndNeuralEngine);
        if let Some(function) = &located.function {
            config.setFunctionName(Some(&NSString::from_str(function)));
        }
        MLModel::modelWithContentsOfURL_configuration_error(&url, &config).map_err(|e| {
            match &located.function {
                Some(f) => anyhow::anyhow!("loading function {f} of {}: {e}", path.display()),
                None => anyhow::anyhow!("loading {}: {e}", path.display()),
            }
        })
    }
}

/// A compiled bundle on disk, and whether this run is the one that made it.
struct Compiled {
    path: PathBuf,
    from_cache: bool,
}

/// Compile a `.mlpackage`, reusing a previous run's result when there is one.
///
/// Compiling a bucket takes on the order of 20 seconds, and without a cache
/// every run pays it again for every bucket. That is why a converted directory
/// normally ships a `compiled/` copy of each bucket alongside the package,
/// doubling what a publisher stores and a user downloads. Caching here is what
/// lets a repository carry the portable form alone.
///
/// The cache lives in `$KOHAGI_COREML_CACHE`, or `~/Library/Caches/kohagi/coreml`.
/// Nothing here is load-bearing: any failure to read or write it falls back to
/// the old behaviour of compiling into a temporary directory, because a slow
/// start is better than a failed one.
fn compile_cached(pkg: &Path) -> Result<Compiled> {
    let Some(entry) = cache_entry(pkg) else {
        return Ok(Compiled {
            path: compile_package(pkg)?,
            from_cache: false,
        });
    };
    if entry.is_dir() {
        return Ok(Compiled {
            path: entry,
            from_cache: true,
        });
    }

    let fresh = compile_package(pkg)?;
    // Publish under a temporary name in the cache directory and rename into
    // place, so a crash or a second process mid-compile cannot leave a partial
    // bundle behind that looks like a hit.
    let staged = entry.with_extension(format!("mlmodelc.partial-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&staged);
    let published = move_dir(&fresh, &staged).and_then(|()| {
        let _ = std::fs::remove_dir_all(&entry);
        std::fs::rename(&staged, &entry)
    });
    match published {
        Ok(()) => {
            evict_superseded(&entry);
            Ok(Compiled {
                path: entry,
                from_cache: false,
            })
        }
        Err(e) => {
            let _ = std::fs::remove_dir_all(&staged);
            eprintln!(
                "kohagi: could not cache the compiled model in {} ({e}); \
                 this run will use a temporary copy",
                entry.display()
            );
            Ok(Compiled {
                path: fresh,
                from_cache: false,
            })
        }
    }
}

/// Where a package's compiled form belongs, or `None` if there is no usable
/// cache directory.
///
/// The key covers the package's path and the size and modification time of every
/// file in it. Hashing 260MB of weights on every start would cost more than it
/// saves, and metadata is enough to notice a re-conversion: the converter
/// rewrites the whole bundle.
fn cache_entry(pkg: &Path) -> Option<PathBuf> {
    let dir = cache_root()?.join("compiled");
    std::fs::create_dir_all(&dir).ok()?;
    entry_in(&dir, pkg)
}

/// Kohagi's CoreML cache directory, holding `compiled/` and — when the emitter is
/// built in — `converted/`. `$KOHAGI_COREML_CACHE` overrides it, which is how the
/// tests get a directory of their own.
///
/// Not created here: each caller makes only the subdirectory it needs.
pub(crate) fn cache_root() -> Option<PathBuf> {
    match std::env::var_os("KOHAGI_COREML_CACHE") {
        Some(v) => Some(PathBuf::from(v)),
        None => Some(PathBuf::from(std::env::var_os("HOME")?).join("Library/Caches/kohagi/coreml")),
    }
}

/// The entry `pkg` maps to inside `dir`, as `<bucket>-<where>-<what>.mlmodelc`.
///
/// The name carries the bucket so the cache can be read by a person, and two
/// hashes rather than one so a superseded entry can be told from an unrelated
/// model's. `where` is the package's location and `what` is its contents: two
/// models that both ship a `seq-128.mlpackage` differ in the first, and a
/// re-conversion of one of them differs only in the second.
fn entry_in(dir: &Path, pkg: &Path) -> Option<PathBuf> {
    let stem = pkg.file_stem()?.to_str()?;
    let (place, contents) = (identity(pkg)?, contents(pkg)?);
    Some(dir.join(format!("{stem}-{place:016x}-{contents:016x}.mlmodelc")))
}

/// Delete this package's earlier entries, which a re-conversion has superseded.
///
/// Only entries with the same bucket *and* the same location are touched, so
/// another model's compile of the same bucket length survives. Without this the
/// cache grows by the size of the model on every re-conversion, and nothing else
/// would ever clean it up.
fn evict_superseded(keep: &Path) {
    let Some((dir, name)) = keep.parent().zip(keep.file_name().and_then(|n| n.to_str())) else {
        return;
    };
    // `<stem>-<place>-`: everything up to and including the second separator.
    let Some(prefix) = name.rfind('-').map(|i| &name[..=i]) else {
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
        if other != name && other.starts_with(prefix) && other.ends_with(".mlmodelc") {
            let _ = std::fs::remove_dir_all(&path);
        }
    }
}

/// FNV-1a. Written out rather than taken from `DefaultHasher`, whose algorithm is
/// explicitly allowed to change between Rust releases — that would silently
/// invalidate every user's cache on a toolchain bump.
fn mix(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= u64::from(b);
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// Which package this is: its canonical path.
fn identity(pkg: &Path) -> Option<u64> {
    let mut h = FNV_OFFSET;
    mix(
        &mut h,
        pkg.canonicalize().ok()?.as_os_str().as_encoded_bytes(),
    );
    Some(h)
}

/// What is in it: every file's name, size and modification time.
fn contents(pkg: &Path) -> Option<u64> {
    fn walk(dir: &Path, h: &mut u64) -> std::io::Result<()> {
        // Sorted, so the hash does not depend on directory order.
        let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .collect();
        entries.sort();
        for path in entries {
            mix(h, path.file_name().unwrap_or_default().as_encoded_bytes());
            let meta = std::fs::metadata(&path)?;
            if meta.is_dir() {
                walk(&path, h)?;
            } else {
                mix(h, &meta.len().to_le_bytes());
                if let Ok(t) = meta.modified() {
                    if let Ok(d) = t.duration_since(std::time::UNIX_EPOCH) {
                        mix(h, &d.as_nanos().to_le_bytes());
                    }
                }
            }
        }
        Ok(())
    }
    let mut h = FNV_OFFSET;
    walk(pkg, &mut h).ok()?;
    Some(h)
}

/// Move a directory, falling back to copy-then-delete when `rename` cannot
/// (the compiler's output and the cache can sit on different volumes).
fn move_dir(from: &Path, to: &Path) -> std::io::Result<()> {
    match std::fs::rename(from, to) {
        Ok(()) => Ok(()),
        Err(_) => {
            copy_deref(from, to)?;
            let _ = std::fs::remove_dir_all(from);
            Ok(())
        }
    }
}

/// A `file://` URL for a local path.
unsafe fn file_url(path: &Path) -> Result<Retained<NSURL>> {
    Ok(NSURL::fileURLWithPath(&NSString::from_str(
        path.to_str().context("model path is not valid UTF-8")?,
    )))
}

/// Compile a `.mlpackage` to a `.mlmodelc` and return its (temporary) path.
///
/// The Hugging Face cache stores a package as a tree of symlinks into its blob
/// store, which the CoreML compiler cannot follow — it fails with a spurious
/// "file doesn't exist". So if the direct compile fails we retry from a
/// dereferenced, symlink-free copy. The first (direct) error is discarded on
/// purpose: for a symlinked package it is the misleading symlink error, and for
/// a genuinely broken package the dereferenced retry reports the real cause.
fn compile_package(pkg: &Path) -> Result<PathBuf> {
    // On the order of 20 seconds per bucket, and silent until now. Only reached
    // when the cache missed, so this prints on a first run and not afterwards.
    eprintln!(
        "kohagi: compiling {} for the Neural Engine — first run only ...",
        pkg.file_name().unwrap_or_default().to_string_lossy()
    );
    if let Ok(out) = compile_at(pkg) {
        return Ok(out);
    }
    let staging = unique_temp_dir("kohagi-coreml-src");
    std::fs::create_dir_all(&staging).with_context(|| format!("creating {}", staging.display()))?;
    let name = pkg.file_name().context("model path has no file name")?;
    let copy = staging.join(name);
    let result = copy_deref(pkg, &copy)
        .with_context(|| format!("dereferencing {}", pkg.display()))
        .and_then(|()| compile_at(&copy).with_context(|| format!("compiling {}", pkg.display())));
    let _ = std::fs::remove_dir_all(&staging);
    result
}

/// One `compileModelAtURL:` call; returns the compiled model's path.
fn compile_at(pkg: &Path) -> Result<PathBuf> {
    unsafe {
        let src = file_url(pkg)?;
        // The async compileModelAtURL:completionHandler: is the current API, but
        // the synchronous one is simpler and fine for a batch CLI.
        #[allow(deprecated)]
        let compiled =
            MLModel::compileModelAtURL_error(&src).map_err(|e| anyhow::anyhow!("{e}"))?;
        let path = compiled.path().context("compiled model URL has no path")?;
        Ok(PathBuf::from(path.to_string()))
    }
}

/// Recursively copy `src` to `dst`, following symlinks so the result has no
/// links — turns a symlinked HF-cache package into a real one the compiler can
/// read.
fn copy_deref(src: &Path, dst: &Path) -> std::io::Result<()> {
    if src.is_dir() {
        std::fs::create_dir_all(dst)?;
        for entry in std::fs::read_dir(src)? {
            let entry = entry?;
            copy_deref(&entry.path(), &dst.join(entry.file_name()))?;
        }
        Ok(())
    } else {
        std::fs::copy(src, dst).map(|_| ())
    }
}

/// A process-unique path under the system temp dir (a per-process counter is
/// enough — one process compiles a handful of buckets).
fn unique_temp_dir(prefix: &str) -> PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{prefix}-{}-{}", std::process::id(), n))
}

/// Download a CoreML model repo into the HF cache and return the snapshot dir.
///
/// A repo may ship, per bucket, a compiled `.mlmodelc`, a portable
/// `.mlpackage`, or both. To avoid downloading the redundant form when both are
/// present, we fetch only the preferred one for each bucket (`prefer`), falling
/// back to the other when a bucket ships just one. The metadata a converted
/// directory needs — `config.json`, `tokenizer.json`, and the
/// `1_Pooling/config.json` a checkpoint declares its pooling in — is always
/// fetched; other repo files are skipped.
pub fn fetch_from_hub(repo: &str, prefer: CoreMlForm) -> Result<PathBuf> {
    let api = hf_hub::api::sync::Api::new().context("initializing Hugging Face Hub client")?;
    let handle = api.model(repo.to_string());
    let info = handle
        .info()
        .with_context(|| format!("querying {repo} on the Hugging Face Hub"))?;

    // First pass: which forms does each bucket ship?
    let mut forms: BTreeMap<usize, (bool, bool)> = BTreeMap::new();
    for sibling in &info.siblings {
        if let Some((seqs, ext)) = bucket_of(&sibling.rfilename) {
            for seq in seqs {
                let seen = forms.entry(seq).or_default();
                match ext {
                    "mlmodelc" => seen.0 = true,
                    "mlpackage" => seen.1 = true,
                    _ => {}
                }
            }
        }
    }

    // Second pass: download config/tokenizer and only the chosen form's files.
    for sibling in &info.siblings {
        let f = &sibling.rfilename;
        if wanted(f, prefer, &forms) {
            handle
                .get(f)
                .with_context(|| format!("fetching {f} from {repo}"))?;
        }
    }

    let config = handle
        .get("config.json")
        .with_context(|| format!("{repo} has no config.json"))?;
    config
        .parent()
        .map(Path::to_path_buf)
        .context("downloaded config.json has no parent directory")
}

/// Whether to download a given repo file: the metadata files always, and for
/// each bucket only the preferred form (or the other one if the bucket ships
/// just that). `forms` maps seq -> (has .mlmodelc, has .mlpackage).
fn wanted(rfilename: &str, prefer: CoreMlForm, forms: &BTreeMap<usize, (bool, bool)>) -> bool {
    match bucket_of(rfilename) {
        // Every length in a multi-function bundle is served by the same file, so
        // the forms available for one of them stand for the whole bundle.
        Some((seqs, ext)) => {
            let (has_compiled, has_package) = forms.get(&seqs[0]).copied().unwrap_or_default();
            let chosen = match prefer {
                CoreMlForm::Compiled if has_compiled => "mlmodelc",
                CoreMlForm::Compiled => "mlpackage",
                CoreMlForm::Package if has_package => "mlpackage",
                CoreMlForm::Package => "mlmodelc",
            };
            ext == chosen
        }
        // `1_Pooling/config.json` is what [`crate::model`] reads the declared
        // pooling from, by the same path convention as a local checkpoint. Skip
        // it and a Hub-hosted conversion warns that it may not be a
        // sentence-embedding model, however faithfully it was converted.
        None => matches!(
            rfilename,
            "config.json" | "tokenizer.json" | "1_Pooling/config.json"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selective_download_picks_one_form_when_both_exist() {
        // A repo shipping both forms of seq-512 and only a package for seq-128.
        let forms = BTreeMap::from([(512usize, (true, true)), (128usize, (false, true))]);
        let get = |f: &str, p: CoreMlForm| wanted(f, p, &forms);

        // Compiled bundles live under compiled/. Prefer compiled: take the
        // .mlmodelc for 512, but the only form (pkg) for 128.
        assert!(get(
            "compiled/seq-512.mlmodelc/weights/weight.bin",
            CoreMlForm::Compiled
        ));
        assert!(!get("seq-512.mlpackage/Data/x", CoreMlForm::Compiled));
        assert!(get("seq-128.mlpackage/Data/x", CoreMlForm::Compiled));

        // Prefer package: take the .mlpackage for 512.
        assert!(get("seq-512.mlpackage/Data/x", CoreMlForm::Package));
        assert!(!get(
            "compiled/seq-512.mlmodelc/weights/weight.bin",
            CoreMlForm::Package
        ));

        // Metadata is always fetched; unrelated repo chrome is not.
        assert!(get("config.json", CoreMlForm::Compiled));
        assert!(get("tokenizer.json", CoreMlForm::Compiled));
        assert!(get("1_Pooling/config.json", CoreMlForm::Compiled));
        assert!(!get("README.md", CoreMlForm::Compiled));
        assert!(!get(".gitattributes", CoreMlForm::Compiled));
    }

    #[test]
    fn parse_bundle_reads_lengths_and_form() {
        let one = |v: Vec<usize>, e| Some((v, e));
        assert_eq!(
            parse_bundle("seq-128.mlpackage"),
            one(vec![128], "mlpackage")
        );
        assert_eq!(parse_bundle("seq-512.mlmodelc"), one(vec![512], "mlmodelc"));
        assert_eq!(parse_bundle("config.json"), None);
        assert_eq!(parse_bundle("seq-xyz.mlpackage"), None);
        assert_eq!(
            bucket_of("compiled/seq-256.mlmodelc/x/y"),
            one(vec![256], "mlmodelc")
        );

        // A multi-function bundle names every length it serves.
        assert_eq!(
            parse_bundle("buckets-128-256-512.mlpackage"),
            one(vec![128, 256, 512], "mlpackage")
        );
        assert_eq!(
            bucket_of("compiled/buckets-128-256.mlmodelc/weights/weight.bin"),
            one(vec![128, 256], "mlmodelc")
        );
        assert_eq!(parse_bundle("buckets-.mlpackage"), None);
        assert_eq!(parse_bundle("buckets-128-x.mlpackage"), None);

        // Only a multi-length bundle needs a function name; a `seq-<N>` bundle
        // is loaded through the model's default function.
        assert_eq!(function_name(128, 1), None);
        assert_eq!(function_name(256, 3), Some("seq_256".to_string()));
    }

    /// A fake package: the cache key only looks at names, sizes and mtimes, so it
    /// does not need to be a real model.
    fn fake_package(dir: &Path, weight: &[u8]) {
        let inner = dir.join("Data/com.apple.CoreML/weights");
        std::fs::create_dir_all(&inner).unwrap();
        std::fs::write(dir.join("Manifest.json"), b"{}").unwrap();
        std::fs::write(dir.join("Data/com.apple.CoreML/model.mlmodel"), b"proto").unwrap();
        std::fs::write(inner.join("weight.bin"), weight).unwrap();
    }

    #[test]
    fn a_cache_key_is_stable_but_follows_the_contents() {
        let root = std::env::temp_dir().join(format!("kohagi-key-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let (pkg, other) = (
            root.join("seq-128.mlpackage"),
            root.join("seq-256.mlpackage"),
        );
        fake_package(&pkg, b"weights");
        fake_package(&other, b"weights");

        let cache = root.join("cache");
        std::fs::create_dir_all(&cache).unwrap();
        let key = |p: &Path| entry_in(&cache, p).unwrap();

        // Same bundle, twice: one entry, and its name says which bucket it is.
        assert_eq!(key(&pkg), key(&pkg));
        let name = key(&pkg).file_name().unwrap().to_string_lossy().to_string();
        assert!(name.starts_with("seq-128-"), "{name}");
        assert!(name.ends_with(".mlmodelc"), "{name}");

        // Two bundles with byte-identical contents still get separate entries,
        // because the location is part of the key: a stale entry from another
        // directory must not be served for this one.
        assert_ne!(key(&pkg), key(&other));

        // Re-converting changes a weight's size, so the old entry is not reused.
        let before = key(&pkg);
        std::fs::write(
            pkg.join("Data/com.apple.CoreML/weights/weight.bin"),
            b"different weights",
        )
        .unwrap();
        assert_ne!(before, key(&pkg));

        // As does adding a file the conversion did not have before.
        let before = key(&pkg);
        std::fs::write(pkg.join("Data/com.apple.CoreML/extra"), b"x").unwrap();
        assert_ne!(before, key(&pkg));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn eviction_removes_only_this_packages_earlier_compiles() {
        let root = std::env::temp_dir().join(format!("kohagi-evict-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let (mine, theirs) = (
            root.join("a/seq-128.mlpackage"),
            root.join("b/seq-128.mlpackage"),
        );
        fake_package(&mine, b"weights");
        fake_package(&theirs, b"weights");
        let cache = root.join("cache");
        std::fs::create_dir_all(&cache).unwrap();

        // Three entries: the one to keep, an earlier compile of the same package,
        // and another directory's compile of the same bucket length.
        let keep = entry_in(&cache, &mine).unwrap();
        let stale = {
            let name = keep.file_name().unwrap().to_string_lossy();
            let (prefix, _) = name.rsplit_once('-').unwrap();
            cache.join(format!("{prefix}-0000000000000000.mlmodelc"))
        };
        let neighbour = entry_in(&cache, &theirs).unwrap();
        for d in [&keep, &stale, &neighbour] {
            std::fs::create_dir_all(d).unwrap();
        }
        assert_ne!(stale, keep);
        assert_ne!(neighbour, keep);

        evict_superseded(&keep);
        assert!(keep.is_dir(), "the entry being published must survive");
        assert!(
            !stale.exists(),
            "an earlier compile of the same package goes"
        );
        assert!(
            neighbour.is_dir(),
            "another directory's seq-128 must not be evicted"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn collect_prefers_a_dedicated_bundle_over_a_multi_function_one() {
        let dir = std::env::temp_dir().join(format!("kohagi-collect-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("buckets-128-256.mlpackage")).unwrap();
        std::fs::create_dir_all(dir.join("seq-128.mlpackage")).unwrap();

        let mut found = BTreeMap::new();
        collect_buckets(&dir, &mut found).unwrap();
        let _ = std::fs::remove_dir_all(&dir);

        // 128 ships both ways, so the dedicated bundle wins and needs no
        // function; 256 only exists inside the multi-function bundle.
        assert_eq!(
            found[&128].package,
            Some(Located {
                path: dir.join("seq-128.mlpackage"),
                function: None,
            })
        );
        assert_eq!(
            found[&256].package,
            Some(Located {
                path: dir.join("buckets-128-256.mlpackage"),
                function: Some("seq_256".into()),
            })
        );
    }
}
