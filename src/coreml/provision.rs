//! Provisioning: turning a model *source* (a local dir or a Hub repo) into
//! loaded [`MLModel`]s ready for the ANE. Covers three steps that all serve the
//! one goal of "get usable bucket models onto disk and into memory":
//!
//! - **download** the preferred form of each bucket from the Hub ([`fetch_from_hub`]),
//! - **locate** the `seq-<N>` bundles in a directory ([`collect_buckets`]),
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

/// A bucket's two possible on-disk forms: the compiled `.mlmodelc` and the
/// portable `.mlpackage`.
type BucketForms = (Option<PathBuf>, Option<PathBuf>);

/// Parse a bucket bundle name like `seq-128.mlmodelc` into `(length, form)`.
/// The single source of truth for the `seq-<N>.<ext>` naming scheme.
fn parse_bucket(name: &str) -> Option<(usize, &str)> {
    let (stem, ext) = name.rsplit_once('.')?;
    if ext != "mlmodelc" && ext != "mlpackage" {
        return None;
    }
    let seq = stem.strip_prefix("seq-")?.parse().ok()?;
    Some((seq, ext))
}

/// Parse a repo-relative path (`compiled/seq-128.mlmodelc/...` or
/// `seq-128.mlpackage/...`) into its bucket `(length, form)`.
fn bucket_of(rfilename: &str) -> Option<(usize, &str)> {
    let rel = rfilename.strip_prefix("compiled/").unwrap_or(rfilename);
    parse_bucket(rel.split('/').next()?)
}

/// Scan one directory for `seq-<N>` bucket bundles, recording each into `found`
/// keyed by sequence length: `.mlmodelc` in the compiled slot, `.mlpackage` in
/// the package slot.
pub(super) fn collect_buckets(
    dir: &Path,
    found: &mut BTreeMap<usize, BucketForms>,
) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        let Some((seq, ext)) = path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(parse_bucket)
        else {
            continue;
        };
        let slot = found.entry(seq).or_default();
        match ext {
            "mlmodelc" => slot.0 = Some(path),
            "mlpackage" => slot.1 = Some(path),
            _ => {}
        }
    }
    Ok(())
}

/// Load one bucket, preferring the compiled `.mlmodelc` and falling back to the
/// portable `.mlpackage`. At least one of the two is `Some` (the caller only
/// inserts a bucket when it finds a file).
pub(super) fn load_bucket(
    seq: usize,
    compiled: Option<&Path>,
    package: Option<&Path>,
) -> Result<Retained<MLModel>> {
    if let Some(c) = compiled {
        match load_model(c) {
            Ok(model) => return Ok(model),
            Err(e) if package.is_some() => {
                eprintln!(
                    "kohagi: seq-{seq}.mlmodelc did not load ({e:#}); \
                     compiling seq-{seq}.mlpackage instead"
                );
            }
            Err(e) => return Err(e).with_context(|| format!("loading {}", c.display())),
        }
    }
    let package = package.expect("load_bucket called with neither model form");
    load_model(package).with_context(|| format!("loading {}", package.display()))
}

/// Load one model, pinned to CPU+ANE. A `.mlpackage` is compiled to a
/// (temporary) `.mlmodelc` first; a `.mlmodelc` is loaded directly.
fn load_model(path: &Path) -> Result<Retained<MLModel>> {
    let compiled;
    let target = if path.extension().and_then(|e| e.to_str()) == Some("mlpackage") {
        compiled = compile_package(path)?;
        compiled.as_path()
    } else {
        path
    };
    unsafe {
        let url = file_url(target)?;
        let config = MLModelConfiguration::new();
        config.setComputeUnits(MLComputeUnits::CPUAndNeuralEngine);
        MLModel::modelWithContentsOfURL_configuration_error(&url, &config)
            .map_err(|e| anyhow::anyhow!("loading {}: {e}", path.display()))
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
/// back to the other when a bucket ships just one. `config.json` and
/// `tokenizer.json` are always fetched; other repo files are skipped.
pub fn fetch_from_hub(repo: &str, prefer: CoreMlForm) -> Result<PathBuf> {
    let api = hf_hub::api::sync::Api::new().context("initializing Hugging Face Hub client")?;
    let handle = api.model(repo.to_string());
    let info = handle
        .info()
        .with_context(|| format!("querying {repo} on the Hugging Face Hub"))?;

    // First pass: which forms does each bucket ship?
    let mut forms: BTreeMap<usize, (bool, bool)> = BTreeMap::new();
    for sibling in &info.siblings {
        if let Some((seq, ext)) = bucket_of(&sibling.rfilename) {
            let seen = forms.entry(seq).or_default();
            match ext {
                "mlmodelc" => seen.0 = true,
                "mlpackage" => seen.1 = true,
                _ => {}
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

/// Whether to download a given repo file: `config.json` / `tokenizer.json`
/// always, and for each bucket only the preferred form (or the other one if the
/// bucket ships just that). `forms` maps seq -> (has .mlmodelc, has .mlpackage).
fn wanted(rfilename: &str, prefer: CoreMlForm, forms: &BTreeMap<usize, (bool, bool)>) -> bool {
    match bucket_of(rfilename) {
        Some((seq, ext)) => {
            let (has_compiled, has_package) = forms.get(&seq).copied().unwrap_or_default();
            let chosen = match prefer {
                CoreMlForm::Compiled if has_compiled => "mlmodelc",
                CoreMlForm::Compiled => "mlpackage",
                CoreMlForm::Package if has_package => "mlpackage",
                CoreMlForm::Package => "mlmodelc",
            };
            ext == chosen
        }
        None => rfilename == "config.json" || rfilename == "tokenizer.json",
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
        assert!(!get("README.md", CoreMlForm::Compiled));
        assert!(!get(".gitattributes", CoreMlForm::Compiled));
    }

    #[test]
    fn parse_bucket_reads_length_and_form() {
        assert_eq!(parse_bucket("seq-128.mlpackage"), Some((128, "mlpackage")));
        assert_eq!(parse_bucket("seq-512.mlmodelc"), Some((512, "mlmodelc")));
        assert_eq!(parse_bucket("config.json"), None);
        assert_eq!(parse_bucket("seq-xyz.mlpackage"), None);
        assert_eq!(
            bucket_of("compiled/seq-256.mlmodelc/x/y"),
            Some((256, "mlmodelc"))
        );
    }
}
