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
/// (temporary) `.mlmodelc` first; a `.mlmodelc` is loaded directly.
fn load_model(located: &Located) -> Result<Retained<MLModel>> {
    let compiled;
    let path = located.path.as_path();
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
