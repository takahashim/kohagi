//! Kohagi's cross-encoder CLI: `{"id","query","text"}` JSONL in, `{"id","score"}`
//! out. See PROTOCOL-rerank.md.
//!
//! A separate binary rather than a mode of `kohagi`, because `kohagi` is a pure
//! function from texts to vectors and this one is a function from pairs to
//! numbers: different input record, different output record, different model.
//! They share the encoder underneath (the library's `rerank` module) and every
//! protocol rule around it.
//!
//! Exit codes match `kohagi`: 0 = every pair scored, 2 = finished with skipped
//! lines, 1 = fatal.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, ValueEnum};

use kohagi::rerank::{self, Reranker};
use kohagi::{Backend, CoreMlForm, ModelSource, Precision};

#[derive(Clone, Copy, ValueEnum)]
enum PrecisionArg {
    /// Matches the PyTorch reference; works on every CPU.
    F32,
    /// ~2x faster on x86_64 CPUs with AVX512-BF16.
    Bf16,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum BackendArg {
    /// Apple Accelerate on macOS, candle's own gemm elsewhere.
    Cpu,
    /// Apple GPU. Needs a binary built with `--features metal`.
    Metal,
    /// NVIDIA GPU via CUDA. Needs a binary built with `--features cuda`.
    Cuda,
    /// Apple Neural Engine. Needs `--features coreml`; converts --model-id
    /// itself unless given a converted bundle.
    Coreml,
}

#[derive(Clone, Copy, ValueEnum)]
enum CoreMlFormArg {
    /// Compiled `.mlmodelc` — no per-run compile (default).
    Compiled,
    /// Portable `.mlpackage` — compiled on load, robust across OS versions.
    Package,
}

impl From<CoreMlFormArg> for CoreMlForm {
    fn from(f: CoreMlFormArg) -> Self {
        match f {
            CoreMlFormArg::Compiled => CoreMlForm::Compiled,
            CoreMlFormArg::Package => CoreMlForm::Package,
        }
    }
}

impl From<PrecisionArg> for Precision {
    fn from(p: PrecisionArg) -> Self {
        match p {
            PrecisionArg::F32 => Precision::F32,
            PrecisionArg::Bf16 => Precision::Bf16,
        }
    }
}

impl From<BackendArg> for Backend {
    fn from(b: BackendArg) -> Self {
        match b {
            BackendArg::Cpu => Backend::Cpu,
            BackendArg::Metal => Backend::Metal,
            BackendArg::Cuda => Backend::Cuda,
            BackendArg::Coreml => Backend::CoreML,
        }
    }
}

/// Score query/document pairs with a Japanese ModernBERT cross-encoder.
///
/// Reads {"id","query","text"} JSONL on stdin and writes {"id","score"} JSONL
/// on stdout. Retrieval finds candidates; this reorders them.
#[derive(Parser)]
#[command(name = "kohagi-rerank", version)]
struct Args {
    /// Hugging Face reranker repo to download (ignored with --model-path).
    #[arg(long, default_value = "cl-nagoya/ruri-v3-reranker-310m")]
    model_id: String,
    /// Local safetensors weights (offline mode; config.json must sit next to
    /// it). Requires --tokenizer-path.
    #[arg(long, requires = "tokenizer_path")]
    model_path: Option<PathBuf>,
    /// Local tokenizer.json (offline mode).
    #[arg(long, requires = "model_path")]
    tokenizer_path: Option<PathBuf>,
    /// Numeric precision of the forward pass.
    #[arg(long, value_enum, default_value_t = PrecisionArg::F32)]
    precision: PrecisionArg,
    /// Device for the forward pass. `coreml` runs the encoder on the Apple
    /// Neural Engine and the classification head on the CPU, which is where it
    /// runs on every backend.
    #[arg(long, value_enum, default_value_t = BackendArg::Cpu)]
    device: BackendArg,
    /// Directory of a converted CoreML bundle for `--device coreml`, as
    /// `coreml-convert` writes it: the bucket models, tokenizer.json,
    /// config.json, and the head.safetensors a reranker needs. Omit it to
    /// convert --model-id on first use.
    #[arg(long)]
    coreml_dir: Option<PathBuf>,
    /// Hugging Face repo holding that same layout, downloaded and cached on
    /// first use. `--coreml-dir` wins if both are set.
    #[arg(long)]
    coreml_model_id: Option<String>,
    /// Fixed sequence lengths to emit when `--device coreml` converts a
    /// checkpoint itself. A pair fills more of a bucket than a single text
    /// does, so these run longer than the embedding defaults.
    #[arg(long, value_delimiter = ',', default_values_t = [128usize, 256, 512])]
    coreml_buckets: Vec<usize>,
    /// When a --coreml-model-id repo ships both forms of a bucket, which to
    /// download.
    #[arg(long, value_enum, default_value_t = CoreMlFormArg::Compiled)]
    coreml_prefer: CoreMlFormArg,
    /// Report the raw logit instead of its sigmoid. The default matches
    /// sentence-transformers' CrossEncoder for a one-label model, so
    /// thresholds tuned there carry over unchanged.
    #[arg(long)]
    raw_logits: bool,
    /// Add `n_tokens` (tokens scored, both halves and specials included) and
    /// `truncated` to each output record.
    #[arg(long)]
    report_tokens: bool,
    /// Token-level truncation length for the pair. The longer half is trimmed
    /// first, as in the reference implementation.
    #[arg(long, default_value_t = 512)]
    max_seq_length: usize,
    /// Bucketing granularity; memory stays bounded regardless.
    #[arg(long, default_value_t = 64)]
    batch_size: usize,
    /// Score these pairs and exit, instead of reading stdin. Takes two values
    /// (query, then text); repeatable. Output ids are the pair positions.
    #[arg(long, num_args = 2, value_names = ["QUERY", "TEXT"])]
    pair: Vec<String>,
    /// Load the model, print one line of JSON describing it on stdout, and
    /// exit without reading stdin.
    #[arg(long, conflicts_with = "pair")]
    print_model_info: bool,
}

impl Args {
    fn options(&self) -> rerank::Options {
        rerank::Options {
            max_seq_length: self.max_seq_length,
            batch_size: self.batch_size,
            precision: self.precision.into(),
            backend: self.device.into(),
            sigmoid: !self.raw_logits,
            coreml_form: self.coreml_prefer.into(),
        }
    }

    /// Where the model comes from, plus the name to show in the summary.
    fn source(&self) -> anyhow::Result<(ModelSource, String)> {
        // CoreML loads converted fixed-shape models — a directory, a Hub repo,
        // or a checkpoint converted on first use — rather than safetensors.
        if self.device == BackendArg::Coreml {
            if let Some(dir) = self.coreml_dir.clone() {
                let label = label_of(&dir);
                return Ok((ModelSource::CoreMl { dir }, label));
            }
            if let Some(repo) = self.coreml_model_id.clone() {
                let label = repo.clone();
                return Ok((ModelSource::CoreMlHub { repo }, label));
            }
            let mut buckets = self.coreml_buckets.clone();
            buckets.sort_unstable();
            buckets.dedup();
            anyhow::ensure!(
                !buckets.is_empty(),
                "`--coreml-buckets` is empty; give at least one sequence length"
            );
            let (checkpoint, label) = self.checkpoint_source();
            return Ok((
                ModelSource::CoreMlConvert {
                    checkpoint: Box::new(checkpoint),
                    buckets,
                    quantize: kohagi::CoreMlQuantize::None,
                },
                label,
            ));
        }
        Ok(self.checkpoint_source())
    }

    fn checkpoint_source(&self) -> (ModelSource, String) {
        match (&self.model_path, &self.tokenizer_path) {
            // clap's `requires` guarantees these two arrive together.
            (Some(model), Some(tokenizer)) => {
                let label = label_of(model);
                (
                    ModelSource::Files {
                        model: model.clone(),
                        tokenizer: tokenizer.clone(),
                    },
                    label,
                )
            }
            _ => (
                ModelSource::Hub {
                    repo: self.model_id.clone(),
                },
                self.model_id.clone(),
            ),
        }
    }
}

/// A checkpoint's file is nearly always `model.safetensors`; the directory is
/// the part a caller chose. Same rule as `kohagi`'s.
fn label_of(path: &Path) -> String {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string());
    if name == "model.safetensors" {
        if let Some(dir) = path.parent().and_then(Path::file_name) {
            return dir.to_string_lossy().into_owned();
        }
    }
    name
}

fn print_model_info(reranker: &Reranker, label: &str) -> anyhow::Result<()> {
    #[derive(serde::Serialize)]
    struct Printed<'a> {
        model: &'a str,
        #[serde(flatten)]
        info: kohagi::ModelInfo,
    }

    println!(
        "{}",
        serde_json::to_string(&Printed {
            model: label,
            info: reranker.info(),
        })?
    );
    Ok(())
}

/// `--pair` mode: score the arguments and print what stdio mode would, with
/// the pair positions as ids.
fn score_arguments(args: &Args, reranker: &Reranker) -> anyhow::Result<()> {
    let pairs: Vec<(&str, &str)> = args
        .pair
        .chunks(2)
        .map(|p| (p[0].as_str(), p[1].as_str()))
        .collect();
    let (scores, _) = reranker.score(&pairs)?;
    for (id, score) in scores.iter().enumerate() {
        println!("{}", serde_json::json!({"id": id, "score": score}));
    }
    Ok(())
}

fn run(args: Args) -> anyhow::Result<usize> {
    let (source, label) = args.source()?;
    let opts = args.options();

    if args.print_model_info || !args.pair.is_empty() {
        let reranker = Reranker::load(&source, opts)?;
        if args.print_model_info {
            print_model_info(&reranker, &label)?;
        } else {
            score_arguments(&args, &reranker)?;
        }
        return Ok(0);
    }

    rerank::stdio::run(|| Reranker::load(&source, opts), args.report_tokens, &label)
}

fn main() -> ExitCode {
    match run(Args::parse()) {
        Ok(0) => ExitCode::SUCCESS,
        Ok(_) => ExitCode::from(2),
        Err(e) => {
            eprintln!("kohagi-rerank: error: {e:#}");
            if e.chain().any(|c| c.is::<kohagi::UnsupportedRequest>()) {
                ExitCode::from(3)
            } else {
                ExitCode::FAILURE
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_help_and_flags_are_well_formed() {
        use clap::CommandFactory;
        Args::command().debug_assert();
    }

    #[test]
    fn a_checkpoint_is_labelled_by_its_directory() {
        assert_eq!(
            label_of(Path::new("/m/rerankers/xsmall-v2/model.safetensors")),
            "xsmall-v2"
        );
        assert_eq!(
            label_of(Path::new("/m/reranker.safetensors")),
            "reranker.safetensors"
        );
    }
}
