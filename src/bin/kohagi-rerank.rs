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
//! lines, 1 = fatal, 3 = the requested CoreML backend cannot serve this request.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;

use kohagi::cli::{self, BackendArg, CoreMlFormArg, PrecisionArg};
use kohagi::rerank::{self, Reranker};
use kohagi::ModelSource;

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
    /// Refuse to score anything unless the loaded weights' sha256 starts with
    /// this hex prefix — paste the 12 digits from a summary line or the full
    /// digest from --print-model-info. A threshold belongs to the weights it
    /// was tuned on, and this stops the wrong ones with exit 1 before any pair
    /// is answered. With --device coreml the bundle's recorded source_sha256
    /// is checked instead, and a bundle that recorded none is refused.
    #[arg(long, value_name = "HEX")]
    expect_sha256: Option<String>,
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
        let checkpoint = cli::checkpoint_source(
            self.model_path.as_ref(),
            self.tokenizer_path.as_ref(),
            &self.model_id,
        );
        // CoreML loads converted fixed-shape models rather than safetensors.
        // Never quantized: a reranker's output is one number being compared
        // against a threshold, and int8 moves it further than fp16 already does.
        if self.device == BackendArg::Coreml {
            return cli::coreml_source(
                self.coreml_dir.as_ref(),
                self.coreml_model_id.as_deref(),
                &self.coreml_buckets,
                kohagi::CoreMlQuantize::None,
                checkpoint,
            );
        }
        Ok(checkpoint)
    }
}

/// Load the model and, when `--expect-sha256` pinned a digest, refuse weights
/// that do not carry it — before anything is scored, whichever mode loads.
fn load_checked(args: &Args, source: &ModelSource) -> anyhow::Result<Reranker> {
    let reranker = Reranker::load(source, args.options())?;
    cli::verify_fingerprint(args.expect_sha256.as_deref(), &reranker.info())?;
    Ok(reranker)
}

/// `--pair` mode: send argument pairs to the protocol, which writes them like
/// stdin input.
fn score_arguments(
    pair: &[String],
    reranker: &Reranker,
    report_tokens: bool,
) -> anyhow::Result<()> {
    // `num_args = 2` accepts exactly two values or rejects the flag, so this
    // slice has an even length and `as_chunks` leaves no remainder.
    let pairs: Vec<(&str, &str)> = pair
        .as_chunks::<2>()
        .0
        .iter()
        .map(|[q, t]| (q.as_str(), t.as_str()))
        .collect();
    rerank::stdio::run_pairs(reranker, &pairs, report_tokens)
}

fn run(args: Args) -> anyhow::Result<usize> {
    let (source, label) = args.source()?;

    if args.print_model_info || !args.pair.is_empty() {
        let reranker = load_checked(&args, &source)?;
        if args.print_model_info {
            cli::print_model_info(&label, &reranker.info())?;
        } else {
            score_arguments(&args.pair, &reranker, args.report_tokens)?;
        }
        return Ok(0);
    }

    rerank::stdio::run(|| load_checked(&args, &source), args.report_tokens, &label)
}

fn main() -> ExitCode {
    kohagi::program::set("kohagi-rerank");
    cli::exit_code(run(Args::parse()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_help_and_flags_are_well_formed() {
        use clap::CommandFactory;
        Args::command().debug_assert();
    }
}
