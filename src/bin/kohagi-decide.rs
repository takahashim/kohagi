//! Kohagi's typed-decision CLI: a state plus typed questions in, answers out.
//! See PROTOCOL-decide.md.
//!
//! A separate binary from `kohagi-rerank`: this one takes a state and named
//! questions, and returns the `choice`, `score` or `noul` that a closed set of
//! candidates implies. The checkpoint (`--model-id`) and the head (`--head`) are
//! chosen independently — a `cross-encoder` scores options one pair at a time, a
//! packed head reads them together.
//!
//! Exit codes match the other binaries: 0 = every line answered, 2 = finished
//! with skipped lines, 1 = fatal, 3 = the requested CoreML backend cannot serve
//! this request.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;

use kohagi::cli::{self, BackendArg, CoreMlFormArg, PrecisionArg};
use kohagi::decide as protocol;
use kohagi::decide::engine::{Engine, EngineOptions};

/// Answer typed questions about a state.
///
/// Reads {"id","state","questions"} JSONL on stdin and writes {"id","answers"}
/// JSONL on stdout. Each question is "choice", "score" or "noul".
#[derive(Parser)]
#[command(name = "kohagi-decide", version)]
struct Args {
    /// Hugging Face repo to download (ignored with --model-path).
    #[arg(long, default_value = "argos1111/modernbert-ja-310m-jev")]
    model_id: String,
    /// Which decision head reads the checkpoint. `cross-encoder` scores each
    /// (context, option) pair on its own (the argos Jev-local shape); `laya`
    /// reads state and options together (needs the `head-laya` feature).
    #[arg(long, default_value = "cross-encoder", value_name = "NAME")]
    head: String,
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
    /// Neural Engine and the head on the CPU (cross-encoder head only).
    #[arg(long, value_enum, default_value_t = BackendArg::Cpu)]
    device: BackendArg,
    /// Directory of a converted CoreML bundle for `--device coreml`, as
    /// `coreml-convert` writes it. Omit it to convert --model-id on first use.
    #[arg(long)]
    coreml_dir: Option<PathBuf>,
    /// Hugging Face repo holding that same layout, downloaded and cached on
    /// first use. `--coreml-dir` wins if both are set.
    #[arg(long)]
    coreml_model_id: Option<String>,
    /// Fixed sequence lengths to emit when `--device coreml` converts a
    /// checkpoint itself.
    #[arg(long, value_delimiter = ',', default_values_t = cli::RERANK_COREML_BUCKETS)]
    coreml_buckets: Vec<usize>,
    /// When a --coreml-model-id repo ships both forms of a bucket, which to
    /// download.
    #[arg(long, value_enum, default_value_t = CoreMlFormArg::Compiled)]
    coreml_prefer: CoreMlFormArg,
    /// Token-level truncation length for one (context, option) pair.
    #[arg(long, default_value_t = 512)]
    max_seq_length: usize,
    /// Bucketing granularity; memory stays bounded regardless.
    #[arg(long, default_value_t = 64)]
    batch_size: usize,
    /// Load the model, print one line of JSON describing it on stdout, and
    /// exit without reading stdin.
    /// Subfolder inside the repo or local dir that holds the checkpoint, for a
    /// model whose files are nested (e.g. `--subfolder multilingual`). Head
    /// dependent; the cross-encoder head is always flat.
    #[arg(long)]
    subfolder: Option<String>,
    /// Local model directory, laid out as the head expects (nested files under
    /// it). An alternative to --model-path, which names a flat safetensors file.
    #[arg(long)]
    model_dir: Option<PathBuf>,
    #[arg(long)]
    print_model_info: bool,
    /// Refuse to answer anything unless the loaded weights' sha256 starts with
    /// this hex prefix, as in `kohagi-rerank`.
    #[arg(long, value_name = "HEX")]
    expect_sha256: Option<String>,
}

impl Args {
    /// Where the checkpoint comes from, plus the name to show for it. The
    /// loading itself is [`Engine`]'s, so `--head` picks the reading and this
    /// only picks the file.
    fn source(&self) -> anyhow::Result<(kohagi::ModelSource, String)> {
        if let Some(dir) = &self.model_dir {
            return Ok((
                kohagi::ModelSource::Dir { root: dir.clone() },
                cli::label_of(dir),
            ));
        }
        cli::RerankModel {
            model_path: self.model_path.as_ref(),
            tokenizer_path: self.tokenizer_path.as_ref(),
            model_id: &self.model_id,
            device: self.device,
            precision: self.precision,
            coreml_dir: self.coreml_dir.as_ref(),
            coreml_model_id: self.coreml_model_id.as_deref(),
            coreml_buckets: &self.coreml_buckets,
            coreml_prefer: self.coreml_prefer,
            max_seq_length: self.max_seq_length,
            batch_size: self.batch_size,
            sigmoid: false,
            expect_sha256: None,
        }
        .source()
    }

    fn options(&self) -> EngineOptions {
        EngineOptions {
            max_seq_length: self.max_seq_length,
            batch_size: self.batch_size,
            precision: self.precision.into(),
            backend: self.device.into(),
            coreml_form: self.coreml_prefer.into(),
        }
    }
}

fn run(args: Args) -> anyhow::Result<usize> {
    let (source, label) = args.source()?;
    let options = args.options();
    let head = args.head.as_str();

    if args.print_model_info {
        let engine = Engine::load(&source, options, head, args.subfolder.as_deref())?;
        cli::print_model_info(&label, &engine.info())?;
        return Ok(0);
    }

    let expected = args.expect_sha256.as_deref();
    protocol::stdio::run(
        || {
            let engine = Engine::load(&source, options, head, args.subfolder.as_deref())?;
            cli::verify_fingerprint(expected, &engine.info())?;
            Ok(engine)
        },
        &label,
    )
}

fn main() -> ExitCode {
    kohagi::program::set("kohagi-decide");
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
