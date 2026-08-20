//! Convert a ModernBERT checkpoint into the CoreML layout Kohagi's
//! `--device coreml` reads, from Rust rather than through
//! `scripts/convert_coreml.py`.
//!
//! ```console
//! cargo run --release --bin coreml-convert --features coreml-export -- \
//!     --model-id cl-nagoya/ruri-v3-130m \
//!     --out-dir models/ruri-v3-130m-coreml \
//!     --sequence-lengths 64,128,256,512
//! ```
//!
//! A separate binary rather than a `kohagi` subcommand: writing a bundle to a
//! directory is something a publisher does once per model, while `--device coreml`
//! converts into its own cache without going through here. `required-features`
//! keeps it out of the default build, and only `kohagi` is packaged for release.
//!
//! What it writes, which is the layout `src/coreml.rs` looks for:
//!
//! ```text
//! <out-dir>/
//!   buckets-128-256-512.mlpackage    # one CoreML function per length
//!   config.json
//!   tokenizer.json
//!   1_Pooling/config.json            # if the checkpoint ships one
//! ```
//!
//! Every length shares one copy of the weights, so adding a bucket costs almost
//! nothing on disk.
//! No `compiled/` directory: Kohagi compiles a package on first use and caches the
//! result, so shipping one only moves ~20 s off the first run at the cost of
//! doubling the download.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{bail, Context, Result};
use clap::Parser;
use kohagi::coreml_export::{bundle_name, convert, encoder::Options, Checkpoint};

#[derive(Parser)]
#[command(
    about = "Convert a ModernBERT encoder to Kohagi's CoreML layout",
    long_about = None
)]
struct Args {
    /// Hugging Face model repo to convert. Downloaded and cached on first use.
    #[arg(long, conflicts_with = "model_path")]
    model_id: Option<String>,

    /// Local `model.safetensors`, instead of `--model-id`. Needs `--config-path`.
    #[arg(long, requires = "config_path")]
    model_path: Option<PathBuf>,

    /// Local `config.json`, for use with `--model-path`.
    #[arg(long)]
    config_path: Option<PathBuf>,

    /// Local `tokenizer.json` to copy into the output. Taken from the Hub repo
    /// when `--model-id` is used.
    #[arg(long)]
    tokenizer_path: Option<PathBuf>,

    /// Where to write the converted layout. Created if missing; an existing
    /// bundle of the same name is replaced.
    #[arg(long)]
    out_dir: PathBuf,

    /// Fixed sequence lengths to serve, as CoreML functions in one bundle.
    ///
    /// The largest is the ceiling on Kohagi's `--max-seq-length`. The lengths
    /// share one copy of the weights, so a longer list costs no disk — but each
    /// is a model to open, so a bucket nothing lands in is pure overhead. ANE
    /// latency is also not monotonic in length, so measure a candidate set with
    /// `tools/coreml-jigs`' `bucket-latency` rather than assuming.
    #[arg(long, value_delimiter = ',', default_value = "64,128,256,512")]
    sequence_lengths: Vec<usize>,

    /// Store the embedding table as int8 instead of fp16, dequantized inside the
    /// graph.
    ///
    /// For a large vocabulary that is most of the bytes. Measured on bekko: a
    /// two-bucket bundle went from 247MB to 149MB at an unchanged JaCWIR MAP@10.
    /// Vectors move by about 7e-4, so a quantized
    /// bundle and an fp16 one must not share a search index.
    #[arg(long)]
    quantize_embeddings: bool,

    /// Also store every projection weight as int8, with a scale per output
    /// channel. Implies `--quantize-embeddings`.
    ///
    /// Measured on bekko as 149MB to 125MB with JaCWIR MAP@10 moving 0.8843 to
    /// 0.8838. Retrieval quality has not been
    /// measured for this implementation — only the cosine distance to the CPU path.
    #[arg(long)]
    quantize_all: bool,

    /// Also emit `compiled/<name>.mlmodelc`, so a user's first run does not pay the
    /// ~20 s compile.
    ///
    /// Doubles the output. Kohagi caches its own compile
    /// (`~/Library/Caches/kohagi/coreml`), so this only moves the cost off the first
    /// run. Needs a build with
    /// `--features coreml,coreml-export`.
    #[arg(long)]
    compiled: bool,
}

/// Resolve the checkpoint's files, downloading if needed.
///
/// The shape of the result is `coreml_export::Checkpoint`, the same thing
/// `--device coreml` builds from an already-resolved model, so both go through
/// one conversion.
fn gather(args: &Args) -> Result<Checkpoint> {
    if let Some(repo) = &args.model_id {
        let api =
            hf_hub::api::sync::Api::new().context("initializing the Hugging Face Hub client")?;
        let handle = api.model(repo.clone());
        let fetch = |name: &str| -> Result<PathBuf> {
            handle
                .get(name)
                .with_context(|| format!("fetching {name} from {repo}"))
        };
        // Optional: a reranker or a base LM ships no pooling declaration, and a
        // 404 there is information rather than an error.
        let pooling = fetch("1_Pooling/config.json")
            .inspect_err(|e| eprintln!("  no 1_Pooling/config.json ({e:#})"))
            .ok();
        return Ok(Checkpoint {
            weights: fetch("model.safetensors")?,
            config: fetch("config.json")?,
            tokenizer: fetch("tokenizer.json")?,
            pooling,
            source: repo.clone(),
        });
    }

    let Some(weights) = args.model_path.clone() else {
        bail!("pass either --model-id or --model-path with --config-path");
    };
    let Some(tokenizer) = args.tokenizer_path.clone() else {
        bail!(
            "--model-path needs --tokenizer-path: a converted directory Kohagi can \
             load carries its own tokenizer.json"
        );
    };
    Ok(Checkpoint {
        config: args.config_path.clone().expect("clap requires it"),
        pooling: weights
            .parent()
            .map(|d| d.join("1_Pooling").join("config.json"))
            .filter(|p| p.is_file()),
        source: weights.display().to_string(),
        weights,
        tokenizer,
    })
}

fn run(args: &Args) -> Result<()> {
    let mut lengths = args.sequence_lengths.clone();
    lengths.sort_unstable();
    lengths.dedup();
    if lengths.is_empty() || lengths[0] == 0 {
        bail!("--sequence-lengths must be positive");
    }
    // Before the download and the emit, not after: a build that cannot compile
    // should cost a message rather than 260MB of bundle it then refuses to finish.
    #[cfg(not(feature = "coreml"))]
    if args.compiled {
        bail!("--compiled needs a build with `--features coreml,coreml-export`");
    }

    let checkpoint = gather(args)?;
    let opts = Options {
        quantize_embeddings: args.quantize_embeddings || args.quantize_all,
        quantize_projections: args.quantize_all,
    };
    eprintln!(
        "emitting: {} as one bundle from {} ...",
        lengths
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", "),
        checkpoint.source
    );

    let cfg = convert(&args.out_dir, &checkpoint, &lengths, &opts)?;
    let name = bundle_name(&lengths);
    eprintln!(
        "config  : hidden {}, {} layers, {} heads, vocab {}, gate {}",
        cfg.hidden,
        cfg.layers,
        cfg.heads,
        cfg.vocab,
        cfg.activation.name()
    );
    eprintln!("wrote   : {name}");

    if args.compiled {
        #[cfg(feature = "coreml")]
        {
            eprintln!("compiling (this takes ~20s per bucket) ...");
            let out = kohagi::coreml_export::compile_beside(&args.out_dir.join(&name))?;
            eprintln!(
                "  compiled {}",
                out.strip_prefix(&args.out_dir).unwrap_or(&out).display()
            );
        }
        // Unreachable: `run` refuses `--compiled` on such a build before it starts.
        #[cfg(not(feature = "coreml"))]
        bail!("--compiled needs a build with `--features coreml,coreml-export`");
    }

    eprintln!("\ndone -> {}", args.out_dir.display());
    eprintln!(
        "try: kohagi --device coreml --coreml-dir {} --text '瑠璃も玻璃も照らせば光る'",
        args.out_dir.display()
    );
    eprintln!(
        "check: tools/coreml-jigs' coreml-inspect {}",
        args.out_dir.display()
    );
    Ok(())
}

fn main() -> ExitCode {
    // Set the prefix before the emitter can write warnings.
    kohagi::program::set("coreml-convert");
    match run(&Args::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{}: error: {e:#}", kohagi::program::name());
            ExitCode::FAILURE
        }
    }
}
