//! Convert a ModernBERT checkpoint into the CoreML layout Kohagi's
//! `--device coreml` reads, from Rust rather than through
//! `scripts/convert_coreml.py`.
//!
//! ```console
//! cargo run --release --bin coreml-convert --features coreml-export -- \
//!     --model-id cl-nagoya/ruri-v3-130m \
//!     --out-dir models/ruri-v3-130m-coreml \
//!     --sequence-lengths 128,256,512
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

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{bail, Context, Result};
use clap::Parser;
use kohagi::coreml_export::{
    encoder::{self, EncoderConfig, Options},
    modernbert::Activation,
    safetensors::Checkpoint,
    write_package, Provenance,
};

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
    /// The largest is the ceiling on Kohagi's `--max-seq-length`. ANE latency is
    /// not monotonic in length, so measure a candidate set with
    /// `tools/coreml-jigs`' `bucket-latency` rather than assuming.
    #[arg(long, value_delimiter = ',', default_value = "128,256,512")]
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

/// The files a converted directory carries beside the bundle, and whether Kohagi
/// can manage without one.
const METADATA: [(&str, bool); 3] = [
    ("config.json", true),
    ("tokenizer.json", true),
    // Kohagi falls back to mean pooling and warns when this is absent, which is
    // alarming for a faithfully converted encoder — so it is fetched when the
    // checkpoint has one, and merely noted when it does not.
    ("1_Pooling/config.json", false),
];

/// What a conversion needs on disk before it can start.
struct Sources {
    weights: PathBuf,
    config: String,
    /// Metadata files to place beside the bundle, as `(name, source path)`.
    metadata: Vec<(&'static str, PathBuf)>,
}

/// Resolve the checkpoint and the metadata files, downloading if needed.
fn gather(args: &Args) -> Result<Sources> {
    if let Some(repo) = &args.model_id {
        let api =
            hf_hub::api::sync::Api::new().context("initializing the Hugging Face Hub client")?;
        let handle = api.model(repo.clone());
        let fetch = |name: &str| -> Result<PathBuf> {
            handle
                .get(name)
                .with_context(|| format!("fetching {name} from {repo}"))
        };
        let weights = fetch("model.safetensors")?;
        let mut files = Vec::new();
        for (name, required) in METADATA {
            match fetch(name) {
                Ok(path) => files.push((name, path)),
                Err(e) if !required => eprintln!("  no {name} ({e:#})"),
                Err(e) => return Err(e),
            }
        }
        let config = files
            .iter()
            .find(|(name, _)| *name == "config.json")
            .map(|(_, path)| std::fs::read_to_string(path))
            .transpose()?
            .context("the repo has no config.json")?;
        return Ok(Sources {
            weights,
            config,
            metadata: files,
        });
    }

    let Some(weights) = args.model_path.clone() else {
        bail!("pass either --model-id or --model-path with --config-path");
    };
    let config_path = args.config_path.clone().expect("clap requires it");
    let config = std::fs::read_to_string(&config_path)
        .with_context(|| format!("reading {}", config_path.display()))?;
    let mut files = vec![("config.json", config_path)];
    if let Some(tokenizer) = &args.tokenizer_path {
        files.push(("tokenizer.json", tokenizer.clone()));
    } else {
        eprintln!(
            "  no --tokenizer-path; the output will need a tokenizer.json copied in \
             before Kohagi can use it"
        );
    }
    Ok(Sources {
        weights,
        config,
        metadata: files,
    })
}

/// Copy `from` to `<out>/<name>`, creating the parent directory a nested name
/// needs. Dereferences, since the Hub cache is a tree of symlinks and the output
/// is meant to be uploadable as it stands.
fn place(out: &Path, name: &str, from: &Path) -> Result<()> {
    let to = out.join(name);
    if let Some(parent) = to.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    std::fs::copy(from, &to).with_context(|| format!("copying {name} into {}", out.display()))?;
    Ok(())
}

fn run(args: &Args) -> Result<()> {
    let mut lengths = args.sequence_lengths.clone();
    lengths.sort_unstable();
    lengths.dedup();
    if lengths.is_empty() || lengths[0] == 0 {
        bail!("--sequence-lengths must be positive");
    }

    let sources = gather(args)?;
    let cfg = EncoderConfig::from_json(&sources.config)?;
    eprintln!(
        "config  : hidden {}, {} layers, {} heads, vocab {}, gate {}",
        cfg.hidden,
        cfg.layers,
        cfg.heads,
        cfg.vocab,
        cfg.activation.name()
    );

    // Before opening a 500MB checkpoint: a bucket past the trained positions has no
    // RoPE frequencies behind it, and `emit_all` would refuse anyway.
    if let Some(max) = cfg.max_positions {
        if let Some(&over) = lengths.iter().find(|&&s| s > max) {
            bail!(
                "--sequence-lengths {over} is past this checkpoint's \
                 max_position_embeddings ({max})"
            );
        }
    }

    let checkpoint = Checkpoint::open(&sources.weights)?;
    let source = args
        .model_id
        .clone()
        .unwrap_or_else(|| sources.weights.display().to_string());
    let opts = Options {
        quantize_embeddings: args.quantize_embeddings || args.quantize_all,
        quantize_projections: args.quantize_all,
    };
    let provenance = Provenance {
        source: source.clone(),
        lengths: lengths.clone(),
        quantized_embeddings: opts.quantize_embeddings,
        quantized_projections: opts.quantize_projections,
        activation: (cfg.activation != Activation::default()).then(|| cfg.activation.name()),
    };

    eprintln!(
        "emitting: {} as one bundle from {source} ...",
        lengths
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    );
    let (model, blob) = encoder::emit_with(&cfg, &checkpoint, &lengths, &provenance, &opts)?;

    let name = format!(
        "buckets-{}.mlpackage",
        lengths
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("-")
    );
    let bundle = args.out_dir.join(&name);
    std::fs::create_dir_all(&args.out_dir)
        .with_context(|| format!("creating {}", args.out_dir.display()))?;
    write_package(&bundle, &model, &blob)?;
    eprintln!("wrote   : {name} ({:.1} MB)", blob.len() as f64 / 1e6);

    if args.compiled {
        #[cfg(feature = "coreml")]
        {
            eprintln!("compiling (this takes ~20s per bucket) ...");
            let out = kohagi::coreml_export::compile_beside(&bundle)?;
            eprintln!(
                "  compiled {}",
                out.strip_prefix(&args.out_dir).unwrap_or(&out).display()
            );
        }
        #[cfg(not(feature = "coreml"))]
        bail!("--compiled needs a build with `--features coreml,coreml-export`");
    }

    for (file, from) in &sources.metadata {
        place(&args.out_dir, file, from)?;
        eprintln!("  copied {file}");
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
    match run(&Args::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("coreml-convert: error: {e:#}");
            ExitCode::FAILURE
        }
    }
}
