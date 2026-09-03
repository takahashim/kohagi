//! Kohagi CLI: JSONL embedding over stdin/stdout, plus a `--text` one-shot
//! mode for quick checks. See PROTOCOL.md for the full contract.
//!
//! Exit codes: 0 = all input embedded, 2 = finished but some lines were
//! skipped (see stderr), 1 = fatal (model load, I/O, bad flags), 3 = the
//! requested CoreML backend cannot serve this request (built without the
//! feature, no `--coreml-dir`, or `--max-seq-length` beyond the largest
//! converted bucket) — caught before any input is read, so the caller can
//! retry on `--device cpu`.

use std::process::ExitCode;

use clap::{Parser, ValueEnum};

use kohagi::cli::{self, ModelArgs};
use kohagi::{stdio, ModelSource};

/// This binary's own output shape; the rest of the CLI value enums are shared
/// with `kohagi-rerank` in [`kohagi::cli`].
#[derive(Copy, Clone, ValueEnum)]
enum FormatArg {
    /// Kohagi's JSONL protocol: one record per line, ids echoed.
    Jsonl,
    /// One OpenAI `/v1/embeddings` response object for the whole run.
    Openai,
}

impl From<FormatArg> for stdio::Format {
    fn from(f: FormatArg) -> Self {
        match f {
            FormatArg::Jsonl => stdio::Format::Jsonl,
            FormatArg::Openai => stdio::Format::OpenAi,
        }
    }
}

/// Local sentence embeddings for Ruri v3 / ModernBERT models.
///
/// Reads {"id","text"} JSONL on stdin and writes {"id","embedding"} JSONL on
/// stdout; or embeds --text arguments directly. The model is downloaded from
/// the Hugging Face Hub on first use and cached (~/.cache/huggingface).
#[derive(Parser)]
#[command(name = "kohagi", version)]
struct Args {
    #[command(flatten)]
    model: ModelArgs,
    /// Prefix prepended to every text before embedding. Ruri v3 task
    /// prefixes: "検索文書: ", "検索クエリ: ", "トピック: ", or "" (plain
    /// sentence similarity).
    #[arg(long, default_value = "")]
    prefix: String,
    /// Shape of stdout. `jsonl` is Kohagi's protocol, one record per line,
    /// echoing each input's id. `openai` is one `/v1/embeddings` response
    /// object for the whole run, so code written against that API can read it
    /// unchanged — it identifies embeddings by position rather than by id, and
    /// an aborted run leaves an incomplete JSON document.
    #[arg(long, value_enum, default_value_t = FormatArg::Jsonl)]
    format: FormatArg,
    /// Add `n_tokens` (tokens embedded, specials included) and `truncated`
    /// (text ran past --max-seq-length, so its tail was dropped) to each output
    /// record. Off by default, keeping the plain protocol-1 output. The summary
    /// line always reports how many records were truncated regardless.
    #[arg(long)]
    report_tokens: bool,
    /// Embed these texts and exit, instead of reading stdin. Repeatable;
    /// output ids are the argument positions (0, 1, …).
    #[arg(long)]
    text: Vec<String>,
    /// Load the model, print one line of JSON describing it on stdout, and
    /// exit without reading stdin. The weights' sha256 is in there: record it
    /// beside a result and the question "which checkpoint produced this" has
    /// an answer that a renamed directory cannot change.
    #[arg(long, conflicts_with = "text")]
    print_model_info: bool,
}

/// `--text` mode: embed the arguments and print what stdio mode would, with the
/// argument positions as ids.
fn embed_arguments(args: &Args, source: &ModelSource, label: &str) -> anyhow::Result<()> {
    let embedder = args.model.load(source)?;
    let prefixed: Vec<String> = args
        .text
        .iter()
        .map(|t| format!("{}{t}", args.prefix))
        .collect();
    let texts: Vec<&str> = prefixed.iter().map(String::as_str).collect();
    let (vectors, tokens) = embedder.embed_with_tokens(&texts)?;

    // Same output shape as stdio mode; ids are the argument positions.
    let mut out = stdio::Writer::new(
        std::io::stdout().lock(),
        args.format.into(),
        label,
        args.report_tokens,
    );
    for (id, (embedding, info)) in vectors.iter().zip(&tokens).enumerate() {
        out.record(&serde_json::Value::from(id), embedding, info)?;
    }
    out.finish()
}

/// Returns the number of skipped input lines (0 in `--text` mode).
fn run(args: Args) -> anyhow::Result<usize> {
    let (source, label) = args.model.source()?;

    // The OpenAI item shape has nowhere to put per-record token counts, and
    // dropping what was asked for silently is worse than saying so. The total is
    // still reported, as `usage.prompt_tokens`.
    if args.report_tokens && matches!(args.format, FormatArg::Openai) {
        return Err(kohagi::UnsupportedRequest(
            "--report-tokens has no place in the OpenAI response shape; \
             usage.prompt_tokens carries the total, and per-record counts need \
             --format jsonl"
                .to_string(),
        )
        .into());
    }

    if args.print_model_info {
        // Checked here too, so `--print-model-info --expect-sha256 …` is a
        // standalone "is this the checkpoint I think it is" that exits 1.
        let embedder = args.model.load(&source)?;
        cli::print_model_info(&label, &embedder.info())?;
        return Ok(0);
    }

    if !args.text.is_empty() {
        embed_arguments(&args, &source, &label)?;
        return Ok(0);
    }

    stdio::run(
        || args.model.load(&source),
        &args.prefix,
        args.report_tokens,
        &label,
        args.format.into(),
    )
}

fn main() -> ExitCode {
    kohagi::program::set("kohagi");
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
