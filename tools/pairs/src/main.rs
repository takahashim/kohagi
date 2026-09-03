//! An mteb reranking dataset, as the (query, text) pairs a distillation
//! reads.
//!
//! The dataset ships as four parquet files (queries, corpus, qrels,
//! top_ranked) and a distillation wants one flat list of pairs. This walks
//! the candidates each query was given, looks up both texts, and writes one
//! line per pair with the label carried alongside.
//!
//! The label rides along rather than being consumed here: the teacher's
//! score is what the student learns from, and the label is what the
//! experiment is measured against afterwards. Both belong to the same row,
//! and `kohagi-dataset` carries anything it does not recognize through to
//! its output, so a row that starts here arrives whole.
//!
//!     cargo run --release --manifest-path tools/pairs/Cargo.toml -- \
//!       --dataset <ESCIReranking snapshot> --language jp --split test \
//!       --queries 200 --candidates 10 --out pairs.jsonl
//!
//! Queries are taken in file order and candidates in the order the dataset
//! ranked them, so the same arguments give the same file. Nothing here is
//! random.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use arrow::array::{Array, AsArray, StringArray};
use arrow::datatypes::Int64Type;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

const USAGE: &str = "\
usage: kohagi-pairs --dataset <dir> --out <pairs.jsonl>
                    [--language jp] [--split test]
                    [--queries N] [--candidates N]

<dir> is an mteb reranking snapshot: it holds <language>-queries,
<language>-corpus, <language>-qrels and <language>-top_ranked, each with
one parquet file per split.";

fn main() -> Result<()> {
    let cli = Cli::parse()?;
    let queries = strings(&cli.file("queries")?, "_id", "text")?;
    let corpus = strings(&cli.file("corpus")?, "_id", "text")?;
    let labels = scores(&cli.file("qrels")?)?;
    let ranked = candidates(&cli.file("top_ranked")?)?;

    let mut out = std::io::BufWriter::new(
        std::fs::File::create(&cli.out)
            .with_context(|| format!("writing {}", cli.out.display()))?,
    );
    let (mut written, mut used) = (0usize, 0usize);
    for (query_id, docs) in ranked.iter().take(cli.queries) {
        let Some(query) = queries.get(query_id) else { continue };
        used += 1;
        for doc_id in docs.iter().take(cli.candidates) {
            let Some(text) = corpus.get(doc_id) else { continue };
            let row = serde_json::json!({
                "query": query,
                "text": text,
                "query_id": query_id,
                "doc_id": doc_id,
                "relevance": labels.get(&(query_id.clone(), doc_id.clone())).copied().unwrap_or(0),
            });
            writeln!(out, "{row}")?;
            written += 1;
        }
    }
    out.flush()?;
    eprintln!(
        "wrote {} ({written} pairs over {used} queries)",
        cli.out.display()
    );
    Ok(())
}

/// Two string columns as a map, which is what three of the four files are.
fn strings(path: &Path, key: &str, value: &str) -> Result<HashMap<String, String>> {
    let mut map = HashMap::new();
    for batch in reader(path)? {
        let batch = batch?;
        let keys = column::<StringArray>(&batch, key, path)?;
        let values = column::<StringArray>(&batch, value, path)?;
        for i in 0..batch.num_rows() {
            map.insert(keys.value(i).to_string(), values.value(i).to_string());
        }
    }
    Ok(map)
}

/// The relevance of each (query, document) the dataset labelled.
fn scores(path: &Path) -> Result<HashMap<(String, String), i64>> {
    let mut map = HashMap::new();
    for batch in reader(path)? {
        let batch = batch?;
        let queries = column::<StringArray>(&batch, "query-id", path)?;
        let docs = column::<StringArray>(&batch, "corpus-id", path)?;
        let scores = batch
            .column_by_name("score")
            .with_context(|| format!("{} has no score column", path.display()))?
            .as_primitive_opt::<Int64Type>()
            .with_context(|| format!("{}: score is not an integer column", path.display()))?
            .clone();
        for i in 0..batch.num_rows() {
            map.insert(
                (queries.value(i).to_string(), docs.value(i).to_string()),
                scores.value(i),
            );
        }
    }
    Ok(map)
}

/// The candidates each query was given, in the order they were given, and
/// the queries in the order the file holds them.
fn candidates(path: &Path) -> Result<Vec<(String, Vec<String>)>> {
    let mut all = Vec::new();
    for batch in reader(path)? {
        let batch = batch?;
        let queries = column::<StringArray>(&batch, "query-id", path)?;
        let lists = batch
            .column_by_name("corpus-ids")
            .with_context(|| format!("{} has no corpus-ids column", path.display()))?
            .as_list_opt::<i32>()
            .with_context(|| format!("{}: corpus-ids is not a list column", path.display()))?
            .clone();
        for i in 0..batch.num_rows() {
            let ids = lists.value(i);
            let ids = ids
                .as_any()
                .downcast_ref::<StringArray>()
                .with_context(|| format!("{}: corpus-ids holds something else", path.display()))?;
            all.push((
                queries.value(i).to_string(),
                (0..ids.len()).map(|j| ids.value(j).to_string()).collect(),
            ));
        }
    }
    Ok(all)
}

fn reader(path: &Path) -> Result<parquet::arrow::arrow_reader::ParquetRecordBatchReader> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("reading {}", path.display()))?;
    Ok(ParquetRecordBatchReaderBuilder::try_new(file)?.build()?)
}

fn column<T: 'static + Clone>(
    batch: &arrow::record_batch::RecordBatch,
    name: &str,
    path: &Path,
) -> Result<T> {
    batch
        .column_by_name(name)
        .with_context(|| format!("{} has no {name} column", path.display()))?
        .as_any()
        .downcast_ref::<T>()
        .with_context(|| format!("{}: {name} is not the type this expects", path.display()))
        .cloned()
}

struct Cli {
    dataset: PathBuf,
    out: PathBuf,
    language: String,
    split: String,
    queries: usize,
    candidates: usize,
}

impl Cli {
    /// Where one of the four parts of the dataset is. The name is the
    /// dataset's own layout, not a convention invented here.
    fn file(&self, part: &str) -> Result<PathBuf> {
        let dir = self.dataset.join(format!("{}-{part}", self.language));
        let mut found: Vec<PathBuf> = std::fs::read_dir(&dir)
            .with_context(|| format!("reading {}", dir.display()))?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|path| {
                path.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(&self.split) && n.ends_with(".parquet"))
            })
            .collect();
        found.sort();
        found
            .pop()
            .with_context(|| format!("no {} parquet in {}", self.split, dir.display()))
    }

    fn parse() -> Result<Self> {
        let (mut dataset, mut out) = (None, None);
        let mut language = "jp".to_string();
        let mut split = "test".to_string();
        let (mut queries, mut candidates) = (usize::MAX, usize::MAX);

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            let mut next = |what: &str| -> Result<String> {
                args.next().with_context(|| format!("{what} wants a value"))
            };
            match arg.as_str() {
                "--dataset" => dataset = Some(PathBuf::from(next("--dataset")?)),
                "--out" => out = Some(PathBuf::from(next("--out")?)),
                "--language" => language = next("--language")?,
                "--split" => split = next("--split")?,
                "--queries" => queries = next("--queries")?.parse()?,
                "--candidates" => candidates = next("--candidates")?.parse()?,
                "--help" | "-h" => {
                    eprintln!("{USAGE}");
                    std::process::exit(0);
                }
                other => anyhow::bail!("unknown argument {other:?}\n\n{USAGE}"),
            }
        }
        Ok(Self {
            dataset: dataset.context(USAGE)?,
            out: out.context(USAGE)?,
            language,
            split,
            queries,
            candidates,
        })
    }
}
