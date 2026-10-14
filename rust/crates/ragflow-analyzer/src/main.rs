//! `ragflow-analyzer`: run RAGFlow's C++ analyzer over stdin, one JSON object per
//! input line on stdout: `{"text", "tokens", "fine_grained"}`.
//!
//! ```text
//! ragflow-analyzer [--mode query|index] [--language LANG] [--dict PATH]
//!                  [--input lines|jsonl] [--threads N]
//! ```
//!
//! - `--mode query` (default): `Tokenize` then `FineGrainedTokenize`, language `""`
//!   (English), as the Go query path.
//! - `--mode index`: same calls with the dataset's parser language, and the empty
//!   result fallbacks of `tokenizeChunks`.
//! - `--input jsonl`: each line is a JSON string, or an object with `text` (other
//!   fields such as `id` are echoed back). Needed for texts containing newlines.
//! - `--threads N`: tokenize on N threads sharing one pool; output order is the
//!   input order.

use std::io::{self, BufRead, BufWriter, Write};
use std::process::ExitCode;

use ragflow_analyzer::{index_tokens, query_tokens, AnalyzerPool, Dictionary, DEFAULT_DICT_PATH};
use serde_json::{Map, Value};

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Query,
    Index,
}

struct Args {
    mode: Mode,
    language: String,
    dict: String,
    jsonl: bool,
    threads: usize,
}

fn usage() -> ! {
    eprintln!(
        "usage: ragflow-analyzer [--mode query|index] [--language LANG] [--dict PATH] \
         [--input lines|jsonl] [--threads N]"
    );
    std::process::exit(2)
}

fn parse_args() -> Args {
    let mut args = Args {
        mode: Mode::Query,
        language: String::new(),
        dict: std::env::var("RAGFLOW_DICT_PATH").unwrap_or_else(|_| DEFAULT_DICT_PATH.into()),
        jsonl: false,
        threads: 1,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().unwrap_or_else(|| usage());
        match flag.as_str() {
            "--mode" => {
                args.mode = match val().as_str() {
                    "query" => Mode::Query,
                    "index" => Mode::Index,
                    _ => usage(),
                }
            }
            "--language" => args.language = val(),
            "--dict" => args.dict = val(),
            "--input" => {
                args.jsonl = match val().as_str() {
                    "lines" => false,
                    "jsonl" => true,
                    _ => usage(),
                }
            }
            "--threads" => args.threads = val().parse::<usize>().unwrap_or_else(|_| usage()).max(1),
            _ => usage(),
        }
    }
    args
}

fn parse_record(line: &str, jsonl: bool) -> Result<(String, Map<String, Value>), String> {
    if !jsonl {
        return Ok((line.to_string(), Map::new()));
    }
    match serde_json::from_str::<Value>(line).map_err(|e| e.to_string())? {
        Value::String(s) => Ok((s, Map::new())),
        Value::Object(mut m) => match m.remove("text") {
            Some(Value::String(s)) => Ok((s, m)),
            _ => Err("object without string \"text\"".into()),
        },
        _ => Err("expected a JSON string or object".into()),
    }
}

fn analyze(pool: &AnalyzerPool, args: &Args, line: &str) -> Result<Value, String> {
    let (text, extra) = parse_record(line, args.jsonl)?;
    let mut a = pool.acquire(&args.language).map_err(|e| e.to_string())?;
    let (tokens, fine) = match args.mode {
        Mode::Index => index_tokens(&mut a, &text),
        Mode::Query => query_tokens(&mut a, &text),
    }
    .map_err(|e| e.to_string())?;
    let mut out = extra;
    out.insert("text".into(), Value::String(text));
    out.insert("tokens".into(), Value::String(tokens));
    out.insert("fine_grained".into(), Value::String(fine));
    Ok(Value::Object(out))
}

fn main() -> ExitCode {
    let args = parse_args();
    let dict = match Dictionary::load_quietly(&args.dict) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("ragflow-analyzer: loading {}: {e}", args.dict);
            return ExitCode::FAILURE;
        }
    };
    let pool = AnalyzerPool::new(dict);

    let lines: Vec<String> = match io::stdin().lock().lines().collect() {
        Ok(l) => l,
        Err(e) => {
            eprintln!("ragflow-analyzer: reading stdin: {e}");
            return ExitCode::FAILURE;
        }
    };

    let results: Vec<Result<Value, String>> = if args.threads <= 1 {
        lines.iter().map(|l| analyze(&pool, &args, l)).collect()
    } else {
        let mut slots: Vec<Option<Result<Value, String>>> =
            (0..lines.len()).map(|_| None).collect();
        let chunk = lines.len().div_ceil(args.threads).max(1);
        std::thread::scope(|s| {
            for (ls, out) in lines.chunks(chunk).zip(slots.chunks_mut(chunk)) {
                let (pool, args) = (&pool, &args);
                s.spawn(move || {
                    for (l, o) in ls.iter().zip(out.iter_mut()) {
                        *o = Some(analyze(pool, args, l));
                    }
                });
            }
        });
        slots
            .into_iter()
            .map(|o| o.expect("every slot filled"))
            .collect()
    };

    let stdout = io::stdout();
    let mut w = BufWriter::new(stdout.lock());
    let mut failed = false;
    for (i, r) in results.into_iter().enumerate() {
        match r {
            Ok(v) => {
                if writeln!(w, "{v}").is_err() {
                    return ExitCode::FAILURE;
                }
            }
            Err(e) => {
                failed = true;
                eprintln!("ragflow-analyzer: line {}: {e}", i + 1);
            }
        }
    }
    if w.flush().is_err() || failed {
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
