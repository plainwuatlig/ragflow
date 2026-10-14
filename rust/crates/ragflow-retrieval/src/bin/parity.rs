//! `ragflow-retrieval-parity`: the Rust half of the retrieval parity check
//! and its comparator.
//!
//! ```text
//! ragflow-retrieval-parity embed   --cases F --ollama URL --model M > vectors.json
//! ragflow-retrieval-parity run     --cases F --vectors V > rust.jsonl
//! ragflow-retrieval-parity compare --go go.jsonl --rust rust.jsonl
//! ```
//!
//! A case is one JSON line `{"name": ..., "args": {<Retrieval tool args>}}`.
//! `run` reads ES_URL, ES_USER, ES_PASSWORD, MYSQL_HOST, MYSQL_PORT,
//! MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB and RAGFLOW_DICT_PATH from the
//! environment and prints one line per case in oracle-go's shape:
//! `{name, envelope, references, requests, error?}`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::{Arc, Mutex};

use ragflow_retrieval::adapter::NlpRetrievalAdapter;
use ragflow_retrieval::es::SearchEngine;
use ragflow_retrieval::live::{EsHttp, FixedEmbeddings, MysqlStore, OllamaEmbedder};
use ragflow_retrieval::service::{QueryEmbedder, RetrievalService};
use ragflow_runtime::components::retrieval::{invoke_tool, retrieval_reference};
use ragflow_runtime::RunState;
use serde_json::{json, Map, Value};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn need(args: &[String], name: &str) -> Result<String, String> {
    arg(args, name).ok_or_else(|| format!("missing {name}"))
}

fn env(k: &str) -> String {
    std::env::var(k).unwrap_or_default()
}

fn read_cases(path: &str) -> Result<Vec<(String, Value)>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut out = Vec::new();
    for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let v: Value = serde_json::from_str(line).map_err(|e| format!("case: {e}"))?;
        let name = v
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        out.push((name, v.get("args").cloned().unwrap_or(Value::Null)));
    }
    Ok(out)
}

/// The text the server embeds: the query after the `user` prefix strip.
fn embedded_text(query: &str) -> String {
    regex::Regex::new(r"(?i)^user[:：\s]*")
        .unwrap()
        .replace_all(query, "")
        .into_owned()
}

struct Recording {
    inner: EsHttp,
    log: Mutex<Vec<Value>>,
}

impl SearchEngine for Recording {
    fn search(&self, index: &str, body: &Value, track: bool) -> Result<Value, String> {
        self.log.lock().unwrap().push(json!({
            "index": index,
            "query": if track { "track_total_hits=true" } else { "" },
            "body": body,
        }));
        self.inner.search(index, body, track)
    }
}

fn cmd_embed(args: &[String]) -> Result<(), String> {
    let cases = read_cases(&need(args, "--cases")?)?;
    let e = OllamaEmbedder {
        base_url: need(args, "--ollama")?,
        model: need(args, "--model")?,
    };
    let mut out = Map::new();
    for (_, a) in cases {
        let q = embedded_text(a.get("query").and_then(Value::as_str).unwrap_or(""));
        if q.is_empty() || out.contains_key(&q) {
            continue;
        }
        let v = e.embed_query(&q)?;
        out.insert(q, json!(v));
    }
    println!("{}", Value::Object(out));
    Ok(())
}

fn cmd_run(args: &[String]) -> Result<(), String> {
    let cases = read_cases(&need(args, "--cases")?)?;
    let vectors: HashMap<String, Vec<f64>> = match arg(args, "--vectors") {
        Some(p) => serde_json::from_str(&std::fs::read_to_string(&p).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?,
        None => HashMap::new(),
    };
    let dict = PathBuf::from(env("RAGFLOW_DICT_PATH"));
    let qb = ragflow_retrieval::query_builder(&dict, None)?;
    let es = Arc::new(Recording {
        inner: EsHttp::new(
            &env("ES_URL"),
            Some(&env("ES_USER")),
            Some(&env("ES_PASSWORD")),
        ),
        log: Mutex::new(Vec::new()),
    });
    let port = match env("MYSQL_PORT").as_str() {
        "" => "3306".to_string(),
        p => p.to_string(),
    };
    let url = format!(
        "mysql://{}:{}@{}:{}/{}",
        env("MYSQL_USER"),
        env("MYSQL_PASSWORD"),
        env("MYSQL_HOST"),
        port,
        env("MYSQL_DB")
    );
    let store = Arc::new(MysqlStore::new(&url)?);
    let service = Arc::new(RetrievalService {
        engine: es.clone(),
        documents: store.clone(),
        qb,
    });
    let adapter = NlpRetrievalAdapter {
        service,
        kbs: store,
        embeddings: Arc::new(FixedEmbeddings { vectors }),
        rerankers: None,
        retrieve_by_children: true,
    };
    for (name, a) in cases {
        let mut state = RunState::default();
        let res = invoke_tool(&adapter, &a, &mut state);
        let requests: Vec<Value> = std::mem::take(&mut *es.log.lock().unwrap());
        let mut line = Map::new();
        line.insert("name".into(), json!(name));
        line.insert("requests".into(), Value::Array(requests));
        match res {
            Ok(env) => {
                line.insert("envelope".into(), Value::Object(env));
            }
            Err(e) => {
                line.insert("error".into(), json!(e));
            }
        }
        line.insert(
            "references".into(),
            retrieval_reference(&state).unwrap_or(Value::Null),
        );
        println!("{}", Value::Object(line));
    }
    Ok(())
}

fn by_name(path: &str) -> Result<Vec<(String, Value)>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(|l| {
            let v: Value = serde_json::from_str(l).map_err(|e| e.to_string())?;
            Ok((
                v.get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                v,
            ))
        })
        .collect()
}

/// Numbers as f64; arrays under a `filter`/`must_not`/`should` key sorted (ES
/// treats them as sets; Go fills them from map iteration).
fn normalize(v: &Value, key: &str) -> Value {
    match v {
        Value::Number(n) => json!(n.as_f64().unwrap_or(0.0)),
        Value::Array(a) => {
            let mut items: Vec<Value> = a.iter().map(|x| normalize(x, "")).collect();
            if matches!(key, "filter" | "must_not" | "should") {
                items.sort_by_key(|x| x.to_string());
            }
            Value::Array(items)
        }
        Value::Object(m) => Value::Object(
            m.iter()
                .map(|(k, x)| (k.clone(), normalize(x, k)))
                .collect(),
        ),
        other => other.clone(),
    }
}

fn chunk_ids(r: &Value) -> Vec<String> {
    r.pointer("/envelope/chunks")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .map(|c| {
                    c.get("id")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string()
                })
                .collect()
        })
        .unwrap_or_default()
}

fn score_diff(go: &Value, rust: &Value) -> f64 {
    let mut worst = 0.0f64;
    let pairs = |r: &Value, p: &str| {
        r.pointer(p)
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
    };
    for (g, r) in pairs(go, "/references/chunks")
        .iter()
        .zip(pairs(rust, "/references/chunks").iter())
    {
        for k in ["similarity", "term_similarity", "vector_similarity"] {
            let a = g.get(k).and_then(Value::as_f64).unwrap_or(f64::NAN);
            let b = r.get(k).and_then(Value::as_f64).unwrap_or(f64::NAN);
            let d = (a - b).abs();
            worst = if d.is_nan() {
                f64::INFINITY
            } else {
                worst.max(d)
            };
        }
    }
    for (g, r) in pairs(go, "/envelope/chunks")
        .iter()
        .zip(pairs(rust, "/envelope/chunks").iter())
    {
        let a = g.get("score").and_then(Value::as_f64).unwrap_or(0.0);
        let b = r.get("score").and_then(Value::as_f64).unwrap_or(0.0);
        worst = worst.max((a - b).abs());
    }
    worst
}

fn cmd_compare(args: &[String]) -> Result<bool, String> {
    let go = by_name(&need(args, "--go")?)?;
    let rust: HashMap<String, Value> = by_name(&need(args, "--rust")?)?.into_iter().collect();
    let mut pass = 0;
    let mut fail = 0;
    println!(
        "| case | go n | rust n | ids+order | max score diff | formalized | requests | result |"
    );
    println!("|---|---|---|---|---|---|---|---|");
    let mut details = Vec::new();
    for (name, g) in &go {
        let Some(r) = rust.get(name) else {
            println!("| {name} | - | missing | | | | | FAIL |");
            fail += 1;
            continue;
        };
        let (gi, ri) = (chunk_ids(g), chunk_ids(r));
        let ids_ok = gi == ri;
        let diff = score_diff(g, r);
        let fc = |v: &Value| v.pointer("/envelope/formalized_content").cloned();
        let fc_ok = fc(g) == fc(r);
        let refs_ok = g
            .pointer("/references/chunks")
            .and_then(Value::as_array)
            .map(Vec::len)
            == r.pointer("/references/chunks")
                .and_then(Value::as_array)
                .map(Vec::len);
        let err_ok = g.get("error").is_some() == r.get("error").is_some();
        let gq: Vec<Value> = g
            .get("requests")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let rq: Vec<Value> = r
            .get("requests")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let req_ok = gq.len() == rq.len()
            && gq.iter().zip(&rq).all(|(a, b)| {
                normalize(&a["body"], "") == normalize(&b["body"], "") && a["index"] == b["index"]
            });
        let ok = ids_ok && diff <= 1e-9 && fc_ok && refs_ok && err_ok;
        if ok {
            pass += 1;
        } else {
            fail += 1;
        }
        println!(
            "| {name} | {} | {} | {} | {:.3e} | {} | {} | {} |",
            gi.len(),
            ri.len(),
            if ids_ok { "same" } else { "DIFF" },
            diff,
            if fc_ok { "same" } else { "DIFF" },
            if req_ok { "same" } else { "DIFF" },
            if ok { "PASS" } else { "FAIL" }
        );
        if !ok || !req_ok {
            details.push(format!(
                "## {name}\ngo ids:   {gi:?}\nrust ids: {ri:?}\ngo error: {:?}\nrust error: {:?}",
                g.get("error"),
                r.get("error")
            ));
            if !req_ok {
                for (i, (a, b)) in gq.iter().zip(&rq).enumerate() {
                    let (na, nb) = (normalize(&a["body"], ""), normalize(&b["body"], ""));
                    if na != nb {
                        let strip = |v: &Value| {
                            let mut v = v.clone();
                            if let Some(k) = v.pointer_mut("/knn/query_vector") {
                                *k = json!("<vector>");
                            }
                            v
                        };
                        details.push(format!(
                            "request {i} go:   {}\nrequest {i} rust: {}",
                            strip(&na),
                            strip(&nb)
                        ));
                    }
                }
                if gq.len() != rq.len() {
                    details.push(format!("request count go={} rust={}", gq.len(), rq.len()));
                }
            }
        }
    }
    println!("\n{pass} PASS, {fail} FAIL");
    for d in details {
        println!("\n{d}");
    }
    Ok(fail == 0)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let result = match args.get(1).map(String::as_str) {
        Some("embed") => cmd_embed(&args).map(|_| true),
        Some("run") => cmd_run(&args).map(|_| true),
        Some("compare") => cmd_compare(&args),
        _ => Err("usage: ragflow-retrieval-parity embed|run|compare ...".into()),
    };
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("ragflow-retrieval-parity: {e}");
            ExitCode::FAILURE
        }
    }
}
