//! The parity oracle's Rust half.
//!
//! ```text
//! ragflow-parity run    --canvas dsl.json --query Q --recordings rec.json
//! ragflow-parity diff   go.jsonl rust.jsonl [--canvas dsl.json]
//! ragflow-parity corpus <dir> --oracle <oracle-go> --conf <conf/models> [--out <dir>] [--only <substr>]
//! ```
//!
//! `run` prints the Rust runtime's events as JSON lines in the shape
//! `oracle-go` prints; `diff` compares two such files; `corpus` runs every
//! case in a directory through both and prints PASS/FAIL per case.

mod case;
mod compare;
mod rust_run;

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use serde_json::{Map, Value};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("run") => cmd_run(&args[1..]),
        Some("diff") => cmd_diff(&args[1..]),
        Some("corpus") => cmd_corpus(&args[1..]),
        _ => Err(
            "usage: ragflow-parity run|diff|corpus ... (see the crate docs or rust/parity/README.md)"
                .into(),
        ),
    };
    match result {
        Ok(code) => code,
        Err(e) => {
            eprintln!("ragflow-parity: {e}");
            ExitCode::from(2)
        }
    }
}

/// `--flag value` pairs plus positional arguments.
struct Args {
    flags: Map<String, Value>,
    positional: Vec<String>,
}

fn parse_args(args: &[String]) -> Result<Args, String> {
    let mut flags = Map::new();
    let mut positional = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if let Some(name) = a.strip_prefix("--") {
            let v = it.next().ok_or_else(|| format!("--{name} needs a value"))?;
            flags.insert(name.to_string(), Value::String(v.clone()));
        } else {
            positional.push(a.clone());
        }
    }
    Ok(Args { flags, positional })
}

impl Args {
    fn get(&self, name: &str) -> Option<&str> {
        self.flags.get(name).and_then(Value::as_str)
    }
    fn need(&self, name: &str) -> Result<&str, String> {
        self.get(name)
            .ok_or_else(|| format!("--{name} is required"))
    }
}

fn cmd_run(args: &[String]) -> Result<ExitCode, String> {
    let a = parse_args(args)?;
    let dsl = std::fs::read_to_string(a.need("canvas")?).map_err(|e| e.to_string())?;
    let recordings = case::read_json(Path::new(a.need("recordings")?))?;
    let outcome = rust_run::run(
        &dsl,
        a.get("query").unwrap_or_default(),
        recordings.as_object().ok_or("recordings: not an object")?,
    );
    for line in &outcome.lines {
        println!("{line}");
    }
    Ok(if outcome.missing.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(3)
    })
}

fn read_lines(path: &str) -> Result<Vec<Value>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(|e| format!("{path}: {e}")))
        .collect()
}

fn cmd_diff(args: &[String]) -> Result<ExitCode, String> {
    let a = parse_args(args)?;
    let [go, rust] = a.positional.as_slice() else {
        return Err("diff takes two files: go.jsonl rust.jsonl".into());
    };
    // With --canvas, independent components may finish in any order.
    let topo = match a.get("canvas") {
        Some(path) => Some(compare::topology(&case::read_json(Path::new(path))?)),
        None => None,
    };
    let report = compare::compare(
        &compare::parse(&read_lines(go)?),
        &compare::parse(&read_lines(rust)?),
        topo.as_ref(),
    );
    print_report("diff", &report);
    Ok(if report.passed() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

fn print_report(name: &str, report: &compare::Report) {
    for (check, detail) in &report.hard {
        println!("--- {name}: FAIL {check}\n{}", detail.trim_end());
    }
    for (check, detail) in &report.soft {
        println!("--- {name}: soft {check}\n{}", detail.trim_end());
    }
    for (check, detail) in &report.known {
        println!("--- {name}: KNOWN {check}\n{}", detail.trim_end());
    }
}

struct Row {
    name: String,
    verdict: String,
    detail: String,
}

fn cmd_corpus(args: &[String]) -> Result<ExitCode, String> {
    let a = parse_args(args)?;
    let dir = PathBuf::from(a.positional.first().ok_or("corpus needs a directory")?);
    let oracle = a.need("oracle")?;
    let conf = a.need("conf")?;
    let out_root = PathBuf::from(a.get("out").unwrap_or("target/parity"));
    let only = a.get("only");

    let mut cases: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .filter(|p| only.is_none_or(|o| p.to_string_lossy().contains(o)))
        .collect();
    cases.sort();

    let mut rows = Vec::new();
    for path in &cases {
        let row = run_case(path, oracle, conf, &out_root).unwrap_or_else(|e| Row {
            name: path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into(),
            verdict: "ERROR".into(),
            detail: e,
        });
        rows.push(row);
    }

    let width = rows.iter().map(|r| r.name.len()).max().unwrap_or(4).max(4);
    println!("\n{:width$}  result", "case");
    for r in &rows {
        println!("{:width$}  {}", r.name, r.verdict);
    }
    for r in rows.iter().filter(|r| !r.detail.is_empty()) {
        println!("\n=== {} ({})\n{}", r.name, r.verdict, r.detail.trim_end());
    }
    let known = rows
        .iter()
        .filter(|r| r.verdict.starts_with("KNOWN"))
        .count();
    let passed = rows
        .iter()
        .filter(|r| r.verdict.starts_with("PASS"))
        .count();
    let failed = rows.len() - passed - known;
    println!(
        "\n{} cases, {} pass, {} known, {} fail; artifacts in {}",
        rows.len(),
        passed,
        known,
        failed,
        out_root.display()
    );
    Ok(if failed == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

fn run_case(path: &Path, oracle: &str, conf: &str, out_root: &Path) -> Result<Row, String> {
    let case = case::load(path)?;
    let out = out_root.join(&case.name);
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let dsl_path = out.join("dsl.json");
    let rec_path = out.join("recordings.json");
    let dsl_text = serde_json::to_string_pretty(&case.dsl).map_err(|e| e.to_string())?;
    write(&dsl_path, &dsl_text)?;
    write(
        &rec_path,
        &serde_json::to_string_pretty(&case.recordings).map_err(|e| e.to_string())?,
    )?;

    let go = Command::new(oracle)
        .arg("-canvas")
        .arg(&dsl_path)
        .arg("-query")
        .arg(&case.query)
        .arg("-recordings")
        .arg(&rec_path)
        .arg("-conf")
        .arg(conf)
        .output()
        .map_err(|e| format!("running {oracle}: {e}"))?;
    let go_stdout = String::from_utf8_lossy(&go.stdout).to_string();
    let go_stderr = String::from_utf8_lossy(&go.stderr).to_string();
    write(&out.join("go.jsonl"), &go_stdout)?;
    write(&out.join("go.stderr"), &go_stderr)?;
    let go_lines: Vec<Value> = go_stdout
        .lines()
        .filter(|l| l.trim_start().starts_with('{'))
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();

    let rust = rust_run::run(&dsl_text, &case.query, &case.recordings);
    let rust_text: String = rust.lines.iter().map(|l| format!("{l}\n")).collect();
    write(&out.join("rust.jsonl"), &rust_text)?;

    let mut detail = format!("canvas: {}\n", case.canvas_path.display());
    let go_missing = go.status.code() == Some(3);
    if go_missing || !rust.missing.is_empty() {
        let _ = std::fmt::Write::write_fmt(
            &mut detail,
            format_args!(
                "missing recordings: go exit {:?}, rust {:?}\n{}",
                go.status.code(),
                rust.missing,
                go_stderr
                    .lines()
                    .filter(|l| !l.contains("level"))
                    .collect::<Vec<_>>()
                    .join("\n")
            ),
        );
        return Ok(Row {
            name: case.name,
            verdict: "MISSING-RECORDING".into(),
            detail,
        });
    }
    if !go.status.success() {
        detail.push_str(&format!(
            "oracle-go exited {:?}; stderr tail:\n{}",
            go.status.code(),
            tail(&go_stderr, 20)
        ));
        return Ok(Row {
            name: case.name,
            verdict: "ORACLE-ERROR".into(),
            detail,
        });
    }

    let report = compare::compare(
        &compare::parse(&go_lines),
        &compare::parse(&rust.lines),
        Some(&compare::topology(&case.dsl)),
    );
    let mut text = String::new();
    for (check, d) in &report.hard {
        text.push_str(&format!("[FAIL {check}]\n{}\n", d.trim_end()));
    }
    for (check, d) in &report.soft {
        text.push_str(&format!("[soft {check}]\n{}\n", d.trim_end()));
    }
    for (check, d) in &report.known {
        text.push_str(&format!("[KNOWN {check}]\n{}\n", d.trim_end()));
    }
    let soft: Vec<&str> = report.soft.iter().map(|(c, _)| *c).collect();
    let soft = if soft.is_empty() {
        String::new()
    } else {
        format!(" (soft: {})", soft.join(", "))
    };
    let verdict = if !report.passed() {
        let hard: Vec<&str> = report.hard.iter().map(|(c, _)| *c).collect();
        format!("FAIL ({})", hard.join(", "))
    } else if !report.known.is_empty() {
        let mut known: Vec<&str> = report.known.iter().map(|(c, _)| *c).collect();
        known.dedup();
        format!("KNOWN ({}){soft}", known.join(", "))
    } else {
        format!("PASS{soft}")
    };
    Ok(Row {
        name: case.name,
        verdict,
        detail: if text.is_empty() {
            String::new()
        } else {
            detail + &text
        },
    })
}

fn write(path: &Path, text: &str) -> Result<(), String> {
    std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
}

fn tail(text: &str, n: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}
