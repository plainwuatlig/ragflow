//! The conformance runner for the canvas part of the contract.
//!
//! ```text
//! ragflow-conformance canvas check  <suite> [--out <dir>] [--only <substr>]
//! ragflow-conformance canvas record <suite> --oracle <oracle-go> --conf <conf/models> [--only <substr>]
//! ragflow-conformance run    --canvas dsl.json --query Q --recordings rec.json
//! ragflow-conformance diff   golden.jsonl rust.jsonl [--canvas dsl.json]
//! ```
//!
//! A suite directory (`rust/conformance/contract/<version>/canvas`) holds
//! `cases/*.json` and one golden event stream per case in
//! `golden/<case>.jsonl`. `check` runs every case through `ragflow-runtime`
//! and compares the events with the golden; it needs no Go. `record` runs
//! upstream's Go runtime (`oracle-go`) on every case and writes the goldens;
//! it is how a contract version is cut. `run` prints the Rust runtime's
//! events as JSON lines in the golden's shape; `diff` compares two such
//! files.

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
        Some("canvas") => match args.get(1).map(String::as_str) {
            Some("check") => cmd_check(&args[2..]),
            Some("record") => cmd_record(&args[2..]),
            _ => Err("usage: ragflow-conformance canvas check|record <suite> ...".into()),
        },
        _ => Err(
            "usage: ragflow-conformance canvas check|record | run | diff ... (see the crate docs or rust/conformance/README.md)"
                .into(),
        ),
    };
    match result {
        Ok(code) => code,
        Err(e) => {
            eprintln!("ragflow-conformance: {e}");
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

fn read_lines(path: &Path) -> Result<Vec<Value>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(|e| format!("{}: {e}", path.display())))
        .collect()
}

fn cmd_diff(args: &[String]) -> Result<ExitCode, String> {
    let a = parse_args(args)?;
    let [golden, rust] = a.positional.as_slice() else {
        return Err("diff takes two files: golden.jsonl rust.jsonl".into());
    };
    // With --canvas, independent components may finish in any order.
    let topo = match a.get("canvas") {
        Some(path) => Some(compare::topology(&case::read_json(Path::new(path))?)),
        None => None,
    };
    let report = compare::compare(
        &compare::parse(&read_lines(Path::new(golden))?),
        &compare::parse(&read_lines(Path::new(rust))?),
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

/// A suite directory: `cases/` and `golden/`.
struct Suite {
    cases: PathBuf,
    golden: PathBuf,
}

fn suite(a: &Args) -> Result<Suite, String> {
    let dir =
        PathBuf::from(a.positional.first().ok_or(
            "the suite directory is required (rust/conformance/contract/<version>/canvas)",
        )?);
    let cases = dir.join("cases");
    if !cases.is_dir() {
        return Err(format!("{}: no cases/ directory", dir.display()));
    }
    Ok(Suite {
        cases,
        golden: dir.join("golden"),
    })
}

fn list_cases(dir: &Path, only: Option<&str>) -> Result<Vec<PathBuf>, String> {
    let mut cases: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .filter(|p| only.is_none_or(|o| p.to_string_lossy().contains(o)))
        .collect();
    cases.sort();
    if cases.is_empty() {
        return Err(format!("{}: no cases match", dir.display()));
    }
    Ok(cases)
}

struct Row {
    name: String,
    verdict: String,
    detail: String,
}

fn print_rows(rows: &[Row], footer: &str) -> bool {
    let width = rows.iter().map(|r| r.name.len()).max().unwrap_or(4).max(4);
    println!("\n{:width$}  result", "case");
    for r in rows {
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
        .filter(|r| r.verdict.starts_with("PASS") || r.verdict.starts_with("RECORDED"))
        .count();
    let failed = rows.len() - passed - known;
    println!(
        "\n{} cases, {} pass, {} known, {} fail{footer}",
        rows.len(),
        passed,
        known,
        failed
    );
    failed == 0
}

/// `canvas check`: the Rust runtime against the goldens. No Go involved.
fn cmd_check(args: &[String]) -> Result<ExitCode, String> {
    let a = parse_args(args)?;
    let s = suite(&a)?;
    let out_root = PathBuf::from(a.get("out").unwrap_or("target/conformance/canvas"));
    let mut rows = Vec::new();
    for path in list_cases(&s.cases, a.get("only"))? {
        let row = check_case(&path, &s.golden, &out_root).unwrap_or_else(|e| Row {
            name: stem(&path),
            verdict: "ERROR".into(),
            detail: e,
        });
        rows.push(row);
    }
    let ok = print_rows(&rows, &format!("; Rust streams in {}", out_root.display()));
    Ok(if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

fn check_case(path: &Path, golden_dir: &Path, out_root: &Path) -> Result<Row, String> {
    let case = case::load(path)?;
    let golden_path = golden_dir.join(format!("{}.jsonl", case.name));
    if !golden_path.is_file() {
        return Ok(Row {
            name: case.name,
            verdict: "FAIL (no golden)".into(),
            detail: format!(
                "{} does not exist; cut it with `canvas record`",
                golden_path.display()
            ),
        });
    }
    let golden = read_lines(&golden_path)?;
    let dsl_text = serde_json::to_string_pretty(&case.dsl).map_err(|e| e.to_string())?;
    let rust = rust_run::run(&dsl_text, &case.query, &case.recordings);
    let out = out_root.join(&case.name);
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    write(&out.join("dsl.json"), &dsl_text)?;
    let rust_text: String = rust.lines.iter().map(|l| format!("{l}\n")).collect();
    write(&out.join("rust.jsonl"), &rust_text)?;
    let mut detail = format!("canvas: {}\n", case.canvas_path.display());
    if !rust.missing.is_empty() {
        detail.push_str(&format!("missing recordings: {:?}\n", rust.missing));
        return Ok(Row {
            name: case.name,
            verdict: "MISSING-RECORDING".into(),
            detail,
        });
    }
    let report = compare::compare(
        &compare::parse(&golden),
        &compare::parse(&rust.lines),
        Some(&compare::topology(&case.dsl)),
    );
    Ok(verdict_row(case.name, detail, &report))
}

fn verdict_row(name: String, mut detail: String, report: &compare::Report) -> Row {
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
    if text.is_empty() {
        detail.clear();
    } else {
        detail.push_str(&text);
    }
    Row {
        name,
        verdict,
        detail,
    }
}

/// `canvas record`: upstream's Go runtime writes the goldens. The only step
/// that needs Go; run when a contract version is cut.
fn cmd_record(args: &[String]) -> Result<ExitCode, String> {
    let a = parse_args(args)?;
    let s = suite(&a)?;
    let oracle = a.need("oracle")?;
    let conf = a.need("conf")?;
    std::fs::create_dir_all(&s.golden).map_err(|e| format!("{}: {e}", s.golden.display()))?;
    let scratch = std::env::temp_dir().join(format!("ragflow-conformance-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).map_err(|e| e.to_string())?;
    let mut rows = Vec::new();
    for path in list_cases(&s.cases, a.get("only"))? {
        let row = record_case(&path, oracle, conf, &s.golden, &scratch).unwrap_or_else(|e| Row {
            name: stem(&path),
            verdict: "ERROR".into(),
            detail: e,
        });
        rows.push(row);
    }
    let _ = std::fs::remove_dir_all(&scratch);
    let ok = print_rows(&rows, &format!("; goldens in {}", s.golden.display()));
    Ok(if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

fn record_case(
    path: &Path,
    oracle: &str,
    conf: &str,
    golden_dir: &Path,
    scratch: &Path,
) -> Result<Row, String> {
    let case = case::load(path)?;
    let dsl_path = scratch.join(format!("{}.dsl.json", case.name));
    let rec_path = scratch.join(format!("{}.recordings.json", case.name));
    write(
        &dsl_path,
        &serde_json::to_string_pretty(&case.dsl).map_err(|e| e.to_string())?,
    )?;
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
    let stdout = String::from_utf8_lossy(&go.stdout);
    let stderr = String::from_utf8_lossy(&go.stderr);
    if go.status.code() == Some(3) {
        return Ok(Row {
            name: case.name,
            verdict: "MISSING-RECORDING".into(),
            detail: stderr
                .lines()
                .filter(|l| !l.contains("level"))
                .collect::<Vec<_>>()
                .join("\n"),
        });
    }
    if !go.status.success() {
        return Ok(Row {
            name: case.name,
            verdict: "ORACLE-ERROR".into(),
            detail: format!(
                "oracle-go exited {:?}; stderr tail:\n{}",
                go.status.code(),
                tail(&stderr, 20)
            ),
        });
    }
    // Only the event lines; the Go runtime's own logging goes to stderr but
    // the filter keeps a stray line from ever landing in a golden.
    let lines: Vec<&str> = stdout
        .lines()
        .filter(|l| l.trim_start().starts_with('{'))
        .collect();
    if lines.is_empty() {
        return Ok(Row {
            name: case.name,
            verdict: "ORACLE-ERROR".into(),
            detail: "oracle-go printed no events".into(),
        });
    }
    let golden_path = golden_dir.join(format!("{}.jsonl", case.name));
    write(&golden_path, &format!("{}\n", lines.join("\n")))?;
    Ok(Row {
        name: case.name,
        verdict: format!("RECORDED ({} events)", lines.len()),
        detail: String::new(),
    })
}

fn stem(path: &Path) -> String {
    path.file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .into()
}

fn write(path: &Path, text: &str) -> Result<(), String> {
    std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
}

fn tail(text: &str, n: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}
