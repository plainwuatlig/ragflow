//! `ragflow-server`: RAGFlow's agent endpoints, served from RAGFlow's own
//! MySQL, Kvrocks and Elasticsearch. Configured through the environment
//! variables RAGFlow's own deployment already defines:
//!
//! | variable | default |
//! |---|---|
//! | `MYSQL_HOST` `MYSQL_PORT` `MYSQL_USER` `MYSQL_PASSWORD` `MYSQL_DBNAME` | `mysql` `3306` `root` (none) `rag_flow` |
//! | `KVROCKS_HOST` `KVROCKS_PORT` `REDIS_PASSWORD` `KVROCKS_DB` | `kvrocks` `6379` (none) `1` |
//! | `ES_URL` or `ES_HOST`, `ES_USER`, `ELASTIC_PASSWORD` | `http://es01:9200`, `elastic` |
//! | `RAGFLOW_SECRET_KEY` | read from Kvrocks when unset |
//! | `RAGFLOW_MODELS_DIR` | `conf/models` (upstream's provider catalog) |
//! | `RAGFLOW_DICT_PATH` | `/usr/share/infinity/resource` (the analyzer's dictionaries) |
//! | `RAGFLOW_LISTEN` | `0.0.0.0:9380` |
//! | `RAGFLOW_RES_DIR` | unset (the analyzer's resource directory) |
//!
//! `ragflow-server healthcheck` probes `GET /healthz` on `RAGFLOW_LISTEN`'s port
//! and exits 0 when the server answers (the image's `HEALTHCHECK`).

use std::sync::Arc;

use ragflow_server::facts::Facts;
use ragflow_server::http::{router, App};
use ragflow_server::kv::{Kv, Kvrocks};
use ragflow_server::models::Catalog;
use ragflow_server::service::Agent;
use ragflow_server::sessions::SessionStore;
use ragflow_server::store::Store;
use ragflow_server::wiring::{retrieval, SearchConfig};

fn env(name: &str, default: &str) -> String {
    std::env::var(name)
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn encode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}

/// `ragflow-server healthcheck`: GET /healthz on the local listen port.
fn healthcheck() -> Result<(), String> {
    use std::io::{Read, Write};
    let listen = env("RAGFLOW_LISTEN", "0.0.0.0:9380");
    let port = listen.rsplit(':').next().unwrap_or("9380");
    let timeout = std::time::Duration::from_secs(3);
    let addr: std::net::SocketAddr = format!("127.0.0.1:{port}")
        .parse()
        .map_err(|e| format!("{listen}: {e}"))?;
    let mut conn =
        std::net::TcpStream::connect_timeout(&addr, timeout).map_err(|e| e.to_string())?;
    conn.set_read_timeout(Some(timeout))
        .map_err(|e| e.to_string())?;
    conn.write_all(b"GET /healthz HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .map_err(|e| e.to_string())?;
    let mut head = String::new();
    conn.take(64)
        .read_to_string(&mut head)
        .map_err(|e| e.to_string())?;
    if head.starts_with("HTTP/1.1 200") {
        Ok(())
    } else {
        Err(format!("unhealthy: {}", head.lines().next().unwrap_or("")))
    }
}

/// Resolves on SIGTERM (`docker stop`) or Ctrl-C, so the server stops accepting
/// and drains instead of waiting for the runtime's kill.
async fn shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};
    let mut term = signal(SignalKind::terminate()).expect("a SIGTERM handler");
    tokio::select! {
        _ = term.recv() => {}
        _ = tokio::signal::ctrl_c() => {}
    }
}

#[tokio::main]
async fn main() {
    if std::env::args().nth(1).as_deref() == Some("healthcheck") {
        if let Err(e) = healthcheck() {
            eprintln!("ragflow-server healthcheck: {e}");
            std::process::exit(1);
        }
        return;
    }
    if let Err(e) = run().await {
        eprintln!("ragflow-server: {e}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), String> {
    let mysql_url = format!(
        "mysql://{}:{}@{}:{}/{}",
        encode(&env("MYSQL_USER", "root")),
        encode(&env("MYSQL_PASSWORD", "")),
        env("MYSQL_HOST", "mysql"),
        env("MYSQL_PORT", "3306"),
        env("MYSQL_DBNAME", "rag_flow"),
    );
    let store = Arc::new(Store::connect(&mysql_url).map_err(|e| format!("mysql: {e}"))?);
    let password = env("REDIS_PASSWORD", "");
    let kv_url = format!(
        "redis://{}{}:{}/{}",
        if password.is_empty() {
            String::new()
        } else {
            format!(":{}@", encode(&password))
        },
        env("KVROCKS_HOST", "kvrocks"),
        env("KVROCKS_PORT", "6379"),
        env("KVROCKS_DB", "1"),
    );
    let kv: Arc<dyn Kv> = Arc::new(Kvrocks::connect(&kv_url).map_err(|e| format!("kvrocks: {e}"))?);
    let catalog = Arc::new(Catalog::load(std::path::Path::new(&env(
        "RAGFLOW_MODELS_DIR",
        "conf/models",
    )))?);
    let facts: Arc<dyn Facts> = store.clone();
    let sessions: Arc<dyn SessionStore> = store.clone();
    let es_url = std::env::var("ES_URL")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| format!("http://{}:9200", env("ES_HOST", "es01")));
    let search = retrieval(
        &SearchConfig {
            es_url,
            es_user: Some(env("ES_USER", "elastic")),
            es_password: std::env::var("ELASTIC_PASSWORD").ok(),
            mysql_url,
            dict_path: env("RAGFLOW_DICT_PATH", "/usr/share/infinity/resource").into(),
            res_dir: std::env::var("RAGFLOW_RES_DIR").ok().map(Into::into),
        },
        facts.clone(),
        catalog.clone(),
    )?;
    let agent = Arc::new(Agent::new(
        facts.clone(),
        sessions.clone(),
        kv.clone(),
        catalog.clone(),
        Some(search),
    ));
    let app = Arc::new(App {
        agent,
        facts,
        sessions,
        kv,
        catalog,
        secret_env: std::env::var("RAGFLOW_SECRET_KEY")
            .ok()
            .filter(|v| !v.is_empty()),
    });
    let listen = env("RAGFLOW_LISTEN", "0.0.0.0:9380");
    let listener = tokio::net::TcpListener::bind(&listen)
        .await
        .map_err(|e| format!("{listen}: {e}"))?;
    eprintln!("ragflow-server listening on {listen}");
    axum::serve(listener, router(app))
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| e.to_string())
}
