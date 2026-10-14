//! Live store clients: Elasticsearch over HTTP, MySQL (`knowledgebase`,
//! `document`, `tenant_model*`), and an Ollama query embedder.

use std::collections::HashMap;
use std::sync::Arc;

use mysql::prelude::Queryable;
use serde_json::{json, Value};

use crate::adapter::{EmbeddingResolver, Knowledgebase, KnowledgebaseStore};
use crate::es::SearchEngine;
use crate::service::{DocumentStore, QueryEmbedder};

fn base64(input: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in input.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            T[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            T[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// Elasticsearch `_search` over HTTP with basic auth.
pub struct EsHttp {
    base_url: String,
    auth: Option<String>,
    agent: ureq::Agent,
}

impl EsHttp {
    pub fn new(base_url: &str, user: Option<&str>, password: Option<&str>) -> Self {
        let auth = user.map(|u| {
            format!(
                "Basic {}",
                base64(format!("{u}:{}", password.unwrap_or("")).as_bytes())
            )
        });
        EsHttp {
            base_url: base_url.trim_end_matches('/').to_string(),
            auth,
            agent: ureq::AgentBuilder::new().build(),
        }
    }
}

impl SearchEngine for EsHttp {
    fn search(&self, index: &str, body: &Value, track_total_hits: bool) -> Result<Value, String> {
        let mut url = format!("{}/{}/_search", self.base_url, index);
        if track_total_hits {
            url.push_str("?track_total_hits=true");
        }
        let mut req = self
            .agent
            .post(&url)
            .set("Content-Type", "application/json");
        if let Some(a) = &self.auth {
            req = req.set("Authorization", a);
        }
        match req.send_string(&body.to_string()) {
            Ok(resp) => resp.into_json::<Value>().map_err(|e| e.to_string()),
            Err(ureq::Error::Status(code, resp)) => Err(format!(
                "status {code}: {}",
                resp.into_string().unwrap_or_default()
            )),
            Err(e) => Err(e.to_string()),
        }
    }
}

/// MySQL-backed dataset/document lookups.
pub struct MysqlStore {
    pool: mysql::Pool,
}

impl MysqlStore {
    pub fn new(url: &str) -> Result<Self, String> {
        let opts = mysql::Opts::from_url(url).map_err(|e| e.to_string())?;
        Ok(MysqlStore {
            pool: mysql::Pool::new(opts).map_err(|e| e.to_string())?,
        })
    }

    fn placeholders(n: usize) -> String {
        vec!["?"; n].join(",")
    }

    fn kb_rows(&self, sql: &str, params: Vec<mysql::Value>) -> Result<Vec<Knowledgebase>, String> {
        let mut conn = self.pool.get_conn().map_err(|e| e.to_string())?;
        let rows: Vec<KbRow> = conn.exec(sql, params).map_err(|e| e.to_string())?;
        Ok(rows
            .into_iter()
            .map(
                |(id, tenant_id, name, embd_id, tenant_embd_id, parser_config)| Knowledgebase {
                    id,
                    tenant_id,
                    name,
                    embd_id: embd_id.unwrap_or_default(),
                    tenant_embd_id,
                    parser_config: parser_config
                        .and_then(|p| serde_json::from_str(&p).ok())
                        .unwrap_or(Value::Null),
                },
            )
            .collect())
    }

    /// `(model_name, provider_name, instance extra)` for a tenant model id.
    pub fn tenant_model(&self, id: &str) -> Result<Option<(String, String, Value)>, String> {
        let mut conn = self.pool.get_conn().map_err(|e| e.to_string())?;
        let row: Option<(String, String, Option<String>)> = conn
            .exec_first(
                "SELECT m.model_name, p.provider_name, i.extra FROM tenant_model m \
                 JOIN tenant_model_provider p ON p.id = m.provider_id \
                 LEFT JOIN tenant_model_instance i ON i.id = m.instance_id WHERE m.id = ?",
                (id,),
            )
            .map_err(|e| e.to_string())?;
        Ok(row.map(|(m, p, e)| {
            (
                m,
                p,
                e.and_then(|x| serde_json::from_str(&x).ok())
                    .unwrap_or(Value::Null),
            )
        }))
    }
}

/// id, tenant_id, name, embd_id, tenant_embd_id, parser_config.
type KbRow = (
    String,
    String,
    String,
    Option<String>,
    Option<String>,
    Option<String>,
);

const KB_COLUMNS: &str =
    "id, tenant_id, name, embd_id, tenant_embd_id, CAST(parser_config AS CHAR)";

impl KnowledgebaseStore for MysqlStore {
    fn get_by_ids(&self, ids: &[String]) -> Result<Vec<Knowledgebase>, String> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let sql = format!(
            "SELECT {KB_COLUMNS} FROM knowledgebase WHERE id IN ({}) AND status = ?",
            Self::placeholders(ids.len())
        );
        let mut params: Vec<mysql::Value> =
            ids.iter().map(|s| mysql::Value::from(s.as_str())).collect();
        params.push(mysql::Value::from("1"));
        self.kb_rows(&sql, params)
    }

    fn get_by_name(&self, name: &str, tenant_id: &str) -> Result<Option<Knowledgebase>, String> {
        let sql = format!(
            "SELECT {KB_COLUMNS} FROM knowledgebase WHERE LOWER(name) = LOWER(?) AND tenant_id = ? AND status = ? LIMIT 1"
        );
        Ok(self
            .kb_rows(&sql, vec![name.into(), tenant_id.into(), "1".into()])?
            .into_iter()
            .next())
    }

    fn tenant_model_name(&self, id: &str) -> Option<String> {
        self.tenant_model(id).ok().flatten().map(|(m, _, _)| m)
    }
}

impl DocumentStore for MysqlStore {
    fn existing_document_ids(&self, ids: &[String]) -> Result<Vec<String>, String> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut conn = self.pool.get_conn().map_err(|e| e.to_string())?;
        let sql = format!(
            "SELECT id FROM document WHERE id IN ({})",
            Self::placeholders(ids.len())
        );
        let params: Vec<mysql::Value> =
            ids.iter().map(|s| mysql::Value::from(s.as_str())).collect();
        conn.exec(sql, params).map_err(|e| e.to_string())
    }
}

/// Ollama `/api/embed` query embeddings.
pub struct OllamaEmbedder {
    pub base_url: String,
    pub model: String,
}

impl QueryEmbedder for OllamaEmbedder {
    fn embed_query(&self, text: &str) -> Result<Vec<f64>, String> {
        let url = format!("{}/api/embed", self.base_url.trim_end_matches('/'));
        let resp: Value = ureq::post(&url)
            .send_json(json!({"model": self.model, "input": [text]}))
            .map_err(|e| e.to_string())?
            .into_json()
            .map_err(|e| e.to_string())?;
        resp.pointer("/embeddings/0")
            .and_then(Value::as_array)
            .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect())
            .ok_or_else(|| "embedding response without embeddings[0]".to_string())
    }
}

/// Resolves a dataset's tenant embedding model from MySQL; Ollama only.
pub struct MysqlEmbeddingResolver {
    pub store: Arc<MysqlStore>,
}

impl EmbeddingResolver for MysqlEmbeddingResolver {
    fn resolve(&self, kb: &Knowledgebase) -> Result<Arc<dyn QueryEmbedder>, String> {
        let id = kb
            .tenant_embd_id
            .clone()
            .filter(|s| !s.trim().is_empty())
            .ok_or("only datasets with a tenant embedding model are supported")?;
        let (model, provider, extra) = self
            .store
            .tenant_model(&id)?
            .ok_or("tenant model not found")?;
        if provider != "Ollama" {
            return Err(format!("embedding provider {provider:?} is not supported"));
        }
        let base_url = extra
            .get("base_url")
            .and_then(Value::as_str)
            .ok_or("Ollama instance without base_url")?
            .to_string();
        Ok(Arc::new(OllamaEmbedder { base_url, model }))
    }
}

/// Replays recorded query vectors (query text → vector).
pub struct FixedEmbeddings {
    pub vectors: HashMap<String, Vec<f64>>,
}

impl QueryEmbedder for FixedEmbeddings {
    fn embed_query(&self, text: &str) -> Result<Vec<f64>, String> {
        self.vectors
            .get(text)
            .cloned()
            .ok_or_else(|| format!("no recorded embedding for {text:?}"))
    }
}

impl EmbeddingResolver for FixedEmbeddings {
    fn resolve(&self, _kb: &Knowledgebase) -> Result<Arc<dyn QueryEmbedder>, String> {
        Ok(Arc::new(FixedEmbeddings {
            vectors: self.vectors.clone(),
        }))
    }
}
