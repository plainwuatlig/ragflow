//! The server's retrieval backend: `ragflow-retrieval`'s search over
//! Elasticsearch and MySQL, with the query embedded by the dataset's own
//! embedding model, resolved for the dataset owner's tenant
//! (`NLPRetrievalAdapter.resolveEmbeddingModel`).

use std::sync::Arc;
use std::time::Duration;

use ragflow_retrieval::adapter::{EmbeddingResolver, Knowledgebase, NlpRetrievalAdapter};
use ragflow_retrieval::live::{EsHttp, MysqlStore};
use ragflow_retrieval::service::{QueryEmbedder, RetrievalService as NlpService};
use ragflow_runtime::retrieval::RetrievalService;
use serde_json::{json, Value};

use crate::facts::Facts;
use crate::llm::base_url;
use crate::models::{Access, Catalog, Resolver, EMBEDDING};
use crate::net;

/// Calls an embedding endpoint with one text.
struct HttpEmbedder {
    url: String,
    api_key: String,
    model: String,
    ollama: bool,
    strict: bool,
}

impl QueryEmbedder for HttpEmbedder {
    fn embed_query(&self, text: &str) -> Result<Vec<f64>, String> {
        let agent = net::agent(self.strict, Duration::from_secs(20 * 60));
        let mut req = agent
            .post(&self.url)
            .set("Content-Type", "application/json");
        if !self.api_key.trim().is_empty() {
            req = req.set("Authorization", &format!("Bearer {}", self.api_key));
        }
        let body = json!({ "model": self.model, "input": [text] });
        let resp: Value = match req.send_string(&body.to_string()) {
            Ok(r) => r
                .into_json()
                .map_err(|e| format!("failed to parse response: {e}"))?,
            Err(ureq::Error::Status(code, r)) => {
                return Err(format!(
                    "embeddings API error: status {code}, body: {}",
                    r.into_string().unwrap_or_default()
                ))
            }
            Err(e) => return Err(format!("failed to send request: {e}")),
        };
        let vector = if self.ollama {
            resp.pointer("/embeddings/0")
        } else {
            resp.pointer("/data/0/embedding")
        };
        vector
            .and_then(Value::as_array)
            .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect())
            .ok_or_else(|| "embedding response without a vector".to_string())
    }
}

/// Resolves a dataset's embedding model.
pub struct ServerEmbeddings {
    pub facts: Arc<dyn Facts>,
    pub catalog: Arc<Catalog>,
}

impl EmbeddingResolver for ServerEmbeddings {
    fn resolve(&self, kb: &Knowledgebase) -> Result<Arc<dyn QueryEmbedder>, String> {
        let wrap = |e: String| {
            format!(
                "retrieval: resolve embedding model for dataset {}: {e}",
                kb.id
            )
        };
        let model_ref = match kb.tenant_embd_id.as_deref().map(str::trim) {
            Some(id) if !id.is_empty() => id.to_string(),
            _ => kb.embd_id.clone(),
        };
        let resolver = Resolver {
            facts: self.facts.as_ref(),
            catalog: &self.catalog,
        };
        let access = Access {
            user_id: String::new(),
            tenant_id: kb.tenant_id.clone(),
        };
        let target = if model_ref.trim().is_empty() {
            resolver.resolve_default(&access, EMBEDDING)
        } else {
            resolver.resolve(&access, EMBEDDING, &model_ref)
        }
        .map_err(|e| wrap(e.to_string()))?;
        let provider = self
            .catalog
            .provider(&target.provider)
            .ok_or_else(|| wrap(format!("provider {:?} is not configured", target.provider)))?;
        let ollama = provider.name.eq_ignore_ascii_case("ollama");
        let suffix = provider.url_suffix.get("embedding").ok_or_else(|| {
            wrap(format!(
                "provider {:?} has no embedding endpoint",
                provider.name
            ))
        })?;
        let root = base_url(provider, &target.base_url, &target.region)
            .ok_or_else(|| wrap("no base URL configured".into()))?;
        let local = provider.class.as_deref() == Some("local");
        if target.api_key.trim().is_empty() && !local {
            return Err(wrap("api key is required".into()));
        }
        Ok(Arc::new(HttpEmbedder {
            url: format!("{root}/{}", suffix.trim_start_matches('/')),
            api_key: target.api_key,
            model: target.model_name,
            ollama,
            strict: !local,
        }))
    }
}

/// Settings of the document store.
pub struct SearchConfig {
    pub es_url: String,
    pub es_user: Option<String>,
    pub es_password: Option<String>,
    pub mysql_url: String,
    pub dict_path: std::path::PathBuf,
    pub res_dir: Option<std::path::PathBuf>,
}

pub fn retrieval(
    cfg: &SearchConfig,
    facts: Arc<dyn Facts>,
    catalog: Arc<Catalog>,
) -> Result<Arc<dyn RetrievalService>, String> {
    let qb = ragflow_retrieval::query_builder(&cfg.dict_path, cfg.res_dir.as_deref())?;
    let store = Arc::new(MysqlStore::new(&cfg.mysql_url)?);
    let engine = Arc::new(EsHttp::new(
        &cfg.es_url,
        cfg.es_user.as_deref(),
        cfg.es_password.as_deref(),
    ));
    let service = Arc::new(NlpService {
        engine,
        documents: store.clone(),
        qb,
    });
    Ok(Arc::new(NlpRetrievalAdapter {
        service,
        kbs: store,
        embeddings: Arc::new(ServerEmbeddings { facts, catalog }),
        rerankers: None,
        retrieve_by_children: true,
    }))
}
