//! `NLPRetrievalAdapter.Search` (internal/agent/tool/retrieval_nlp.go @
//! v1.0.0-rc1) behind `RuntimeAdapter` (forces `AllowDenseFallback=false`),
//! implementing the runtime's [`RetrievalService`].
//!
//! Features that need an LLM or the tag index are refused with an error
//! instead of silently degrading: `meta_data_filter`, `cross_languages`,
//! `toc_enhance`, datasets with a tag source, and a `rerank_id` without a
//! configured resolver.

use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

use ragflow_runtime::retrieval::{
    RetrievalChunk, RetrievalRequest, RetrievalService as ToolRetrievalService,
};
use regex::Regex;
use serde_json::{Map, Value};

use crate::scorer::Reranker;
use crate::service::{QueryEmbedder, RetrievalRequest as NlpRequest, RetrievalService};

/// The `knowledgebase` row fields retrieval reads.
#[derive(Debug, Clone, Default)]
pub struct Knowledgebase {
    pub id: String,
    pub tenant_id: String,
    pub name: String,
    pub embd_id: String,
    pub tenant_embd_id: Option<String>,
    pub parser_config: Value,
}

/// MySQL `knowledgebase` lookups (`KnowledgebaseDAO`).
pub trait KnowledgebaseStore: Send + Sync {
    /// `id IN ? AND status = '1'`.
    fn get_by_ids(&self, ids: &[String]) -> Result<Vec<Knowledgebase>, String>;
    /// `LOWER(name) = LOWER(?) AND tenant_id = ? AND status = '1'`.
    fn get_by_name(&self, name: &str, tenant_id: &str) -> Result<Option<Knowledgebase>, String>;
    /// `tenant_model.model_name` for a tenant model id.
    fn tenant_model_name(&self, id: &str) -> Option<String>;
}

/// Builds the query embedder for a dataset (`resolveEmbeddingModel`).
pub trait EmbeddingResolver: Send + Sync {
    fn resolve(&self, kb: &Knowledgebase) -> Result<Arc<dyn QueryEmbedder>, String>;
}

/// Builds a rerank model from `rerank_id` (`resolveRerankModel`).
pub trait RerankResolver: Send + Sync {
    fn resolve(&self, tenant_id: &str, rerank_id: &str) -> Result<Arc<dyn Reranker>, String>;
}

pub struct NlpRetrievalAdapter {
    pub service: Arc<RetrievalService>,
    pub kbs: Arc<dyn KnowledgebaseStore>,
    pub embeddings: Arc<dyn EmbeddingResolver>,
    pub rerankers: Option<Arc<dyn RerankResolver>>,
    /// `RetrieveByChildren` runs (the production enhancer is installed).
    pub retrieve_by_children: bool,
}

fn user_prefix() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)^user[:：\s]*").unwrap())
}

fn compact(v: &[String]) -> Vec<String> {
    v.iter()
        .map(|s| crate::term_weight::go_trim_space(s))
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// `common.BaseModelName`.
fn base_model_name(name: &str) -> String {
    if let Some(idx) = name.rfind('@').filter(|i| *i > 0) {
        let base = &name[..idx];
        if let Some(i2) = base.rfind('@').filter(|i| *i > 0) {
            return base[..i2].to_string();
        }
        return base.to_string();
    }
    name.to_string()
}

/// `TagFileIDFromParserConfig`.
pub fn tag_file_id(parser_config: &Value) -> String {
    let from_tags = |v: Option<&Value>| -> String {
        v.and_then(|t| t.get("tag_file_id"))
            .and_then(Value::as_str)
            .map(|s| crate::term_weight::go_trim_space(s).to_string())
            .unwrap_or_default()
    };
    let Some(m) = parser_config.as_object() else {
        return String::new();
    };
    let v = from_tags(m.get("tags"));
    if !v.is_empty() {
        return v;
    }
    let mut keys: Vec<&String> = m
        .keys()
        .filter(|k| *k == "Extractor" || k.starts_with("Extractor:"))
        .collect();
    keys.sort();
    for k in keys {
        if let Some(obj) = m.get(k).filter(|x| x.is_object()) {
            let v = from_tags(obj.get("tags"));
            if !v.is_empty() {
                return v;
            }
        }
    }
    String::new()
}

impl NlpRetrievalAdapter {
    fn embedding_key(&self, kb: &Knowledgebase) -> String {
        let raw = kb.embd_id.trim().to_string();
        let tenant_embd = kb
            .tenant_embd_id
            .as_deref()
            .map(str::trim)
            .unwrap_or("")
            .to_string();
        if raw.is_empty() && tenant_embd.is_empty() {
            return format!("default:{}", kb.tenant_id.trim());
        }
        let mut id = tenant_embd;
        if id.is_empty() && !raw.is_empty() && !raw.contains('@') {
            id = raw.clone();
        }
        if id.is_empty() {
            return format!("embedding:{}", base_model_name(&raw));
        }
        let mut base = self
            .kbs
            .tenant_model_name(&id)
            .map(|s| s.trim().to_string())
            .unwrap_or_default();
        if base.is_empty() {
            base = if !raw.is_empty() && raw != id {
                base_model_name(&raw)
            } else {
                id
            };
        }
        format!("embedding:{base}")
    }

    /// `NLPRetrievalAdapter.Search`.
    pub fn search_tool(&self, req: &RetrievalRequest) -> Result<Vec<RetrievalChunk>, String> {
        if req.use_kg {
            return Err("GraphRAG retrieval is not supported".into());
        }
        if req.query.is_empty() {
            return Ok(Vec::new());
        }
        let top_n = if req.top_n <= 0 { 8 } else { req.top_n };
        // resolveDatasets
        let dataset_ids = compact(&req.dataset_ids);
        if dataset_ids.is_empty() {
            return Err("retrieval: dataset_ids is required".into());
        }
        let found = self
            .kbs
            .get_by_ids(&dataset_ids)
            .map_err(|e| format!("retrieval: resolve dataset tenants: {e}"))?;
        let by_id: BTreeMap<String, Knowledgebase> =
            found.into_iter().map(|k| (k.id.clone(), k)).collect();
        let mut kbs = Vec::new();
        for id in &dataset_ids {
            let mut kb = by_id.get(id).cloned();
            if kb.is_none() && !req.tenant_id.trim().is_empty() {
                kb = self
                    .kbs
                    .get_by_name(id, &req.tenant_id)
                    .map_err(|e| format!("retrieval: resolve dataset {id:?} by name: {e}"))?;
            }
            kbs.push(kb.ok_or_else(|| format!("retrieval: dataset {id:?} was not found"))?);
        }
        let mut tenant_ids: Vec<String> = Vec::new();
        for kb in &kbs {
            let t = kb.tenant_id.trim().to_string();
            if !t.is_empty() && !tenant_ids.contains(&t) {
                tenant_ids.push(t);
            }
        }
        if tenant_ids.is_empty() {
            return Err(format!(
                "retrieval: no valid knowledge bases found for dataset_ids {dataset_ids:?}"
            ));
        }
        if tenant_ids.len() != 1 {
            return Err("retrieval: datasets span multiple tenants".into());
        }
        let first = self.embedding_key(&kbs[0]);
        if kbs[1..].iter().any(|kb| self.embedding_key(kb) != first) {
            return Err("retrieval: datasets use different embedding models".into());
        }
        let embedder = self.embeddings.resolve(&kbs[0]).map_err(|e| {
            format!(
                "retrieval: resolve embedding model for dataset {}: {e}",
                kbs[0].id
            )
        })?;
        let method = req
            .meta_data_filter
            .as_ref()
            .and_then(|m| m.get("method"))
            .and_then(Value::as_str)
            .unwrap_or("");
        if req.toc_enhance || method == "auto" || method == "semi_auto" {
            return Err("retrieval: chat-model features (toc_enhance, auto metadata filter) are not supported".into());
        }
        let doc_ids = compact(&req.doc_scope);
        if req.meta_data_filter.as_ref().is_some_and(|m| !m.is_empty()) {
            return Err("retrieval: meta_data_filter is not supported".into());
        }
        if !req.cross_languages.is_empty() {
            return Err("retrieval: cross_languages is not supported".into());
        }
        let query = user_prefix().replace_all(&req.query, "").into_owned();
        let vector_only = req.keywords_similarity_weight.is_some_and(|k| k < 0.01);
        let mut rank_feature: Option<BTreeMap<String, f64>> = None;
        if let Some(rf) = req.rank_feature.as_ref().filter(|m| !m.is_empty()) {
            rank_feature = Some(rf.clone());
        } else if !vector_only {
            // LabelQuestion: nil unless a dataset has a tag source.
            if kbs
                .iter()
                .any(|kb| !tag_file_id(&kb.parser_config).is_empty())
            {
                return Err("retrieval: datasets with a tag source are not supported".into());
            }
        }
        let rerank = if req.rerank_id.is_empty() {
            None
        } else {
            match &self.rerankers {
                Some(r) => Some(
                    r.resolve(&kbs[0].tenant_id, &req.rerank_id)
                        .map_err(|e| format!("retrieval: resolve rerank model: {e}"))?,
                ),
                None => return Err("retrieval: model resolver is not configured".into()),
            }
        };
        // nlpRequestFromRetrieval
        let mut n = NlpRequest {
            question: query,
            tenant_ids: tenant_ids.clone(),
            kb_ids: kbs.iter().map(|k| k.id.clone()).collect(),
            doc_ids,
            page: 1,
            page_size: top_n,
            embedding_model: Some(embedder),
            allow_dense_fallback: req.allow_dense_fallback,
            vector_only,
            ..Default::default()
        };
        if req.exclude_compiled {
            return Err("retrieval: exclude_compiled is not supported".into());
        }
        if req.rerank_candidates_count != 0 {
            n.rerank_candidates_count = Some(req.rerank_candidates_count);
        }
        if req.top_k > 0 {
            n.knn_top_k = Some(req.top_k);
        } else if top_n > 0 {
            n.knn_top_k = Some(top_n * 4);
        }
        if let Some(t) = req.similarity_threshold {
            n.similarity_threshold = Some(t);
        }
        if let Some(kw) = req.keywords_similarity_weight {
            n.vector_similarity_weight = Some(1.0 - kw);
            if kw > 0.99 {
                n.embedding_model = None;
            }
        }
        if req.only_original_text {
            return Err("retrieval: only_original_text is not supported".into());
        }
        n.rerank_model = rerank;
        if rank_feature.is_some() {
            n.rank_feature = rank_feature;
        }
        let res = self.service.retrieval(&mut n)?;
        if res.chunks.is_empty() {
            return Ok(Vec::new());
        }
        let mut raw = res.chunks;
        if self.retrieve_by_children {
            raw = self.service.retrieval_by_children(raw, &tenant_ids);
        }
        Ok(raw.iter().map(translate_chunk).collect())
    }
}

impl ToolRetrievalService for NlpRetrievalAdapter {
    /// `RuntimeAdapter.Search`: dense-only fallbacks off, empty rank feature → nil.
    fn search(&self, req: &RetrievalRequest) -> Result<Vec<RetrievalChunk>, String> {
        let mut r = req.clone();
        r.allow_dense_fallback = Some(false);
        if r.rank_feature.as_ref().is_some_and(|m| m.is_empty()) {
            r.rank_feature = None;
        }
        self.search_tool(&r)
    }

    fn resolve_dataset_id(
        &self,
        name_or_id: &str,
        tenant_id: &str,
        _user_id: &str,
    ) -> Option<String> {
        if let Ok(kbs) = self.kbs.get_by_ids(&[name_or_id.to_string()]) {
            if let Some(kb) = kbs.into_iter().next() {
                return Some(kb.id);
            }
        }
        if !tenant_id.is_empty() {
            if let Ok(Some(kb)) = self.kbs.get_by_name(name_or_id, tenant_id) {
                return Some(kb.id);
            }
        }
        None
    }
}

fn s(raw: &Map<String, Value>, k: &str) -> String {
    raw.get(k).and_then(Value::as_str).unwrap_or("").to_string()
}

fn first_s(raw: &Map<String, Value>, keys: &[&str]) -> String {
    keys.iter()
        .map(|k| s(raw, k))
        .find(|v| !v.is_empty())
        .unwrap_or_default()
}

fn num(raw: &Map<String, Value>, k: &str) -> Option<f64> {
    raw.get(k).and_then(Value::as_f64)
}

/// `translateChunk`.
pub fn translate_chunk(raw: &Map<String, Value>) -> RetrievalChunk {
    let content = {
        let c = s(raw, "content_with_weight");
        if c.is_empty() {
            s(raw, "content_ltks")
        } else {
            c
        }
    };
    let positions = ["positions", "position_int"]
        .iter()
        .find_map(|k| raw.get(*k).filter(|v| !v.is_null()).cloned());
    let score = match num(raw, "similarity") {
        Some(f) => f,
        None => match (num(raw, "term_similarity"), num(raw, "vector_similarity")) {
            (Some(t), Some(v)) => (t + v) / 2.0,
            (Some(t), None) => t,
            (None, Some(v)) => v,
            _ => 0.0,
        },
    };
    RetrievalChunk {
        id: s(raw, "chunk_id"),
        content,
        document_id: s(raw, "doc_id"),
        document_name: s(raw, "docnm_kwd"),
        dataset_id: s(raw, "kb_id"),
        image_id: first_s(raw, &["image_id", "img_id"]),
        doc_type: first_s(raw, &["doc_type_kwd", "doc_type"]),
        url: first_s(raw, &["url", "document_url", "doc_url"]),
        positions,
        mom_id: s(raw, "mom_id"),
        chunk_index: num(raw, "chunk_order_int").map(|f| f as i64).unwrap_or(0),
        page_num: num(raw, "page_num_int").map(|f| f as i64).unwrap_or(0),
        score,
        term_similarity: num(raw, "term_similarity").unwrap_or(0.0),
        vector_similarity: num(raw, "vector_similarity").unwrap_or(0.0),
    }
}
