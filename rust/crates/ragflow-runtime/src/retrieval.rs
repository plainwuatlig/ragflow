//! The retrieval backend the Retrieval component calls, behind a trait so
//! the runtime stays free of store access (upstream's
//! `runtime.RetrievalService`, internal/agent/runtime/retrieval.go).
//! `ragflow-retrieval` implements it over Elasticsearch and MySQL.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

/// `runtime.RetrievalRequest`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RetrievalRequest {
    pub query: String,
    pub dataset_ids: Vec<String>,
    pub memory_ids: Vec<String>,
    pub top_n: i64,
    pub rerank_candidates_count: i64,
    pub top_k: i64,
    pub keywords_similarity_weight: Option<f64>,
    pub use_kg: bool,
    pub similarity_threshold: Option<f64>,
    pub allow_dense_fallback: Option<bool>,
    pub rerank_id: String,
    pub cross_languages: Vec<String>,
    pub toc_enhance: bool,
    pub meta_data_filter: Option<Map<String, Value>>,
    pub retrieval_from: String,
    pub doc_scope: Vec<String>,
    pub tenant_id: String,
    pub rank_feature: Option<BTreeMap<String, f64>>,
    pub user_id: String,
    pub exclude_compiled: bool,
    pub only_original_text: bool,
}

/// `runtime.RetrievalChunk`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RetrievalChunk {
    pub id: String,
    pub content: String,
    pub document_id: String,
    pub document_name: String,
    pub dataset_id: String,
    pub image_id: String,
    pub doc_type: String,
    pub url: String,
    pub positions: Option<Value>,
    pub mom_id: String,
    pub chunk_index: i64,
    pub page_num: i64,
    pub score: f64,
    pub term_similarity: f64,
    pub vector_similarity: f64,
}

/// `runtime.RetrievalService`: dataset search for the canvas.
pub trait RetrievalService: Send + Sync {
    fn search(&self, req: &RetrievalRequest) -> Result<Vec<RetrievalChunk>, String>;

    /// Resolves a dataset reference given by name or id in the legacy
    /// `UserFillUp:` query forms (`resolveRetrievalDatasetID`); `None` when
    /// unknown or unsupported.
    /// Lookup order as upstream: by id, then by name in `tenant_id`, then
    /// by name among datasets visible to `user_id`.
    fn resolve_dataset_id(
        &self,
        name_or_id: &str,
        tenant_id: &str,
        user_id: &str,
    ) -> Option<String> {
        let _ = (name_or_id, tenant_id, user_id);
        None
    }
}
