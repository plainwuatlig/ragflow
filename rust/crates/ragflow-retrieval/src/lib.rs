//! RAGFlow's agent Retrieval over Elasticsearch, reproducing upstream's Go
//! server at v1.0.0-rc1 (`internal/service/nlp`, the ES engine, the agent
//! retrieval adapter), with tokenization through `ragflow-analyzer` (the same
//! C++ analyzer the Go server links).
//!
//! Upstream behaviour is kept even where it is odd; the module docs name each
//! quirk. Store access is behind traits ([`es::SearchEngine`],
//! [`service::DocumentStore`], [`adapter::KnowledgebaseStore`],
//! [`adapter::EmbeddingResolver`]); the `live` feature provides HTTP/MySQL
//! implementations.

pub mod adapter;
pub mod es;
pub mod gocompat;
pub mod jianfan;
pub mod query_builder;
pub mod scorer;
pub mod service;
pub mod synonym;
pub mod term_weight;
pub mod tokenizer;
pub mod wordnet;

#[cfg(feature = "live")]
pub mod live;

use std::path::Path;
use std::sync::Arc;

/// Loads the analyzer dictionaries and builds the query builder the way the
/// Go server boots (`tokenizer.Init`, `InitQueryBuilderFromTokenizer`).
/// `res_dir` plays the server's `rag/res` (relative to its working
/// directory); the official image has no `ner.json`, `term.freq` or
/// `synonym.json` there, which `None` reproduces.
pub fn query_builder(
    dict_path: &Path,
    res_dir: Option<&Path>,
) -> Result<Arc<query_builder::QueryBuilder>, String> {
    let dict = ragflow_analyzer::Dictionary::load_quietly(dict_path).map_err(|e| e.to_string())?;
    let tok = Arc::new(tokenizer::Tokenizer::new(dict));
    Ok(Arc::new(query_builder::QueryBuilder::new(
        tok,
        res_dir,
        Some(&dict_path.join("wordnet")),
    )))
}
