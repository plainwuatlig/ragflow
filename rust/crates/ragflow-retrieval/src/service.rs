//! `nlp.RetrievalService` (internal/service/nlp/retrieval.go @ v1.0.0-rc1),
//! Elasticsearch only: `Retrieval`, `Search` with its retries,
//! `PruneDeletedChunks` (including the field-map rebuild quirk),
//! `scoreSearchResult`, `RetrievalByChildren`.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use serde_json::{json, Map, Value};

use crate::es::{
    chunk_ids, engine_search, get_fields, get_scores, knn_scores_request, Chunk, FilterValue,
    MatchDense, SearchEngine, SearchRequest,
};
use crate::gocompat::{pairwise_sum, sort_slice_stable};
use crate::query_builder::QueryBuilder;
use crate::scorer::{rerank_by_model, rerank_standard, rerank_with_knn, Reranker, Scores};

/// The `src` list `Search` selects (and `GetFields` projects).
pub const SRC: [&str; 18] = [
    "docnm_kwd",
    "content_ltks",
    "kb_id",
    "img_id",
    "title_tks",
    "important_kwd",
    "position_int",
    "doc_id",
    "chunk_order_int",
    "page_num_int",
    "top_int",
    "create_timestamp_flt",
    "knowledge_graph_kwd",
    "content_with_weight",
    "doc_type_kwd",
    "mom_id",
    "row_id()",
    "_score",
];

/// Which document ids still exist (`DocumentDAO.GetByIDs`, no status filter).
pub trait DocumentStore: Send + Sync {
    fn existing_document_ids(&self, ids: &[String]) -> Result<Vec<String>, String>;
}

/// Query embedding (`EmbeddingModel` + `GetVector`).
pub trait QueryEmbedder: Send + Sync {
    /// The query-side embedding of `text`.
    fn embed_query(&self, text: &str) -> Result<Vec<f64>, String>;
}

/// `nlp.RetrievalRequest`.
#[derive(Clone, Default)]
pub struct RetrievalRequest {
    pub question: String,
    pub tenant_ids: Vec<String>,
    pub kb_ids: Vec<String>,
    pub doc_ids: Vec<String>,
    pub page: i64,
    pub page_size: i64,
    pub rerank_candidates_count: Option<i64>,
    pub knn_top_k: Option<i64>,
    pub knn_num_candidates: Option<i64>,
    pub similarity_threshold: Option<f64>,
    pub vector_similarity_weight: Option<f64>,
    pub rank_feature: Option<BTreeMap<String, f64>>,
    pub rerank_model: Option<Arc<dyn Reranker>>,
    pub embedding_model: Option<Arc<dyn QueryEmbedder>>,
    pub allow_dense_fallback: Option<bool>,
    /// `Filter`: extra filter-map entries.
    pub filter: Vec<(String, FilterValue)>,
    pub vector_only: bool,
}

#[derive(Debug, Clone, Default)]
pub struct RetrievalResult {
    /// Result chunk maps, in final order (keys as Go writes them).
    pub chunks: Vec<Map<String, Value>>,
    pub total: i64,
}

/// `RetrievalSearchResult`.
pub struct SearchOutcome {
    pub chunks: Vec<Chunk>,
    pub total: i64,
    pub query_vector: Option<Vec<f64>>,
    pub field: BTreeMap<String, Chunk>,
    pub ids: Vec<String>,
    /// The request bodies sent, in order (diagnostics).
    pub requests: Vec<Value>,
}

pub struct RetrievalService {
    pub engine: Arc<dyn SearchEngine>,
    pub documents: Arc<dyn DocumentStore>,
    pub qb: Arc<QueryBuilder>,
}

/// `minMatch`.
fn min_match(vw: Option<f64>, with: f64) -> f64 {
    if vw.is_some_and(|v| v >= 0.8) {
        0.0
    } else {
        with
    }
}

/// `buildIndexNames`.
pub fn build_index_names(tenant_ids: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for tid in tenant_ids {
        for part in tid.split(',') {
            let part = crate::term_weight::go_trim_space(part);
            if !part.is_empty() {
                out.push(format!("ragflow_{part}"));
            }
        }
    }
    out
}

impl RetrievalService {
    /// `Retrieval`.
    pub fn retrieval(&self, req: &mut RetrievalRequest) -> Result<RetrievalResult, String> {
        if req.question.is_empty() {
            return Ok(RetrievalResult::default());
        }
        let knn_top_k = *req.knn_top_k.get_or_insert(1024);
        let knn_num = *req.knn_num_candidates.get_or_insert(2048);
        let threshold = *req.similarity_threshold.get_or_insert(0.2);
        let vw = *req
            .vector_similarity_weight
            .get_or_insert(if req.vector_only { 1.0 } else { 0.3 });
        let rank_feature = req
            .rank_feature
            .get_or_insert_with(|| BTreeMap::from([("pagerank_fea".to_string(), 10.0)]))
            .clone();
        if req.page <= 0 {
            req.page = 1;
        }
        if req.page_size <= 0 {
            req.page_size = 1;
        }
        let rcc = *req.rerank_candidates_count.get_or_insert(64);
        let page_size = req.page_size;
        if rcc <= 0 || req.page > rcc / page_size {
            return Err(format!(
                "rerank_candidates_count({rcc}) must be greater than or equal to page({}) * page_size({page_size})",
                req.page
            ));
        }
        if req.rerank_model.is_some() && req.page != 1 {
            return Err(format!(
                "Pagination is not supported when rerank_mdl is specified. Please set page=1 to retrieve the top {page_size} results."
            ));
        }

        let sr = self.search(req, rcc, knn_top_k, knn_num, threshold, vw, &rank_feature)?;
        let mut sr = self.prune_deleted_chunks(sr)?;
        if sr.total == 0 {
            return Ok(RetrievalResult::default());
        }
        let scores = self.score_search_result(req, vw, &rank_feature, &mut sr)?;
        if scores.sim.is_empty() {
            return Ok(RetrievalResult::default());
        }
        let mut idx: Vec<(usize, f64)> = scores.sim.iter().copied().enumerate().collect();
        sort_slice_stable(&mut idx, |a, b| a.1 > b.1);
        let mut post = threshold;
        if vw <= 0.0 || req.vector_only {
            post = 0.0;
        }
        let valid: Vec<usize> = idx
            .iter()
            .filter(|(_, s)| *s >= post)
            .map(|(i, _)| *i)
            .collect();
        if valid.is_empty() {
            return Ok(RetrievalResult::default());
        }
        let begin = ((req.page - 1) * page_size) as usize;
        let mut end = begin + page_size as usize;
        let page_idx: &[usize] = if begin < valid.len() {
            if end > valid.len() {
                end = valid.len();
            }
            &valid[begin..end]
        } else {
            &[]
        };
        let dim = sr.query_vector.as_ref().map(Vec::len).unwrap_or(0);
        let mut out = Vec::new();
        for &i in page_idx {
            let Some(chunk_id) = sr.ids.get(i) else {
                continue;
            };
            let Some(c) = sr.field.get(chunk_id) else {
                continue;
            };
            out.push(result_chunk(
                chunk_id,
                c,
                scores.sim[i],
                scores.tsim[i],
                scores.vsim[i],
                dim,
            ));
        }
        Ok(RetrievalResult {
            chunks: out,
            total: valid.len() as i64,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn search(
        &self,
        req: &RetrievalRequest,
        rcc: i64,
        knn_top_k: i64,
        knn_num: i64,
        threshold: f64,
        vw: f64,
        rank_feature: &BTreeMap<String, f64>,
    ) -> Result<SearchOutcome, String> {
        let allow_dense_fallback = req.allow_dense_fallback.unwrap_or(true);
        // GetFilters: kb_id, doc_id, then Filter entries, then available_int.
        let mut filters: Vec<(String, FilterValue)> = Vec::new();
        if !req.kb_ids.is_empty() {
            filters.push(("kb_id".into(), FilterValue::StrList(req.kb_ids.clone())));
        }
        if !req.doc_ids.is_empty() {
            filters.push(("doc_id".into(), FilterValue::StrList(req.doc_ids.clone())));
        }
        for (k, v) in &req.filter {
            if !filters.iter().any(|(fk, _)| fk == k) {
                filters.push((k.clone(), v.clone()));
            }
        }
        if !filters.iter().any(|(k, _)| k == "available_int") {
            filters.push(("available_int".into(), FilterValue::Int(1)));
        }
        let has_doc_filter = filters.iter().any(|(k, _)| k == "doc_id");
        let knn_top_k = if knn_top_k <= 0 { 1024 } else { knn_top_k };
        let num_candidates = if knn_num <= 0 { 2048 } else { knn_num };
        let page_size = if rcc <= 0 { 30 } else { rcc };
        let src: Vec<String> = SRC.iter().map(|s| s.to_string()).collect();
        let mut sreq = SearchRequest {
            index_names: build_index_names(&req.tenant_ids),
            kb_ids: req.kb_ids.clone(),
            offset: 0,
            limit: page_size,
            filter: filters,
            select_fields: src.clone(),
            ..Default::default()
        };
        let mut requests = Vec::new();
        let run = |sreq: &SearchRequest, requests: &mut Vec<Value>| {
            requests.push(crate::es::search_body(sreq));
            engine_search(self.engine.as_ref(), sreq)
        };

        let vwo = Some(vw);
        let (match_text, _keywords) = self.qb.question(&req.question, "", min_match(vwo, 0.3));
        let result;
        let mut query_vector = None;
        match &req.embedding_model {
            None => {
                sreq.match_text = match_text;
                sreq.rank_feature = Some(rank_feature.clone());
                result = run(&sreq, &mut requests)?;
            }
            Some(emb) => {
                let similarity = if threshold <= 0.0 { 0.1 } else { threshold };
                let vector = emb
                    .embed_query(&req.question)
                    .map_err(|e| format!("GetVector failed: {e}"))?;
                let dense = MatchDense {
                    vector_column: format!("q_{}_vec", vector.len()),
                    embedding: vector,
                    top_n: knn_top_k,
                    similarity,
                    num_candidates,
                };
                sreq.select_fields.push(dense.vector_column.clone());
                let dense_only = req.vector_only || match_text.is_none();
                if dense_only {
                    sreq.match_text = None;
                    sreq.fusion_vector_weight = None;
                } else {
                    sreq.match_text = match_text.clone();
                    sreq.fusion_vector_weight = Some(1.0); // "0.001,1"
                }
                sreq.match_dense = Some(dense.clone());
                sreq.rank_feature = Some(rank_feature.clone());
                let mut r = run(&sreq, &mut requests)?;
                let mut final_dense = dense.clone();
                if r.total == 0 {
                    if dense_only {
                        if allow_dense_fallback {
                            final_dense = MatchDense {
                                similarity: 0.17,
                                ..dense.clone()
                            };
                            sreq.match_text = None;
                            sreq.fusion_vector_weight = None;
                            sreq.match_dense = Some(final_dense.clone());
                            r = run(&sreq, &mut requests)?;
                        }
                    } else if has_doc_filter {
                        sreq.select_fields = src.clone();
                        sreq.match_text = None;
                        sreq.match_dense = None;
                        sreq.fusion_vector_weight = None;
                        sreq.rank_feature = None;
                        r = run(&sreq, &mut requests)?;
                    } else {
                        let (mt2, _) = self.qb.question(&req.question, "qa", min_match(vwo, 0.1));
                        final_dense = MatchDense {
                            similarity: 0.17,
                            ..dense.clone()
                        };
                        if req.vector_only || mt2.is_none() {
                            sreq.match_text = None;
                            sreq.fusion_vector_weight = None;
                        } else {
                            sreq.match_text = mt2.clone();
                            sreq.fusion_vector_weight = Some(1.0);
                        }
                        sreq.match_dense = Some(final_dense.clone());
                        sreq.rank_feature = Some(rank_feature.clone());
                        r = run(&sreq, &mut requests)?;
                        if r.total == 0 && mt2.is_some() && allow_dense_fallback {
                            final_dense = MatchDense {
                                similarity: 0.17,
                                ..dense.clone()
                            };
                            sreq.match_text = None;
                            sreq.fusion_vector_weight = None;
                            sreq.match_dense = Some(final_dense.clone());
                            r = run(&sreq, &mut requests)?;
                        }
                    }
                }
                query_vector = Some(final_dense.embedding);
                result = r;
            }
        }
        let mut chunks = result.chunks;
        let ids = chunk_ids(&chunks);
        let field = get_fields(&mut chunks, &SRC);
        Ok(SearchOutcome {
            chunks,
            total: result.total,
            query_vector,
            field,
            ids,
            requests,
        })
    }

    /// `PruneDeletedChunks`. When any document is gone, `Chunks` is rebuilt
    /// from the field maps (no `_id`/`_index`), so the later KNN pass finds
    /// no ids and every vector similarity becomes 0 — as in Go.
    pub fn prune_deleted_chunks(&self, r: SearchOutcome) -> Result<SearchOutcome, String> {
        let mut unique: Vec<String> = Vec::new();
        let mut seen = HashSet::new();
        for c in r.field.values() {
            if let Some(d) = c.get("doc_id").and_then(Value::as_str) {
                if !d.is_empty() && seen.insert(d.to_string()) {
                    unique.push(d.to_string());
                }
            }
        }
        if unique.is_empty() {
            return Ok(r);
        }
        let existing: HashSet<String> = self
            .documents
            .existing_document_ids(&unique)
            .map_err(|e| format!("PruneDeletedChunks failed: GetByIDs failed: {e}"))?
            .into_iter()
            .collect();
        if existing.len() == unique.len() {
            return Ok(r);
        }
        let mut ids = Vec::new();
        let mut chunks = Vec::new();
        let mut field = BTreeMap::new();
        for id in &r.ids {
            let Some(c) = r.field.get(id) else { continue };
            match c.get("doc_id").and_then(Value::as_str) {
                Some(d) if !d.is_empty() => {
                    if !existing.contains(d) {
                        continue;
                    }
                }
                _ => {}
            }
            ids.push(id.clone());
            chunks.push(c.clone());
            field.insert(id.clone(), c.clone());
        }
        Ok(SearchOutcome {
            total: ids.len() as i64,
            chunks,
            query_vector: r.query_vector,
            field,
            ids,
            requests: r.requests,
        })
    }

    fn score_search_result(
        &self,
        req: &RetrievalRequest,
        vw: f64,
        rank_feature: &BTreeMap<String, f64>,
        sr: &mut SearchOutcome,
    ) -> Result<Scores, String> {
        let tk = 1.0 - vw;
        if let (Some(rm), true) = (&req.rerank_model, sr.total > 0) {
            return rerank_by_model(
                rm.as_ref(),
                &sr.chunks,
                &sr.ids,
                &sr.field,
                &req.question,
                tk,
                vw,
                "content_ltks",
                &self.qb,
                rank_feature,
            );
        }
        if req.vector_only {
            let mut scores: BTreeMap<String, f64> = BTreeMap::new();
            for c in &sr.chunks {
                let mut id = c
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                if id.is_empty() {
                    id = c
                        .get("_id")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                }
                if id.is_empty() {
                    continue;
                }
                if let Some(s) = c.get("_score").and_then(Value::as_f64) {
                    scores.insert(id, s);
                }
            }
            let sim: Vec<f64> = sr
                .ids
                .iter()
                .map(|id| scores.get(id).copied().unwrap_or(0.0))
                .collect();
            return Ok(Scores {
                tsim: vec![0.0; sim.len()],
                vsim: sim.clone(),
                sim,
            });
        }
        let qv = sr.query_vector.clone().unwrap_or_default();
        let knn = match knn_scores_request(&sr.chunks, &qv) {
            None => Ok(BTreeMap::new()),
            Some((index, body)) => {
                sr.requests.push(body.clone());
                match self.engine.search(&index, &body, false) {
                    Ok(resp) if resp.get("error").is_none() => Ok(get_scores(&resp)),
                    Ok(resp) => Err(format!(
                        "KNN scores search returned error: {}",
                        resp["error"]
                    )),
                    Err(e) => Err(e),
                }
            }
        };
        match knn {
            Ok(knn) => Ok(rerank_with_knn(
                &sr.chunks,
                &sr.ids,
                &mut sr.field,
                &knn,
                &req.question,
                tk,
                vw,
                "content_ltks",
                &self.qb,
                rank_feature,
            )),
            Err(e) => {
                eprintln!(
                    "ragflow-retrieval: KNNScores failed, falling back to local computation: {e}"
                );
                Ok(rerank_standard(
                    &sr.chunks,
                    &qv,
                    &req.question,
                    tk,
                    vw,
                    "content_ltks",
                    &self.qb,
                    rank_feature,
                ))
            }
        }
    }

    /// `RetrievalByChildren` (via the enhancer): children sharing a `mom_id`
    /// collapse into their parent chunk; a no-op without `mom_id`.
    pub fn retrieval_by_children(
        &self,
        chunks: Vec<Map<String, Value>>,
        tenant_ids: &[String],
    ) -> Vec<Map<String, Value>> {
        let index_names = build_index_names(tenant_ids);
        if chunks.is_empty() || index_names.is_empty() {
            return chunks;
        }
        // Go groups in a map (random iteration); BTreeMap fixes an order.
        let mut groups: BTreeMap<(String, String, String), Vec<Map<String, Value>>> =
            BTreeMap::new();
        let mut remaining = Vec::new();
        for c in &chunks {
            let mom = c.get("mom_id").and_then(Value::as_str).unwrap_or("");
            let doc = c.get("doc_id").and_then(Value::as_str).unwrap_or("");
            if mom.is_empty() || doc.is_empty() {
                remaining.push(c.clone());
                continue;
            }
            let kb = c.get("kb_id").and_then(Value::as_str).unwrap_or("");
            groups
                .entry((mom.to_string(), doc.to_string(), kb.to_string()))
                .or_default()
                .push(c.clone());
        }
        if groups.is_empty() {
            return chunks;
        }
        let mut vector_size = 1024;
        for ((mom, doc, kb), children) in groups {
            let sreq = SearchRequest {
                index_names: vec![index_names[0].clone()],
                kb_ids: vec![kb.clone()],
                limit: 1,
                filter: vec![
                    ("id".into(), FilterValue::Str(mom.clone())),
                    ("doc_id".into(), FilterValue::Str(doc.clone())),
                ],
                ..Default::default()
            };
            let parent = match engine_search(self.engine.as_ref(), &sreq) {
                Ok(r) if !r.chunks.is_empty() => r.chunks.into_iter().next().unwrap(),
                _ => {
                    remaining.extend(children);
                    continue;
                }
            };
            let sims: Vec<f64> = children
                .iter()
                .filter_map(|c| c.get("similarity").and_then(Value::as_f64))
                .collect();
            let avg = pairwise_sum(&sims) / children.len() as f64;
            let content: Vec<&str> = children
                .iter()
                .filter_map(|c| c.get("content_ltks").and_then(Value::as_str))
                .collect();
            let mut kwds = Vec::new();
            for c in &children {
                if let Some(Value::Array(a)) = c.get("important_kwd") {
                    kwds.extend(a.iter().filter(|x| x.is_string()).cloned());
                }
            }
            let doc_type = parent
                .get("doc_type_kwd")
                .and_then(Value::as_str)
                .unwrap_or("");
            let img = match parent.get("img_id") {
                None | Some(Value::Null) => json!(""),
                Some(v) if v.as_str() == Some("") => json!(""),
                Some(v) => v.clone(),
            };
            let mut agg = Map::new();
            agg.insert("chunk_id".into(), json!(mom));
            agg.insert("content_ltks".into(), json!(content.join(" ")));
            agg.insert(
                "content_with_weight".into(),
                parent
                    .get("content_with_weight")
                    .cloned()
                    .unwrap_or(Value::Null),
            );
            agg.insert(
                "doc_id".into(),
                parent.get("doc_id").cloned().unwrap_or(Value::Null),
            );
            agg.insert(
                "docnm_kwd".into(),
                parent.get("docnm_kwd").cloned().unwrap_or(Value::Null),
            );
            agg.insert(
                "kb_id".into(),
                parent.get("kb_id").cloned().unwrap_or(Value::Null),
            );
            agg.insert("important_kwd".into(), Value::Array(kwds));
            agg.insert("image_id".into(), img);
            agg.insert("similarity".into(), json!(avg));
            agg.insert("vector_similarity".into(), json!(avg));
            agg.insert("term_similarity".into(), json!(avg));
            agg.insert("vector".into(), json!(vec![0.0; vector_size]));
            agg.insert(
                "positions".into(),
                parent.get("position_int").cloned().unwrap_or(Value::Null),
            );
            agg.insert("doc_type_kwd".into(), json!(doc_type));
            'outer: for c in &children {
                for (k, v) in c {
                    if k.ends_with("_vec") {
                        if let Value::Array(a) = v {
                            vector_size = a.len();
                            agg.insert("vector".into(), Value::Array(a.clone()));
                            break 'outer;
                        }
                    }
                }
            }
            remaining.push(agg);
        }
        let n = remaining.len();
        for i in 0..n {
            for j in i + 1..n {
                let si = remaining[i]
                    .get("similarity")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0);
                let sj = remaining[j]
                    .get("similarity")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0);
                if sj > si {
                    remaining.swap(i, j);
                }
            }
        }
        remaining
    }
}

/// The chunk map `Retrieval` builds for one page entry.
fn result_chunk(
    chunk_id: &str,
    c: &Chunk,
    sim: f64,
    tsim: f64,
    vsim: f64,
    dim: usize,
) -> Map<String, Value> {
    let mut r = Map::new();
    r.insert("chunk_id".into(), json!(chunk_id));
    for k in [
        "content_ltks",
        "content_with_weight",
        "doc_id",
        "docnm_kwd",
        "kb_id",
        "important_kwd",
    ] {
        if let Some(v) = c.get(k) {
            r.insert(k.into(), v.clone());
        }
    }
    r.insert(
        "image_id".into(),
        c.get("img_id").cloned().unwrap_or(json!("")),
    );
    r.insert(
        "positions".into(),
        match c.get("position_int") {
            Some(v) if !v.is_null() => v.clone(),
            _ => json!([]),
        },
    );
    let doc_type = match c.get("doc_type_kwd") {
        Some(Value::String(s)) if !s.is_empty() => json!(s),
        Some(Value::Array(a)) if !a.is_empty() => Value::Array(a.clone()),
        _ => Value::Null,
    };
    r.insert("doc_type_kwd".into(), doc_type);
    r.insert("similarity".into(), json!(sim));
    r.insert("term_similarity".into(), json!(tsim));
    r.insert("vector_similarity".into(), json!(vsim));
    r.insert(
        "important_kwd".into(),
        c.get("important_kwd").cloned().unwrap_or(json!([])),
    );
    r.insert(
        "mom_id".into(),
        c.get("mom_id").cloned().unwrap_or(json!("")),
    );
    r.insert(
        "row_id".into(),
        c.get("row_id()").cloned().unwrap_or(Value::Null),
    );
    r.insert("tag_kwd".into(), json!([]));
    r.insert("vector".into(), json!(vec![0.0; dim]));
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NoEngine;
    impl SearchEngine for NoEngine {
        fn search(&self, _: &str, _: &Value, _: bool) -> Result<Value, String> {
            Err("no engine".into())
        }
    }

    struct Docs(Vec<String>);
    impl DocumentStore for Docs {
        fn existing_document_ids(&self, ids: &[String]) -> Result<Vec<String>, String> {
            Ok(ids.iter().filter(|i| self.0.contains(i)).cloned().collect())
        }
    }

    fn outcome() -> SearchOutcome {
        let mut chunks: Vec<Chunk> = ["c1", "c2", "c3"]
            .iter()
            .zip(["d1", "d2", "d1"])
            .map(|(id, doc)| {
                json!({"_id": id, "id": id, "_index": "ragflow_t", "_score": 0.9, "doc_id": doc,
                       "content_ltks": "a b"})
                .as_object()
                .unwrap()
                .clone()
            })
            .collect();
        let ids = chunk_ids(&chunks);
        let field = get_fields(&mut chunks, &SRC);
        SearchOutcome {
            chunks,
            total: 3,
            query_vector: Some(vec![0.1, 0.2]),
            field,
            ids,
            requests: vec![],
        }
    }

    /// The prune quirk: one deleted document rebuilds `chunks` from the field
    /// maps, which carry no `_id`, so the KNN pass has nothing to ask.
    #[test]
    fn prune_rebuilds_chunks_without_ids() {
        let Ok(dict) = std::env::var("RAGFLOW_DICT_PATH") else {
            eprintln!("RAGFLOW_DICT_PATH unset; skipping");
            return;
        };
        let svc = RetrievalService {
            engine: Arc::new(NoEngine),
            documents: Arc::new(Docs(vec!["d1".into()])),
            qb: crate::query_builder(std::path::Path::new(&dict), None).unwrap(),
        };
        let all = svc.prune_deleted_chunks(outcome()).unwrap();
        assert_eq!(all.total, 2);
        assert_eq!(all.ids, vec!["c1".to_string(), "c3".to_string()]);
        assert!(all.chunks.iter().all(|c| !c.contains_key("_id")));
        assert!(knn_scores_request(&all.chunks, &[0.1, 0.2]).is_none());

        let svc = RetrievalService {
            documents: Arc::new(Docs(vec!["d1".into(), "d2".into()])),
            ..svc
        };
        let kept = svc.prune_deleted_chunks(outcome()).unwrap();
        assert_eq!(kept.total, 3);
        assert!(knn_scores_request(&kept.chunks, &[0.1, 0.2]).is_some());
    }
}
