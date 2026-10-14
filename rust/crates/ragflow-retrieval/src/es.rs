//! The Elasticsearch side of `internal/engine/elasticsearch/chunk.go` @
//! v1.0.0-rc1 that retrieval uses: request bodies (`Search`, `KNNScores`),
//! response conversion, `calculateScores`/`sortByScore`, `GetChunkIDs`,
//! `GetFields`, `GetScores`.
//!
//! The bool query is one object used both as `query` and as `knn.filter`, so
//! the text clause and any rank_feature clauses restrict the kNN leg too.

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

use crate::gocompat::fmt_v;
use crate::query_builder::{MatchText, QUERY_FIELDS};

pub type Chunk = Map<String, Value>;

pub const PAGERANK_FLD: &str = "pagerank_fea";
pub const TAG_FLD: &str = "tag_feas";

/// The engine boundary: one `_search` call against one index.
pub trait SearchEngine: Send + Sync {
    /// `POST /<index>/_search` (with `?track_total_hits=true` when asked);
    /// returns the decoded response body or the error text.
    fn search(&self, index: &str, body: &Value, track_total_hits: bool) -> Result<Value, String>;
}

#[derive(Debug, Clone, PartialEq)]
pub enum FilterValue {
    StrList(Vec<String>),
    Str(String),
    Int(i64),
}

/// `types.MatchDenseExpr`.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchDense {
    pub vector_column: String,
    pub embedding: Vec<f64>,
    pub top_n: i64,
    pub similarity: f64,
    pub num_candidates: i64,
}

/// `types.SearchRequest` with its `MatchExprs` spelled out.
#[derive(Debug, Clone, Default)]
pub struct SearchRequest {
    pub index_names: Vec<String>,
    pub kb_ids: Vec<String>,
    pub offset: i64,
    pub limit: i64,
    /// Filter map in insertion order (Go iterates its map randomly; clause
    /// order inside `filter` does not change ES semantics).
    pub filter: Vec<(String, FilterValue)>,
    pub select_fields: Vec<String>,
    /// `MatchExprs` contains a `MatchTextExpr` slot (possibly nil).
    pub match_text: Option<MatchText>,
    pub match_dense: Option<MatchDense>,
    /// `FusionExpr` weighted_sum weights (`"0.001,1"` → vector weight 1).
    pub fusion_vector_weight: Option<f64>,
    pub rank_feature: Option<BTreeMap<String, f64>>,
}

pub struct SearchResult {
    pub chunks: Vec<Chunk>,
    pub total: i64,
}

/// `common.FormatMinimumShouldMatchPercent`.
pub fn format_msm(fraction: f64) -> String {
    format!("{}%", (fraction * 100.0 + 0.5 + 1e-9).floor() as i64)
}

/// `lowerCaseQueryText`.
pub fn lower_case_query_text(text: &str) -> String {
    let b = text.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(b.len());
    let mut in_quote = false;
    let mut escaped = false;
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if escaped {
            out.push(c);
            escaped = false;
            i += 1;
            continue;
        }
        if c == b'\\' {
            out.push(c);
            escaped = true;
            i += 1;
        } else if c == b'"' {
            in_quote = !in_quote;
            out.push(c);
            i += 1;
        } else if c.is_ascii_alphabetic() {
            let mut j = i;
            while j < b.len() && b[j].is_ascii_alphabetic() {
                j += 1;
            }
            let word = &b[i..j];
            if !in_quote && (word == b"AND" || word == b"OR" || word == b"NOT") {
                out.extend_from_slice(word);
            } else {
                out.extend(word.iter().map(|c| c.to_ascii_lowercase()));
            }
            i = j;
        } else {
            out.push(c);
            i += 1;
        }
    }
    String::from_utf8(out).expect("ASCII-only edits keep UTF-8")
}

fn build_bool_query(
    filter: &[(String, FilterValue)],
    kb_ids: &[String],
) -> Option<Map<String, Value>> {
    let mut filter_clauses: Vec<Value> = Vec::new();
    let mut should_clauses: Vec<Value> = Vec::new();
    if !kb_ids.is_empty() {
        filter_clauses.push(json!({"terms": {"kb_id": kb_ids}}));
    }
    for (k, v) in filter {
        if k == "available_int" {
            let n = match v {
                FilterValue::Int(n) => *n as f64,
                _ => continue,
            };
            if n == 0.0 {
                filter_clauses.push(json!({"range": {"available_int": {"lt": 1}}}));
            } else {
                filter_clauses
                    .push(json!({"bool": {"must_not": [{"range": {"available_int": {"lt": 1}}}]}}));
            }
            continue;
        }
        if k == "id" {
            match v {
                FilterValue::StrList(l) if !l.is_empty() => {
                    should_clauses.push(json!({"terms": {"id": l}}));
                    should_clauses.push(json!({"terms": {"_id": l}}));
                }
                FilterValue::Str(s) if !s.is_empty() => {
                    should_clauses.push(json!({"term": {"id": s}}));
                    should_clauses.push(json!({"term": {"_id": s}}));
                }
                FilterValue::Int(n) if *n != 0 => {
                    should_clauses.push(json!({"term": {"id": n}}));
                    should_clauses.push(json!({"term": {"_id": n}}));
                }
                _ => {}
            }
            continue;
        }
        match v {
            FilterValue::Str(s) if s.is_empty() => {}
            FilterValue::StrList(l) => filter_clauses.push(json!({"terms": {k.clone(): l}})),
            FilterValue::Str(s) => filter_clauses.push(json!({"term": {k.clone(): s}})),
            FilterValue::Int(n) => filter_clauses.push(json!({"term": {k.clone(): n}})),
        }
    }
    let mut b = Map::new();
    if !filter_clauses.is_empty() {
        b.insert("filter".into(), Value::Array(filter_clauses));
    }
    if !should_clauses.is_empty() {
        b.insert("should".into(), Value::Array(should_clauses));
        b.insert("minimum_should_match".into(), json!(1));
    }
    if b.is_empty() {
        None
    } else {
        Some(b)
    }
}

fn query_string(mt: &MatchText) -> Value {
    let msm = match mt.minimum_should_match {
        Some(f) => format_msm(f),
        None => "0%".to_string(),
    };
    json!({"query_string": {
        "fields": QUERY_FIELDS,
        "type": "best_fields",
        "query": lower_case_query_text(&mt.matching_text),
        "minimum_should_match": msm,
        "boost": 1.0,
    }})
}

fn rank_feature_queries(rf: &BTreeMap<String, f64>) -> Vec<Value> {
    rf.iter()
        .filter(|(k, _)| k.as_str() != PAGERANK_FLD)
        .map(|(k, v)| json!({"rank_feature": {"field": format!("{TAG_FLD}.{k}"), "linear": {}, "boost": v}}))
        .collect()
}

/// The body `Engine.Search` sends for `req` (identical for every index).
pub fn search_body(req: &SearchRequest) -> Value {
    let offset = req.offset.max(0);
    let limit = if req.limit <= 0 { 30 } else { req.limit };
    let mut bool_q = build_bool_query(&req.filter, &req.kb_ids);
    let vector_weight = req.fusion_vector_weight.unwrap_or(0.5);
    let has_vector = req
        .match_dense
        .as_ref()
        .is_some_and(|d| !d.embedding.is_empty());
    // `query` holds either a bool map or a bare query_string object.
    let mut top_query: Option<Value> = None;
    let mut bool_is_query = true;
    if let Some(mt) = &req.match_text {
        let tq = query_string(mt);
        match bool_q.as_mut() {
            Some(b) => {
                b.insert("must".into(), Value::Array(vec![tq]));
                if has_vector {
                    b.insert("boost".into(), json!(1.0 - vector_weight));
                }
            }
            None => {
                bool_is_query = false;
                top_query = Some(tq);
            }
        }
    }
    let wrap = |b: &Map<String, Value>| json!({"bool": b});
    let mut body = Map::new();
    let mut query_present = false;
    if has_vector {
        let d = req.match_dense.as_ref().unwrap();
        let mut k = d.top_n;
        if k <= 0 {
            k = limit;
        }
        if k <= 0 {
            k = 1024;
        }
        k = k.min(10000);
        let num_candidates = k.max(d.num_candidates.min(10000));
        let filter = if bool_is_query {
            bool_q.as_ref().map(wrap).unwrap_or(Value::Null)
        } else {
            top_query.clone().unwrap_or(Value::Null)
        };
        // rank features are appended to the shared map below, after this
        // snapshot; patch them into knn.filter there too.
        body.insert(
            "knn".into(),
            json!({
                "field": d.vector_column,
                "query_vector": d.embedding,
                "k": k,
                "num_candidates": num_candidates,
                "similarity": d.similarity,
                "filter": filter,
            }),
        );
        if req.match_text.is_some() && (bool_q.is_some() || top_query.is_some()) {
            query_present = true;
        }
    } else {
        query_present = true;
        if bool_q.is_none() && top_query.is_none() {
            top_query = Some(json!({"match_all": {}}));
            bool_is_query = false;
        }
    }
    // Rank features join the bool map that `query` points at.
    if let Some(rf) = &req.rank_feature {
        if !rf.is_empty() && query_present && bool_is_query {
            let qs = rank_feature_queries(rf);
            if !qs.is_empty() {
                if let Some(b) = bool_q.as_mut() {
                    // Go appends inside a loop that keeps re-reading the old
                    // slice: only the last clause survives when `should`
                    // already existed.
                    match b.get("should").and_then(Value::as_array).cloned() {
                        Some(mut existing) => {
                            existing.push(qs.last().unwrap().clone());
                            b.insert("should".into(), Value::Array(existing));
                        }
                        None => {
                            b.insert("should".into(), Value::Array(qs));
                        }
                    }
                }
                if let Some(knn) = body.get_mut("knn").and_then(Value::as_object_mut) {
                    knn.insert("filter".into(), wrap(bool_q.as_ref().unwrap()));
                }
            }
        }
    }
    if query_present {
        let q = if bool_is_query {
            bool_q.as_ref().map(wrap).unwrap_or(Value::Null)
        } else {
            top_query.clone().unwrap_or(Value::Null)
        };
        body.insert("query".into(), q);
    }
    body.insert("size".into(), json!(limit));
    body.insert("from".into(), json!(offset));
    let has_text = req.match_text.is_some();
    if !req.select_fields.is_empty() {
        let mut src = req.select_fields.clone();
        if has_text || has_vector {
            if !src.iter().any(|f| f == PAGERANK_FLD) {
                src.push(PAGERANK_FLD.into());
            }
            if !src.iter().any(|f| f == TAG_FLD) {
                src.push(TAG_FLD.into());
            }
        }
        let vec_fields: Vec<&String> = req
            .select_fields
            .iter()
            .filter(|f| f.ends_with("_vec"))
            .collect();
        body.insert("_source".into(), json!(src));
        if !vec_fields.is_empty() {
            body.insert("fields".into(), json!(vec_fields));
        }
    } else if has_text || has_vector {
        body.insert("_source".into(), json!([PAGERANK_FLD, TAG_FLD]));
    }
    Value::Object(body)
}

fn to_f64(v: Option<&Value>) -> Option<f64> {
    v.and_then(Value::as_f64)
}

/// `convertESResponse`: `_source` plus `_score`, `_id`, `id`, `_index`.
pub fn convert_response(resp: &Value) -> (Vec<Chunk>, i64) {
    let total = resp
        .pointer("/hits/total/value")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let mut chunks = Vec::new();
    for hit in resp
        .pointer("/hits/hits")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let mut c = hit
            .get("_source")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let id = hit
            .get("_id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        c.insert(
            "_score".into(),
            json!(to_f64(hit.get("_score")).unwrap_or(0.0)),
        );
        c.insert("_id".into(), json!(id));
        c.insert("id".into(), json!(id));
        c.insert(
            "_index".into(),
            json!(hit.get("_index").and_then(Value::as_str).unwrap_or("")),
        );
        if let Some(h) = hit.get("highlight").and_then(Value::as_object) {
            if !h.is_empty() {
                c.insert("highlight".into(), Value::Object(h.clone()));
            }
        }
        chunks.push(c);
    }
    (chunks, total)
}

/// `Engine.Search`: one request per index; a failing index is skipped (Go
/// logs and continues), then scores get `pagerank_fea` added and a stable
/// sort by `_score` desc cuts the window to `limit`.
pub fn engine_search(
    engine: &dyn SearchEngine,
    req: &SearchRequest,
) -> Result<SearchResult, String> {
    if req.index_names.is_empty() {
        return Err("index names cannot be empty".into());
    }
    let limit = if req.limit <= 0 { 30 } else { req.limit };
    let body = search_body(req);
    let mut all = Vec::new();
    let mut total = 0i64;
    for index in &req.index_names {
        match engine.search(index, &body, true) {
            Ok(resp) => {
                if let Some(err) = resp.get("error") {
                    eprintln!("ragflow-retrieval: Elasticsearch error response on {index}: {err}");
                    continue;
                }
                let (chunks, t) = convert_response(&resp);
                total += t;
                all.extend(chunks);
            }
            Err(e) => {
                eprintln!("ragflow-retrieval: Elasticsearch query failed on {index}: {e}");
            }
        }
    }
    let has_vector = req
        .match_dense
        .as_ref()
        .is_some_and(|d| !d.embedding.is_empty());
    if !all.is_empty() && (req.match_text.is_some() || has_vector) {
        for c in &mut all {
            let mut score = 0.0;
            if let Some(f) = to_f64(c.get("_score")) {
                score += f;
            }
            if let Some(f) = c.get(PAGERANK_FLD).and_then(Value::as_f64) {
                score += f;
            }
            c.insert("_score".into(), json!(score));
        }
        crate::gocompat::sort_slice_stable(&mut all, |a, b| {
            to_f64(a.get("_score")).unwrap_or(0.0) > to_f64(b.get("_score")).unwrap_or(0.0)
        });
        if all.len() as i64 > limit && limit > 0 {
            all.truncate(limit as usize);
        }
    }
    Ok(SearchResult { chunks: all, total })
}

/// `elasticsearchChunkID`.
pub fn chunk_id(c: &Chunk) -> Option<String> {
    for k in ["id", "_id"] {
        if let Some(s) = c.get(k).and_then(Value::as_str) {
            if !s.is_empty() {
                return Some(s.to_string());
            }
        }
    }
    None
}

/// `GetChunkIDs`.
pub fn chunk_ids(chunks: &[Chunk]) -> Vec<String> {
    chunks.iter().filter_map(chunk_id).collect()
}

/// Go `fmt.Sprintf("%v", v)` for a JSON-decoded value.
pub fn go_sprint_v(v: &Value) -> String {
    match v {
        Value::Null => "<nil>".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => fmt_v(n.as_f64().unwrap_or(0.0)),
        Value::String(s) => s.clone(),
        Value::Array(a) => format!(
            "[{}]",
            a.iter().map(go_sprint_v).collect::<Vec<_>>().join(" ")
        ),
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            let parts: Vec<String> = keys
                .iter()
                .map(|k| format!("{}:{}", k, go_sprint_v(&m[*k])))
                .collect();
            format!("map[{}]", parts.join(" "))
        }
    }
}

/// `GetFields(chunks, fields)`: id → projected field map. Also writes `id`
/// into chunks that lack it, as Go does.
pub fn get_fields(chunks: &mut [Chunk], fields: &[&str]) -> BTreeMap<String, Chunk> {
    let mut result = BTreeMap::new();
    if fields.is_empty() || chunks.is_empty() {
        return result;
    }
    for c in chunks.iter_mut() {
        let Some(id) = chunk_id(c) else { continue };
        if c.get("id")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
        {
            c.insert("id".into(), json!(id));
        }
        let mut m = Map::new();
        for field in fields {
            let Some(mut val) = c.get(*field).cloned() else {
                continue;
            };
            if val.is_null() {
                continue;
            }
            if let Value::Array(a) = &val {
                if a.len() == 1 && !a[0].is_array() {
                    val = a[0].clone();
                }
            }
            if val.is_array() {
                m.insert(field.to_string(), val);
                continue;
            }
            if *field == "available_int" && val.is_number() {
                m.insert(field.to_string(), val);
                continue;
            }
            if !val.is_string() {
                val = Value::String(go_sprint_v(&val));
            }
            m.insert(field.to_string(), val);
        }
        if !m.is_empty() {
            result.insert(id, m);
        }
    }
    result
}

/// `KNNScores` body and target index; `None` when there is nothing to ask.
pub fn knn_scores_request(chunks: &[Chunk], query_vector: &[f64]) -> Option<(String, Value)> {
    if chunks.is_empty() || query_vector.is_empty() {
        return None;
    }
    let ids: Vec<String> = chunks
        .iter()
        .filter_map(|c| c.get("_id").and_then(Value::as_str).map(str::to_string))
        .collect();
    if ids.is_empty() {
        return None;
    }
    let k = ids.len();
    let index = chunks[0]
        .get("_index")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let body = json!({
        "knn": {
            "field": format!("q_{}_vec", query_vector.len()),
            "query_vector": query_vector,
            "k": k,
            "num_candidates": k * 2,
            "similarity": 0.0,
            "filter": {"terms": {"id": ids}},
        },
        "size": k,
        "_source": false,
    });
    Some((index, body))
}

/// `GetScores`: `_id` → `_score`.
pub fn get_scores(resp: &Value) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    for hit in resp
        .pointer("/hits/hits")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if let Some(id) = hit.get("_id").and_then(Value::as_str) {
            if !id.is_empty() {
                out.insert(
                    id.to_string(),
                    hit.get("_score").and_then(Value::as_f64).unwrap_or(0.0),
                );
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowercases_terms_but_not_operators() {
        assert_eq!(
            lower_case_query_text("Foo OR \"Bar OR\" AND x\\Y"),
            "foo OR \"bar or\" AND x\\Y"
        );
    }

    #[test]
    fn msm_percent() {
        assert_eq!(format_msm(0.3), "30%");
        assert_eq!(format_msm(0.1), "10%");
        assert_eq!(format_msm(0.0), "0%");
    }
}
