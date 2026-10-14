//! Retrieval: searches datasets through an injected [`RetrievalService`].
//!
//! Mirrors upstream @ v1.0.0-rc1: the component wrapper
//! (`retrievalComponent`, component/universe_a_wrappers.go: node params as
//! defaults, `kb_ids` → `dataset_ids`, legacy `UserFillUp:` forms, the "No
//! dataset is selected." guard) and the tool it delegates to
//! (`RetrievalTool.InvokableRun`, tool/retrieval.go: argument merge, template
//! resolution, `formalized_content`, the chunk envelope, citation references).
//! Memory retrieval and GraphRAG are not supported.

/// How a refused dataset dependency check begins (`universe_a_wrappers.go`).
pub const DEPENDENCY_ERROR: &str = "agent Retrieval dependency permission:";

use std::sync::{Arc, OnceLock};

use regex::Regex;
use serde_json::{json, Map, Value};

use crate::graph::Node;
use crate::retrieval::{RetrievalChunk, RetrievalRequest, RetrievalService};
use crate::run::{Component, NodeCtx};
use crate::state::{Outputs, RunState};
use crate::template::render_strict;

/// `retrievalParams`.
#[derive(Debug, Clone, Default)]
struct Params {
    query: String,
    kb_ids: Vec<String>,
    memory_ids: Vec<String>,
    user_id: String,
    top_n: i64,
    top_k: i64,
    similarity_threshold: Option<f64>,
    keywords_similarity_weight: Option<f64>,
    rerank_id: String,
    empty_response: String,
    cross_languages: Vec<String>,
    toc_enhance: bool,
    use_kg: bool,
    meta_data_filter: Option<Map<String, Value>>,
    retrieval_from: String,
}

fn to_int_param(v: &Value) -> i64 {
    match v {
        Value::Number(n) => n
            .as_i64()
            .unwrap_or_else(|| n.as_f64().unwrap_or(0.0) as i64),
        Value::String(s) => s.parse().unwrap_or(0),
        _ => 0,
    }
}

fn to_float_param(v: &Value) -> f64 {
    match v {
        Value::Number(n) => n.as_f64().unwrap_or(0.0),
        Value::String(s) => s.parse().unwrap_or(0.0),
        _ => 0.0,
    }
}

/// `toStringSlice`.
fn to_string_slice(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::String(s)) => s
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect(),
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(|x| x.as_str().map(str::to_string))
            .collect(),
        _ => Vec::new(),
    }
}

fn parse_params(params: &Value) -> Params {
    let mut out = Params::default();
    let Some(p) = params.as_object() else {
        return out;
    };
    let mut p = p.clone();
    if let Some(ids) = p.get("dataset_ids").cloned() {
        p.insert("kb_ids".into(), ids);
    }
    if let Some(Value::String(s)) = p.get("query") {
        out.query = s.clone();
    }
    if let Some(Value::Array(a)) = p.get("kb_ids") {
        out.kb_ids = a
            .iter()
            .filter_map(|x| x.as_str().map(str::to_string))
            .collect();
    }
    out.memory_ids = to_string_slice(p.get("memory_ids"));
    if let Some(Value::String(s)) = p.get("user_id") {
        out.user_id = s.clone();
    }
    if let Some(v) = p.get("top_n") {
        out.top_n = to_int_param(v);
    }
    if let Some(v) = p.get("top_k") {
        out.top_k = to_int_param(v);
    }
    if let Some(v) = p.get("similarity_threshold") {
        out.similarity_threshold = Some(to_float_param(v));
    }
    if let Some(v) = p.get("keywords_similarity_weight") {
        out.keywords_similarity_weight = Some(to_float_param(v));
    }
    if let Some(Value::String(s)) = p.get("rerank_id") {
        out.rerank_id = s.clone();
    }
    if let Some(Value::String(s)) = p.get("empty_response") {
        out.empty_response = s.clone();
    }
    out.cross_languages = to_string_slice(p.get("cross_languages"));
    if let Some(Value::Bool(b)) = p.get("toc_enhance") {
        out.toc_enhance = *b;
    }
    if let Some(Value::Bool(b)) = p.get("use_kg") {
        out.use_kg = *b;
    }
    if let Some(Value::Object(m)) = p.get("meta_data_filter") {
        out.meta_data_filter = Some(m.clone());
    }
    if let Some(Value::String(s)) = p.get("retrieval_from") {
        out.retrieval_from = s.clone();
    }
    out
}

pub struct Retrieval {
    params: Params,
    service: Arc<dyn RetrievalService>,
}

pub fn factory(
    node: &Node,
    service: Arc<dyn RetrievalService>,
) -> Result<Box<dyn Component>, String> {
    Ok(Box::new(Retrieval {
        params: parse_params(&node.params),
        service,
    }))
}

fn legacy_query_pattern() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?s)^\s*UserFillUp:\s*(.*?)\s+Input\s+(.*?)\s*$").unwrap())
}

impl Retrieval {
    /// `applyDefaults`.
    fn apply_defaults(&self, inputs: &Outputs) -> Map<String, Value> {
        let p = &self.params;
        let mut out = inputs.clone();
        let put = |k: &str, v: Value, cond: bool, out: &mut Map<String, Value>| {
            if cond && !out.contains_key(k) {
                out.insert(k.into(), v);
            }
        };
        put("query", json!(p.query), !p.query.is_empty(), &mut out);
        put("kb_ids", json!(p.kb_ids), !p.kb_ids.is_empty(), &mut out);
        put("top_n", json!(p.top_n), p.top_n > 0, &mut out);
        put("top_k", json!(p.top_k), p.top_k > 0, &mut out);
        if let Some(t) = p.similarity_threshold {
            put("similarity_threshold", json!(t), true, &mut out);
        }
        if let Some(w) = p.keywords_similarity_weight {
            put("keywords_similarity_weight", json!(w), true, &mut out);
        }
        put(
            "rerank_id",
            json!(p.rerank_id),
            !p.rerank_id.is_empty(),
            &mut out,
        );
        put(
            "empty_response",
            json!(p.empty_response),
            !p.empty_response.is_empty(),
            &mut out,
        );
        put(
            "memory_ids",
            json!(p.memory_ids),
            !p.memory_ids.is_empty(),
            &mut out,
        );
        put("user_id", json!(p.user_id), !p.user_id.is_empty(), &mut out);
        put(
            "cross_languages",
            json!(p.cross_languages),
            !p.cross_languages.is_empty(),
            &mut out,
        );
        put("toc_enhance", json!(true), p.toc_enhance, &mut out);
        put("use_kg", json!(true), p.use_kg, &mut out);
        if let Some(m) = &p.meta_data_filter {
            put("meta_data_filter", Value::Object(m.clone()), true, &mut out);
        }
        put(
            "retrieval_from",
            json!(p.retrieval_from),
            !p.retrieval_from.is_empty(),
            &mut out,
        );
        if let Some(kb) = out.remove("kb_ids") {
            if !out.contains_key("dataset_ids") {
                out.insert("dataset_ids".into(), kb);
            }
        }
        out
    }

    fn resolve_dataset(&self, name: &str, state: &RunState) -> Option<String> {
        if name.is_empty() {
            return None;
        }
        let tenant = state
            .sys
            .get("tenant_id")
            .and_then(Value::as_str)
            .unwrap_or("");
        let user = state
            .sys
            .get("user_id")
            .and_then(Value::as_str)
            .unwrap_or("");
        self.service
            .resolve_dataset_id(name, tenant, user)
            .filter(|s| !s.is_empty())
    }

    /// `normalizeLegacyRetrievalInputs` (+ the structured form).
    fn normalize_legacy(&self, out: &mut Map<String, Value>, state: &RunState) {
        if self.normalize_structured(out, state) {
            return;
        }
        let raw = out
            .get("query")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        if raw.is_empty() {
            return;
        }
        let Some(m) = legacy_query_pattern().captures(&raw) else {
            return;
        };
        let kb_name = m[1].trim().to_string();
        let query = m[2].trim().to_string();
        if !query.is_empty() {
            out.insert("query".into(), json!(query));
        }
        if out.contains_key("dataset_ids") || kb_name.is_empty() {
            return;
        }
        if let Some(id) = self.resolve_dataset(&kb_name, state) {
            out.insert("dataset_ids".into(), json!([id]));
        }
    }

    fn normalize_structured(&self, out: &mut Map<String, Value>, state: &RunState) -> bool {
        let has_ids = out.contains_key("dataset_ids");
        let mut candidates: Vec<Map<String, Value>> = Vec::new();
        if let Some(Value::Object(st)) = out.get("state") {
            if let Some(Value::Object(raw)) = st.get("UserFillUp:KBInput") {
                candidates.push(raw.clone());
            }
        }
        candidates.push(out.clone());
        let mut consumed = false;
        for c in candidates {
            let kb = c
                .get("kb")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let q = c
                .get("query")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            if kb.is_empty() && legacy_query_pattern().is_match(q.trim()) {
                continue;
            }
            if kb.is_empty() && q.is_empty() {
                continue;
            }
            consumed = true;
            if !q.is_empty() {
                out.insert("query".into(), json!(q));
            }
            if !kb.is_empty() && !has_ids {
                if let Some(id) = self.resolve_dataset(kb.trim(), state) {
                    out.insert("dataset_ids".into(), json!([id]));
                }
            }
            if !q.is_empty() {
                return true;
            }
            if !kb.is_empty() && out.get("dataset_ids").is_some_and(|v| !v.is_null()) {
                return true;
            }
        }
        consumed
    }
}

impl Component for Retrieval {
    fn invoke(&self, ctx: &mut NodeCtx<'_>) -> Result<Outputs, String> {
        let mut merged = self.apply_defaults(ctx.inputs);
        self.normalize_legacy(&mut merged, ctx.state);
        let mut query = merged
            .get("query")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if let Ok(resolved) = render_strict(&query, ctx.state) {
            query = resolved;
        }
        if !query.is_empty()
            && merged.get("retrieval_from").and_then(Value::as_str) == Some("dataset")
        {
            let empty = match merged.get("dataset_ids") {
                None | Some(Value::Null) => true,
                Some(Value::Array(a)) => a.is_empty(),
                _ => false,
            };
            if empty {
                let mut out = Map::new();
                out.insert("_ERROR".into(), json!("No dataset is selected."));
                return Ok(out);
            }
        }
        let envelope = invoke_tool(self.service.as_ref(), &Value::Object(merged), ctx.state)
            .map_err(|e| {
                // Upstream checks the datasets first and reports that
                // failure as is; a failed search is `agent: Retrieval: ...`.
                if e.starts_with(DEPENDENCY_ERROR) {
                    e
                } else {
                    format!("agent: Retrieval: {e}")
                }
            })?;
        let decoded = normalize_outputs(envelope);
        Ok(decoded)
    }
}

/// `normalizeRetrievalOutputs`: an empty result still has the arrays a
/// downstream reference iterates over.
fn normalize_outputs(mut decoded: Map<String, Value>) -> Map<String, Value> {
    let chunks = match decoded.get("chunks") {
        None | Some(Value::Null) => json!([]),
        Some(c) => c.clone(),
    };
    decoded.insert("chunks".into(), chunks.clone());
    if matches!(decoded.get("json"), None | Some(Value::Null)) {
        decoded.insert("json".into(), chunks);
    }
    decoded
}

/// `retrievalArgs` after JSON decoding (Go `json.Unmarshal` type rules).
#[derive(Debug, Default)]
struct Args {
    query: String,
    dataset_ids: Vec<String>,
    kb_ids: Vec<String>,
    memory_ids: Vec<String>,
    user_id: String,
    top_n: i64,
    rerank_candidates_count: i64,
    top_k: i64,
    keywords_similarity_weight: Option<f64>,
    use_kg: bool,
    similarity_threshold: Option<f64>,
    rerank_id: String,
    cross_languages: Vec<String>,
    toc_enhance: bool,
    meta_data_filter: Option<Map<String, Value>>,
    retrieval_from: String,
    empty_response: String,
}

fn parse_args(v: &Value) -> Result<Args, String> {
    let err =
        |k: &str| format!("retrieval: parse arguments: json: cannot unmarshal into field {k}");
    let mut a = Args::default();
    let Some(m) = v.as_object() else {
        return Err("retrieval: parse arguments: not an object".into());
    };
    let string = |k: &str| -> Result<String, String> {
        match m.get(k) {
            None | Some(Value::Null) => Ok(String::new()),
            Some(Value::String(s)) => Ok(s.clone()),
            _ => Err(err(k)),
        }
    };
    let strings = |k: &str| -> Result<Vec<String>, String> {
        match m.get(k) {
            None | Some(Value::Null) => Ok(Vec::new()),
            Some(Value::Array(xs)) => xs
                .iter()
                .map(|x| x.as_str().map(str::to_string).ok_or_else(|| err(k)))
                .collect(),
            _ => Err(err(k)),
        }
    };
    let int = |k: &str| -> Result<i64, String> {
        match m.get(k) {
            None | Some(Value::Null) => Ok(0),
            Some(Value::Number(n)) => n.as_i64().ok_or_else(|| err(k)),
            _ => Err(err(k)),
        }
    };
    let float = |k: &str| -> Result<Option<f64>, String> {
        match m.get(k) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Number(n)) => Ok(n.as_f64()),
            _ => Err(err(k)),
        }
    };
    let boolean = |k: &str| -> Result<bool, String> {
        match m.get(k) {
            None | Some(Value::Null) => Ok(false),
            Some(Value::Bool(b)) => Ok(*b),
            _ => Err(err(k)),
        }
    };
    a.query = string("query")?;
    a.dataset_ids = strings("dataset_ids")?;
    a.kb_ids = strings("kb_ids")?;
    a.memory_ids = strings("memory_ids")?;
    a.user_id = string("user_id")?;
    a.top_n = int("top_n")?;
    a.rerank_candidates_count = int("rerank_candidates_count")?;
    a.top_k = int("top_k")?;
    a.keywords_similarity_weight = float("keywords_similarity_weight")?;
    a.use_kg = boolean("use_kg")?;
    a.similarity_threshold = float("similarity_threshold")?;
    a.rerank_id = string("rerank_id")?;
    a.cross_languages = strings("cross_languages")?;
    a.toc_enhance = boolean("toc_enhance")?;
    a.meta_data_filter = match m.get("meta_data_filter") {
        None | Some(Value::Null) => None,
        Some(Value::Object(o)) => Some(o.clone()),
        _ => return Err(err("meta_data_filter")),
    };
    a.retrieval_from = string("retrieval_from")?;
    a.empty_response = string("empty_response")?;
    Ok(a)
}

fn compact(v: &[String]) -> Vec<String> {
    v.iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

fn resolve_value(v: &Value, state: &RunState) -> Result<Value, String> {
    Ok(match v {
        Value::String(s) => Value::String(
            render_strict(s, state)
                .map_err(|e| format!("retrieval: resolve metadata filter value: {e}"))?,
        ),
        Value::Object(m) => {
            let mut out = Map::new();
            for (k, x) in m {
                out.insert(k.clone(), resolve_value(x, state)?);
            }
            Value::Object(out)
        }
        Value::Array(a) => Value::Array(
            a.iter()
                .map(|x| resolve_value(x, state))
                .collect::<Result<_, _>>()?,
        ),
        other => other.clone(),
    })
}

/// `RetrievalTool.InvokableRun` with an empty tool-level default set (the
/// component path). Returns the decoded envelope and records references.
pub fn invoke_tool(
    service: &dyn RetrievalService,
    args: &Value,
    state: &mut RunState,
) -> Result<Map<String, Value>, String> {
    let mut a = parse_args(args)?;
    // mergeDefaults with empty defaults.
    if a.dataset_ids.is_empty() && !a.kb_ids.is_empty() {
        a.dataset_ids = a.kb_ids.clone();
    }
    if a.retrieval_from.is_empty() && !a.dataset_ids.is_empty() {
        a.retrieval_from = "dataset".into();
    }
    if a.retrieval_from.is_empty() && !a.memory_ids.is_empty() {
        a.retrieval_from = "memory".into();
    }
    a.query = render_strict(&a.query, state)
        .map_err(|e| format!("retrieval: resolve query variables: {e}"))?;
    // resolveRetrievalUserID
    let uid = a.user_id.trim().to_string();
    a.user_id = if uid.is_empty() {
        String::new()
    } else if uid.contains('{') || uid.contains('}') {
        render_strict(&uid, state)
            .map_err(|e| format!("retrieval: resolve user_id variable: {e}"))?
            .trim()
            .to_string()
    } else if let Some(v) = state.lookup(&uid) {
        match v {
            Value::String(s) => s,
            other => crate::template::go_v(&other),
        }
    } else if uid.starts_with("sys.") || uid.starts_with("env.") {
        String::new()
    } else {
        uid
    };
    let empty_envelope = |text: &str| {
        let mut m = Map::new();
        m.insert("formalized_content".into(), json!(text));
        m
    };
    if a.query.is_empty() {
        return Ok(empty_envelope(&a.empty_response));
    }
    if a.use_kg {
        return Err("GraphRAG retrieval is not supported".into());
    }
    if a.retrieval_from.is_empty() {
        return Ok(empty_envelope(&a.empty_response));
    }
    if a.retrieval_from != "dataset" && a.retrieval_from != "memory" {
        return Err(format!(
            "retrieval: unsupported retrieval_from {:?}",
            a.retrieval_from
        ));
    }
    if a.retrieval_from == "dataset" && a.dataset_ids.is_empty() {
        return Err("retrieval: dataset_ids is required".into());
    }
    if a.retrieval_from == "memory" {
        if a.memory_ids.is_empty() {
            return Err("retrieval: memory_ids is required".into());
        }
        return Err("retrieval: memory retrieval is not supported".into());
    }
    // resolveRetrievalDatasetIDs
    let mut ids = Vec::new();
    for id in &a.dataset_ids {
        if !id.contains('@') {
            ids.push(id.clone());
            continue;
        }
        match state.lookup(id) {
            None => return Err(format!("retrieval: dataset variable {id:?} is empty")),
            Some(Value::String(s)) => ids.push(s),
            Some(Value::Array(xs)) => {
                for x in xs {
                    ids.push(
                        x.as_str()
                            .ok_or_else(|| {
                                format!(
                                    "retrieval: dataset variable {id:?} contains non-string value"
                                )
                            })?
                            .to_string(),
                    );
                }
            }
            Some(_) => {
                return Err(format!(
                    "retrieval: dataset variable {id:?} must be a string or string list"
                ))
            }
        }
    }
    let dataset_ids = compact(&ids);
    let meta = match &a.meta_data_filter {
        None => None,
        Some(m) => match resolve_value(&Value::Object(m.clone()), state)? {
            Value::Object(o) => Some(o),
            _ => return Err("retrieval: metadata filter must be an object".into()),
        },
    };
    let tenant_id = match state.sys.get("tenant_id").and_then(Value::as_str) {
        Some(t) if !t.is_empty() => t.to_string(),
        _ => state
            .sys
            .get("user_id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
    };
    let req = RetrievalRequest {
        query: a.query.clone(),
        dataset_ids,
        memory_ids: a.memory_ids.clone(),
        top_n: a.top_n,
        rerank_candidates_count: a.rerank_candidates_count,
        top_k: a.top_k,
        keywords_similarity_weight: a.keywords_similarity_weight,
        use_kg: a.use_kg,
        similarity_threshold: a.similarity_threshold,
        rerank_id: a.rerank_id.clone(),
        cross_languages: a.cross_languages.clone(),
        toc_enhance: a.toc_enhance,
        meta_data_filter: meta,
        retrieval_from: a.retrieval_from.clone(),
        user_id: a.user_id.clone(),
        tenant_id,
        ..Default::default()
    };
    let chunks = service.search(&req)?;
    let mut out = Map::new();
    let formalized = if chunks.is_empty() {
        a.empty_response.clone()
    } else {
        render_chunks(&chunks)
    };
    out.insert("formalized_content".into(), json!(formalized));
    if !chunks.is_empty() {
        out.insert(
            "chunks".into(),
            Value::Array(chunks.iter().map(chunk_payload).collect()),
        );
        record_references(state, &chunks);
    }
    Ok(out)
}

/// `renderChunks`: `[ID:<id>] <content>\n` per chunk.
pub fn render_chunks(chunks: &[RetrievalChunk]) -> String {
    let mut s = String::new();
    for c in chunks {
        s.push_str(&format!("[ID:{}] {}\n", c.id, c.content));
    }
    s
}

/// `chunkPayload` with Go's `omitempty`.
fn chunk_payload(c: &RetrievalChunk) -> Value {
    let mut m = Map::new();
    if !c.id.is_empty() {
        m.insert("id".into(), json!(c.id));
    }
    if !c.content.is_empty() {
        m.insert("content".into(), json!(c.content));
    }
    if !c.document_id.is_empty() {
        m.insert("document_id".into(), json!(c.document_id));
    }
    if c.score != 0.0 {
        m.insert("score".into(), json!(c.score));
    }
    Value::Object(m)
}

/// `referenceChunksFromRetrieval` / `referenceDocAggsFromRetrieval` merged
/// into the run's references (`SetRetrievalReferences`).
pub fn record_references(state: &mut RunState, chunks: &[RetrievalChunk]) {
    let refs = reference_chunks(chunks);
    let aggs = reference_doc_aggs(chunks);
    let mut existing: Vec<Value> = state
        .retrieval
        .get("chunks")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut seen: std::collections::HashSet<String> = existing
        .iter()
        .filter_map(|c| c.get("id").map(crate::template::go_v))
        .filter(|s| !s.is_empty())
        .collect();
    for c in refs {
        if let Some(id) = c
            .get("id")
            .map(crate::template::go_v)
            .filter(|s| !s.is_empty())
        {
            if !seen.insert(id) {
                continue;
            }
        }
        existing.push(c);
    }
    let mut doc_aggs = state
        .retrieval
        .get("doc_aggs")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    for agg in aggs {
        let name = agg
            .get("doc_name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if name.is_empty() || doc_aggs.contains_key(&name) {
            continue;
        }
        doc_aggs.insert(name, agg);
    }
    state
        .retrieval
        .insert("chunks".into(), Value::Array(existing));
    state
        .retrieval
        .insert("doc_aggs".into(), Value::Object(doc_aggs));
}

pub fn reference_chunks(chunks: &[RetrievalChunk]) -> Vec<Value> {
    chunks
        .iter()
        .enumerate()
        .map(|(idx, c)| {
            let id = if c.id.is_empty() {
                idx.to_string()
            } else {
                c.id.clone()
            };
            let mut m = Map::new();
            m.insert("id".into(), json!(id));
            m.insert("chunk_id".into(), json!(c.id));
            m.insert("content".into(), json!(c.content));
            m.insert("content_with_weight".into(), json!(c.content));
            m.insert("document_id".into(), json!(c.document_id));
            m.insert("doc_id".into(), json!(c.document_id));
            m.insert("document_name".into(), json!(c.document_name));
            m.insert("docnm_kwd".into(), json!(c.document_name));
            m.insert("dataset_id".into(), json!(c.dataset_id));
            m.insert("kb_id".into(), json!(c.dataset_id));
            m.insert("image_id".into(), json!(c.image_id));
            m.insert("img_id".into(), json!(c.image_id));
            m.insert("similarity".into(), json!(c.score));
            m.insert("term_similarity".into(), json!(c.term_similarity));
            m.insert("vector_similarity".into(), json!(c.vector_similarity));
            if !c.url.is_empty() {
                m.insert("url".into(), json!(c.url));
                m.insert("document_url".into(), json!(c.url));
            }
            if let Some(p) = &c.positions {
                m.insert("positions".into(), p.clone());
                m.insert("position_int".into(), p.clone());
            }
            Value::Object(m)
        })
        .collect()
}

pub fn reference_doc_aggs(chunks: &[RetrievalChunk]) -> Vec<Value> {
    let mut order: Vec<String> = Vec::new();
    let mut by: std::collections::HashMap<String, Map<String, Value>> =
        std::collections::HashMap::new();
    for c in chunks {
        if c.document_id.is_empty() && c.document_name.is_empty() {
            continue;
        }
        let key = if c.document_id.is_empty() {
            c.document_name.clone()
        } else {
            c.document_id.clone()
        };
        let agg = by.entry(key.clone()).or_insert_with(|| {
            order.push(key.clone());
            let mut m = Map::new();
            m.insert("count".into(), json!(0));
            m.insert("doc_id".into(), json!(c.document_id));
            m.insert("doc_name".into(), json!(c.document_name));
            if !c.url.is_empty() {
                m.insert("url".into(), json!(c.url));
            }
            m
        });
        let n = agg.get("count").and_then(Value::as_i64).unwrap_or(0);
        agg.insert("count".into(), json!(n + 1));
    }
    order
        .into_iter()
        .map(|k| Value::Object(by.remove(&k).unwrap()))
        .collect()
}

/// `CanvasState.GetRetrievalReference`: `{chunks, doc_aggs (sorted by doc
/// name), total}`, or `None` when nothing was recorded.
pub fn retrieval_reference(state: &RunState) -> Option<Value> {
    let chunks = state
        .retrieval
        .get("chunks")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut aggs: Vec<(String, Value)> = state
        .retrieval
        .get("doc_aggs")
        .and_then(Value::as_object)
        .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default();
    aggs.sort_by(|a, b| a.0.cmp(&b.0));
    if chunks.is_empty() && aggs.is_empty() {
        return None;
    }
    let total = chunks.len();
    Some(json!({
        "chunks": chunks,
        "doc_aggs": aggs.into_iter().map(|(_, v)| v).collect::<Vec<_>>(),
        "total": total,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct Fake {
        seen: Mutex<Vec<RetrievalRequest>>,
        chunks: Vec<RetrievalChunk>,
    }

    impl RetrievalService for Fake {
        fn search(&self, req: &RetrievalRequest) -> Result<Vec<RetrievalChunk>, String> {
            self.seen.lock().unwrap().push(req.clone());
            Ok(self.chunks.clone())
        }
    }

    fn chunk(id: &str, score: f64) -> RetrievalChunk {
        RetrievalChunk {
            id: id.into(),
            content: format!("text {id}"),
            document_id: "d1".into(),
            document_name: "doc.md".into(),
            dataset_id: "kb1".into(),
            positions: Some(json!([])),
            score,
            ..Default::default()
        }
    }

    #[test]
    fn tool_renders_envelope_and_references() {
        let fake = Fake {
            seen: Mutex::new(Vec::new()),
            chunks: vec![chunk("a", 0.5), chunk("b", 0.0)],
        };
        let mut state = RunState::default();
        let out = invoke_tool(
            &fake,
            &json!({"query": "q", "kb_ids": ["kb1"], "top_n": 3}),
            &mut state,
        )
        .unwrap();
        assert_eq!(
            out["formalized_content"],
            json!("[ID:a] text a\n[ID:b] text b\n")
        );
        assert_eq!(
            out["chunks"][1],
            json!({"id": "b", "content": "text b", "document_id": "d1"})
        );
        let req = &fake.seen.lock().unwrap()[0];
        assert_eq!(req.dataset_ids, vec!["kb1".to_string()]);
        assert_eq!(req.retrieval_from, "dataset");
        assert_eq!(req.top_n, 3);
        let refs = retrieval_reference(&state).unwrap();
        assert_eq!(refs["total"], json!(2));
        assert_eq!(refs["doc_aggs"][0]["count"], json!(2));
    }

    #[test]
    fn an_empty_result_keeps_its_arrays() {
        let out = normalize_outputs(
            json!({"formalized_content": "x"})
                .as_object()
                .unwrap()
                .clone(),
        );
        assert_eq!(
            Value::Object(out),
            json!({"formalized_content": "x", "chunks": [], "json": []})
        );
        let kept = normalize_outputs(
            json!({"chunks": [{"a": 1}], "json": null})
                .as_object()
                .unwrap()
                .clone(),
        );
        assert_eq!(kept["json"], json!([{"a": 1}]));
    }

    #[test]
    fn empty_result_uses_empty_response() {
        let fake = Fake {
            seen: Mutex::new(Vec::new()),
            chunks: vec![],
        };
        let mut state = RunState::default();
        let out = invoke_tool(
            &fake,
            &json!({"query": "q", "dataset_ids": ["kb1"], "empty_response": "none"}),
            &mut state,
        )
        .unwrap();
        assert_eq!(
            out,
            json!({"formalized_content": "none"})
                .as_object()
                .unwrap()
                .clone()
        );
        assert!(retrieval_reference(&state).is_none());
    }

    #[test]
    fn node_params_fill_defaults_and_kb_ids_become_dataset_ids() {
        let r = Retrieval {
            params: parse_params(
                &json!({"kb_ids": ["kb1"], "top_n": 4, "keywords_similarity_weight": "0.7"}),
            ),
            service: Arc::new(Fake {
                seen: Mutex::new(Vec::new()),
                chunks: vec![],
            }),
        };
        let mut inputs = Map::new();
        inputs.insert("query".into(), json!("hello"));
        let merged = r.apply_defaults(&inputs);
        assert_eq!(merged["dataset_ids"], json!(["kb1"]));
        assert!(!merged.contains_key("kb_ids"));
        assert_eq!(merged["top_n"], json!(4));
        assert_eq!(merged["keywords_similarity_weight"], json!(0.7));
    }
}
