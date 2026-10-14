//! Client-side scoring (internal/service/nlp/reranker.go @ v1.0.0-rc1):
//! `RerankWithKNN`, `RerankStandard`, `RerankByModel`, `TokenSimilarity`,
//! rank features.
//!
//! In the canvas path the field maps carry neither `pagerank_fea` nor
//! `tag_feas` (they are not in `src`), so rank features add 0 there; the
//! arithmetic is still implemented as Go has it.

use std::collections::{BTreeMap, HashMap};

use serde_json::Value;

use crate::es::{Chunk, PAGERANK_FLD, TAG_FLD};
use crate::gocompat::{fields, pairwise_sum, py_sqrt};
use crate::query_builder::QueryBuilder;

/// `extractContentTokens`: whitespace split, deduped in order.
fn content_tokens(c: &Chunk, cfield: &str) -> Vec<String> {
    let Some(v) = c.get(cfield).and_then(Value::as_str) else {
        return Vec::new();
    };
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for t in fields(v) {
        if seen.insert(t) {
            out.push(t.to_string());
        }
    }
    out
}

fn split_tokens(c: &Chunk, key: &str) -> Vec<String> {
    match c.get(key).and_then(Value::as_str) {
        Some(v) => fields(v).into_iter().map(str::to_string).collect(),
        None => Vec::new(),
    }
}

/// `extractImportantKeywords`.
fn important_keywords(c: &Chunk) -> Vec<String> {
    match c.get("important_kwd") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(|x| x.as_str().map(str::to_string))
            .collect(),
        _ => Vec::new(),
    }
}

/// content + title×2 + important_kwd×5 + question_tks×6.
fn weighted_tokens(c: &Chunk, cfield: &str) -> Vec<String> {
    let content = content_tokens(c, cfield);
    let title = split_tokens(c, "title_tks");
    let question = split_tokens(c, "question_tks");
    let important = important_keywords(c);
    let mut tks = content;
    for _ in 0..2 {
        tks.extend(title.iter().cloned());
    }
    for _ in 0..5 {
        tks.extend(important.iter().cloned());
    }
    for _ in 0..6 {
        tks.extend(question.iter().cloned());
    }
    tks
}

/// `tokensToDict`: weights with insertion order of keys.
fn tokens_to_dict(tks: &[String], qb: &QueryBuilder) -> (HashMap<String, f64>, Vec<String>) {
    let mut d: HashMap<String, f64> = HashMap::new();
    let mut order = Vec::new();
    let wts = qb.term_weight.weights(tks, false);
    for (i, tw) in wts.iter().enumerate() {
        let t = &tw.term;
        let c = tw.weight;
        if !d.contains_key(t) {
            order.push(t.clone());
        }
        *d.entry(t.clone()).or_insert(0.0) += c * 0.4;
        if i + 1 < wts.len() {
            let nt = &wts[i + 1].term;
            let nc = wts[i + 1].weight;
            let k = format!("{t}{nt}");
            if !d.contains_key(&k) {
                order.push(k.clone());
            }
            *d.entry(k).or_insert(0.0) += c.max(nc) * 0.6;
        }
    }
    (d, order)
}

/// `TokenSimilarity`.
pub fn token_similarity(atks: &[String], btkss: &[Vec<String>], qb: &QueryBuilder) -> Vec<f64> {
    let (q, q_order) = tokens_to_dict(atks, qb);
    btkss
        .iter()
        .map(|b| {
            let (d, _) = tokens_to_dict(b, qb);
            if q.is_empty() || d.is_empty() {
                return 0.0;
            }
            let keys: Vec<String> = if q_order.is_empty() {
                let mut k: Vec<String> = q.keys().cloned().collect();
                k.sort();
                k
            } else {
                q_order.clone()
            };
            let mut s = 1e-9;
            for t in &keys {
                if d.contains_key(t) {
                    s += q[t];
                }
            }
            let mut total = 1e-9;
            for t in &keys {
                total += q[t];
            }
            s / total
        })
        .collect()
}

fn as_float(v: &Value) -> Option<f64> {
    v.as_f64()
}

fn tag_feas_map(v: Option<&Value>) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    match v {
        Some(Value::String(s)) => {
            if s.is_empty() || s == "{}" {
                return out;
            }
            if let Ok(Value::Object(m)) = serde_json::from_str::<Value>(s) {
                for (k, v) in m {
                    if let Some(f) = v.as_f64() {
                        out.insert(k, f);
                    }
                }
            }
        }
        Some(Value::Object(m)) => {
            for (k, v) in m {
                if let Some(f) = v.as_f64() {
                    out.insert(k.clone(), f);
                }
            }
        }
        _ => {}
    }
    out
}

/// Shared body of `applyRankFeatureScores` / `applyRankFeatureScoresForIDs`.
fn apply_rank_features(
    sources: &[Option<&Chunk>],
    sim: &mut [f64],
    rank_feature: &BTreeMap<String, f64>,
) {
    if sources.is_empty() || sim.is_empty() {
        return;
    }
    let pageranks: Vec<f64> = sources
        .iter()
        .map(|c| {
            c.and_then(|c| c.get(PAGERANK_FLD))
                .and_then(as_float)
                .unwrap_or(0.0)
        })
        .collect();
    if rank_feature.is_empty() {
        for (s, p) in sim.iter_mut().zip(&pageranks) {
            *s += p;
        }
        return;
    }
    let qbuf: Vec<f64> = rank_feature
        .iter()
        .filter(|(k, _)| k.as_str() != PAGERANK_FLD)
        .map(|(_, s)| s * s)
        .collect();
    let q_denor = py_sqrt(pairwise_sum(&qbuf));
    if q_denor == 0.0 {
        for (s, p) in sim.iter_mut().zip(&pageranks) {
            *s += p;
        }
        return;
    }
    for (i, c) in sources.iter().enumerate() {
        let tags = tag_feas_map(c.and_then(|c| c.get(TAG_FLD)));
        let mut tag_score = 0.0;
        if !tags.is_empty() {
            let mut nor = 0.0;
            let mut denor = 0.0;
            for (t, sc) in &tags {
                if let Some(w) = rank_feature.get(t) {
                    nor += w * sc;
                }
                denor += sc * sc;
            }
            if denor != 0.0 {
                tag_score = nor / py_sqrt(denor) / q_denor;
            }
        }
        sim[i] += tag_score * 10.0 + pageranks[i];
    }
}

pub struct Scores {
    pub sim: Vec<f64>,
    pub tsim: Vec<f64>,
    pub vsim: Vec<f64>,
}

/// `RerankWithKNN`.
#[allow(clippy::too_many_arguments)]
pub fn rerank_with_knn(
    chunks: &[Chunk],
    ids: &[String],
    field: &mut BTreeMap<String, Chunk>,
    knn_scores: &BTreeMap<String, f64>,
    query: &str,
    tk_weight: f64,
    vt_weight: f64,
    cfield: &str,
    qb: &QueryBuilder,
    rank_feature: &BTreeMap<String, f64>,
) -> Scores {
    if ids.is_empty() {
        return Scores {
            sim: vec![],
            tsim: vec![],
            vsim: vec![],
        };
    }
    for id in ids {
        if let Some(c) = field.get_mut(id) {
            if let Some(Value::String(s)) = c.get("important_kwd").cloned() {
                c.insert("important_kwd".into(), Value::Array(vec![Value::String(s)]));
            }
        }
    }
    let (_, keywords) = qb.question(query, "qa", 0.6);
    let mut ins_tw: Vec<Vec<String>> = Vec::with_capacity(ids.len());
    for (i, id) in ids.iter().enumerate() {
        let c = match field.get(id) {
            Some(c) => c,
            None => match chunks.get(i) {
                Some(c) => c,
                None => {
                    ins_tw.push(Vec::new());
                    continue;
                }
            },
        };
        ins_tw.push(weighted_tokens(c, cfield));
    }
    let tsim = token_similarity(&keywords, &ins_tw, qb);
    let vsim: Vec<f64> = ids
        .iter()
        .map(|id| knn_scores.get(id).copied().unwrap_or(0.0))
        .collect();
    let mut sim: Vec<f64> = tsim
        .iter()
        .zip(&vsim)
        .map(|(t, v)| tk_weight * t + vt_weight * v)
        .collect();
    let sources: Vec<Option<&Chunk>> = ids.iter().map(|id| field.get(id)).collect();
    apply_rank_features(&sources, &mut sim, rank_feature);
    Scores { sim, tsim, vsim }
}

fn extract_vector(c: &Chunk, column: &str, dim: usize) -> Vec<f64> {
    match c.get(column) {
        Some(Value::Array(a)) => a.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect(),
        _ => vec![0.0; dim],
    }
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x * y).collect();
    let aa: Vec<f64> = a.iter().map(|x| x * x).collect();
    let bb: Vec<f64> = b.iter().map(|x| x * x).collect();
    let dot = pairwise_sum(&d);
    let na = pairwise_sum(&aa);
    let nb = pairwise_sum(&bb);
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (py_sqrt(na) * py_sqrt(nb))
}

/// `RerankStandard` (the fallback when the KNN pass fails).
#[allow(clippy::too_many_arguments)]
pub fn rerank_standard(
    chunks: &[Chunk],
    question_vector: &[f64],
    query: &str,
    tk_weight: f64,
    vt_weight: f64,
    cfield: &str,
    qb: &QueryBuilder,
    rank_feature: &BTreeMap<String, f64>,
) -> Scores {
    if chunks.is_empty() {
        return Scores {
            sim: vec![],
            tsim: vec![],
            vsim: vec![],
        };
    }
    let (_, keywords) = qb.question(query, "qa", 0.6);
    let dim = question_vector.len();
    let column = format!("q_{dim}_vec");
    let embd: Vec<Vec<f64>> = chunks
        .iter()
        .map(|c| extract_vector(c, &column, dim))
        .collect();
    let tw: Vec<Vec<String>> = chunks.iter().map(|c| weighted_tokens(c, cfield)).collect();
    let vsim: Vec<f64> = embd.iter().map(|b| cosine(question_vector, b)).collect();
    let tsim = token_similarity(&keywords, &tw, qb);
    let mut sim = if vsim.iter().all(|s| *s == 0.0) {
        tsim.clone()
    } else {
        tsim.iter()
            .zip(&vsim)
            .map(|(t, v)| tk_weight * t + vt_weight * v)
            .collect()
    };
    let sources: Vec<Option<&Chunk>> = chunks.iter().map(Some).collect();
    apply_rank_features(&sources, &mut sim, rank_feature);
    let tsim_out = if vsim.iter().all(|s| *s == 0.0) {
        tsim.clone()
    } else {
        tsim
    };
    Scores {
        sim,
        tsim: tsim_out,
        vsim,
    }
}

/// A rerank model: relevance score per document index.
pub trait Reranker: Send + Sync {
    fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<(usize, f64)>, String>;
}

/// `NormalizeRerankScores`.
pub fn normalize_rerank_scores(mut scores: Vec<f64>) -> Vec<f64> {
    if scores.is_empty() {
        return scores;
    }
    let mut min = scores[0];
    let mut max = scores[0];
    for s in &scores[1..] {
        if *s < min {
            min = *s;
        }
        if *s > max {
            max = *s;
        }
    }
    if min >= 0.0 && max <= 1.0 {
        return scores;
    }
    let span = max - min;
    if span < 1e-3 {
        for s in &mut scores {
            *s = s.clamp(0.0, 1.0);
        }
        return scores;
    }
    let inv = 1.0 / span;
    for s in &mut scores {
        *s = (*s - min) * inv;
    }
    scores
}

/// Go `common.RemoveRedundantSpaces` approximation: collapse whitespace runs.
fn remove_redundant_spaces(s: &str) -> String {
    fields(s).join(" ")
}

/// `RerankByModel`.
#[allow(clippy::too_many_arguments)]
pub fn rerank_by_model(
    reranker: &dyn Reranker,
    chunks: &[Chunk],
    ids: &[String],
    field: &BTreeMap<String, Chunk>,
    query: &str,
    tk_weight: f64,
    vt_weight: f64,
    cfield: &str,
    qb: &QueryBuilder,
    rank_feature: &BTreeMap<String, f64>,
) -> Result<Scores, String> {
    if chunks.is_empty() {
        return Ok(Scores {
            sim: vec![],
            tsim: vec![],
            vsim: vec![],
        });
    }
    let (_, keywords) = qb.question(query, "qa", 0.6);
    let mut ins_tw = Vec::new();
    let mut docs = Vec::new();
    for (i, id) in ids.iter().enumerate() {
        let c = match field.get(id) {
            Some(c) => c,
            None => match chunks.get(i) {
                Some(c) => c,
                None => continue,
            },
        };
        let mut tks = content_tokens(c, cfield);
        tks.extend(split_tokens(c, "title_tks"));
        tks.extend(important_keywords(c));
        tks.extend(split_tokens(c, "question_tks"));
        let mut doc = c
            .get("content_with_weight")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if doc.is_empty() {
            doc = remove_redundant_spaces(&tks.join(" "));
        }
        ins_tw.push(tks);
        docs.push(doc);
    }
    let tsim = token_similarity(&keywords, &ins_tw, qb);
    let mut model = vec![0.0; ins_tw.len()];
    for (idx, score) in reranker.rerank(query, &docs)? {
        if idx < model.len() {
            model[idx] = score;
        }
    }
    let model = normalize_rerank_scores(model);
    let mut sim: Vec<f64> = tsim
        .iter()
        .zip(&model)
        .map(|(t, m)| tk_weight * t + vt_weight * m)
        .collect();
    let sources: Vec<Option<&Chunk>> = ids.iter().map(|id| field.get(id)).collect();
    apply_rank_features(&sources, &mut sim, rank_feature);
    Ok(Scores {
        sim,
        tsim,
        vsim: model,
    })
}
