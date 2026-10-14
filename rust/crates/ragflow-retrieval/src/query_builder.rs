//! `QueryBuilder.Question` (internal/service/nlp/query_builder.go @
//! v1.0.0-rc1): the query_string text, keywords and minimum_should_match.
//!
//! Go quirks kept on purpose: `IsChinese` is true for any text of ≤3
//! whitespace fields, so short English queries take the "Chinese" branch; the
//! term list of a segment is ordered with Go's unstable `sort.Slice`.

use std::path::Path;
use std::sync::Arc;

use regex::{Captures, Regex};

use crate::gocompat::{fields, fmt_f4, fmt_v, rune_len, sort_slice, to_lower};
use crate::jianfan::t2s;
use crate::synonym::Synonym;
use crate::term_weight::{go_trim_space, TermWeightDealer};
use crate::tokenizer::Tokenizer;

/// The seven query_string fields (`InitQueryBuilder`).
pub const QUERY_FIELDS: [&str; 7] = [
    "title_tks^10",
    "title_sm_tks^5",
    "important_kwd^30",
    "important_tks^20",
    "question_tks^20",
    "content_ltks^2",
    "content_sm_ltks",
];

/// `types.MatchTextExpr` as Question builds it.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchText {
    pub matching_text: String,
    /// `ExtraOptions["minimum_should_match"]`: set on the Chinese branch only.
    pub minimum_should_match: Option<f64>,
    pub original_query: String,
}

pub struct QueryBuilder {
    tok: Arc<Tokenizer>,
    pub term_weight: TermWeightDealer,
    synonym: Synonym,
    re_eng_alpha: Regex,
    re_sub_special: Regex,
    re_stop_zh: Regex,
    re_stop_en1: Regex,
    re_stop_en2: Regex,
    re_eng_zh_num: Regex,
    re_eng_zh: Regex,
    re_zh_eng_num: Regex,
    re_zh_eng: Regex,
    re_simple_token: Regex,
    re_punct: Regex,
    re_clean_quote: Regex,
    re_single_char: Regex,
    re_lead_sign: Regex,
    re_query_special: Regex,
    re_special_char: Regex,
    re_clean_term: Regex,
}

impl QueryBuilder {
    /// `InitQueryBuilder(<dict>/wordnet)`. `res_dir` plays the server's
    /// `rag/res` (None, or a directory without the files, mirrors the image).
    pub fn new(tok: Arc<Tokenizer>, res_dir: Option<&Path>, wordnet_dir: Option<&Path>) -> Self {
        QueryBuilder {
            term_weight: TermWeightDealer::new(tok.clone(), res_dir),
            synonym: Synonym::new(res_dir, wordnet_dir),
            tok,
            re_eng_alpha: Regex::new(r"^[a-zA-Z]+$").unwrap(),
            re_sub_special: Regex::new(r#"([:{}/\[\]\-\*"\(\)\|\+~\^])"#).unwrap(),
            re_stop_zh: Regex::new(
                "(?i)\u{662f}*(\u{600e}\u{4e48}\u{529e}|\u{4ec0}\u{4e48}\u{6837}\u{7684}|\u{54ea}\u{5bb6}|\u{4e00}\u{4e0b}|\u{90a3}\u{5bb6}|\u{8bf7}\u{95ee}|\u{5565}\u{6837}|\u{548b}\u{6837}\u{4e86}|\u{4ec0}\u{4e48}\u{65f6}\u{5019}|\u{4f55}\u{65f6}|\u{4f55}\u{5730}|\u{4f55}\u{4eba}|\u{662f}\u{5426}|\u{662f}\u{4e0d}\u{662f}|\u{591a}\u{5c11}|\u{54ea}\u{91cc}|\u{600e}\u{4e48}|\u{54ea}\u{513f}|\u{600e}\u{4e48}\u{6837}|\u{5982}\u{4f55}|\u{54ea}\u{4e9b}|\u{662f}\u{5565}|\u{5565}\u{662f}|\u{554a}|\u{5417}|\u{5462}|\u{5427}|\u{548b}|\u{4ec0}\u{4e48}|\u{6709}\u{6ca1}\u{6709}|\u{5440}|\u{8c01}|\u{54ea}\u{4f4d}|\u{54ea}\u{4e2a})\u{662f}*",
            )
            .unwrap(),
            re_stop_en1: Regex::new(r"(?i)(^| )(what|who|how|which|where|why)('re|'s)? ").unwrap(),
            re_stop_en2: Regex::new(r"(?i)(^| )('s|'re|is|are|were|was|do|does|did|don't|doesn't|didn't|has|have|be|there|you|me|your|my|mine|just|please|may|i|should|would|wouldn't|will|won't|done|go|for|with|so|the|a|an|by|i'm|it's|he's|she's|they|they're|you're|as|by|on|in|at|up|out|down|of|to|or|and|if) ").unwrap(),
            re_eng_zh_num: Regex::new(r"([A-Za-z]+[0-9]*)([\x{4e00}-\x{9fa5}]+)").unwrap(),
            re_eng_zh: Regex::new(r"([A-Za-z])([\x{4e00}-\x{9fa5}]+)").unwrap(),
            re_zh_eng_num: Regex::new(r"([\x{4e00}-\x{9fa5}]+)([A-Za-z]+[0-9]*)").unwrap(),
            re_zh_eng: Regex::new(r"([\x{4e00}-\x{9fa5}]+)([A-Za-z])").unwrap(),
            re_simple_token: Regex::new(r"^[0-9a-z\.\+#_\*-]+$").unwrap(),
            re_punct: Regex::new(
                "[ :|\\r\\n\\t,\u{ff0c}.\u{3002}?\u{ff1f}/\\`!\u{ff01}&^%()\\[\\]{}<>*~'\"\\\\]+",
            )
            .unwrap(),
            re_clean_quote: Regex::new(r#"[ "'^]+"#).unwrap(),
            re_single_char: Regex::new(r"^[a-z0-9]$").unwrap(),
            re_lead_sign: Regex::new(r"^[\+\-]+").unwrap(),
            re_query_special: Regex::new(r"[.^+\(\)-]").unwrap(),
            re_special_char: Regex::new(
                "[,\\.\\/;'\\[\\]\\\\\\`~!@#$%\\^&\\*\\(\\)=\\+_<>\\?:\"\\{\\}\\|\u{ff0c}\u{3002}\u{ff1b}'\u{2018}\u{2019}\u{3010}\u{3011}\u{3001}\u{ff01}\u{ffe5}\u{2026}\u{2026}\u{ff08}\u{ff09}\u{2014}\u{2014}\u{300a}\u{300b}\u{ff1f}\u{ff1a}\"\"\"-]+",
            )
            .unwrap(),
            re_clean_term: Regex::new(r#"[ "']+"#).unwrap(),
        }
    }

    /// `IsChinese`.
    pub fn is_chinese(&self, line: &str) -> bool {
        let fs = fields(line);
        if fs.len() <= 3 {
            return true;
        }
        let non_alpha = fs.iter().filter(|f| !self.re_eng_alpha.is_match(f)).count();
        non_alpha as f64 / fs.len() as f64 >= 0.7
    }

    /// `SubSpecialChar`: backslash before `: { } / [ ] - * " ( ) | + ~ ^`.
    pub fn sub_special_char(&self, line: &str) -> String {
        self.re_sub_special
            .replace_all(line, |c: &Captures| format!("\\{}", &c[1]))
            .into_owned()
    }

    /// `RmWWW`.
    pub fn rm_www(&self, txt: &str) -> String {
        let original = txt.to_string();
        let t = self.re_stop_zh.replace_all(txt, "").into_owned();
        let t = self.re_stop_en1.replace_all(&t, " ").into_owned();
        let t = self.re_stop_en2.replace_all(&t, " ").into_owned();
        if t.is_empty() {
            original
        } else {
            t
        }
    }

    /// `AddSpaceBetweenEngZh`.
    pub fn add_space_between_eng_zh(&self, txt: &str) -> String {
        let t = self
            .re_eng_zh_num
            .replace_all(txt, "${1} ${2}")
            .into_owned();
        let t = self.re_eng_zh.replace_all(&t, "${1} ${2}").into_owned();
        let t = self.re_zh_eng_num.replace_all(&t, "${1} ${2}").into_owned();
        self.re_zh_eng.replace_all(&t, "${1} ${2}").into_owned()
    }

    /// `StrFullWidth2HalfWidth`.
    pub fn str_full_width_to_half_width(s: &str) -> String {
        s.chars()
            .map(|c| {
                let code = c as i64;
                let inside = if code == 0x3000 { 0x20 } else { code - 0xFEE0 };
                if !(0x20..=0x7E).contains(&inside) {
                    c
                } else {
                    char::from_u32(inside as u32).unwrap_or(c)
                }
            })
            .collect()
    }

    /// `NeedFineGrainedTokenize`.
    pub fn need_fine_grained(&self, tk: &str) -> bool {
        rune_len(tk) >= 3 && !self.re_simple_token.is_match(tk)
    }

    fn tokenize_or(&self, s: &str) -> String {
        self.tok.tokenize(s).unwrap_or_else(|_| s.to_string())
    }

    /// `Question(txt, tbl, minMatch)` → (match expression, keywords).
    pub fn question(
        &self,
        txt: &str,
        _tbl: &str,
        min_match: f64,
    ) -> (Option<MatchText>, Vec<String>) {
        let original_query = txt.to_string();
        let t = self.add_space_between_eng_zh(txt);
        let t = to_lower(&t);
        let t = Self::str_full_width_to_half_width(&t);
        let t = t2s(&t);
        let t = self.re_punct.replace_all(&t, " ").into_owned();
        let txt_no_stop = self.rm_www(&t);

        if !self.is_chinese(&txt_no_stop) {
            let txt_final = self.rm_www(&txt_no_stop);
            let tokenized = self.tokenize_or(&txt_final);
            let tks: Vec<String> = fields(&tokenized).into_iter().map(str::to_string).collect();
            let mut keywords: Vec<String> = tks.iter().filter(|t| !t.is_empty()).cloned().collect();
            let tws = self.term_weight.weights(&tks, false);
            let mut tks_w: Vec<(String, f64)> = Vec::new();
            for tw in tws {
                let tk = self.re_clean_quote.replace_all(&tw.term, "").into_owned();
                let tk = self.re_single_char.replace_all(&tk, "").into_owned();
                let tk = self.re_lead_sign.replace_all(&tk, "").into_owned();
                let tk = go_trim_space(&tk).to_string();
                if tk.is_empty() {
                    continue;
                }
                tks_w.push((tk, tw.weight));
            }
            tks_w.truncate(256);
            let mut syns = vec![String::new(); tks_w.len()];
            for (i, (tk, w)) in tks_w.iter().enumerate() {
                let tk_syns = self.synonym.lookup(tk, 8);
                if !tk_syns.is_empty() {
                    let mut parts = Vec::new();
                    for syn in &tk_syns {
                        let syn = go_trim_space(syn);
                        if !syn.is_empty() {
                            parts.push(format!("\"{}\"^{}", syn, fmt_f4(w / 4.0)));
                        }
                    }
                    syns[i] = parts.join(" ");
                    keywords.extend(tk_syns);
                }
            }
            let mut q: Vec<String> = Vec::new();
            for (i, (tk, w)) in tks_w.iter().enumerate() {
                if self.re_query_special.is_match(tk) {
                    continue;
                }
                q.push(format!("({}^{} {})", tk, fmt_f4(*w), syns[i]));
            }
            for i in 1..tks_w.len() {
                let left = go_trim_space(&tks_w[i - 1].0);
                let right = go_trim_space(&tks_w[i].0);
                if left.is_empty() || right.is_empty() {
                    continue;
                }
                let mut max_w = tks_w[i - 1].1;
                if tks_w[i].1 > max_w {
                    max_w = tks_w[i].1;
                }
                q.push(format!("\"{} {}\"^{}", left, right, fmt_f4(max_w * 2.0)));
            }
            if q.is_empty() {
                q.push(txt_final.clone());
            }
            return (
                Some(MatchText {
                    matching_text: q.join(" "),
                    minimum_should_match: None,
                    original_query,
                }),
                keywords,
            );
        }

        let otxt = txt_no_stop.clone();
        let txt_chinese = self.rm_www(&txt_no_stop);
        let mut qs: Vec<String> = Vec::new();
        let mut keywords: Vec<String> = Vec::new();
        let mut segments = self.term_weight.split(&txt_chinese);
        segments.truncate(256);

        for segment in &segments {
            if segment.is_empty() {
                continue;
            }
            keywords.push(segment.clone());
            let mut twl = self
                .term_weight
                .weights(std::slice::from_ref(segment), true);
            let syns = self.synonym.lookup(segment, 8);
            if !syns.is_empty() && keywords.len() < 32 {
                keywords.extend(syns.iter().cloned());
            }
            sort_slice(&mut twl, |a, b| a.weight > b.weight);

            let mut terms: Vec<(String, f64)> = Vec::new();
            for tw in &twl {
                let mut term = tw.term.clone();
                let weight = tw.weight;
                let mut sm: Vec<String> = Vec::new();
                if self.need_fine_grained(&term) {
                    if let Ok(fg) = self.tok.fine_grained(&term) {
                        if !fg.is_empty() {
                            sm = fields(&fg).into_iter().map(str::to_string).collect();
                        }
                    }
                }
                let mut clean_sm = Vec::new();
                for m in &sm {
                    let m = self.re_special_char.replace_all(m, "").into_owned();
                    let m = self.sub_special_char(&m);
                    if rune_len(&m) > 1 {
                        clean_sm.push(m);
                    }
                }
                let sm = clean_sm;
                if keywords.len() < 32 {
                    let clean_tk = self.re_clean_term.replace_all(&term, "").into_owned();
                    if !clean_tk.is_empty() {
                        keywords.push(clean_tk);
                    }
                    keywords.extend(sm.iter().cloned());
                }
                let tk_syns: Vec<String> = self
                    .synonym
                    .lookup(&term, 8)
                    .iter()
                    .map(|s| self.sub_special_char(s))
                    .collect();
                if keywords.len() < 32 {
                    for s in &tk_syns {
                        if !s.is_empty() {
                            keywords.push(s.clone());
                        }
                    }
                }
                let mut fg_syns: Vec<String> = Vec::new();
                for s in &tk_syns {
                    if s.is_empty() {
                        continue;
                    }
                    if let Ok(fg) = self.tok.fine_grained(s) {
                        if !fg.is_empty() {
                            let fg = if fg.contains(' ') {
                                format!("\"{fg}\"")
                            } else {
                                fg
                            };
                            fg_syns.push(fg);
                        }
                    }
                }
                if keywords.len() >= 32 {
                    break;
                }
                term = self.sub_special_char(&term);
                if term.is_empty() {
                    continue;
                }
                if term.contains(' ') {
                    term = format!("\"{term}\"");
                }
                if !fg_syns.is_empty() {
                    term = format!("({} OR ({})^0.2)", term, fg_syns.join(" "));
                }
                if !sm.is_empty() {
                    let sm_str = sm.join(" ");
                    term = format!("{term} OR \"{sm_str}\" OR (\"{sm_str}\"~2)^0.5");
                }
                terms.push((term, weight));
            }

            let mut tms = terms
                .iter()
                .map(|(t, w)| format!("({})^{}", t, fmt_v(*w)))
                .collect::<Vec<_>>()
                .join(" ");
            if twl.len() > 1 {
                let tokenized = self.tok.tokenize(segment).unwrap_or_default();
                if !tokenized.is_empty() {
                    tms.push_str(&format!(" (\"{tokenized}\"~2)^1.5"));
                }
            }
            if !syns.is_empty() && !tms.is_empty() {
                let mut parts = Vec::new();
                for s in &syns {
                    let s = self.sub_special_char(s);
                    if !s.is_empty() {
                        let tokenized = self.tok.tokenize(&s).unwrap_or_default();
                        if !tokenized.is_empty() {
                            parts.push(format!("\"{tokenized}\""));
                        }
                    }
                }
                if !parts.is_empty() {
                    tms = format!("({})^5 OR ({})^0.7", tms, parts.join(" OR "));
                }
            }
            if !tms.is_empty() {
                qs.push(tms);
            }
        }

        if !qs.is_empty() {
            let parts: Vec<String> = qs
                .iter()
                .filter(|q| !q.is_empty())
                .map(|q| format!("({q})"))
                .collect();
            let mut query = parts.join(" OR ");
            if query.is_empty() {
                query = otxt;
            }
            return (
                Some(MatchText {
                    matching_text: query,
                    minimum_should_match: Some(min_match),
                    original_query,
                }),
                keywords,
            );
        }
        (None, keywords)
    }
}
