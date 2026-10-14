//! `TermWeightDealer` (internal/service/nlp/term_weight.go @ v1.0.0-rc1).
//!
//! Data files are read relative to a resource directory (`rag/res` from the
//! server's working directory in Go) and are all optional: the official image
//! ships neither `ner.json` nor `term.freq`, so both maps are empty there.
//! `Pretoken`'s punctuation pattern never compiles in Go (its last `\]`
//! escapes the closing bracket and `regexp.MatchString` errors), so tokens are
//! never replaced by `#`; that is reproduced by not matching at all.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use regex::Regex;

use crate::gocompat::{fields, pairwise_sum, py_log10, rune_len};
use crate::tokenizer::Tokenizer;

const STOP_WORDS: &[&str] = &[
    "请问", "您", "你", "我", "他", "是", "的", "就", "有", "于", "及", "即", "在", "为", "最",
    "有", "从", "以", "了", "将", "与", "吗", "吧", "中", "#", "什么", "怎么", "哪个", "哪些",
    "啥", "相关",
];

pub struct TermWeightDealer {
    tok: Arc<Tokenizer>,
    stop_words: HashSet<&'static str>,
    ne: HashMap<String, String>,
    df: HashMap<String, i64>,
    re_digit: Regex,
    re_short_alnum: Regex,
    re_leading_alnum: Regex,
    re_space_run: Regex,
    re_ends_letter: Regex,
    re_num_like: Regex,
    re_short_letters: Regex,
    re_num_space: Regex,
    re_tag_numeric: Regex,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TermWeight {
    pub term: String,
    pub weight: f64,
}

impl TermWeightDealer {
    /// `NewTermWeightDealer(resPath)`: `res_dir` plays `rag/res`.
    pub fn new(tok: Arc<Tokenizer>, res_dir: Option<&Path>) -> Self {
        let mut ne = HashMap::new();
        let mut df = HashMap::new();
        if let Some(dir) = res_dir {
            if let Ok(data) = std::fs::read(dir.join("ner.json")) {
                if let Ok(m) = serde_json::from_slice::<HashMap<String, String>>(&data) {
                    ne = m;
                }
            }
            df = load_dict(&dir.join("term.freq"));
        }
        TermWeightDealer {
            tok,
            stop_words: STOP_WORDS.iter().copied().collect(),
            ne,
            df,
            re_digit: Regex::new(r"^[0-9]$").unwrap(),
            re_short_alnum: Regex::new(r"^[0-9a-z]{1,2}$").unwrap(),
            re_leading_alnum: Regex::new(r"^[0-9a-zA-Z]").unwrap(),
            re_space_run: Regex::new(r"[ \t]+").unwrap(),
            re_ends_letter: Regex::new(r"(?s).*[a-zA-Z]$").unwrap(),
            re_num_like: Regex::new(r"^[0-9,.]{2,}$").unwrap(),
            re_short_letters: Regex::new(r"^[a-z]{1,2}$").unwrap(),
            re_num_space: Regex::new(r"^[0-9. -]{2,}$").unwrap(),
            re_tag_numeric: Regex::new(r"^[0-9-]+").unwrap(),
        }
    }

    pub fn ne(&self) -> &HashMap<String, String> {
        &self.ne
    }

    fn is_stop(&self, t: &str) -> bool {
        self.stop_words.contains(t)
    }

    /// `Pretoken(txt, num, stpwd)`.
    pub fn pretoken(&self, txt: &str, num: bool, stpwd: bool) -> Vec<String> {
        let tokenized = self.tok.tokenize(txt).unwrap_or_else(|_| txt.to_string());
        let mut res = Vec::new();
        for t in fields(&tokenized) {
            if stpwd && self.is_stop(t) {
                continue;
            }
            if self.re_digit.is_match(t) && !num {
                continue;
            }
            // The punctuation pattern never compiles in Go: no "#" mapping.
            if t != "#" && !t.is_empty() {
                res.push(t.to_string());
            }
        }
        res
    }

    fn one_term(&self, t: &str) -> bool {
        rune_len(t) == 1 || self.re_short_alnum.is_match(t)
    }

    /// `TokenMerge(tks)`.
    pub fn token_merge(&self, tks: &[String]) -> Vec<String> {
        if tks.is_empty() {
            return Vec::new();
        }
        let mut res: Vec<String> = Vec::new();
        let mut i = 0;
        while i < tks.len() {
            if i == 0 && tks.len() > 1 && self.one_term(&tks[i]) {
                let is_next_multi = rune_len(&tks[i + 1]) > 1;
                let is_next_alnum = self.re_leading_alnum.is_match(&tks[i + 1]);
                if is_next_multi && !is_next_alnum {
                    res.push(format!("{} {}", tks[0], tks[1]));
                    i = 2;
                    continue;
                }
            }
            let mut j = i;
            while j < tks.len() && !tks[j].is_empty() {
                if self.is_stop(&tks[j]) {
                    break;
                }
                if !self.one_term(&tks[j]) {
                    break;
                }
                j += 1;
            }
            if j - i > 1 {
                if j - i < 5 {
                    res.push(tks[i..j].join(" "));
                    i = j;
                } else {
                    let mut k = i;
                    while k < j {
                        if k + 1 < j {
                            res.push(format!("{} {}", tks[k], tks[k + 1]));
                        } else {
                            res.push(tks[k].clone());
                        }
                        k += 2;
                    }
                    i = j;
                }
            } else {
                if !tks[i].is_empty() {
                    res.push(tks[i].clone());
                }
                i += 1;
            }
        }
        res.into_iter().filter(|t| !t.is_empty()).collect()
    }

    /// `Split(txt)`.
    pub fn split(&self, txt: &str) -> Vec<String> {
        if txt.is_empty() {
            return vec![String::new()];
        }
        let txt = self.re_space_run.replace_all(txt, " ");
        let txt = go_trim_space(&txt);
        let mut tks: Vec<String> = Vec::new();
        for t in txt.split(' ') {
            let t = go_trim_space(t);
            if t.is_empty() {
                continue;
            }
            if let Some(prev) = tks.last() {
                let prev_ends = self.re_ends_letter.is_match(prev);
                let curr_ends = self.re_ends_letter.is_match(t);
                let prev_ne = self.ne.get(prev.as_str()).map(String::as_str).unwrap_or("");
                let curr_ne = self.ne.get(t).map(String::as_str).unwrap_or("");
                if prev_ends && curr_ends && curr_ne != "func" && prev_ne != "func" {
                    let last = tks.last_mut().unwrap();
                    last.push(' ');
                    last.push_str(t);
                    continue;
                }
            }
            tks.push(t.to_string());
        }
        tks
    }

    fn ner_weight(&self, t: &str) -> f64 {
        if self.re_num_like.is_match(t) {
            return 2.0;
        }
        if self.re_short_letters.is_match(t) {
            return 0.01;
        }
        match self.ne.get(t).map(String::as_str) {
            Some("toxic") => 2.0,
            Some("func") => 1.0,
            Some("corp") | Some("loca") | Some("sch") | Some("stock") => 3.0,
            Some("firstnm") => 1.0,
            _ => 1.0,
        }
    }

    fn postag_weight(&self, t: &str) -> f64 {
        let tag = self.tok.tag(t);
        match tag.as_str() {
            "r" | "c" | "d" => 0.3,
            "ns" | "nt" => 3.0,
            "n" => 2.0,
            _ if self.re_tag_numeric.is_match(&tag) => 2.0,
            _ => 1.0,
        }
    }

    fn fine_pieces(&self, t: &str) -> Vec<String> {
        let fg = self.tok.fine_grained(t).unwrap_or_default();
        fields(&fg)
            .into_iter()
            .filter(|x| rune_len(x) > 1)
            .map(str::to_string)
            .collect()
    }

    fn freq(&self, t: &str) -> f64 {
        if self.re_num_space.is_match(t) {
            return 3.0;
        }
        let s = self.tok.freq(t);
        if s == 0 {
            if let Some(f) = alphabetic_oov_frequency(t) {
                return f;
            }
        }
        if s == 0 && rune_len(t) >= 4 {
            let pieces = self.fine_pieces(t);
            if pieces.len() > 1 {
                let vals: Vec<f64> = pieces.iter().map(|p| self.freq(p)).collect();
                let mut min = vals[0];
                for v in &vals[1..] {
                    if *v < min {
                        min = *v;
                    }
                }
                return min / 6.0;
            }
            return 10.0;
        }
        (s as f64).max(10.0)
    }

    fn df(&self, t: &str) -> f64 {
        if self.re_num_space.is_match(t) {
            return 5.0;
        }
        if let Some(v) = self.df.get(t) {
            return *v as f64 + 3.0;
        }
        if let Some(f) = alphabetic_oov_frequency(t) {
            return f;
        }
        if rune_len(t) >= 4 {
            let pieces = self.fine_pieces(t);
            if pieces.len() > 1 {
                let vals: Vec<f64> = pieces.iter().map(|p| self.df(p)).collect();
                let mut min = vals[0];
                for v in &vals[1..] {
                    if *v < min {
                        min = *v;
                    }
                }
                return (min / 6.0).max(3.0);
            }
        }
        3.0
    }

    fn raw_weights(&self, tt: &[String], out: &mut Vec<TermWeight>) {
        let idf = |s: f64, n: f64| py_log10(10.0 + ((n - s + 0.5) / (s + 0.5)));
        for t in tt {
            let idf1 = idf(self.freq(t), 10_000_000.0);
            let idf2 = idf(self.df(t), 1_000_000_000.0);
            let ner_pos = self.ner_weight(t) * self.postag_weight(t);
            out.push(TermWeight {
                term: t.clone(),
                weight: (0.3 * idf1 + 0.7 * idf2) * ner_pos,
            });
        }
    }

    /// `Weights(tks, preprocess)`.
    pub fn weights(&self, tks: &[String], preprocess: bool) -> Vec<TermWeight> {
        let mut tw = Vec::new();
        if !preprocess {
            self.raw_weights(tks, &mut tw);
        } else {
            for tk in tks {
                let tokens = self.pretoken(tk, true, true);
                let tt = self.token_merge(&tokens);
                if tt.is_empty() {
                    continue;
                }
                self.raw_weights(&tt, &mut tw);
            }
        }
        if tw.is_empty() {
            return tw;
        }
        let buf: Vec<f64> = tw.iter().map(|t| t.weight).collect();
        let s = pairwise_sum(&buf);
        if s > 0.0 {
            for t in &mut tw {
                t.weight /= s;
            }
        }
        tw
    }
}

/// Go `strings.TrimSpace`.
pub fn go_trim_space(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace())
}

fn load_dict(path: &Path) -> HashMap<String, i64> {
    let mut res = HashMap::new();
    let Ok(data) = std::fs::read_to_string(path) else {
        return res;
    };
    for line in data.split('\n') {
        let line = go_trim_space(line);
        if line.is_empty() {
            continue;
        }
        let arr: Vec<&str> = line.split('\t').collect();
        if arr.len() >= 2 {
            if let Ok(f) = arr[1].parse::<i64>() {
                res.insert(arr[0].to_string(), f);
            }
        } else {
            res.insert(arr[0].to_string(), 0);
        }
    }
    res
}

/// `alphabeticOOVFrequency`.
pub fn alphabetic_oov_frequency(term: &str) -> Option<f64> {
    let mut letters = 0i64;
    for c in term.chars() {
        if is_lgc_letter(c) {
            letters += 1;
            continue;
        }
        if c != ' ' && c != '.' && c != '-' {
            return None;
        }
    }
    if letters == 0 {
        return None;
    }
    let exponent = (letters - 3).max(0) as f64 / 2.0;
    let frequency = (300.0 / 2f64.powf(exponent)).round();
    Some(frequency.max(10.0))
}

/// `unicode.IsLetter(r) && unicode.In(r, Latin, Greek, Cyrillic)`.
fn is_lgc_letter(c: char) -> bool {
    if !c.is_alphabetic() {
        return false;
    }
    let u = c as u32;
    // Alphabetic but not a letter (Nl numbers, combining marks) inside the
    // script ranges below.
    if (0x2160..=0x2188).contains(&u)
        || (0x2DE0..=0x2DFF).contains(&u)
        || (0x10140..=0x10174).contains(&u)
    {
        return false;
    }
    const RANGES: &[(u32, u32)] = &[
        // Latin
        (0x41, 0x5A),
        (0x61, 0x7A),
        (0xAA, 0xAA),
        (0xBA, 0xBA),
        (0xC0, 0xD6),
        (0xD8, 0xF6),
        (0xF8, 0x2B8),
        (0x2E0, 0x2E4),
        (0x1D00, 0x1D25),
        (0x1D2C, 0x1D5C),
        (0x1D62, 0x1D65),
        (0x1D6B, 0x1D77),
        (0x1D79, 0x1DBE),
        (0x1E00, 0x1EFF),
        (0x2071, 0x2071),
        (0x207F, 0x207F),
        (0x2090, 0x209C),
        (0x212A, 0x212B),
        (0x2132, 0x2132),
        (0x214E, 0x214E),
        (0x2C60, 0x2C7F),
        (0xA722, 0xA787),
        (0xA78B, 0xA7CA),
        (0xA7D0, 0xA7D9),
        (0xA7F2, 0xA7FF),
        (0xAB30, 0xAB5A),
        (0xAB5C, 0xAB64),
        (0xAB66, 0xAB69),
        (0xFB00, 0xFB06),
        (0xFF21, 0xFF3A),
        (0xFF41, 0xFF5A),
        (0x10780, 0x107BA),
        (0x1DF00, 0x1DF2A),
        // Greek
        (0x370, 0x373),
        (0x376, 0x377),
        (0x37A, 0x37D),
        (0x37F, 0x37F),
        (0x386, 0x386),
        (0x388, 0x3E1),
        (0x3F0, 0x3FF),
        (0x1D26, 0x1D2A),
        (0x1D5D, 0x1D61),
        (0x1D66, 0x1D6A),
        (0x1DBF, 0x1DBF),
        (0x1F00, 0x1FFE),
        (0x2126, 0x2126),
        (0xAB65, 0xAB65),
        (0x10140, 0x1018E),
        (0x1D200, 0x1D245),
        // Cyrillic
        (0x400, 0x52F),
        (0x1C80, 0x1C88),
        (0x1D2B, 0x1D2B),
        (0x1D78, 0x1D78),
        (0xA640, 0xA69F),
        (0x1E030, 0x1E08F),
    ];
    RANGES.iter().any(|&(lo, hi)| (lo..=hi).contains(&u))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oov_frequency() {
        assert_eq!(alphabetic_oov_frequency("abc"), Some(300.0));
        assert_eq!(alphabetic_oov_frequency("abcde"), Some(150.0));
        assert_eq!(alphabetic_oov_frequency("ab1"), None);
        assert_eq!(alphabetic_oov_frequency("停车"), None);
    }
}
