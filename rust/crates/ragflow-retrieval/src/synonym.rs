//! `Synonym.Lookup` (internal/service/nlp/synonym.go @ v1.0.0-rc1): the
//! optional `synonym.json` dictionary first (absent from the official image),
//! then WordNet for purely alphabetic tokens.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use regex::Regex;

use crate::gocompat::to_lower;
use crate::term_weight::go_trim_space;
use crate::wordnet::WordNet;

pub struct Synonym {
    dictionary: HashMap<String, Vec<String>>,
    wordnet: Option<WordNet>,
    re_space: Regex,
    re_alpha: Regex,
}

impl Synonym {
    /// `NewSynonym(nil, resPath, wordnetDir)`.
    pub fn new(res_dir: Option<&Path>, wordnet_dir: Option<&Path>) -> Self {
        let wordnet = wordnet_dir.and_then(|d| WordNet::new(d).ok());
        let mut dictionary = HashMap::new();
        if let Some(dir) = res_dir {
            if let Ok(data) = std::fs::read(dir.join("synonym.json")) {
                if let Ok(serde_json::Value::Object(m)) = serde_json::from_slice(&data) {
                    for (k, v) in m {
                        let key = to_lower(&k);
                        match v {
                            serde_json::Value::String(s) => {
                                dictionary.insert(key, vec![s]);
                            }
                            serde_json::Value::Array(a) => {
                                let list = a
                                    .into_iter()
                                    .filter_map(|x| x.as_str().map(str::to_string))
                                    .collect();
                                dictionary.insert(key, list);
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        Synonym {
            dictionary,
            wordnet,
            re_space: Regex::new(r"[ \t]+").unwrap(),
            re_alpha: Regex::new(r"^[a-z]+$").unwrap(),
        }
    }

    /// `Lookup(tk, topN)`.
    pub fn lookup(&self, tk: &str, top_n: usize) -> Vec<String> {
        if tk.is_empty() {
            return Vec::new();
        }
        let top_n = if top_n == 0 { 8 } else { top_n };
        let key = to_lower(&self.re_space.replace_all(go_trim_space(tk), " "));
        if let Some(res) = self.dictionary.get(&key) {
            return res.iter().take(top_n).cloned().collect();
        }
        if let (true, Some(wn)) = (self.re_alpha.is_match(tk), &self.wordnet) {
            let mut set = BTreeSet::new();
            for syn in wn.synsets(tk) {
                let word = syn.name.split('.').next().unwrap_or("").replace('_', " ");
                set.insert(word);
            }
            set.remove(tk);
            // BTreeSet iterates in byte order, as sort.Strings does.
            return set
                .into_iter()
                .filter(|w| !w.is_empty())
                .take(top_n)
                .collect();
        }
        Vec::new()
    }
}
