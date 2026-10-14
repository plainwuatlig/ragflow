//! `internal/tokenizer` as the retrieval path calls it: every call takes a
//! pooled analyzer and sets language `""` → `"English"` (`withAnalyzerResult`).

use ragflow_analyzer::{AnalyzerPool, Dictionary};

pub struct Tokenizer {
    pool: AnalyzerPool,
}

impl Tokenizer {
    pub fn new(dict: Dictionary) -> Self {
        Tokenizer {
            pool: AnalyzerPool::new(dict),
        }
    }

    /// `tokenizer.Tokenize`; on error Go callers fall back to the input.
    pub fn tokenize(&self, text: &str) -> Result<String, String> {
        let mut a = self.pool.acquire("").map_err(|e| e.to_string())?;
        a.tokenize(text).map_err(|e| e.to_string())
    }

    /// `tokenizer.FineGrainedTokenize`.
    pub fn fine_grained(&self, tokens: &str) -> Result<String, String> {
        let mut a = self.pool.acquire("").map_err(|e| e.to_string())?;
        a.fine_grained_tokenize(tokens).map_err(|e| e.to_string())
    }

    /// `tokenizer.GetTermFreq` (0 when the pool is unavailable).
    pub fn freq(&self, term: &str) -> i32 {
        match self.pool.acquire("") {
            Ok(mut a) => a.term_freq(term),
            Err(_) => 0,
        }
    }

    /// `tokenizer.GetTermTag` ("" when the pool is unavailable).
    pub fn tag(&self, term: &str) -> String {
        match self.pool.acquire("") {
            Ok(mut a) => a.term_tag(term),
            Err(_) => String::new(),
        }
    }
}
