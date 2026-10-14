//! Safe Rust bindings to RAGFlow's C++ text analyzer
//! (`internal/binding/cpp`, C API `rag_analyzer_c_api.h`), the same static
//! library the Go server links through cgo (`internal/binding/rag_analyzer.go`).
//!
//! # Model
//!
//! - [`Dictionary`] is the root analyzer: `RAGAnalyzer_Create` + `RAGAnalyzer_Load`.
//!   It owns the trie, POS table, WordNet lemmatizer and OpenCC converter.
//! - [`Analyzer`] is a `RAGAnalyzer_Copy` of the root, the per-request instance
//!   upstream's pool hands out (`internal/tokenizer/tokenizer.go`). A copy borrows
//!   the root's dictionaries, so it holds an `Arc` to the root: the root is
//!   destroyed only after the last copy.
//! - [`AnalyzerPool`] mirrors the Go pool's contract: copies are reused, and the
//!   language is set on every acquire (`""` means `"English"`).
//!
//! # Memory ownership
//!
//! Strings returned by `RAGAnalyzer_Tokenize`, `RAGAnalyzer_FineGrainedTokenize`
//! and `RAGAnalyzer_GetTermTag` are `malloc`ed by the C++ side; they are copied
//! into a Rust `String` and released with libc `free`, as Go does with `C.free`.
//! Input strings are passed as NUL-terminated copies; like Go's `C.CString`, the C
//! side sees the text only up to the first interior NUL.
//!
//! # Thread safety
//!
//! An [`Analyzer`] carries mutable per-instance state (stemmer, buffers), so it is
//! `Send` but not `Sync`; its methods take `&mut self`.
//!
//! Copies also share one OpenCC converter, whose shortest-path segmentation keeps a
//! scratch buffer inside the converter and reallocates it when a longer input
//! arrives (`opencc/converter.c`, `sp_seg`). Two copies running `Tokenize` at the
//! same time therefore race on that buffer. Upstream's Go pool does not guard this;
//! here [`Analyzer::tokenize`] takes a per-[`Dictionary`] lock around the C call.
//! `FineGrainedTokenize`, `GetTermFreq` and `GetTermTag` do not touch OpenCC and run
//! unlocked.

mod ffi;

use std::ffi::{CStr, CString};
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::os::raw::c_char;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

/// Language applied when the caller passes `""`, as `defaultLanguage` in Go.
pub const DEFAULT_LANGUAGE: &str = "English";

/// Default dictionary root, as Go's tokenizer `Init` (overridden by `RAGFLOW_DICT_PATH`).
pub const DEFAULT_DICT_PATH: &str = "/usr/share/infinity/resource";

#[derive(Debug)]
pub enum Error {
    /// `RAGAnalyzer_Create` returned NULL.
    Create,
    /// `RAGAnalyzer_Load` returned a non-zero code.
    Load(i32),
    /// `RAGAnalyzer_Copy` returned NULL.
    Copy,
    /// A C call that returns a string returned NULL.
    NullResult(&'static str),
    /// The dictionary path is not valid UTF-8 or contains NUL.
    Path,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Create => write!(f, "RAGAnalyzer_Create failed"),
            Error::Load(code) => write!(f, "RAGAnalyzer_Load failed with code {code}"),
            Error::Copy => write!(f, "RAGAnalyzer_Copy failed"),
            Error::NullResult(call) => write!(f, "{call} returned NULL"),
            Error::Path => write!(f, "dictionary path is not a valid C string"),
        }
    }
}

impl std::error::Error for Error {}

/// The root analyzer that owns the loaded dictionaries.
struct Root {
    handle: ffi::RAGAnalyzerHandle,
    /// Serialises `RAGAnalyzer_Tokenize` across all copies (shared OpenCC scratch).
    tokenize_lock: Mutex<()>,
}

// SAFETY: after `Load` the root is only read (by `RAGAnalyzer_Copy`, whose copy
// constructor reads pointers and flags) and finally destroyed by the last `Arc`.
unsafe impl Send for Root {}
unsafe impl Sync for Root {}

impl Drop for Root {
    fn drop(&mut self) {
        // SAFETY: handle came from RAGAnalyzer_Create; every copy holds an Arc to
        // this Root, so none outlives it.
        unsafe { ffi::RAGAnalyzer_Destroy(self.handle) }
    }
}

/// Loaded analyzer dictionaries. Cheap to clone (shared root).
#[derive(Clone)]
pub struct Dictionary {
    root: Arc<Root>,
}

impl Dictionary {
    /// `RAGAnalyzer_Create(path)` + `RAGAnalyzer_Load()`, as Go's pool `Init`.
    ///
    /// `Load` reports progress with `printf` to stdout.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref().to_str().ok_or(Error::Path)?;
        let c_path = CString::new(path).map_err(|_| Error::Path)?;
        // SAFETY: c_path is a valid NUL-terminated string for the call's duration.
        let handle = unsafe { ffi::RAGAnalyzer_Create(c_path.as_ptr()) };
        if handle.is_null() {
            return Err(Error::Create);
        }
        let root = Root {
            handle,
            tokenize_lock: Mutex::new(()),
        };
        // SAFETY: handle is live and not yet shared.
        let code = unsafe { ffi::RAGAnalyzer_Load(root.handle) };
        if code != 0 {
            return Err(Error::Load(code)); // Root's Drop destroys the handle.
        }
        Ok(Dictionary {
            root: Arc::new(root),
        })
    }

    /// Like [`Dictionary::load`], with the process's stdout pointed at stderr while
    /// `Load` runs, so its `printf` progress lines cannot mix into stdout output.
    /// Changes process-wide file descriptors for the duration; call it before other
    /// threads write to stdout.
    pub fn load_quietly(path: impl AsRef<Path>) -> Result<Self, Error> {
        // SAFETY: only dup/dup2/close on fds 1 and 2; fflush(NULL) flushes all C stdio streams.
        unsafe {
            ffi::fflush(std::ptr::null_mut());
            let saved = ffi::dup(1);
            if saved >= 0 {
                ffi::dup2(2, 1);
            }
            let result = Self::load(path);
            ffi::fflush(std::ptr::null_mut());
            if saved >= 0 {
                ffi::dup2(saved, 1);
                ffi::close(saved);
            }
            result
        }
    }

    /// A new independent instance (`RAGAnalyzer_Copy`), English stemmer.
    pub fn analyzer(&self) -> Result<Analyzer, Error> {
        // SAFETY: root handle is live for as long as self.root.
        let handle = unsafe { ffi::RAGAnalyzer_Copy(self.root.handle) };
        if handle.is_null() {
            return Err(Error::Copy);
        }
        Ok(Analyzer {
            handle,
            root: Arc::clone(&self.root),
        })
    }
}

/// One analyzer instance (a copy of the root). `Send`, not `Sync`.
pub struct Analyzer {
    handle: ffi::RAGAnalyzerHandle,
    root: Arc<Root>,
}

// SAFETY: the instance's own state is only touched through &mut self; the state it
// shares with other copies is read-only except OpenCC, guarded in `tokenize`.
unsafe impl Send for Analyzer {}

impl Drop for Analyzer {
    fn drop(&mut self) {
        // SAFETY: handle came from RAGAnalyzer_Copy and is dropped once; the root
        // (owner of the borrowed dictionaries) is still alive via self.root.
        unsafe { ffi::RAGAnalyzer_Destroy(self.handle) }
    }
}

/// NUL-terminated copy of `s`, truncated at the first interior NUL (what the C side
/// would read from Go's `C.CString`).
fn c_string(s: &str) -> CString {
    let end = s.find('\0').unwrap_or(s.len());
    CString::new(&s[..end]).expect("no interior NUL after truncation")
}

/// Copy a malloc'ed C string into Rust and free it.
///
/// # Safety
/// `p` must be NULL or a NUL-terminated buffer allocated with malloc that the
/// caller owns.
unsafe fn take_c_string(p: *mut c_char) -> Option<String> {
    if p.is_null() {
        return None;
    }
    let s = CStr::from_ptr(p).to_string_lossy().into_owned();
    ffi::free(p.cast());
    Some(s)
}

impl Analyzer {
    /// `RAGAnalyzer_SetLanguage`: a Snowball language name (case-insensitive).
    /// Unknown names (including "Chinese") leave the current stemmer unchanged.
    pub fn set_language(&mut self, language: &str) {
        let c = c_string(language);
        // SAFETY: live handle, valid C string.
        unsafe { ffi::RAGAnalyzer_SetLanguage(self.handle, c.as_ptr()) }
    }

    /// `RAGAnalyzer_SetFineGrained`. The Go server never calls it (its wrapper is a
    /// no-op), so instances stay at the default `false`.
    pub fn set_fine_grained(&mut self, fine_grained: bool) {
        // SAFETY: live handle.
        unsafe { ffi::RAGAnalyzer_SetFineGrained(self.handle, fine_grained) }
    }

    /// `RAGAnalyzer_Tokenize`: space-separated coarse tokens.
    pub fn tokenize(&mut self, text: &str) -> Result<String, Error> {
        let c = c_string(text);
        let _guard = lock(&self.root.tokenize_lock);
        // SAFETY: live handle, valid C string; result is malloc'ed and ours.
        unsafe { take_c_string(ffi::RAGAnalyzer_Tokenize(self.handle, c.as_ptr())) }
            .ok_or(Error::NullResult("RAGAnalyzer_Tokenize"))
    }

    /// `RAGAnalyzer_FineGrainedTokenize` over space-separated tokens.
    pub fn fine_grained_tokenize(&mut self, tokens: &str) -> Result<String, Error> {
        let c = c_string(tokens);
        // SAFETY: live handle, valid C string; result is malloc'ed and ours.
        unsafe {
            take_c_string(ffi::RAGAnalyzer_FineGrainedTokenize(
                self.handle,
                c.as_ptr(),
            ))
        }
        .ok_or(Error::NullResult("RAGAnalyzer_FineGrainedTokenize"))
    }

    /// `RAGAnalyzer_GetTermFreq`; 0 when the term is absent.
    pub fn term_freq(&mut self, term: &str) -> i32 {
        let c = c_string(term);
        // SAFETY: live handle, valid C string.
        unsafe { ffi::RAGAnalyzer_GetTermFreq(self.handle, c.as_ptr()) }
    }

    /// `RAGAnalyzer_GetTermTag`; `""` when absent or untagged.
    pub fn term_tag(&mut self, term: &str) -> String {
        let c = c_string(term);
        // SAFETY: live handle, valid C string; result is NULL or malloc'ed and ours.
        unsafe { take_c_string(ffi::RAGAnalyzer_GetTermTag(self.handle, c.as_ptr())) }
            .unwrap_or_default()
    }
}

fn lock(m: &Mutex<()>) -> MutexGuard<'_, ()> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// A pool of [`Analyzer`] copies with the Go pool's per-acquire contract.
pub struct AnalyzerPool {
    dict: Dictionary,
    idle: Mutex<Vec<Analyzer>>,
}

impl AnalyzerPool {
    pub fn new(dict: Dictionary) -> Self {
        AnalyzerPool {
            dict,
            idle: Mutex::new(Vec::new()),
        }
    }

    /// Take an instance (reused or freshly copied) and set its language;
    /// `""` means [`DEFAULT_LANGUAGE`]. Returned to the pool on drop.
    ///
    /// Like Go, a language the stemmer does not know (e.g. "Chinese") leaves the
    /// instance's previous stemmer in place, so the result can depend on which
    /// language the reused instance last served.
    pub fn acquire(&self, language: &str) -> Result<Pooled<'_>, Error> {
        let reused = self.idle.lock().unwrap_or_else(|e| e.into_inner()).pop();
        let mut analyzer = match reused {
            Some(a) => a,
            None => self.dict.analyzer()?,
        };
        analyzer.set_language(if language.is_empty() {
            DEFAULT_LANGUAGE
        } else {
            language
        });
        Ok(Pooled {
            pool: self,
            analyzer: Some(analyzer),
        })
    }

    /// Instances currently idle in the pool.
    pub fn idle_len(&self) -> usize {
        self.idle.lock().unwrap_or_else(|e| e.into_inner()).len()
    }
}

/// An [`Analyzer`] on loan from an [`AnalyzerPool`].
pub struct Pooled<'a> {
    pool: &'a AnalyzerPool,
    analyzer: Option<Analyzer>,
}

impl Deref for Pooled<'_> {
    type Target = Analyzer;
    fn deref(&self) -> &Analyzer {
        self.analyzer.as_ref().expect("present until drop")
    }
}

impl DerefMut for Pooled<'_> {
    fn deref_mut(&mut self) -> &mut Analyzer {
        self.analyzer.as_mut().expect("present until drop")
    }
}

impl Drop for Pooled<'_> {
    fn drop(&mut self) {
        if let Some(a) = self.analyzer.take() {
            self.pool
                .idle
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(a);
        }
    }
}

/// Indexing-path tokenization of one text, as `tokenizeChunks` in
/// `internal/ingestion/component/tokenizer.go`: coarse tokens, then fine-grained
/// tokens of those; an empty result falls back to the input.
pub fn index_tokens(a: &mut Analyzer, text: &str) -> Result<(String, String), Error> {
    let mut ltks = a.tokenize(text)?;
    if ltks.is_empty() {
        ltks = text.to_string();
    }
    let mut sm = a.fine_grained_tokenize(&ltks)?;
    if sm.is_empty() {
        sm = ltks.clone();
    }
    Ok((ltks, sm))
}

/// Query-path tokenization: `tokenizer.Tokenize` and `tokenizer.FineGrainedTokenize`
/// with no fallbacks (callers in `internal/service/nlp/query_builder.go` decide).
pub fn query_tokens(a: &mut Analyzer, text: &str) -> Result<(String, String), Error> {
    let ltks = a.tokenize(text)?;
    let sm = a.fine_grained_tokenize(&ltks)?;
    Ok((ltks, sm))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c_string_truncates_at_nul() {
        assert_eq!(c_string("ab\0cd").as_bytes(), b"ab");
        assert_eq!(c_string("abc").as_bytes(), b"abc");
    }

    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    #[test]
    fn auto_traits() {
        assert_send::<Analyzer>();
        assert_send::<Dictionary>();
        assert_sync::<Dictionary>();
        assert_sync::<AnalyzerPool>();
    }

    /// Needs dictionaries: set RAGFLOW_DICT_PATH (skipped otherwise).
    #[test]
    fn tokenizes_with_dictionaries() {
        let Ok(path) = std::env::var("RAGFLOW_DICT_PATH") else {
            eprintln!("RAGFLOW_DICT_PATH unset; skipping");
            return;
        };
        let pool = AnalyzerPool::new(Dictionary::load_quietly(path).unwrap());
        let mut a = pool.acquire("").unwrap();
        let t = a.tokenize("Hello World 你好世界").unwrap();
        assert!(!t.is_empty());
        assert!(a.term_freq("世界") > 0);
        drop(a);
        assert_eq!(pool.idle_len(), 1);
    }
}
