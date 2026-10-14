//! Raw declarations for `internal/binding/cpp/rag_analyzer_c_api.h`.
//!
//! Only the subset the Go server calls is declared. Ownership rules, from the
//! header and `rag_analyzer_c_api.cpp`:
//!
//! - `RAGAnalyzer_Tokenize`, `RAGAnalyzer_FineGrainedTokenize` and
//!   `RAGAnalyzer_GetTermTag` return a `malloc`ed, NUL-terminated buffer the
//!   caller releases with libc `free` (Go: `C.free`).
//! - `RAGAnalyzer_Copy` returns a handle whose dictionaries (trie, POS table,
//!   WordNet, OpenCC) are borrowed from the source handle's root owner; the
//!   owner must outlive every copy.

use std::os::raw::{c_char, c_int, c_void};

pub type RAGAnalyzerHandle = *mut c_void;

extern "C" {
    pub fn RAGAnalyzer_Create(path: *const c_char) -> RAGAnalyzerHandle;
    pub fn RAGAnalyzer_Destroy(handle: RAGAnalyzerHandle);
    pub fn RAGAnalyzer_Load(handle: RAGAnalyzerHandle) -> c_int;
    pub fn RAGAnalyzer_SetFineGrained(handle: RAGAnalyzerHandle, fine_grained: bool);
    pub fn RAGAnalyzer_SetLanguage(handle: RAGAnalyzerHandle, language: *const c_char);
    pub fn RAGAnalyzer_Tokenize(handle: RAGAnalyzerHandle, text: *const c_char) -> *mut c_char;
    pub fn RAGAnalyzer_FineGrainedTokenize(
        handle: RAGAnalyzerHandle,
        tokens: *const c_char,
    ) -> *mut c_char;
    pub fn RAGAnalyzer_GetTermFreq(handle: RAGAnalyzerHandle, term: *const c_char) -> i32;
    pub fn RAGAnalyzer_GetTermTag(handle: RAGAnalyzerHandle, term: *const c_char) -> *mut c_char;
    pub fn RAGAnalyzer_Copy(handle: RAGAnalyzerHandle) -> RAGAnalyzerHandle;
}

// libc, declared directly to avoid a dependency for three symbols.
extern "C" {
    pub fn free(ptr: *mut c_void);
    pub fn fflush(stream: *mut c_void) -> c_int;
    pub fn dup(fd: c_int) -> c_int;
    pub fn dup2(oldfd: c_int, newfd: c_int) -> c_int;
    pub fn close(fd: c_int) -> c_int;
}
