//! Links the prebuilt `librag_tokenizer_c_api.a` that upstream's `build.sh --cpp`
//! produces in `internal/binding/cpp/cmake-build-release/`.
//!
//! The archive is not built here: it must come from the same toolchain upstream
//! uses (see `rust/docker/analyzer.Dockerfile`). Inputs:
//!
//! - `RAG_ANALYZER_LIB_DIR` — directory holding `librag_tokenizer_c_api.a`.
//!   Defaults to upstream's build directory relative to this crate.
//! - `RAG_ANALYZER_PCRE2_LIB` — optional full path to `libpcre2-8.a`. When unset,
//!   `pcre2-8` is resolved by the linker's search path, which is what upstream's
//!   CMake (bare `libpcre2-8.a`) and cgo (`-lpcre2-8`) rely on.
//!
//! The link line mirrors cgo's in `internal/binding/rag_analyzer.go`:
//! `librag_tokenizer_c_api.a -lstdc++ -lm -lpthread -lpcre2-8`.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=RAG_ANALYZER_LIB_DIR");
    println!("cargo:rerun-if-env-changed=RAG_ANALYZER_PCRE2_LIB");
    println!("cargo:rerun-if-env-changed=DOCS_RS");

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let lib_dir = env::var_os("RAG_ANALYZER_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest.join("../../../internal/binding/cpp/cmake-build-release"));
    let archive = lib_dir.join("librag_tokenizer_c_api.a");

    if !archive.exists() {
        // `cargo check`/clippy on a machine without the archive: allow type-checking,
        // fail only when something actually links.
        println!(
            "cargo:warning=ragflow-analyzer: {} not found; set RAG_ANALYZER_LIB_DIR \
             (build it with upstream `./build.sh --cpp`). Linking will fail.",
            archive.display()
        );
    }
    println!("cargo:rerun-if-changed={}", archive.display());
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=rag_tokenizer_c_api");

    match env::var_os("RAG_ANALYZER_PCRE2_LIB") {
        Some(p) => {
            let p = PathBuf::from(p);
            let dir = p.parent().expect("RAG_ANALYZER_PCRE2_LIB has no parent");
            println!("cargo:rustc-link-search=native={}", dir.display());
            println!("cargo:rustc-link-lib=static=pcre2-8");
        }
        None => println!("cargo:rustc-link-lib=pcre2-8"),
    }
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=pthread");
}
