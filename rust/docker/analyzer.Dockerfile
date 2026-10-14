# syntax=docker/dockerfile:1
#
# ragflow-analyzer: Rust over RAGFlow's C++ text analyzer, via FFI.
#
# Build from the repository root (the context needs build.sh,
# internal/binding/cpp and rust/; see analyzer.Dockerfile.dockerignore):
#
#   docker build -f rust/docker/analyzer.Dockerfile -t ragflow-analyzer .
#   echo '你好世界 Hello World' | docker run --rm -i ragflow-analyzer
#
# Every base is pinned by digest. To re-pin, resolve each tag with
# `docker buildx imagetools inspect <tag>` and replace the digest.

# Upstream's Go builder image (Dockerfile, stage go-builder), linux/amd64 only.
ARG BUILDER_IMAGE=infiniflow/github_action_runner:latest@sha256:b712330a93b685625e4e4e0f3092b255eb96db9120a01ebfd3c392f7e644d47c
# The release whose analyzer dictionaries are copied (/usr/share/infinity/resource).
ARG RAGFLOW_IMAGE=infiniflow/ragflow:v1.0.0-rc1@sha256:31ccf0336d8910f6d90ac7d4fff766cc0483ec10042b906a377f27bfe09f818b
# Source of the Rust toolchain only (copied into the builder).
ARG RUST_IMAGE=rust:1.90.0-slim-bookworm@sha256:64232e656c058f4468e8d024e990acff04f0fd5a5c0a88a574dc37773d7325c9
# Runtime: the builder's distribution (Ubuntu 24.04), so glibc, libstdc++ and
# libpcre2-8 match what the archive and binary were linked against.
ARG RUNTIME_IMAGE=ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55

FROM ${RUST_IMAGE} AS rust-toolchain
FROM ${RAGFLOW_IMAGE} AS ragflow-dicts

# ── (a) C++ archive, built exactly as upstream: `./build.sh --cpp` ──
# (cmake -DCMAKE_BUILD_TYPE=Release; make rag_tokenizer_c_api; re2 symbol rename)
FROM ${BUILDER_IMAGE} AS cpp-builder
USER root
SHELL ["/bin/bash", "-c"]
WORKDIR /ragflow
# build.sh reads dependency versions from go.mod at startup, even for --cpp.
COPY build.sh go.mod ./
COPY internal/binding/cpp internal/binding/cpp
RUN git config --global safe.directory "*" && ./build.sh --cpp \
    && ls -l internal/binding/cpp/cmake-build-release/librag_tokenizer_c_api.a

# ── (b) Rust: link the archive through build.rs ──
FROM cpp-builder AS rust-builder
COPY --from=rust-toolchain /usr/local/cargo /usr/local/cargo
COPY --from=rust-toolchain /usr/local/rustup /usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:${PATH} \
    RAG_ANALYZER_LIB_DIR=/ragflow/internal/binding/cpp/cmake-build-release \
    RAGFLOW_DICT_PATH=/usr/share/infinity/resource
# The archive leaves pcre2 to the final link (cgo: -lpcre2-8). The builder image
# ships only the runtime libpcre2-8.so.0, so add the development symlink/headers.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpcre2-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ragflow-dicts /usr/share/infinity/resource /usr/share/infinity/resource
COPY rust rust
WORKDIR /ragflow/rust
RUN --mount=type=cache,id=ragflow-analyzer-cargo,target=/usr/local/cargo/registry \
    cargo build --release --locked -p ragflow-analyzer \
    && cargo test --release --locked -p ragflow-analyzer

# ── (c) runtime: binary + dictionaries ──
FROM ${RUNTIME_IMAGE}
COPY --from=ragflow-dicts /usr/share/infinity/resource /usr/share/infinity/resource
COPY --from=rust-builder /ragflow/rust/target/release/ragflow-analyzer /usr/local/bin/ragflow-analyzer
ENV RAGFLOW_DICT_PATH=/usr/share/infinity/resource
USER 65534:65534
ENTRYPOINT ["/usr/local/bin/ragflow-analyzer"]
