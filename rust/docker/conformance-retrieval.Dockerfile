# syntax=docker/dockerfile:1
#
# Retrieval conformance image: the Rust retrieval (ragflow-retrieval-conformance)
# and, for cutting a contract version, upstream's Go retrieval (oracle-go,
# built inside the upstream tree the contract version names), both linking
# the same C++ analyzer archive, plus that tree's dictionaries.
#
# Build from the repository root with the contract version's upstream tree
# as a named context (rust/conformance/cut.sh does this):
#
#   mkdir -p /tmp/upstream && git archive <upstream ref> build.sh go.mod go.sum internal conf \
#     | tar -x -C /tmp/upstream
#   docker build -f rust/docker/conformance-retrieval.Dockerfile \
#     --build-context upstream=/tmp/upstream -t ragflow-conformance-retrieval .
#
# Run it with rust/conformance/run.sh retrieval (check) or cut.sh (record).

ARG BUILDER_IMAGE=infiniflow/github_action_runner:latest@sha256:b712330a93b685625e4e4e0f3092b255eb96db9120a01ebfd3c392f7e644d47c
ARG RAGFLOW_IMAGE=infiniflow/ragflow:v1.0.0-rc1@sha256:31ccf0336d8910f6d90ac7d4fff766cc0483ec10042b906a377f27bfe09f818b
ARG RUST_IMAGE=rust:1.90.0-slim-bookworm@sha256:64232e656c058f4468e8d024e990acff04f0fd5a5c0a88a574dc37773d7325c9
ARG RUNTIME_IMAGE=ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55

FROM ${RUST_IMAGE} AS rust-toolchain
FROM ${RAGFLOW_IMAGE} AS ragflow-dicts

FROM ${BUILDER_IMAGE} AS builder
USER root
SHELL ["/bin/bash", "-c"]
WORKDIR /upstream
COPY --from=upstream . .
# The analyzer archive, exactly as upstream builds it.
RUN git config --global safe.directory "*" && ./build.sh --cpp
# The builder image ships only the runtime libpcre2-8.so.0; cgo and rustc
# both link -lpcre2-8.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpcre2-dev \
    && rm -rf /var/lib/apt/lists/*

# Go half: oracle-go compiled as part of the upstream module.
COPY rust/conformance/oracle-go/retrieval rust/conformance/oracle-go/retrieval
ENV GOMODCACHE=/root/.cache/gomod GOCACHE=/root/.cache/gobuild GOTOOLCHAIN=local
RUN --mount=type=cache,id=ragflow-conformance-gomod,target=/root/.cache/gomod,sharing=locked \
    --mount=type=cache,id=ragflow-conformance-gobuild,target=/root/.cache/gobuild \
    GOPROXY=https://proxy.golang.org,direct CGO_ENABLED=1 \
    go build -o /out/oracle-go ./rust/conformance/oracle-go/retrieval

# Rust half, linking the archive built above.
COPY --from=rust-toolchain /usr/local/cargo /usr/local/cargo
COPY --from=rust-toolchain /usr/local/rustup /usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:${PATH} \
    RAG_ANALYZER_LIB_DIR=/upstream/internal/binding/cpp/cmake-build-release \
    RAGFLOW_DICT_PATH=/usr/share/infinity/resource
COPY --from=ragflow-dicts /usr/share/infinity/resource /usr/share/infinity/resource
COPY rust /rust
WORKDIR /rust
RUN --mount=type=cache,id=ragflow-conformance-cargo,target=/usr/local/cargo/registry \
    cargo build --release --locked -p ragflow-retrieval --features live \
    && cargo test --release --locked -p ragflow-analyzer -p ragflow-retrieval \
    && cp target/release/ragflow-retrieval-conformance /out/

FROM ${RUNTIME_IMAGE}
COPY --from=ragflow-dicts /usr/share/infinity/resource /usr/share/infinity/resource
COPY --from=builder /out/oracle-go /oracle/bin/oracle-go
COPY --from=builder /upstream/conf/mapping.json /upstream/conf/doc_meta_es_mapping.json /oracle/conf/
COPY --from=builder /out/ragflow-retrieval-conformance /usr/local/bin/ragflow-retrieval-conformance
ENV RAGFLOW_DICT_PATH=/usr/share/infinity/resource \
    RAG_PROJECT_BASE=/oracle
# An empty working directory: like the server's /ragflow in the official
# image, it has no rag/res/{ner.json,term.freq,synonym.json}.
WORKDIR /work
USER 65534:65534
