---
why: ragflow-retrieval claims upstream's Go retrieval semantics; this harness runs both against the same live stores and diffs the results
lane: pull
verify: rust/retrieval-parity/run.sh CASES.jsonl OUT  (prints a PASS/FAIL table; exit 0 when every case passes)
invalidates-when: the upstream pin moves off v1.0.0-rc1, or upstream reshapes RetrievalTool, NLPRetrievalAdapter, nlp.RetrievalService or the ES engine's Search/KNNScores
---

# rust/retrieval-parity — Go vs Rust retrieval

Runs each case through:

- `oracle-go/`: upstream's Retrieval tool (`RetrievalTool.InvokableRun` →
  `NLPRetrievalAdapter` → `nlp.RetrievalService` → ES engine), compiled
  inside the upstream release tree;
- `ragflow-retrieval-parity run`: the Rust port through the runtime's
  Retrieval tool path;

against the same Elasticsearch and MySQL, with one recorded query embedding
per query given to both, so differences can only come from query building
and scoring. `compare` then checks, per case: chunk ids and order (exact),
`similarity` / `term_similarity` / `vector_similarity` and the envelope score
(within 1e-9), `formalized_content` (byte for byte), and reports whether the
ES request bodies match (filter arrays compared as sets).

## Run

    mkdir -p /tmp/upstream && git archive v1.0.0-rc1 build.sh go.mod go.sum internal conf \
      | tar -x -C /tmp/upstream
    docker build -f rust/docker/retrieval-parity.Dockerfile \
      --build-context upstream=/tmp/upstream -t ragflow-retrieval-parity .
    rust/retrieval-parity/run.sh cases.jsonl out/

`run.sh` documents its environment (ES and MySQL addresses and credentials,
the Docker network they are reachable from, the embedding endpoint used once
to record `out/vectors.json`). A case is one JSON line:

    {"name": "short_en", "args": {"query": "parking fee", "dataset_ids": ["<kb id>"],
                                  "top_k": 64, "keywords_similarity_weight": 0.3}}

Use `top_k >= 64` (the candidate window): with a smaller k the window's tail
is text-only hits at score 0, which ES orders by its own tiebreak.

## Read-only

oracle-go reaches Elasticsearch through an in-process proxy that forwards
only `GET /` and `POST */_search` and answers the rest locally (the engine
constructor's index-template PUTs). The Rust side only issues `_search`.
Both only `SELECT` from MySQL.

## What it does not cover

- Rerank models, `meta_data_filter`, `cross_languages`, `toc_enhance`, tag
  features and GraphRAG: the Rust adapter refuses them; oracle-go's enhancer
  refuses the LLM-backed ones.
- Deleted-document pruning needs a document missing from MySQL while its
  chunks remain in ES; the harness never writes, so that path is covered by a
  unit test (`service::tests::prune_rebuilds_chunks_without_ids`).
- The server's working directory has no `rag/res/{ner.json,synonym.json}`
  in the official image; both sides run without them. A deployment that
  ships them would need `res_dir` set on the Rust side.
