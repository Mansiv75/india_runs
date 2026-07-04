# STACK.md — Technology choices for the Redrob ranker

Companion to `DESIGN.md`. The project is now being built as a production-grade learning product rather than a one-off hackathon submission. The stack is split by responsibility: application/API runtime, indexing and embedding jobs, evaluation, and local infrastructure.

---

## 0. The one decision that shapes everything: dependency groups

Keep dependencies separated so the app stays understandable and deployable:

- `requirements.txt` — application/runtime set: API server, Qdrant client, numeric scoring, config loading, CSV export.
- `requirements-indexing.txt` — indexing and embedding jobs: model stack, feature extraction helpers, parquet tooling.
- `requirements-dev.txt` — tests, linting, type checks, notebooks or exploration helpers if needed.

The ranking API should not directly load heavyweight embedding models unless we intentionally add an online embedding service. Indexing and model execution stay behind explicit service or job boundaries.

---

## 1. Language & runtime

| Concern | Choice | Why |
|---|---|---|
| Language | **Python 3.11** (pinned) | Mature CPU wheels for every dep below; avoids bleeding-edge 3.12/3.13 wheel gaps for sentence-transformers/torch. |
| Env / install | **uv** (locked) | Fast, lockable, reproducible installs. `requirements*.txt` committed regardless so the project remains easy to run. |
| Determinism | pinned `--as-of` date, fixed seeds, pinned model revision | Ranking experiments need repeatability; production also needs model/config versioning. |

---

## 2. OFFLINE — ingestion & normalization

| Concern | Choice | Why |
|---|---|---|
| Read `candidates.jsonl.gz` | stdlib `gzip` (text mode) + **`orjson`** | Stream line-by-line, never materialize 100K raw records in RAM (DESIGN 5). orjson parses ~2-3x faster than stdlib json. |
| Schema validation | **`fastjsonschema`** against `candidate_schema.json` | Compiles the provided schema once, validates 100K rows fast; failures go to `quarantine.jsonl` instead of crashing the batch. Using the given schema directly avoids model drift. |
| Dataframe / compute | **`polars`** (fall back to `pandas` if it raises dev complexity) | Lower memory + faster than pandas at 100K with nested lists (career_history, skills). Decision: start with polars; if nested-list handling or team familiarity slows the build, switch to pandas — output parquet contract is identical either way. |
| Parquet IO | **`pyarrow`** | Backing engine for `normalized.parquet` / `features.parquet`; columnar, typed, compact. |

---

## 3. OFFLINE — feature extraction (the core IP)

| Concern | Choice | Why |
|---|---|---|
| Text mining over descriptions | **regex + curated YAML lexicons** | `shipped_ranking_evidence`, `domain_score`, company-type all key off verb/object/scale cues and curated term lists (DESIGN 6.2). Deterministic, inspectable, defensible — no model needed. |
| Fuzzy lexicon / company matching | **`rapidfuzz`** | Fast C++ fuzzy match for the services/consulting lexicon and company-type classification; tolerant of spelling/format variance. |
| Lemmatization (only if needed) | `spaCy en_core_web_sm` — **deferred** | Add only if regex lexicons prove too brittle. Extra model download; keep out unless eval shows we need it. |

All feature outputs are plain numeric/boolean columns in `features.parquet` so the online path is pure arithmetic.

---

## 4. OFFLINE — representation & indexing

| Concern | Choice | Why |
|---|---|---|
| Dense embeddings | **`sentence-transformers`** + **EmbeddingGemma-300m** (locked) | 300M-param on-device embedding model, encoded OFFLINE only; **Matryoshka-truncated to 256-dim** (float32, L2-normalized) to keep storage/search near small-model cost — raise toward 768 only if eval shows recall needs it. `torch` (CPU wheel) lives here, OFFLINE only. **Apply the model's task-prompt prefixes** identically to candidates and the JD; pin the revision (Gemma license acceptance required) for reproducibility. |
| Vector store | **Qdrant** | Production-oriented vector database with payload filters, named vectors, HNSW search, Docker-friendly local development, and a clear path to managed or self-hosted deployment. |
| Lexical index | **`bm25s`** | numpy/scipy-sparse BM25 — far faster than pure-python `rank_bm25` for scoring 100K at online time; serializes to `bm25.pkl`. |

Vectors are precomputed into a Qdrant collection; **the ranking runtime does not need to load torch or the embedding model** for indexed candidate search. For arbitrary user-supplied job descriptions, embedding can be handled by an explicit embedding service/job rather than hidden inside the ranker.

---

## 5. ONLINE — ranking service / CLI

| Concern | Choice | Why |
|---|---|---|
| API layer | **FastAPI** | Simple Python service boundary for ranking requests, health checks, experiment endpoints, and future UI integration. |
| Numerics / scoring | **`numpy`** | Vectorized rerank, RRF fusion, and shortlist scoring. |
| Sparse / BM25 query | **`scipy`** (via `bm25s`) | Lexical leg of hybrid retrieve. |
| Vector retrieval | **`qdrant-client` + Qdrant service** | Qdrant owns vector search, payload filters, collection schema, and scalable nearest-neighbor retrieval. Local dev runs Qdrant in Docker; production can use managed or self-hosted Qdrant. |
| Config load | **`PyYAML`** (`safe_load`) | Reads `jd_rubric.yaml`, `weights.yaml`. |
| Reasoning generation | **pure-Python clause assembly first** | Deterministic, field-sourced, varied by which features fired (DESIGN 11). Keeps explanations grounded before introducing any LLM summarization. |
| CSV write | **stdlib `csv`** (UTF-8, explicit quoting) | Precise control for batch exports and compatibility with the original submission format. |
| Validation | **pytest + custom QA checks** | Schema, deterministic ranking, grounded reasoning, retrieval recall, and regression checks. |

Runtime memory target: keep feature tables and BM25 artifacts small enough for local development; Qdrant owns vector storage and search.

---

## 6. Evaluation harness

| Concern | Choice | Why |
|---|---|---|
| Ranking metrics | **hand-rolled numpy** (NDCG@K, MAP, P@K, recall@K) | Small, transparent, and easy to reason about while learning. `scikit-learn` can be an optional cross-check. |
| Weight tuning output | emits **`weights.yaml`** | Closes the loop drawn in DESIGN 2: gold-set scoring + ablations produce the tuned `w1..w10` + availability floor that the ranker loads. |

---

## 7. Delivery & local development

| Concern | Choice | Why |
|---|---|---|
| Local infra | **Docker Compose** | Runs Qdrant and any app dependencies consistently while learning and developing. |
| API service | **FastAPI + Uvicorn** | Straightforward Python service for ranking endpoints and future UI integration. |
| UI | **deferred** | Start with API/CLI and inspectable outputs. Add Streamlit, Gradio, or a web frontend after ranking quality is observable. |
| Lint / format | **`ruff`** | Fast, single tool; optional but keeps the repo clean for review. |
| Tests | **`pytest`** | Gold-set ordering, ranking invariants, honeypot-style assertions, Qdrant indexing smoke tests, and API tests. |

---

## 8. Deliberately NOT in the stack

- **No LLM-generated candidate claims** in reasoning until we have grounding checks strong enough to prevent hallucination.
- **No GPU requirement** for the baseline product; CPU should be enough for local learning and small indexing jobs.
- **No embedded vector DB as the primary store**. Qdrant is the vector system of record.
- **No torch/transformers in the API runtime by default**. Keep model execution in indexing jobs or a dedicated embedding service.

---

## 9. Decision log

| Decision | Status | Notes |
|---|---|---|
| Install path | **Locked: uv** | Commit `requirements*.txt` too. |
| Dataframe | **Locked: polars, pandas fallback** | Switch to pandas only if polars raises dev complexity; parquet contract unchanged. |
| Embedding model | **Starting point: EmbeddingGemma-300m @ 256-dim (MRL)** | Local encode path for learning; 768-dim is the recall lever if eval needs it. |
| API framework | **Locked: FastAPI** | Simple service boundary and good fit for Python ranking logic. |
| Vector store / search | **Locked: Qdrant** | Primary vector DB for local Docker and future production deployment. Stores vectors with candidate payloads and supports filtered vector search. |
