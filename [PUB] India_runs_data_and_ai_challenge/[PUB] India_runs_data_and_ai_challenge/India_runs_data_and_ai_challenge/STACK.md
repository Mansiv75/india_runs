# STACK.md — Technology choices for the Redrob ranker

Companion to `DESIGN.md`. Every choice here is justified against the hard constraints in `CLAUDE.md` (H1-H6). Read the firewall rule first: the OFFLINE precompute path may use anything (network, local models, unbounded time); the ONLINE `rank.py` path is CPU-only, network-off, <= 5 min, <= 16 GB RAM. The stack is split to enforce that physically, not by discipline alone.

---

## 0. The one decision that shapes everything: two dependency sets

`rank.py` must never import a heavy/networked library. We enforce this with **two pinned requirement files**, not one:

- `requirements.txt` — the ONLINE/runtime set. Small, pure-CPU, no torch, no transformers, no HTTP client. This is what the Stage-3 sandbox installs and times.
- `requirements-precompute.txt` — the OFFLINE set. Inherits runtime + adds the embedding model stack and extraction helpers.

If `rank.py` ever needs something only in the precompute set, that is a design smell and a potential H1 violation — stop and ask.

---

## 1. Language & runtime

| Concern | Choice | Why |
|---|---|---|
| Language | **Python 3.11** (pinned) | Mature CPU wheels for every dep below; avoids bleeding-edge 3.12/3.13 wheel gaps for sentence-transformers/torch. |
| Env / install | **uv** (locked) | Fast, lockable, reproducible installs (H6). `requirements*.txt` committed regardless so reviewers need no extra tooling. |
| Determinism | pinned `--as-of` date, fixed seeds, pinned model revision | No `Date.now()`/random in the scored path; reproducibility is H6. |

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
| Dense embeddings | **`sentence-transformers`**; model **OPEN** (`bge-small-en-v1.5` vs `e5-small-v2`) | CPU-fast 384-d models, encoded offline -> `emb.npy`, `jd_emb.npy` (float32, L2-normalized). `torch` (CPU wheel) lives here, OFFLINE only. Model choice deferred — decide during indexing build, then pin the exact revision for reproducibility. |
| Lexical index | **`bm25s`** | numpy/scipy-sparse BM25 — far faster than pure-python `rank_bm25` for scoring 100K at online time; serializes to `bm25.pkl`. |

Embeddings are precomputed to `.npy`; **the runtime never loads torch or the model** — it does cosine in numpy.

---

## 5. ONLINE — `rank.py` (the only timed step)

| Concern | Choice | Why |
|---|---|---|
| Numerics / scoring | **`numpy`** | Vectorized rerank, RRF fusion, cosine over the shortlist. 100K x 384 float32 cosine is milliseconds on CPU. |
| Sparse / BM25 query | **`scipy`** (via `bm25s`) | Lexical leg of hybrid retrieve. |
| ANN index | **none (brute-force numpy)** | At 100K, exact cosine beats the complexity/dependency cost of FAISS. Note FAISS as the scale-out path for 200K+, but not used in v1 — keeps the runtime lean and H1-safe. |
| Config load | **`PyYAML`** (`safe_load`) | Reads `jd_rubric.yaml`, `weights.yaml`. |
| Reasoning generation | **pure-Python clause assembly** (no Jinja, no LLM) | Deterministic, field-sourced, varied by which features fired (DESIGN 11, H4). No runtime model — H1 + anti-hallucination. |
| CSV write | **stdlib `csv`** (UTF-8, explicit quoting) | Precise control over the exact H3 format; avoids pandas formatting surprises. |
| Final gate | invoke **`validate_submission.py`** | Last step of every run; fail loudly on any error (H3). |

Runtime memory check: `emb.npy` ~147 MB (100K x 384 x 4B) + flat features parquet + sparse BM25 index — comfortably under 16 GB.

---

## 6. OFFLINE — eval harness

| Concern | Choice | Why |
|---|---|---|
| Ranking metrics | **hand-rolled numpy** (NDCG@10/@50, MAP, P@K) | Small, transparent, matches the exact scoring formula in CLAUDE.md; no heavy dep. `scikit-learn` only as an optional cross-check, never in runtime. |
| Weight tuning output | emits **`weights.yaml`** | Closes the loop drawn in DESIGN 2: gold-set scoring + ablations produce the tuned `w1..w10` + availability floor that `rank.py` loads. |

---

## 7. Delivery & sandbox (H5/H6)

| Concern | Choice | Why |
|---|---|---|
| Sandbox | **HF Spaces + Gradio** | Free CPU tier; runs the ranker on a <= 100-candidate sample within budget. Gradio chosen: native HF Spaces fit, minimal boilerplate for a single input -> output (run ranker -> show top-100 table) demo. |
| Lint / format | **`ruff`** | Fast, single tool; optional but keeps the repo clean for review. |
| Tests | **`pytest`** | Gold-set ordering, H3 invariants, honeypot-rate assertion, validator pass. |

---

## 8. Deliberately NOT in the stack

- **No runtime LLM / hosted API** anywhere in `rank.py` (H1 network-off; H4 hallucination).
- **No GPU / CUDA** dependency (H1 CPU-only).
- **No FAISS/Milvus/vector DB** in v1 — exact numpy cosine suffices at 100K.
- **No torch/transformers in `requirements.txt`** — OFFLINE-only.

---

## 9. Decision log

| Decision | Status | Notes |
|---|---|---|
| Install path | **Locked: uv** | Commit `requirements*.txt` too. |
| Dataframe | **Locked: polars, pandas fallback** | Switch to pandas only if polars raises dev complexity; parquet contract unchanged. |
| Embedding model | **OPEN** | `bge-small-en-v1.5` vs `e5-small-v2`; decide at indexing build, pin revision. |
| Sandbox framework | **Locked: Gradio** | On HF Spaces; native fit + minimal boilerplate for the input -> output demo. |

> Per CLAUDE.md: if any pin or addition risks pulling network/torch into the timed path, stop and ask before committing it.
