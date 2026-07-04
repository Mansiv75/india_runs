# DAY1_PLAN.md - Low-level implementation plan (Day 1)

Goal for end of Day 1: repo scaffolded; extraction contract + prompt finalized and
validated on the 50 `sample_candidates.json` profiles; the resumable full-pool batch
LAUNCHED (running overnight); embedding pass running; LanceDB store schema defined and
smoke-tested on the 50-sample. Nothing in the timed `rank.py` path is touched today.

Decision baked in (your call): a single embedded LanceDB dataset stores BOTH the
dense embeddings AND the extracted feature columns + the raw fields `rank.py` needs.
Feature source: OFFLINE LLM over the FULL 100K pool. `rank.py` stays pure numpy/LanceDB,
network OFF.

> H1 watchpoint carried from CLAUDE.md: LanceDB ships usage telemetry. We disable it
> (env var, see 7) and the budget-check step verifies `rank.py` makes zero network
> calls. This is a hard requirement, not optional.

---

## 0. Repo scaffold (30 min)

```
precompute/
  __init__.py
  config.py            # paths, model id, as-of date, dims, shard size, telemetry-off
  io_utils.py          # stream candidates.jsonl, shard, checkpoint manifest
  schema.py            # the extraction JSON schema (the core IP) + a python validator
  prompt.py            # SYSTEM_PROMPT (rubric+instructions+schema+few-shot), build_user(candidate)
  extract_llm.py       # full-pool Batch API extraction, resumable
  embed.py             # offline embedding pass -> vectors
  build_store.py       # write LanceDB: vectors + feature columns + raw fields
  validate_extract.py  # sanity checks over extracted features (honeypot rate, coverage)
rank.py                # STUB only today (the timed step; build Day 2)
artifacts/             # gitignored big files; small ones committed
  raw_extractions/     # one .jsonl per shard, custom_id=candidate_id  (cache of LLM output)
  manifest.json        # which candidate_ids are done (resume state)
  vectors.lance/       # the unified LanceDB dataset
eval/
  gold.jsonl           # ~30-40 hand-labeled cases (build Day 3, stub today)
requirements.txt           # runtime: lancedb, numpy, pyyaml  (NO torch, NO LLM SDK, NO http client)
requirements-precompute.txt# offline: google-genai, sentence-transformers, torch(cpu), orjson, lancedb, rapidfuzz
.env.example               # GEMINI_API_KEY=   (a.k.a. GOOGLE_API_KEY)
README.md                  # reproduce commands (fill in as we go)
```

`.gitignore`: `artifacts/raw_extractions/`, `candidates.jsonl`, `*.lance/` if large; commit
`manifest.json` and a regenerator note per H6.

---

## 1. config.py

```python
from pathlib import Path

AS_OF_DATE = "2026-06-27"          # pinned for reproducibility (no Date.now in scored path)
CANDIDATES = Path("[PUB] India_runs_data_and_ai_challenge/.../candidates.jsonl")  # real path
ARTIFACTS = Path("artifacts")
RAW_DIR = ARTIFACTS / "raw_extractions"
MANIFEST = ARTIFACTS / "manifest.json"
LANCE_DIR = ARTIFACTS / "vectors.lance"

EXTRACT_MODEL = "gemini-2.5-flash"  # offline extraction provider = Google Gemini (user choice).
                                    # VERIFY current model id + pricing vs Google docs before launch;
                                    # flash-lite is cheaper, 2.5-pro is the quality lever.
SHARD_SIZE = 15000                  # candidates per batch (keeps each batch < 256MB / 100k-req cap)
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # 384-dim; swap to EmbeddingGemma only if eval shows recall gap
EMBED_DIM = 384

# H1: kill LanceDB telemetry BEFORE importing lancedb anywhere.
import os
os.environ["LANCE_TELEMETRY"] = "0"
os.environ["DO_NOT_TRACK"] = "1"
```

---

## 2. io_utils.py - streaming + resumability

- `iter_candidates()`: open `candidates.jsonl` in text mode, `orjson.loads` per line, yield
  `(candidate_id, record)`. Never materialize 100K records (DESIGN 5).
- `load_manifest() -> set[str]`: read `manifest.json` -> set of done candidate_ids (empty if absent).
- `append_results(shard_idx, results: list[dict])`: write/append to `raw_extractions/shard_{idx}.jsonl`,
  then update manifest atomically (write tmp, os.replace).
- `pending_candidates() -> iterator`: `iter_candidates()` filtered by `id not in manifest`.

Resume contract: a crash at candidate 90K loses nothing - rerun `extract_llm.py`, it
skips everything already in the manifest and submits only the remainder.

---

## 3. schema.py - the extraction contract (CORE IP)

Use enums everywhere (the structured-output JSON schema does NOT support numeric min/max or
string length - only types, enum, const, anyOf, $ref; every object needs
`additionalProperties: false` + `required`). Ordinal judgments become enums, not 0-1 floats.

Per-candidate object (sketch - each nested object: additionalProperties false, all keys required):

```
candidate_id: string
musthaves:
  embedding_retrieval_prod      {level: [none, mentioned, hands_on, production], evidence: string}
  vector_db_or_hybrid_search    {level: [...same...], evidence: string}
  ranking_eval_experience       {level: [none, mentioned, used_ndcg_mrr_map, ab_testing], evidence: string}
  shipped_ranking_search_recsys {shipped: bool, scale: [none, small, meaningful, large_scale], evidence: string}
  python_depth                  {level: [none, basic, strong, expert], evidence: string}
career:
  dominant_company_type         [product, services_consulting, mixed, unknown]
  applied_ml_years_at_product   [none, under_2y, 2_to_4y, 4_plus_y]
  production_vs_research        [production, mixed, research_only]
  recent_hands_on_coding        {recent_18mo: bool, evidence: string}
  pre_llm_ml_experience         {present: bool, evidence: string}
  domain                        [nlp_ir, cv, speech, robotics, general_ml, data_eng, other]
disqualifiers (each: {flag: bool, confidence: [low, medium, high], evidence: string}):
  pure_research_no_prod, recent_langchain_only, no_code_18mo, title_chaser,
  pure_consulting_career, cv_speech_robotics_only, closed_source_no_validation
nice_to_haves: {llm_finetuning, learning_to_rank, hrtech_marketplace, distributed_systems, oss: all bool}
external_validation             {present: bool, kind: [none, oss, papers, talks, multiple], evidence: string}
consistency:
  honeypot_suspected            {flag: bool, confidence: [low, medium, high]}
  issues: [ {type: [tenure_exceeds_company, expert_zero_months, skills_exceed_experience,
                    assessment_contradicts_proficiency, duration_sum_mismatch, impossible_dates, other],
             detail: string} ]
  keyword_stuffing              {flag: bool, evidence: string}
grounded_notes:
  strengths: [string]   # each must be sourced from a populated field (feeds H4 reasoning)
  gaps: [string]
```

`evidence` fields force the model to quote from the record - our anti-hallucination audit hook (H4)
and the raw material the Day 2 reasoning generator will draw on.

Also in schema.py: `validate(obj) -> list[str]` - a plain python check (enums in range, required
keys present) so a malformed LLM row goes to a quarantine list instead of poisoning the store.

---

## 4. prompt.py - prompt design (cached prefix)

- `SYSTEM_PROMPT` (static, cached): role + the JD rubric (must-haves / 7 disqualifiers / context
  prefs, lifted from `job_description.txt`) + extraction instructions + grounding rules
  ("use ONLY the provided JSON; if absent, mark none/unknown; quote evidence verbatim; do not infer
  skills not present") + 2 short few-shot examples (one clear-fit, one keyword-stuffer/honeypot).
  Target length >= 4,096 tokens so Haiku prompt-caching engages (count with
  `client.messages.count_tokens`, NOT tiktoken).
- `build_user(record) -> str`: compact JSON of the candidate (`orjson`, no whitespace).
- Request shape (Gemini): `config={"response_mime_type": "application/json",
  "response_schema": EXTRACTION_SCHEMA, "temperature": 0, "system_instruction": SYSTEM_PROMPT}`.
  Gemini structured output accepts a Pydantic model or an OpenAPI-subset dict; our all-enum design
  maps cleanly (use `enum` + string type). Gemini has implicit context caching on repeated prefixes
  (and explicit cached-content for the rubric if we want guaranteed cache hits).

---

## 5. extract_llm.py - full-pool Batch extraction (resumable, the long pole)

```python
from google import genai          # the current unified SDK (`pip install google-genai`)
client = genai.Client()           # reads GEMINI_API_KEY / GOOGLE_API_KEY

# Two execution modes - pick per blocker answer:
# (A) Gemini Batch Mode (cost discount, async): client.batches.create(model=EXTRACT_MODEL,
#     src=[{ "custom_id": cid, "request": {...generate_content args...} }, ...]); poll; collect.
# (B) Concurrent live calls with a bounded thread/async pool + retry/backoff - simpler, fine at 100K
#     if rate limits allow; AI Studio free tier has generous quota for Flash.

# Per request: client.models.generate_content(model=EXTRACT_MODEL, contents=build_user(rec),
#   config={"system_instruction": SYSTEM_PROMPT, "response_mime_type":"application/json",
#           "response_schema": EXTRACTION_SCHEMA, "temperature": 0})
# Parse response.text as JSON -> schema.validate() -> append_results() -> update manifest.
# Results keyed by candidate_id (batch results may arrive unordered).
# On errors/empty: collect failed ids, resubmit in a follow-up pass.
```

Notes:
- Verify current Gemini model id, Batch Mode availability, and pricing against Google docs before
  launch (my training may lag). Flash tier on 100K is cheap; AI Studio free quota may cover it.
- Determinism for H6: pin model + `temperature=0` + the schema; commit `raw_extractions/` +
  `manifest.json` as the reproducible artifact. The extraction script is documented but NOT in the
  timed path. Declare "Gemini" honestly in `submission_metadata.yaml` (it is an allowed AI-tool option).

---

## 6. embed.py - offline embedding pass (runs in parallel with the batch)

- Load `EMBED_MODEL` via sentence-transformers (CPU fine offline).
- Text per candidate = headline + summary + concatenated `career_history[].description`.
  Apply the model's prefix convention if needed.
- Encode in batches -> float32, L2-normalize -> 384-dim.
- Also embed the JD must-have block -> `jd_emb` (store alongside or as a sidecar).
- Keep keyed by candidate_id for the join in build_store.

---

## 7. build_store.py - the unified LanceDB dataset

```python
import os; os.environ["LANCE_TELEMETRY"]="0"; os.environ["DO_NOT_TRACK"]="1"
import lancedb
db = lancedb.connect("artifacts/vectors.lance")
# one row per candidate:
#   candidate_id: str
#   vector: fixed-size-list<float32>[384]
#   <flattened feature columns from extraction: enums->small ints/strings, bools, etc.>
#   <raw fields rank.py / reasoning need: current_title, years_of_experience,
#    notice_period_days, last_active_date, recruiter_response_rate, willing_to_relocate,
#    location, github_activity_score, ... (sentinel -1 handled: treat as missing, not negative)>
table = db.create_table("candidates", data=rows, mode="overwrite")
```

- Smoke test today on the 50-sample: write 50 rows, reopen, run a flat cosine query with `jd_emb`,
  confirm it returns and the feature columns round-trip. Verify with network physically off.
- Sentinel handling: `github_activity_score` and `offer_acceptance_rate` use -1 = "no data".
  Map -1 -> null/NaN on ingest so downstream scoring never reads it as a strong negative.

---

## 8. validate_extract.py - extraction sanity (gate before trusting 100K)

Run against whatever is extracted (50-sample today, full pool Day 3):
- honeypot_suspected rate (sanity: a handful, not ~0 and not huge).
- coverage: % of records with at least one must-have at hands_on/production.
- spot-print 5 random extractions next to their raw record to eyeball grounding (H4).

---

## DRY-RUN GATE (do this BEFORE spending tokens on 100K)

1. Run `extract_llm.py` in a `--sample 50` mode (live `messages.create`, not batch) over
   `sample_candidates.json`.
2. Eyeball: do consulting careers get `pure_consulting_career`? Does a
   "Marketing Manager + expert RAG" get `keyword_stuffing`? Do impossible-date profiles get
   `honeypot_suspected`? Are `evidence` quotes actually from the record?
3. Only when extraction looks right -> launch the full-pool batch (step 5).

---

## BLOCKERS / decisions needed before launching the batch

1. GEMINI_API_KEY (Google AI Studio / Google API key) available, with quota for ~100K Flash calls.
2. Gemini model tier: gemini-2.5-flash (default, cheap) vs 2.5-pro (quality lever for the top-20
   judgments) vs flash-lite (cheapest). Confirm the exact current model id + pricing first.
3. Execution mode: Gemini Batch Mode (discount, async) vs bounded concurrent live calls. Depends on
   your quota/rate limits - I will verify both before launching.
4. Embedding model: default bge-small-en-v1.5 (384-dim, no license, instant). EmbeddingGemma only
   if eval later shows a recall gap.
```
