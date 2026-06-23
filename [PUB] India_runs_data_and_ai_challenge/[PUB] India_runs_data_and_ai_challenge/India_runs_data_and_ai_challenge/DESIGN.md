# DESIGN.md — Redrob Candidate Ranking System

Design target for the Intelligent Candidate Discovery & Ranking Challenge. Read `CLAUDE.md` first — the hard constraints (H1-H6) bound everything here.

This document is the thing we build to and defend at the Stage 5 interview. It encodes the JD as a machine-readable rubric, the feature set, the scoring formula, the trap/honeypot logic, and the reproducibility plan. Weights marked `[tunable]` are initial values to be set by the self-eval harness (Section 11), not magic numbers.

---

## 1. Problem framing (why the design looks like this)

Three facts drive every decision:

1. **No ground-truth labels.** Scores are hidden until close. This is therefore an **unsupervised, JD-derived heuristic ranking** problem, not supervised learning-to-rank. The intelligence is in the rubric (Section 4), not in a trained model. Any learned component would need self-labeled data and is out of scope for v1.
2. **Rank what the JD *means*, not what it *says*.** Keyword/embedding match is an explicitly built trap. The core is **structured reasoning over career history**; embeddings are a supporting signal and a retrieval leg only.
3. **Hard compute firewall (H1).** Everything expensive is OFFLINE precompute (no time limit). The ONLINE `rank.py` step (<= 5 min, CPU, no network) only loads compact artifacts, scores vectorized, sorts, writes CSV, self-validates.

---

## 2. Architecture overview

```
OFFLINE  (precompute/ — unbounded time, may use network/local models)
  --as-of DATE  (pinned for reproducibility; baked into temporal/recency features)
  candidates.jsonl.gz
     -> [1] ingest + normalize        -> normalized.parquet  (+ quarantine.jsonl for malformed rows)
     -> [2] feature extraction        -> features.parquet   (the core IP)
     -> [3] embeddings + BM25 index   -> emb.npy, bm25.pkl
  job_description.md
     -> [0] JD rubric spec            -> jd_rubric.yaml, jd_emb.npy

  eval/  (self-eval harness — Section 13)
     gold mini-set + ablations
     -> [E] tune weights & availability floor -> weights.yaml
            ^________________________ reads features.parquet, scores gold set,
                                      feeds tuned w1..w10 + floor back to [4b]

ONLINE  (rank.py — <= 5 min, CPU only, network OFF)
  load features.parquet + emb.npy + bm25.pkl + jd_rubric.yaml + jd_emb.npy + weights.yaml
     -> [4a] hybrid retrieve top-K shortlist (BM25 + dense, uses jd_emb.npy)
     -> [4b] deep feature rerank  (fit x availability - penalties; weights from weights.yaml)
     -> [5]  honeypot/trap consistency layer
     -> [6]  grounded reasoning generation
     -> [7]  assemble: sort -> assign ranks 1-100 -> enforce non-increasing score (H3)
             -> top-100 CSV -> run validate_submission.py
```

Repo layout enforces the firewall:
```
precompute/   build_features.py, build_index.py, build_rubric.py
rank.py       the ONLY timed step; produces submission.csv
artifacts/    features.parquet, emb.npy, bm25.pkl, jd_rubric.yaml, jd_emb.npy, weights.yaml  (committed or regenerated)
eval/         gold set + self-eval harness (produces weights.yaml)
```

---

## 3. Data model (from candidate_schema.json)

Fields the ranker consumes (per candidate):
- `profile`: years_of_experience, current_title, current_company, current_company_size, current_industry, headline, summary, location, country.
- `career_history[]`: company, title, start/end_date, duration_months, is_current, industry, company_size, description.
- `education[]`: degree, field_of_study, institution, tier, end_year.
- `skills[]`: name, proficiency {beginner..expert}, endorsements, duration_months.
- `certifications[]`, `languages[]`.
- `redrob_signals`: 23 fields — see Section 8 for the ones we use.

---

## 4. The JD rubric (`jd_rubric.yaml`) — machine-readable JD intent

Derived from `job_description.md`. This is the system's brain. Structured so the scorer and the reasoning generator read the same source of truth.

### 4.1 Must-haves (high positive weight)
- `embedding_retrieval_prod`: production embeddings-based retrieval (sentence-transformers, BGE, E5, OpenAI embeddings, etc.). Operational, not toy.
- `vector_db_or_hybrid_search`: Pinecone, Weaviate, Qdrant, Milvus, FAISS, OpenSearch, Elasticsearch, or similar, in production.
- `strong_python`: real Python depth.
- `ranking_eval_experience`: NDCG/MRR/MAP, offline-to-online, A/B interpretation.
- `shipped_ranking_search_recsys`: shipped >=1 end-to-end ranking/search/rec system to real users at meaningful scale. **Mined from career descriptions**, not the skills list.

### 4.2 Nice-to-haves (small positive weight, never required)
LLM fine-tuning (LoRA/QLoRA/PEFT); learning-to-rank (XGBoost/neural); HR-tech/recruiting/marketplace domain; distributed systems / large-scale inference; OSS contributions.

### 4.3 Hard disqualifiers (strong negative; see Section 9 for how applied)
- `pure_research_no_prod`: academic/research-only, no production deployment.
- `recent_langchain_only`: AI experience is primarily <12mo LangChain->OpenAI, with no substantial pre-LLM ML production history.
- `no_code_18mo`: senior who moved to architecture/TL and hasn't shipped code in 18mo.
- `title_chaser`: company switches ~every <=1.5 yrs optimizing for title.
- `pure_consulting_career`: entire career at services/consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, Mindtree, HCL, Tech Mahindra, LTIMindtree, Mphasis, ...). Exception: has prior product-company experience.
- `cv_speech_robotics_only`: primary expertise CV/speech/robotics without significant NLP/IR exposure.
- `closed_source_no_validation`: 5+ yrs entirely closed-source proprietary with no external validation (papers/talks/OSS).

### 4.4 Context preferences (moderate weight / multipliers)
- `india_tier1_location`: in or willing to relocate to Noida/Pune (also Hyderabad, Mumbai, Delhi NCR, Bangalore tier-1). Outside India = case-by-case, down-weight.
- `notice_period`: <=30d preferred; 30+ raises the bar (soft penalty scaling).
- `experience_band`: 6-8 yrs ideal, 5-9 in-band, outside-band considered if other signals strong (soft, not a gate).
- `applied_ml_at_product`: 4-5 yrs applied ML at product (not services) companies.
- `availability`: active on platform / clear job-market signal (behavioral, Section 8).
- `external_validation`: OSS/papers/talks present (also negates closed-source DQ).

---

## 5. Phase 1 — Ingestion & normalization (OFFLINE)

- Stream `candidates.jsonl.gz` line-by-line (gzip text mode). Never materialize the full raw file in RAM.
- Validate each record against `candidate_schema.json`; log and quarantine malformed rows (do not crash the batch).
- Derived temporals: per-role tenure, total tenure, employment gaps, recency of most recent role, count of distinct employers, median tenure (job-hop signal), months since `last_active_date` (uses a fixed "as-of" date passed in, since Date.now is non-deterministic — pin it for reproducibility).
- Output: `normalized.parquet`, one row per candidate, typed columns + nested lists kept as needed.

---

## 6. Phase 2 — Feature extraction (OFFLINE) — core IP

All features interpretable and individually inspectable (needed for reasoning generation and the interview). Grouped:

### 6.1 Skill-fit features
- `musthave_skill_coverage` [0-1]: fraction of must-have skill clusters the candidate evidences, weighted by proficiency, `duration_months`, and matching `skill_assessment_scores`. **Discounts "expert + 0 months"** (honeypot guard).
- `skill_depth`: endorsements + assessment-score corroboration per claimed skill.
- `keyword_stuffing_penalty`: many high-proficiency AI skills that are contradicted by title/career (e.g. "expert RAG" + title "Marketing Manager").

### 6.2 Career-substance features (the anti-trap engine)
- `company_type_score`: classify each employer as product / services-consulting / unknown via a curated lexicon + `industry` field + size heuristics. Career dominated by services/consulting -> negative; product-company ML roles -> positive.
- `shipped_ranking_evidence` [0-1]: NLP/IR over `career_history[].description` for evidence of building ranking/search/recsys/retrieval at scale (verbs + objects + scale cues like "real-time", "millions", "production").
- `domain_score`: NLP/IR vs CV/speech/robotics classification from titles, descriptions, skills. NLP/IR positive; CV/speech/robotics-only negative.
- `applied_ml_years_at_product`: years in applied-ML roles at product companies.

### 6.3 Trajectory features
- `experience_band_fit`: smooth bump centered on 6-8 yrs, gentle falloff (soft).
- `job_hop_score`: penalize median tenure < ~18 months (title-chaser signal).
- `recent_hands_on`: is current/recent role hands-on engineering vs pure architecture/TL (drives `no_code_18mo`).
- `pre_llm_ml_experience`: ML production evidence older than 12 months (negates `recent_langchain_only`).
- `external_validation_score`: OSS (github_activity_score), papers/talks cues in text (negates `closed_source_no_validation`).

### 6.4 Context-fit features
- `location_fit`: India tier-1 / Noida-Pune / willing_to_relocate. Outside India down-weight.
- `notice_fit`: from `notice_period_days`, soft penalty above 30.

### 6.5 Behavioral availability (from redrob_signals) -> Section 8.

Output: `features.parquet`, one row per candidate, all features + raw fields the reasoning generator needs.

---

## 7. Phase 3 — Representation & indexing (OFFLINE)

- **Dense:** local CPU sentence-transformer (e.g. BGE-small / E5-small class) over `summary` + concatenated `career_history[].description`. Store `emb.npy` (float32, L2-normalized). Embed the JD must-have block -> `jd_emb.npy`.
- **Lexical:** BM25 over the same text -> `bm25.pkl`.
- Embeddings are a **retrieval leg and a supporting rerank feature**, never the decider — deliberate, to dodge the keyword/embedding trap.

---

## 8. Behavioral availability multiplier (redrob_signals)

Applied **multiplicatively** on the fit score, not additively: a perfect-on-paper candidate who is unreachable is not hireable.

`availability in [floor, 1.0]` (floor ~0.5 [tunable], so a strong profile is dampened but never zeroed by behavior alone), built from:
- `last_active_date` recency (months since active) — primary.
- `recruiter_response_rate`, `avg_response_time_hours`.
- `open_to_work_flag`, `interview_completion_rate`.
- Secondary corroboration: `saved_by_recruiters_30d`, `profile_views_received_30d`.
Ignored as low-signal/noise: connection_count, raw endorsements (already in skill_depth).

Example shape: a 6-months-inactive, 5%-response candidate lands near the floor; an active, responsive, open-to-work candidate stays ~1.0.

---

## 9. Phase 4 — Retrieval + scoring (ONLINE, multi-step)

### 9.1 Stage 1 — hybrid retrieve (top-K shortlist)
- Lexical BM25(JD must-haves) and dense cosine(jd_emb, emb) over all 100K.
- Combine via reciprocal-rank fusion -> shortlist top-K (`K ~ 1000-2000` [tunable]).
- Purpose: production-faithful, scales to 200K. **Recall guard:** because plain-language Tier-5s may be lexically thin, K is set generously and dense recall is checked in eval (Section 11). If recall risk shows up, raise K or union in a structured-prefilter leg.

### 9.2 Stage 2 — deep rerank (the JD-fit score)
For each shortlisted candidate:

```
fit_raw   = w1*musthave_skill_coverage
          + w2*shipped_ranking_evidence
          + w3*company_type_score
          + w4*domain_score
          + w5*ranking_eval_experience
          + w6*experience_band_fit
          + w7*location_fit
          + w8*external_validation_score
          + w9*semantic_sim            (supporting, deliberately small)
          + w10*nice_to_haves
          - keyword_stuffing_penalty
          - job_hop_penalty
          - notice_penalty

disqualifier_factor = product of soft-gates in [low..1.0] for each DQ in 4.3
                      (soft, evidence-weighted — not a hard 0, to tolerate noisy
                       extraction; a clear DQ drives the factor very low)

score = fit_raw * disqualifier_factor * availability_multiplier
```

- Weights `w*` `[tunable]` via eval, loaded at runtime from `weights.yaml` (emitted by the eval harness, Section 13); initial guess emphasizes career-substance (w2, w3, w4) over skill-list and semantic (w1 moderate, w9 small).
- Fully vectorized in numpy over the shortlist.
- **Tie-break (matches validator):** secondary signal (e.g. availability) then `candidate_id` ascending, so equal scores never violate H3.

---

## 10. Phase 5 — Trap & honeypot consistency layer (ONLINE)

Honeypots (~80, forced tier 0; >10% in top-100 = DQ per H2). Detected by **internal-consistency checks**, never by hardcoded IDs:
- tenure at a company exceeding the company's plausible age / role dates.
- `proficiency = expert` with skill `duration_months = 0` (or near 0).
- skill count / seniority implausible for `years_of_experience`.
- `skill_assessment_scores` contradicting claimed proficiency.
- sum of role durations inconsistent with `years_of_experience`.
- impossible date ordering (end before start, future dates).

Each check contributes to a `consistency_penalty`; a high penalty pushes the candidate out of the top ranks. After ranking, assert measured honeypot-style flag rate in top-100 is well under 10% (alarm in logs if not). This also catches keyword-stuffers and behavioral-twin edge cases as a side effect.

---

## 11. Phase 6 — Grounded reasoning generation (ONLINE, no network)

Decision (locked): **deterministic, fact-templated** from extracted features only. No LLM at runtime; no offline LLM polish. The spec rewards "specific and honest," penalizes hallucination and templated name-insertion.

Each `reasoning` cell is assembled to satisfy all six Stage-4 checks:
- **Specific facts:** cite real values — years_of_experience, current_title, named skills, specific signal numbers.
- **JD connection:** name the concrete JD requirement matched (e.g. "shipped production retrieval", "eval frameworks").
- **Honest concern:** surface the candidate's top gap (notice period, services background, thin production evidence) when present.
- **No hallucination:** every clause is sourced from a field that exists in the record; generator can only reference populated features.
- **Variation:** sentence skeleton + clause selection driven by which features fired, so different candidates read differently (not name-insertion).
- **Rank consistency:** tone scales with rank band (top: strong-fit framing; mid/low: "adjacent / included as filler with caveats"). A rank-5 reads positive, a rank-95 reads hedged — enforced by binding tone to the score.

---

## 12. Phase 7 — Assembly & validation (ONLINE -> output)

- Take top-100 by score; assign ranks 1-100; enforce non-increasing score.
- Write `submission.csv`: `candidate_id,rank,score,reasoning`, UTF-8.
- **Last step of the run:** invoke `validate_submission.py` on the output; fail the run loudly if it reports any error (H3). Never ship unvalidated.

---

## 13. Evaluation harness (OFFLINE) — replaces the missing leaderboard

No ground truth exists, so we validate by methodology, not by submitting:
- **Gold mini-set:** hand-label ~30-50 candidates from `sample_candidates.json` and constructed cases as clear-fit / clear-nonfit / honeypot. Sanity-check that the ranker orders them correctly.
- **Retrieval recall check:** confirm known good candidates survive Stage-1 shortlisting (guards the multi-step recall risk in 9.1).
- **Ablations:** toggle each weight group; confirm career-substance dominates skill-keywords (the anti-trap thesis).
- **Honeypot monitor:** report flagged-honeypot rate in top-100 every run.
- **Reasoning spot-check:** sample 10 rows, run them against the six checks manually (mirrors Stage 4).
- **Budget check:** time `rank.py` end-to-end on a 16 GB CPU box; assert < 5 min.

---

## 14. Reproducibility & delivery (OFFLINE) — H5/H6

- `README.md`: single reproduce command, e.g. `python rank.py --candidates ./candidates.jsonl --out ./submission.csv`, plus the precompute commands and as-of date.
- Pinned `requirements.txt`; committed `artifacts/` (or a one-command regenerator).
- `submission_metadata.yaml` at repo root from the bundle template.
- **Sandbox:** HF Spaces / Streamlit / Colab running the ranker on a <=100 candidate sample within budget.
- **Git history:** genuine commit-per-phase iteration, not a single dump.
- AI-tool usage declared honestly.

---

## 15. Open items / risks (decide during build)

- **Build sequencing** (thin vertical slice vs full feature depth) — deferred per earlier decision; choose when coding starts.
- **Embedding model choice** — pick a CPU-fast model whose 100K encode fits the offline budget; lock the version for reproducibility.
- **Weight setting** — all `[tunable]` weights come from the eval harness, not hand-waving; document the final values and the rationale for the interview.
- **Company lexicon coverage** — services/consulting list is non-exhaustive; back it with the `industry` field and review misses in eval.
- **Soft-gate vs hard-gate on disqualifiers** — starting soft (multiplicative factor) to tolerate noisy extraction; revisit if DQ profiles leak into top-100.

> Per CLAUDE.md: before any step that risks H1-H6, stop and ask.
