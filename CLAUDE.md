# CLAUDE.md — Redrob Hackathon: Intelligent Candidate Discovery & Ranking

This file governs how Claude works in this repo. It exists to encode the **hard, non-negotiable constraints** of the hackathon so that automated help never quietly violates a rule that gets us disqualified.

> **GOLDEN RULE — STOP AND ASK** If any change, command, or suggestion would break (or even risk breaking) a **HARD CONSTRAINT** listed below, **STOP. Do not proceed. Ask the user first**, explaining which constraint is at risk and why. Never trade a disqualification risk for convenience, speed, or a higher score. When in doubt, ask.

> **NO EMOJIS — HARD RULE** Do not use emojis anywhere — not in code, comments, commit messages, CSV reasoning, docs, or chat replies. Plain text only.

---

## The task (context)

Produce a CSV ranking the **top 100 candidates** from `candidates.jsonl` (100,000-candidate pool) for the released `job_description`, best-fit first, each with a 1–2 sentence reasoning. Scored offline against hidden ground truth.

---

## HARD CONSTRAINTS — never break without explicit user approval

These are disqualification-grade rules. Before any action that touches one, **stop and ask**.

### H1. Compute budget (Stage 3 reproduction is enforced in a sandbox)
The **ranking step** that produces the CSV must run within:
- **≤ 5 minutes** wall-clock
- **≤ 16 GB** RAM
- **CPU only** — no GPU
- **Network OFF** — no external/hosted LLM API calls during ranking (no OpenAI, Anthropic/Claude, Cohere, Gemini, or any hosted model service)
- **≤ 5 GB** intermediate disk state

**STOP AND ASK** before introducing: any per-candidate LLM/API call in the ranking path, GPU dependencies, anything that loads the full pool in a way that risks the 16 GB / 5 min budget, or any network call inside `rank.py`. Pre-computation (embeddings/indexes) may exceed 5 min, but the **ranking step itself must not** — keep the two clearly separated.

### H2. Honeypots (Stage 3 filter)
~80 candidates have subtly impossible profiles (e.g. 8 yrs experience at a company founded 3 yrs ago; "expert" in 10 skills with 0 years used). They are forced to tier 0 in ground truth.
- **Honeypot rate > 10% in the top 100 = disqualification.**
- Do **not** special-case/hardcode honeypot IDs — the ranker should avoid them by actually reading profiles. Profile-consistency checking is the right lever.

**STOP AND ASK** before any approach that would plausibly rank impossible profiles highly (e.g. pure keyword/embedding match with no internal-consistency validation).

### H3. Submission CSV format (Stage 1 auto-validator — hard reject on any miss)
- Filename: `<participant_id>.csv`, **UTF-8**, **`.csv` only** (never `.xlsx`/`.json`).
- Columns, in this exact order: `candidate_id,rank,score,reasoning`
- **Exactly 100 data rows** + 1 header. Not 99, not 101.
- `rank`: integers **1–100, each used exactly once** (starts at 1, not 0).
- `candidate_id`: each appears once; every ID **must exist** in `candidates.jsonl`.
- `score`: float, **monotonically non-increasing** as rank increases (rank 1 ≥ rank 2 ≥ … ≥ rank 100). Ties allowed but ranks stay unique — break ties deterministically (secondary signal, else `candidate_id` ascending).
- Scores must **differentiate** — not all identical.
- **Always run `validate_submission.py` on the output before declaring it ready.**

**STOP AND ASK** before changing the column set/order, the row count, the ranking range, or the file format.

### H4. No hallucination in `reasoning` (Stage 4 manual review)
Every claim in a `reasoning` cell must correspond to something **actually in that candidate's profile** (skills, employers, titles, signal values, years).
- No invented skills/employers/experience.
- Reasonings must be **varied** (not templated/name-insertion), **specific**, **connected to JD requirements**, **honest about gaps**, and **consistent with the rank** (no glowing rank-95 / critical rank-5).

**STOP AND ASK** before generating reasoning text from anything other than the candidate's own record, or templating it.

### H5. Submission cap & process
- **Max 3 submissions** total; last valid one counts. There is **no live leaderboard** and no per-submission feedback. Validate locally via methodology, not by burning submissions.
- A **working sandbox link** (HF Spaces / Streamlit / Replit / Colab / public Docker / Binder) is **mandatory** and must run the ranker on a ≤100-candidate sample within the CPU budget.

**STOP AND ASK** before doing anything that would consume a submission slot, or before finalizing if the sandbox path is unverified.

### H6. Authentic engineering (Stages 3–5)
The repo must be reproducible and defensible:
- Single documented command reproduces the CSV (e.g. `python rank.py --candidates ./candidates.jsonl --out ./submission.csv`).
- Real, pinned dependencies (`requirements.txt`/`pyproject.toml`).
- **Genuine git history with real iteration** — not a single dump. A flat history or a codebase that is "entirely LLM API calls" fails review.
- AI tool use is allowed and must be **declared honestly**; finalists defend their architecture in a 30-min interview.

**STOP AND ASK** before squashing/rewriting history into a single commit, removing reproducibility, or any step that would make the work undefensible.

---

## Scoring (optimize for this, within the constraints above)
`Composite = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`
Tiebreaks: higher P@5, then P@10, then earlier timestamp. Top-10 quality dominates — invest there, but never by violating H1–H6.

## Data notes
- Pool: `candidates.jsonl` (100K; gz provided). Each record has 23 `redrob_signals` behavioral fields (response rate, recency, assessment scores, etc.) — usable as a multiplier/modifier on skill-match, not a replacement for reading the profile.
- Other traps beyond honeypots: keyword stuffers, plain-language Tier-5s, behavioral twins. See `redrob_signals_doc` and `job_description`.

## Working conventions
- Keep **pre-computation** and the **time-boxed ranking step** in separate, clearly-labeled code paths.
- Treat the source bundle docs (`*.docx`, `candidate_schema.json`, `submission_spec`, `redrob_signals_doc`) as **read-only reference** — do not edit them.
- Before saying "done": validator passed, row/rank/score invariants hold, honeypot exposure considered, reasoning grounded, ranking step within budget.
