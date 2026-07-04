# token.md — Candidate token-count measurements

Output of `count_tokens.py` (in the bundle folder). Purpose: size the **offline LLM feature-extraction** step — per-candidate token count drives cost, context-fit, and batching.

## Method

- Tokenizer: **tiktoken `o200k_base`** (GPT-4o family). A non-OpenAI model will differ slightly; this is a planning proxy.
- Each candidate serialized as **compact JSON** (`separators=(",",":")`, no whitespace) — the canonical form being tokenized.
- Sample: **200 candidates** streamed from `candidates.jsonl` (the 100K pool).
- Date measured: 2026-06-24.

## Results (200-candidate sample)

| Metric | Tokens / candidate |
|---|---|
| min | 718 |
| p50 (median) | 1,074 |
| mean | 1,091 |
| p90 | 1,333 |
| p95 | 1,495 |
| p99 | 1,582 |
| max | 1,651 |

## What it means

- **Profiles are small and uniform.** The largest is ~1,650 tokens, so every candidate fits any modern context window with room to spare — no chunking needed, and multiple candidates can be batched per call if desired.
- **Payload-only token estimates** (the candidate JSON, nothing else):
  - Full pool (100,000): **~109M tokens**
  - Generous shortlist (5,000): **~5.5M tokens** (the ~20x-cheaper hackathon path)

## Budgeting caveat — these are INPUT payload tokens only

Real extraction also pays for:
- **Instruction prompt** per call: ~300-500 tokens
- **Structured output** per call: ~200-500 tokens

So budget closer to **~1,500-2,000 total tokens per candidate**:
- Full pool: **~150-200M tokens**
- 5,000 shortlist: **~8-10M tokens**

Multiply by the chosen model's per-token price for cost. Output tokens are usually priced higher than input — account for both.

## Reproduce

From the bundle folder (`.../India_runs_data_and_ai_challenge/`):

```
pip install tiktoken          # planning utility only — NOT a pipeline dep
python count_tokens.py --sample 200          # this measurement
python count_tokens.py --sample 1000         # tighter estimate
python count_tokens.py --all                 # full file (slow)
python count_tokens.py --encoding cl100k_base  # GPT-4 / 3.5 tokenizer
python count_tokens.py --show-one            # single record
```

Note: `tiktoken` is a planning utility and must stay out of the pinned runtime `requirements.txt` (it has nothing to do with the timed `rank.py` path).
