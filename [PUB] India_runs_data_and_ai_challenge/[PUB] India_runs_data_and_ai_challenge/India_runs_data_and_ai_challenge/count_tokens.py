"""
count_tokens.py — estimate token counts for candidate JSON records.

Why this exists: we plan to run an OFFLINE LLM over candidate profiles for
feature extraction. Per-candidate token count drives (a) cost, (b) whether a
profile fits the model context, and (c) whether we batch. This script samples
the pool and reports the token distribution, plus a full-pool estimate.

Tokenizer: tiktoken (OpenAI). cl100k_base = GPT-4/3.5-turbo; o200k_base =
GPT-4o/4o-mini. If you use a non-OpenAI model the exact counts differ, but
tiktoken is a solid proxy for planning. Use --encoding to switch.

Usage (run from this folder):
    python count_tokens.py                         # sample 100 from candidates.jsonl
    python count_tokens.py --sample 1000           # larger sample
    python count_tokens.py --file sample_candidates.json
    python count_tokens.py --encoding cl100k_base  # GPT-4/3.5 tokenizer
    python count_tokens.py --all                   # scan the whole file (slow)
    python count_tokens.py --show-one              # print one record's token count

Install:
    pip install tiktoken          (or: uv pip install tiktoken)

Note: streams the file line-by-line — never loads the full 100K pool into RAM.
"""

import argparse
import json
import os
import statistics
import sys

try:
    import tiktoken
except ImportError:
    sys.exit("tiktoken not installed. Run: pip install tiktoken")


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FILE = os.path.join(HERE, "candidates.jsonl")
FULL_POOL_SIZE = 100_000  # candidates.jsonl is the 100K pool


def iter_records(path):
    """Yield parsed JSON records from a .jsonl (one object per line) or a .json
    file that holds a list of objects. Streams jsonl; json list is read whole."""
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    else:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            for rec in data:
                yield rec
        else:
            yield data


def record_to_text(rec):
    """Serialize a record the way an LLM prompt would see it. We use compact
    JSON (no extra whitespace) as the canonical form being tokenized."""
    return json.dumps(rec, ensure_ascii=False, separators=(",", ":"))


def human(n):
    return f"{n:,}"


def main():
    ap = argparse.ArgumentParser(description="Token counter for candidate JSON records.")
    ap.add_argument("--file", default=DEFAULT_FILE, help="path to .jsonl or .json (default: candidates.jsonl)")
    ap.add_argument("--sample", type=int, default=100, help="how many records to sample (default: 100)")
    ap.add_argument("--all", action="store_true", help="scan every record (overrides --sample; slow on 100K)")
    ap.add_argument("--encoding", default="o200k_base", help="tiktoken encoding (default: o200k_base = GPT-4o)")
    ap.add_argument("--show-one", action="store_true", help="print the token count of the first record and exit")
    args = ap.parse_args()

    if not os.path.exists(args.file):
        sys.exit(f"File not found: {args.file}")

    try:
        enc = tiktoken.get_encoding(args.encoding)
    except Exception as e:
        sys.exit(f"Unknown encoding '{args.encoding}': {e}")

    if args.show_one:
        rec = next(iter_records(args.file))
        text = record_to_text(rec)
        n = len(enc.encode(text))
        print(f"First record: {human(n)} tokens ({human(len(text))} chars) using {args.encoding}")
        return

    limit = None if args.all else args.sample
    counts = []
    for i, rec in enumerate(iter_records(args.file)):
        if limit is not None and i >= limit:
            break
        counts.append(len(enc.encode(record_to_text(rec))))

    if not counts:
        sys.exit("No records read.")

    n = len(counts)
    total = sum(counts)
    counts_sorted = sorted(counts)

    def pct(p):
        idx = min(len(counts_sorted) - 1, int(round((p / 100) * (len(counts_sorted) - 1))))
        return counts_sorted[idx]

    mean = total / n
    print(f"File:        {args.file}")
    print(f"Encoding:    {args.encoding}")
    print(f"Sampled:     {human(n)} records" + ("  (FULL FILE)" if args.all else ""))
    print("-" * 48)
    print(f"min:         {human(min(counts))}")
    print(f"p50 (median):{human(int(statistics.median(counts)))}")
    print(f"mean:        {human(int(mean))}")
    print(f"p90:         {human(pct(90))}")
    print(f"p95:         {human(pct(95))}")
    print(f"p99:         {human(pct(99))}")
    print(f"max:         {human(max(counts))}")
    print("-" * 48)
    print(f"mean tokens/candidate: {human(int(mean))}")
    print(f"est. full pool ({human(FULL_POOL_SIZE)}): {human(int(mean * FULL_POOL_SIZE))} tokens")
    print(f"est. shortlist (5,000):                {human(int(mean * 5000))} tokens")


if __name__ == "__main__":
    main()
