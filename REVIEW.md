# How to Reproduce the Test Results

## What was changed

The SQL optimization lives in two repos:

| Repo | Branch | What changed |
|---|---|---|
| `nn-dataset` | `sql-opt-clean5` | `Query.py`, `Read.py`, `Init.py` — variable-N SQL path, index optimization |
| `nn-gpt` | `sql-opt` (this repo) | `test.py` — full integration test suite |

## Setup (one-time)

```bash
# 1. Install the optimized nn-dataset (includes the database)
pip install git+https://github.com/i-am-manishasamal/nn-dataset@sql-opt-clean5 --no-deps

# 2. Install remaining test dependencies
pip install pandas
```

## Run the tests

```bash
cd nn-gpt   # this repo, sql-opt branch
python test.py
```

Expected output: **6/6 tests pass**, cold-start variable-N ~3–7s (well under 30s threshold).

## Proof files (no setup needed)

The benchmark results are already committed in this repo:

| File | What it shows |
|---|---|
| `baseline_smoke.txt` | Baseline: 45.74s → **FAILED** (>30s limit) |
| `candidate_smoke.txt` | Candidate: 0.33s → **PASSED** |
| `proof_final_warmup0.txt` | Full suite, 6/6 pass, cold-start runs=[2.97s, 0.00s, 0.00s] |
| `proof_summary_for_prof.txt` | Summary of all timings |

## Key result

| | Baseline | Candidate | Speedup |
|---|---|---|---|
| Variable-N query | 45.74s → FAILED | ~3–7s → PASSED | **7–15x faster** |
| Algorithm | O(N²) correlated subquery | O(N log N) window function | — |
