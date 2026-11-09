#!/usr/bin/env bash
# CPU-only pre-flight over the LLM candidates. Requests NO GPU on purpose: it builds and
# forwards each model on the CPU, which is the whole reason it is cheap.
# Pod-local LEMUR root is not needed -- preflight execs the source directly and never goes
# through mutator.lemur_io / out/nn/tmp -- but cwd must still be the nn-dataset tree so any
# incidental ab.nn import resolves to the right package.
set -uo pipefail
GA=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
GEN=/a/mm/sweep_out/llm_gen2
cd /a/mm/nn-dataset || exit 1
echo "=== preflight $(date -Is) ==="
PYTHONPATH=/a/mm/nn-dataset python -u "$GA/preflight.py" \
    --run-dirs "$GEN/cold/run" "$GEN/seeded/run" \
    --out /a/mm/sweep_out/llm_pre
echo "=== done rc=$? $(date -Is) ==="
