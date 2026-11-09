#!/usr/bin/env bash
# GATE 1 (pre-production): full-train the top-3 per ablation arm at 50 epochs — the
# PROMOTION AUTHORITY the ep5 proxy is not (10F/7F late-bloomer blind spot). Confirms
# whether the proxy ranking D>C>B>A holds at the real-dB level, and produces the
# headline table (incl. each arm's best deployable within-budget model) to take to
# the professor BEFORE committing the production run. save_to_db OFF. Resumable:
# fulltrain_results.json caches by checksum, so a re-run trains only what's missing.
#
# Run when GPU allocation is up:  bash launch_fulltrain.sh
# (~2h per 50ep train x ~14 candidates ~= long; runs resume cleanly if preempted.)
set -euo pipefail
cd /a/mm/nn-dataset
export PYTHONPATH=/a/mm/nn-dataset
export BRUTE_DEBUG=0
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
PY=python
"$PY" "$D/fulltrain_topk.py" \
  --arms A0 B0 C0 D0 \
  --k 3 --epoch 50 \
  --batch 16 --lr 1e-4 \
  --num-workers 6 \
  --results "$D/fulltrain_results.json"
