#!/usr/bin/env bash
# CLOSED-LOOP test of the source-level eviction gate, on the arm that actually has the problem.
#
# WHY THIS ARM. An earlier A/B ran the gate inside a GENETIC run and it rejected 0 of 161
# candidates -- correctly, because the genetic seed families have measured eviction rates of
# 0-6% (DnSeed2Fast 0 of 697, DnSeed2Act 0 of 357) and so have nothing for the gate to catch.
# Eviction is concentrated in the LLM-written arm at 36.8%. That is where a deployability gate
# can be shown to do anything, and running it anywhere else is a null result by construction.
#
# WHAT IS DIFFERENT FROM THE RETROSPECTIVE ANALYSIS. Applying the gate to already-measured
# output shows what the corpus would have been. This instead puts the gate INSIDE the loop, at
# preflight, before the GPU screen -- so a predicted-evicted candidate never costs a training
# run, and each generation's elite pool is conditioned on a gated population. The generator can
# therefore adapt to the gate, which is the one thing the retrospective analysis cannot show.
#
# DESIGN. Two arms, identical seeds/gens/N/K/floor/epochs, differing only in DEPLOY_THR:
#   ctrl   no gate
#   gate   DEPLOY_THR=0.5   (the measured operating point)
# Both write to their own OUT tree and their own DB prefix, so neither can contaminate the other.
#
# COST. Sequential by necessity -- generation N+1 depends on N's trained results, and steps 4-5
# own the LEMUR SQLite DB on NFSv3 where concurrent writers corrupt rather than block. One GPU
# per arm, the two arms in parallel with each other.
set -euo pipefail
ARM="${ARM:?set ARM=ctrl or ARM=gate}"
case "$ARM" in
  ctrl) export DEPLOY_THR=0 ;;
  gate) export DEPLOY_THR=0.5 ;;
  *) echo "ARM must be ctrl or gate"; exit 2 ;;
esac
export OUT_OVERRIDE="/a/mm/sweep_out/llmgate_${ARM}"
export GENS="${GENS:-4}"
export N="${N:-80}"
export K="${K:-8}"
export FLOOR="${FLOOR:-4}"
export EPOCHS=5
mkdir -p "$OUT_OVERRIDE"
echo "=== LLM gate A/B: arm=$ARM DEPLOY_THR=$DEPLOY_THR GENS=$GENS N=$N ==="
exec bash /a/mm/nn-gpt/ab/gpt/brute/ga/denoise/llm_evolve.sh
