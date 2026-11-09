#!/usr/bin/env bash
# RUN 3 -- iterative fine-tune loop (the classification recipe, denoise edition).
#
# Per cycle:  SFT 1 epoch -- cycle 1 on the >=34 dB corpus, later cycles ONLY on the
#             models the previous cycle produced and accepted (score in prompt, H100)
#          -> generate 40 candidates WITH the fresh adapter (4 RTX shards)
#          -> preflight (arch-dedupe vs whole DB, dead-code strip, class repair)
#          -> 1-epoch screen = the measurement stage (50-epoch training REMOVED 2026-08-03)
#          -> accept 1-epoch PSNR >= 32.5 dB (fallback top-K >= 31.5) into the SFT data;
#             accepted code + scores persist under cycle_<k>/{run,eval}, NOT in the DB
#
# Differences from llm_evolve_par.sh (which this mirrors): a fine-tune stage exists and
# its adapter is CHAINED cycle to cycle (merge_and_unload + fresh LoRA, the repo's
# convention); candidates are judged at the 1-epoch screen only, so cycles are fast
# (~1.5-2 h) at the cost of rank noise -- epoch-1 ordering is a weak predictor of the
# 50-epoch outcome (KPN measured 34.60 @50 from a bottom-tier epoch-1 score).
#
# Every cycle's adapter is kept at cycle_<k>/adapter -- nothing is overwritten.
#
# Coordinator runs ON THE LOGIN NODE (pods cannot create jobs; see llm_evolve_par.sh).
# Run detached:
#   nohup bash run3_evolve.sh > /shared/ssd/.../sweep_out/run3/run3.log 2>&1 &
set -uo pipefail

HOST=/shared/ssd/home/b-b-ahmed/thesis
POD=/a/mm
GA_H="$HOST/nn-gpt/ab/gpt/brute/ga/denoise"
GA="$POD/nn-gpt/ab/gpt/brute/ga/denoise"
SRC="$POD/nn-dataset"
NS="${NS:-b-b-ahmed}"

CYCLES="${CYCLES:-20}"
N="${N:-40}"                    # candidates per cycle
SUPPORTING="${SUPPORTING:-3}"
BASE_MIN_DB="${BASE_MIN_DB:-34.0}"  # cycle-1 base dataset bar -- CORPUS models, 50-ep scores
MIN_DB="${MIN_DB:-32.5}"        # per-cycle acceptance bar -- measured at the 1-EPOCH screen
                                # (the 50-epoch stage is removed; 34@50ep maps to ~32.5@1ep:
                                # epoch-1 median for hand-written models is 32.6, eventual-35+
                                # families measure 29-33 there)
FALLBACK_DB="${FALLBACK_DB:-31.5}"
MIN_K="${MIN_K:-5}"
SCREEN_GATE="${SCREEN_GATE:-30.0}"   # screen job's own pass marker; acceptance ignores it
COSMAX="${COSMAX:-0.90}"
SHARDS="${SHARDS:-4}"
TAG="${TAG:-run3}"
OUT_H="$HOST/sweep_out/$TAG"
OUT="$POD/sweep_out/$TAG"
MAINDB_H="${MAINDB_H:-$HOST/nn-dataset/db/ab.nn.db}"
MAINDB="${MAINDB:-$SRC/db/ab.nn.db}"
# Cycle 1 warm-starts from run3/cycle10, the best-MEASURED adapter, not from the KTO run.
# llmprod generated ~425 candidates per adapter from cycles 1/5/10/12 simultaneously against a
# frozen DB with an identical prompt config -- the only clean head-to-head we have. Preflight
# yield: cycle1 10.8%, cycle5 12.1%, cycle10 20.2%, cycle12 15.5%; cycle10 beats cycle1 by
# +9.4 pp [+4.6,+14.3] (Newcombe), and the progression TURNS OVER after 10, so chaining further
# is not automatically better. run3's own 20 per-cycle preflights cannot arbitrate this -- n~37
# each, Wilson intervals 20-30 pp wide, and every cycle overlaps the 29.0% pooled rate.
# Caveat kept deliberately: cycle10 is the best of the FOUR tested, not established best of 20.
PEFT0="${PEFT0:-$POD/sweep_out/run3/cycle10/adapter}"

mkdir -p "$OUT_H"

ROOTSETUP="R=/scratch/root; mkdir -p \$R/out; \
for l in ab README.md data train validation; do ln -sfn $SRC/\$l \$R/\$l; done"

wait_job() {   # wait_job <name> <completions>
  local name="$1" want="$2" tick=0 ok fail
  while :; do
    ok=$(kubectl get job "$name" -n "$NS" -o jsonpath='{.status.succeeded}' 2>/dev/null)
    fail=$(kubectl get job "$name" -n "$NS" -o jsonpath='{.status.failed}' 2>/dev/null)
    if [ "${ok:-0}" -ge "$want" ] 2>/dev/null; then
      echo "    $name: $ok/$want ok"; return 0
    fi
    if [ "${fail:-0}" -gt 0 ] 2>/dev/null; then
      echo "    $name: $fail FAILED (ok=${ok:-0}/$want)"
      kubectl logs -n "$NS" "job/$name" --tail=25 2>&1 | sed 's/^/      /'
      return 1
    fi
    tick=$((tick+1))
    [ $((tick % 60)) -eq 0 ] && echo "      ... $name ${ok:-0}/$want  $((tick*15))s"
    sleep 15
  done
}

run_job() {   # run_job <name> <completions> <gpus> <command>
  local name="$1" comp="$2" gpus="$3" cmd="$4"
  kubectl delete job "$name" -n "$NS" --ignore-not-found >/dev/null 2>&1
  sleep 3
  python3 "$GA_H/mkjob.py" --name "$name" --completions "$comp" --gpus "$gpus" \
      --command "$cmd" --out "/tmp/$name.json" >/dev/null || return 1
  kubectl apply -f "/tmp/$name.json" -n "$NS" >/dev/null || return 1
  wait_job "$name" "$comp"
}

# The SFT stage needs the H100 (7B fine-tune at 8.7k tokens); mkjob.py targets the RTX
# fleet, so this stage carries its own manifest -- same nodeSelector/toleration as the
# proven lp-ktotrain job.
run_h100_job() {   # run_h100_job <name> <command>
  local name="$1" cmd="$2"
  kubectl delete job "$name" -n "$NS" --ignore-not-found >/dev/null 2>&1
  sleep 3
  cat > "/tmp/$name.yaml" <<EOF
apiVersion: batch/v1
kind: Job
metadata: {name: $name, namespace: $NS}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 604800
  template:
    spec:
      restartPolicy: Never
      nodeSelector: {kubernetes.io/hostname: dgx-h100}
      tolerations:
      - {key: dgx, operator: Equal, value: access, effect: NoSchedule}
      containers:
      - name: ml
        image: abrainone/ai-linux:cu12.6.3-latest
        command: ["/bin/bash","-lc"]
        args: ["$cmd"]
        env:
        - {name: HF_HOME, value: /a/mm/hf_cache}
        - {name: TORCH_HOME, value: /a/mm/torch_cache}
        - {name: TRANSFORMERS_OFFLINE, value: "1"}
        - {name: HF_HUB_OFFLINE, value: "1"}
        - {name: PYTORCH_CUDA_ALLOC_CONF, value: "expandable_segments:True"}
        - {name: TOKENIZERS_PARALLELISM, value: "false"}
        resources: {limits: {cpu: 12, memory: 96Gi, nvidia.com/gpu: 1}, requests: {cpu: 6, memory: 96Gi, nvidia.com/gpu: 1}}
        securityContext: {runAsUser: 1067, runAsGroup: 1376, allowPrivilegeEscalation: false}
        volumeMounts:
        - {mountPath: /a/mm, name: v0}
        - {mountPath: /scratch, name: scratch}
        - {mountPath: /dev/shm, name: shm}
      volumes:
      - {name: v0, hostPath: {path: $HOST, type: Directory}}
      - {name: scratch, emptyDir: {sizeLimit: 40Gi}}
      - {name: shm, emptyDir: {medium: Memory, sizeLimit: 16Gi}}
EOF
  kubectl apply -f "/tmp/$name.yaml" >/dev/null || return 1
  wait_job "$name" 1
}

echo "######## RUN3  cycles=$CYCLES n=$N base-bar=${BASE_MIN_DB}dB@50ep accept=${MIN_DB}dB@1ep  $(date -Is) ########"
echo "coordinator: $(hostname)   out: $OUT_H   warm start: $PEFT0"

# Cycle 1 must not begin until the KTO run has released the H100 and written its adapter.
if kubectl get job lp-ktotrain -n "$NS" >/dev/null 2>&1; then
  echo "--- waiting for lp-ktotrain (KTO run 1) to finish ---"
  wait_job lp-ktotrain 1 || { echo "FATAL: KTO run failed; no warm-start adapter"; exit 1; }
fi
[ -d "${PEFT0/$POD/$HOST}" ] || { echo "FATAL: warm-start adapter missing: ${PEFT0/$POD/$HOST}"; exit 1; }

# One-time: the arch-class map for deduping the base dataset (from the archuid audit).
UIDMAP_H="$OUT_H/uid_map.json"
[ -f "$UIDMAP_H" ] || { echo "FATAL: $UIDMAP_H missing -- copy the audit uid map there first"; exit 1; }

PREV_ADAPTER="$PEFT0"
for k in $(seq 1 "$CYCLES"); do
  CD="$OUT/cycle$k"; CD_H="$OUT_H/cycle$k"; mkdir -p "$CD_H"
  if [ -f "$CD_H/DONE" ]; then
    echo "cycle $k already DONE -- skipping"; PREV_ADAPTER="$CD/adapter"; continue
  fi
  echo; echo "######## CYCLE $k/$CYCLES  $(date -Is) ########"

  # PER-CYCLE DATA, NOT CUMULATIVE. Cycle 1 trains on the >=bar corpus once; every later
  # cycle trains ONLY on the models IT just produced and measured. The old data is already
  # in the chained adapter's weights -- re-showing the fixed base every cycle would be ~20
  # epochs over the same 350 examples through a chained model, which is memorisation, not
  # feedback. A cycle that accepted nothing does no SFT and carries its adapter forward.
  echo "--- [$k.0] SFT dataset (cycle-local) ---"
  SFT_DATA=""
  if [ "$k" -eq 1 ]; then
    run_job "lp-r3-data-$k" 1 0 "cd $GA && PYTHONPATH=$POD/nn-gpt python -u run3_sft_data.py \
      --db $MAINDB --min-db $BASE_MIN_DB --uid-map $OUT/uid_map.json \
      --supporting $SUPPORTING --out $CD/sft_data" \
      || { echo "FATAL cycle$k: dataset build"; exit 1; }
    SFT_DATA="$CD/sft_data"
  elif [ -s "$OUT_H/cycle$((k-1))/accepted.jsonl" ]; then
    run_job "lp-r3-data-$k" 1 0 "cd $GA && PYTHONPATH=$POD/nn-gpt python -u run3_sft_data.py \
      --db $MAINDB --append $OUT/cycle$((k-1))/accepted.jsonl --supporting $SUPPORTING \
      --out $CD/sft_data" \
      || { echo "FATAL cycle$k: dataset build"; exit 1; }
    # a build that found nothing writes no files -- treat like an empty acceptance and
    # carry the adapter forward instead of crashing the SFT stage on a missing train.jsonl
    [ -s "$CD_H/sft_data/train.jsonl" ] && SFT_DATA="$CD/sft_data" \
      || echo "    dataset came out empty; skipping SFT this cycle"
  fi

  if [ -n "$SFT_DATA" ]; then
    echo "--- [$k.1] SFT (1 epoch, H100, chained from $(basename "$(dirname "$PREV_ADAPTER")")/adapter) ---"
    # repo-format examples are ~2k tokens (one baseline, no reference blocks), so the
    # sequence cap drops to 6144 and the batch rises -- measured 3 min/step at 12k/batch 2
    run_h100_job "lp-r3-sft-$k" "cd $GA && PYTHONPATH=$POD/nn-gpt python -u run3_sft.py \
      --data $SFT_DATA --out $CD/adapter --peft $PREV_ADAPTER \
      --max_seq_length 6144 --per_device_train_batch_size 4 --gradient_accumulation_steps 2" \
      || { echo "FATAL cycle$k: SFT"; exit 1; }
  else
    echo "--- [$k.1] nothing accepted last cycle -> no SFT; adapter carried forward ---"
    ln -sfn "../cycle$((k-1))/adapter" "$CD_H/adapter"
  fi

  echo "--- [$k.2] generate $N with cycle adapter ---"
  FB=""
  if [ "$k" -gt 1 ] && [ -f "$OUT_H/cycle$((k-1))/pre/preflight.json" ]; then
    python3 "$GA_H/rejection_digest.py" --pre "$OUT_H/cycle$((k-1))/pre" \
      --run "$OUT_H/cycle$((k-1))/run" --examples 2 --cos-max "$COSMAX" \
      --out "$CD_H/feedback.txt" 2>&1 | sed 's/^/    /' || true
    [ -s "$CD_H/feedback.txt" ] && FB="--feedback-file $CD/feedback.txt"
  fi
  run_job "lp-r3-gen-$k" "$SHARDS" 1 \
    "cd $POD/nn-gpt && PYTHONPATH=$POD/nn-gpt python -u $GA/llm_denoise.py --mode seeded \
     --n $N --supporting $SUPPORTING --seed-set db --adapter $CD/adapter \
     --tournament 3 --elite-share 0 $FB \
     --out $CD/run --id-prefix r3c$k --shard \$JOB_COMPLETION_INDEX --num-shards $SHARDS" \
    || { echo "FATAL cycle$k: generation"; exit 1; }

  # CUDA_VISIBLE_DEVICES= FORCES THE GRADED CHECK ONTO THE CPU. run_job passes gpus=0, but
  # omitting nvidia.com/gpu is a SCHEDULING statement, not isolation -- a pod that lands on a
  # GPU node still sees the cards. The graded path builds every candidate through
  # mutator/lemur_io.py:169, which is unconditionally
  #   torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # so preflight silently constructs on whatever card it can see. On 2026-08-05 that card was
  # a shared H100 another tenant had filled to 78.53 of 79.18 GiB; the llmprod preflight died
  # at candidate 50/1699. Worse than the crash: the OutOfMemoryError is CAUGHT and degrades to
  # "untraceable for the graded check; exact check still applies", so a contended GPU makes
  # preflight silently UNDER-DEDUPE and report near-duplicates as novel. The measured cost of
  # CPU is 0.62 s/candidate vs 0.42 s/candidate -- about six minutes over a whole batch.
  echo "--- [$k.3] preflight (CPU-forced) ---"
  PRE_OK=1
  run_job "lp-r3-pre-$k" 1 0 \
    "$ROOTSETUP; cd \$R && CUDA_VISIBLE_DEVICES= PYTHONPATH=\$R:$POD/nn-gpt python -u $GA/preflight.py \
     --run-dirs $CD/run --out $CD/pre --arch-dedupe --arch-graded --cos-max $COSMAX \
     --arch-dedupe-db $MAINDB --arch-store $OUT/diversity_store.pt --strip-dead-code" \
    || PRE_OK=0

  # A CRASHED PREFLIGHT MUST NOT LOOK LIKE A BARREN ONE. This block used to be a single
  # `[ -s pass_ids.txt ] || { touch DONE; PREV_ADAPTER=...; continue; }`, which is the most
  # dangerous form of the pattern in this repo: a preflight that died on candidate 3 got a
  # DONE marker, advanced the adapter chain, and moved to the next cycle -- recorded as a
  # cycle that completed and happened to yield nothing. Over 20 chained cycles that would be
  # indistinguishable from genuine collapse of the generator, and it would corrupt the very
  # measurement the run exists to make. (Audited 2026-08-06: all 20 cycles of the completed
  # run wrote non-zero pass_ids, so this path never actually fired and that result stands.)
  if [ "$PRE_OK" -eq 0 ]; then
    echo "FATAL cycle$k: preflight job did not succeed -- refusing to advance the adapter chain"
    echo "  a crashed preflight and an empty one are different events; treating this as fatal"
    exit 1
  fi
  [ -s "$CD_H/pre/pass_ids.txt" ] || { echo "cycle$k: preflight completed, nothing was novel"; touch "$CD_H/DONE"; PREV_ADAPTER="$CD/adapter"; continue; }
  echo "    $(wc -l < "$CD_H/pre/pass_ids.txt") of $N survived preflight"

  # ---------------------------------------------------------------- NPU DEPLOYABILITY GATE
  # WHY IT IS HERE. Preflight proves a candidate builds, runs a forward pass, is architecturally
  # novel and fits MMac <= 9053. It cannot prove the thing will run on a phone, and the MAC budget
  # is blind to that: of 128 models profiled on a Galaxy S26 Ultra, 33 (26%) exceed 10 ms at
  # 256x256 against a 1.72 ms median, and every one passes the budget. The LLM arm is worst hit --
  # 8 of 11 profiled LLM models execute entirely off the NPU at a median 55.21 ms against 1.58 ms
  # for DenoiseGA -- so without this gate the arm admits its most accurate models precisely when
  # they are its least deployable.
  #
  # THIS IS A CORPUS-ADMISSION FILTER, NOT A COMPUTE SAVER. 50-epoch training was removed from
  # this loop on 2026-08-03, so candidates are judged at the 1-epoch screen and the GPU saved by
  # rejecting early is small. What the gate changes is WHAT ENTERS THE CORPUS: everything
  # downstream of here is deployable by construction, so the SFT data the next cycle learns from
  # contains only architectures that actually run on the NPU.
  #
  # RUNS ON THE LOGIN NODE, COSTS NO GPU. It needs two things pods do not have: the TFLite
  # converter (the ai-linux image ships neither litert_torch nor ai_edge_torch, hence the shared
  # venv at thesis/venv_tflite) and the AI Hub API token. Both halves are CPU-only. Conversion is
  # ~3.4 s per candidate and profile jobs run in parallel on the farm.
  #
  # DELIBERATELY NOT A STATIC OPERATOR CHECK. That was measured and rejected: an operator rule
  # scores 100% recall against the label "npu_time_frac == 0" but only 36% against the label that
  # matters, "latency > 10 ms", because the ~60 ms cluster is slow for reasons no operator
  # signature can see. See npu_gate.py --validate, which re-runs that comparison on demand.
  # NPU_GATE=0 disables the stage; the run then behaves exactly as it did before it existed.
  if [ "${NPU_GATE:-1}" = "1" ]; then
    echo "--- [$k.3b] NPU deployability gate (CPU + device farm, no GPU) ---"
    GD_H="$CD_H/npu"; mkdir -p "$GD_H"
    NPU_IDS="$GD_H/pass_ids_npu.txt"
    if python3 "$GA_H/npu_gate.py" --stage-ids --ids-file "$CD_H/pre/pass_ids.txt" \
         --run-dirs "$CD_H/run" --work "$GD_H" 2>&1 | sed 's/^/    /' \
       && bash "$GA_H/npu_gate_stage.sh" "$GD_H/src" "$GD_H" "${NPU_MAX_MS:-10}" 2>&1 | tail -6 | sed 's/^/    /' \
       && python3 "$GA_H/npu_gate.py" --filter-ids --ids-file "$CD_H/pre/pass_ids.txt" \
            --work "$GD_H" --out-ids "$NPU_IDS" 2>&1 | sed 's/^/    /'; then
      SCREEN_IDS="$CD/npu/pass_ids_npu.txt"
      echo "    gate applied: $(wc -l < "$NPU_IDS") of $(wc -l < "$CD_H/pre/pass_ids.txt") deployable"
      # An empty pass list is a real verdict, but there is nothing left to measure, so close the
      # cycle the same way a barren preflight does rather than submitting an empty screen job.
      [ -s "$NPU_IDS" ] || { echo "cycle$k: no deployable candidates"; touch "$CD_H/DONE"; PREV_ADAPTER="$CD/adapter"; continue; }
    else
      # FAIL OPEN, LOUDLY. A broken converter or an expired API token must not silently discard a
      # whole cycle's candidates -- that would look identical to "the generator produced nothing
      # deployable" and would corrupt the very trend this 20-cycle run exists to measure.
      echo "    WARN cycle$k: NPU gate could not run -- proceeding UNGATED on all preflight survivors"
      echo "    (check $GD_H/stage.log; deployability is unverified for this cycle)"
      SCREEN_IDS="$CD/pre/pass_ids.txt"
    fi
  else
    SCREEN_IDS="$CD/pre/pass_ids.txt"
  fi

  echo "--- [$k.4] 1-epoch screen -- the measurement stage ---"
  run_job "lp-r3-scr-$k" "$SHARDS" 1 \
    "$ROOTSETUP; cd \$R && PYTHONPATH=\$R:$POD/nn-gpt python -u $GA/screen_and_train.py \
     --run-dirs $CD/run --only-ids $SCREEN_IDS --out $CD/eval --gate $SCREEN_GATE \
     --screen-only --shard \$JOB_COMPLETION_INDEX --num-shards $SHARDS" \
    || echo "    WARN cycle$k: screen had failures"

  # 50-EPOCH TRAINING REMOVED (2026-08-03, user decision). Acceptance now reads the screen's
  # 1-epoch scores directly; accepted candidates' code is taken from the run dir since these
  # models no longer enter the DB. Their code + scores persist under cycle$k/{run,eval} --
  # full-recipe training can be applied to any of them later, out of band.
  echo "--- [$k.5] accept into SFT data (1-ep bar $MIN_DB dB, fallback $FALLBACK_DB) ---"
  python3 "$GA_H/run3_accept.py" --eval-dir "$CD_H/eval" --run-dir "$CD_H/run" \
      --emit-root "$CD/run" \
      --min-db "$MIN_DB" --fallback-db "$FALLBACK_DB" --min-k "$MIN_K" \
      --out "$CD_H/accepted.jsonl"
  echo "    $(wc -l < "$CD_H/accepted.jsonl") models accepted for cycle $((k+1))'s data"

  PREV_ADAPTER="$CD/adapter"
  touch "$CD_H/DONE"
  echo "--- cycle $k done $(date -Is) ---"
done

echo; echo "######## RUN3 COMPLETE $(date -Is) ########"
echo "adapters: $OUT_H/cycle_*/adapter (one per cycle, all kept)"
