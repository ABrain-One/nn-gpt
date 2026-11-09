#!/usr/bin/env bash
# Chain the re-export: converted artifacts -> AI Hub profiling -> published-protocol report.
# Login node, because the qai-hub token lives in the user's config and cluster pods have no
# network credentials.
#
# THE TWO HALVES ARRIVE HOURS APART AND THAT IS FINE. Latency (reexport-cpu, no GPU) finishes in
# minutes; full-resolution PSNR (reexport-gpu) waits for ft50 to release 5 cards around 05:20 and
# then retrains ten models. So the report is run TWICE: once as soon as latency exists, which prints
# a latency-only preview, and again once fullres.jsonl appears, which is the real thing. Neither
# pass is destructive -- reexport_report.py reads and rewrites one JSON.
set -uo pipefail

NS=b-b-ahmed
H=/shared/ssd/home/b-b-ahmed/thesis
GA=$H/nn-gpt/ab/gpt/brute/ga/denoise
RX=$H/sweep_out/reexport
LOG=$RX/finish.log
mkdir -p "$RX"
say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

say "=== reexport finisher ==="

say "waiting for reexport-cpu (16 models x 2 resolutions)"
# READ .status.phase, NOT THE STATUS COLUMN. `kubectl get pods` prints "Completed" for a pod whose
# phase is "Succeeded", so matching on the column text against Succeeded never fires -- the first
# version of this loop sat through its full 60-minute wait on an already-finished job before
# falling through. jsonpath returns the API value and the case matches both spellings anyway.
for i in $(seq 1 120); do
  s=$(kubectl get pods -l job-name=reexport-cpu -n $NS \
        -o jsonpath='{.items[0].status.phase}' 2>/dev/null)
  a=$(ls "$RX"/tflite_480x720/*.tflite 2>/dev/null | wc -l)
  b=$(ls "$RX"/tflite_1088x1920/*.tflite 2>/dev/null | wc -l)
  case "$s" in
    Succeeded|Completed) say "reexport-cpu SUCCEEDED (480x720=$a, 1088x1920=$b)"; break;;
    Error|Failed|"") say "reexport-cpu ended status=${s:-gone} (480x720=$a, 1088x1920=$b)"
                     # Partial artifacts are still worth profiling; zero is not.
                     [ $((a+b)) -gt 0 ] || { say "FATAL no artifacts at all -- stopping"; exit 1; }
                     say "proceeding with partial artifacts"; break;;
  esac
  sleep 30
done

n=$(( $(ls "$RX"/tflite_480x720/*.tflite 2>/dev/null | wc -l) + $(ls "$RX"/tflite_1088x1920/*.tflite 2>/dev/null | wc -l) ))
say "--- submitting $n artifacts x 2 devices to the AI Hub farm ---"
python3 "$GA/reexport_submit.py" --submit 2>&1 | tail -6 | tee -a "$LOG"

say "--- collecting (profile jobs finish in minutes; poll until nothing is pending) ---"
for i in $(seq 1 40); do
  out=$(python3 "$GA/reexport_submit.py" --collect 2>&1)
  echo "$out" | tail -8 >> "$LOG"
  pend=$(echo "$out" | grep -o '[0-9]* pending' | grep -o '^[0-9]*' || echo 0)
  say "  collect pass $i: ${pend:-?} pending"
  [ "${pend:-1}" -eq 0 ] && { say "all profile jobs collected"; break; }
  sleep 60
done

say "--- report pass 1: latency only (full-res PSNR still pending) ---"
python3 "$GA/reexport_report.py" --dir "$RX" 2>&1 | tee -a "$LOG"

say "--- waiting for reexport-gpu to produce fullres.jsonl ---"
for i in $(seq 1 720); do
  if [ -s "$RX/fullres.jsonl" ]; then
    got=$(grep -c '"status": "OK"' "$RX/fullres.jsonl" 2>/dev/null || echo 0)
    want=$(python3 -c "import json;print(json.load(open('$RX/subset.json'))['n'])" 2>/dev/null || echo 16)
    [ "${got:-0}" -ge "${want:-16}" ] && { say "full-res complete ($got/$want)"; break; }
    [ $((i % 20)) -eq 0 ] && say "  full-res $got/$want after $((i*2)) min"
  fi
  # If the GPU job is gone and nothing more is arriving, report what exists rather than hang.
  live=$(kubectl get pods -l job-name=reexport-gpu -n $NS --no-headers 2>/dev/null | grep -c Running)
  if [ "${live:-0}" -eq 0 ] && [ -s "$RX/fullres.jsonl" ] && [ $i -gt 10 ]; then
    say "reexport-gpu has no running pods -- reporting what exists"; break
  fi
  sleep 120
done

say "--- report pass 2: the real one ---"
python3 "$GA/reexport_report.py" --dir "$RX" 2>&1 | tee -a "$LOG"
say "=== reexport complete ==="
say "READ THE ANCHOR BLOCK FIRST. If LiteDenoiseNet's re-measured full-res RGB PSNR is not within"
say "about a dB below its published 37.66, the protocol is not reproduced and no other row counts."
