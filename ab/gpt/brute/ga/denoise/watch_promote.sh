#!/usr/bin/env bash
# Block until run R<N>'s promotion completes (growth_loop logs "promote R<N> ->"),
# then report the real 50-epoch PSNRs and which models earned seed slots.
# Usage: watch_promote.sh <N>
set -uo pipefail
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
PY=python
N="${1:?run number}"
for i in $(seq 1 1200); do   # up to 10h
  if grep -qa "promote R${N} ->" "$D/growth_loop.log" 2>/dev/null; then break; fi
  sleep 30
done
echo "[R${N} PROMOTION COMPLETE]"
grep -a "promote R${N} ->" "$D/growth_loop.log" | tail -1
echo "--- full-train @50ep results (psnr50, sorted) ---"
"$PY" - "$D/fulltrain_R${N}.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    runs = sorted(d.get("runs", []), key=lambda r: r.get("psnr50", 0), reverse=True)
    print(f"{'name':40}{'psnr50':>8}{'proxy5':>8}{'MMac':>8}{'deploy':>7}")
    for r in runs:
        print(f"{(r.get('name') or '')[:39]:40}{r.get('psnr50',0):8.3f}"
              f"{(r.get('psnr5_proxy') or 0):8.3f}{(r.get('macs') or 0)/1e6:8.0f}"
              f"{('YES' if r.get('deployable') else 'no'):>7}")
    if runs:
        print(f"\nbest psnr50: {runs[0].get('psnr50',0):.3f} dB  ({(runs[0].get('name') or '')[:44]})")
except Exception as e:
    print("no fulltrain results:", e)
PY
echo "--- promoted seeds ---"
ls "$D/promoted_pool_r${N}"/*.py 2>/dev/null | wc -l | xargs echo "promoted_pool_r${N} count:"
