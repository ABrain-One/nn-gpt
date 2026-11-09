#!/usr/bin/env bash
# Both similarity views over the d10v2 GA output, in one pass.
#
#   1. WL EXACT HASH  -- the structural gate the GA actually enforces. Counts how many of
#      the generated sources are distinct GRAPHS, and how many collapse together.
#   2. RESIDUAL CKA   -- functional_similarity.py. Compares what each model REMOVES
#      (out - noisy) rather than what it outputs: the shared clean signal dominates raw
#      output-CKA and saturates it near 1.0 for every competent denoiser, so residual is
#      the discriminative view.
#
# CAVEAT THAT MUST TRAVEL WITH THE NUMBERS: model_outputs() builds each net from source
# with RANDOM INIT -- no trained weights exist for GA mutants. So CKA here measures the
# architecture's inductive bias at initialisation, not learned behaviour. Two models can
# look functionally close at init and diverge once trained, and vice versa. Treat it as a
# structural probe, not a claim about trained function.
set -uo pipefail
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
MODELS="${MODELS:-$D/seed_ga_D10v2_models}"
OUT=/a/mm/sweep_out/similarity
mkdir -p "$OUT"
cd "$D" || exit 1
export PYTHONPATH=/a/mm/nn-dataset

echo "=== models dir: $MODELS ($(ls "$MODELS"/*.py 2>/dev/null | wc -l) sources) ==="

echo
echo "=== 1. WL EXACT-HASH distinctness ==="
python -u - "$MODELS" <<'PYEOF'
import glob, os, sys, collections
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "/a/mm/nn-gpt/ab/gpt/brute/ast")
from diversity_gate import DiversityGate
g = DiversityGate(store_path=None, log=lambda *a: None, graded=True, cos_max=0.95)
groups = collections.defaultdict(list)
bad = 0
for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.py"))):
    src = open(p).read()
    first = src.splitlines()[0] if src else ""
    lbl = first.split("name=", 1)[1].split()[0] if first.startswith("# name=") \
        else os.path.basename(p)[:-3]
    try:
        h, _ = g._fingerprint(src)
    except Exception:
        h = None
    if h is None:
        bad += 1
        continue
    groups[h].append(lbl)
n = sum(len(v) for v in groups.values())
print(f"sources hashed : {n}  (untraceable: {bad})")
print(f"distinct graphs: {len(groups)}")
dupes = {k: v for k, v in groups.items() if len(v) > 1}
print(f"hash collisions: {len(dupes)} group(s) holding {sum(len(v) for v in dupes.values())} models")
for k, v in list(dupes.items())[:10]:
    print(f"   {k[:12]}  x{len(v)}  {', '.join(x[:34] for x in v[:4])}")
PYEOF

echo
echo "=== 2. RESIDUAL CKA (random init — see header caveat) ==="
python -u functional_similarity.py \
    --models-dir "$MODELS" \
    --n 32 \
    --out "$OUT/functional_similarity_d10v2.json"
