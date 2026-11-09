"""Lateral-skip PLACEMENT arm — 75 runs that the 2x2 factorial never touched.

`ablation_factorial.py` analyses the 8 cells of the G x B x L design (192 of the 267 completed
runs). The remaining 75 are a separate question: given that a lateral encoder->decoder skip
exists, WHERE should it attach? Three placements, 5 bases, 5 repeats:

    L@identity   the skip carries the encoder tensor through unchanged
    L@t          it passes through a transform before being merged
    L@s          it passes through a scaled/gated transform

Design note: 5 repeats per cell here against 3 in the 2x2, which matters because this arm is
noisier -- pooled within-cell sd 0.388 dB vs 0.142 dB. Contrasts are paired within
(base, repeat), so the pairing absorbs both base identity and repeat-level noise; all 25 blocks
are complete.

THE CAVEAT THAT MUST TRAVEL WITH THE HEADLINE. L@identity's per-base means run 33.50-34.21 dB,
and the no-op (identity-function) PSNR on this crop is 33.021. So `LLM_Pg1_seeded` at 33.500 sits
0.48 dB above doing nothing at all. The +1.19 dB advantage of L@s is therefore partly "L@s is
better" and partly "L@identity degrades some bases toward collapse" -- those are different
claims and this arm cannot separate them. The degeneracy screen used elsewhere (exact equality
with 33.021) does NOT fire here, so these are not hard-collapsed runs, but the margin is thin
enough that the effect size should not be quoted as a pure placement benefit. Reported below as
a per-base distance-to-no-op column so the reader can see which bases are at risk.
"""
import collections
import glob
import json
import math
import os
import statistics as st

ROOT = "/a/mm" if os.path.isdir("/a/mm/nn-gpt") else "/shared/ssd/home/b-b-ahmed/thesis"
TRAIN = f"{ROOT}/sweep_out/ablation/train"
NOOP_DB = 33.021
CELLS = ["L@identity", "L@t", "L@s"]
FIELD = "reported_dB"   # same field ablation_factorial.py uses


def load():
    rows = []
    for f in sorted(glob.glob(f"{TRAIN}/result_shard*.jsonl")):
        for line in open(f):
            if line.strip():
                rows.append(json.loads(line))
    return [x for x in rows
            if x.get("cell", "").startswith("L@")
            and str(x.get("status", "")).lower() == "ok"
            and isinstance(x.get(FIELD), (int, float))]


def paired(blocks, a, b):
    ds = [v[a] - v[b] for v in blocks]
    m = st.mean(ds)
    sem = st.stdev(ds) / math.sqrt(len(ds)) if len(ds) > 1 else float("nan")
    agree = sum(1 for x in ds if (x > 0) == (m > 0))
    return m, sem, m - 1.96 * sem, m + 1.96 * sem, agree, len(ds)


lat = load()
print("=" * 92)
print("LATERAL-SKIP PLACEMENT ARM")
print("=" * 92)
by = collections.defaultdict(dict)
per = collections.defaultdict(list)
for x in lat:
    by[(x["base"], x["repeat"])][x["cell"]] = x[FIELD]
    per[(x["base"], x["cell"])].append(x[FIELD])
blocks = [v for v in by.values() if len(v) == len(CELLS)]
bases = sorted({x["base"] for x in lat})
print(f"\n{len(lat)} runs | {len(bases)} bases | {len(blocks)} complete (base,repeat) blocks "
      f"of {len(by)}")

sds = [st.stdev(v) for v in per.values() if len(v) > 1]
pooled = math.sqrt(sum(s * s for s in sds) / len(sds))
mde = 2.8 * pooled / math.sqrt(len(blocks))
print(f"pooled within-cell sd {pooled:.3f} dB (the 2x2 arm measured 0.142) -> MDE ~{mde:.3f} dB")

print(f"\n{'base':32}" + "".join(f"{c:>13}" for c in CELLS) + f"{'min - no-op':>13}")
for b in bases:
    ms = {c: st.mean(per[(b, c)]) for c in CELLS if per.get((b, c))}
    line = f"{b[:32]:32}" + "".join(f"{ms.get(c, float('nan')):>13.3f}" for c in CELLS)
    gap = min(ms.values()) - NOOP_DB
    flag = "  <-- thin" if gap < 0.75 else ""
    print(line + f"{gap:>13.3f}{flag}")
print(f"  no-op (identity function) reference: {NOOP_DB:.3f} dB")

print(f"\nPAIRED CONTRASTS (within base+repeat, n={len(blocks)})")
for a, b in (("L@t", "L@identity"), ("L@s", "L@identity"), ("L@s", "L@t")):
    m, sem, lo, hi, agree, n = paired(blocks, a, b)
    v = "RESOLVED" if lo > 0 or hi < 0 else "not resolved"
    print(f"  {a:11} - {b:11} {m:>+7.3f}  sem {sem:.3f}  [{lo:>+7.3f},{hi:>+7.3f}]  "
          f"{v:12} sign agree {agree}/{n}")

print("\nRANKED (per-base centred, averaged over bases)")
cent = collections.defaultdict(list)
for b in bases:
    ms = {c: st.mean(per[(b, c)]) for c in CELLS if per.get((b, c))}
    if len(ms) < len(CELLS):
        continue
    gm = st.mean(ms.values())
    for c, val in ms.items():
        cent[c].append(val - gm)
means = {c: st.mean(v) for c, v in cent.items()}
for c, m in sorted(means.items(), key=lambda x: -x[1]):
    print(f"  {c:12} {m:>+7.3f} dB  (n={len(cent[c])} bases)")
print(f"  span {max(means.values()) - min(means.values()):.3f} dB")

print(f"""
READING IT. Placement is a LARGER lever than presence: the span here is
{max(means.values()) - min(means.values()):.3f} dB against the 0.307 dB span of the whole G0 2x2 in
ablation_factorial.py, and every contrast against L@identity agrees in sign on all 25 blocks. But see
the module docstring -- L@identity sits close enough to the {NOOP_DB} dB no-op on some bases that part
of this gap is degradation rather than improvement, so the honest claim is "do not use the bare
identity placement", not "L@s is worth 1.19 dB".""")
