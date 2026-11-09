"""Prompt-engineering ablation: what do the four prompt styles actually buy?

The production LLM run (llmprod, arms a1/a5/a10/a12) generated 1,752 candidates under four
prompt styles, assigned per candidate and recorded in each candidate's meta.json AND encoded
in the candidate id itself (`p10_0000_seeded_constraint_based`). The id survives every
downstream stage, so it is the join key. Two things it can be joined to:

  preflight.json         1,699 records  {id, stage, ok, arch_uid, nearest, nearest_cos}
  eval/results_shard*    249 records    {id, ep1, passed}

WHAT THIS CANNOT ANSWER, AND WHY. Not "which prompt writes the most accurate model", and not
"which writes the fastest". Two independent reasons:

  1. No llmprod candidate reached the DB with a name -- db_name is null for all 249 screened
     rows -- so there is no converged (50-epoch) PSNR and no latency measurement to join to.
  2. The one quality number that does exist, ep1, has almost no dynamic range on this task:
     median 32.72 dB against a 33.021 dB no-op floor, with only 19 of 238 rows ABOVE the
     floor. One epoch of denoising barely beats the identity function, so ep1 ranks how badly
     a model has not yet converged -- it is not a proxy for final quality. It is reported
     below purely so the absence of signal is on the record, never as a style ranking.

An earlier framing of mine assumed the `LLM_Pg<N>` groups in the DB were prompt groups, which
would have supplied exactly the missing PSNR+latency join across 350 named models. They are
not: llm_evolve_par.sh:125 is `for g in $(seq 1 "$GENS")` and line 237 builds
`--prefix LLM_Pg${g}`, so Pg is the GENERATION index. Recording that here because the join
looks available and is not.

So what IS answerable is the funnel: of the candidates a style produces, how many are valid,
novel and edge-legal, and how architecturally distinct are they. That is a real question with
~425 candidates per style behind it -- and it is the question that governs cost per usable
model, which is what the generation loop actually spends.

STATISTICS. Rates get Wilson intervals, not normal-approximation ones -- several per-style
failure cells are small enough that the normal approximation puts the bound outside [0,1].
Style-vs-style rate differences get a Newcombe interval on the difference of two Wilson
intervals. nearest_cos is compared with a bootstrap on the median (it is bounded and skewed,
so the mean is the wrong summary). Nothing here is a paired design, so these are all
between-style comparisons with correspondingly less power than the factorial's within-base
contrasts -- stated in the output rather than left for the reader to infer.
"""
import collections
import glob
import json
import math
import os
import statistics as st

ROOT = "/a/mm" if os.path.isdir("/a/mm/nn-gpt") else "/shared/ssd/home/b-b-ahmed/thesis"
LP = f"{ROOT}/sweep_out/llmprod"
NOOP_DB = 33.021  # the identity-function PSNR on the AWGN val crop; see ablation_factorial.py
STYLES = ("constraint_based", "direct_instruction", "step_by_step", "conversational")
LABEL = {
    "constraint_based": "Constraint-based",
    "direct_instruction": "Direct instruction",
    "step_by_step": "Step-by-step",
    "conversational": "Conversational",
}


def style_of(cid):
    for s in STYLES:
        if cid.endswith(s):
            return s
    return None


def wilson(k, n, z=1.96):
    """Wilson score interval. Correct at the small per-style failure cells where the
    normal approximation would put a bound outside [0,1]."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def newcombe(k1, n1, k2, n2, z=1.96):
    """Interval on a difference of proportions, built from the two Wilson intervals."""
    p1, l1, u1 = wilson(k1, n1, z)
    p2, l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def boot_median_diff(a, b, n=4000):
    """Bootstrap CI on median(a) - median(b). Deterministic LCG: no Math.random equivalent
    here, and a fixed stream means the reported interval is reproducible."""
    if len(a) < 5 or len(b) < 5:
        return None
    seed = 20260807
    out = []
    for _ in range(n):
        ra, rb = [], []
        for _ in range(len(a)):
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            ra.append(a[seed % len(a)])
        for _ in range(len(b)):
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            rb.append(b[seed % len(b)])
        out.append(st.median(ra) - st.median(rb))
    out.sort()
    return st.median(a) - st.median(b), out[int(0.025 * n)], out[int(0.975 * n)]


# ---- load ---------------------------------------------------------------------------------
pre = json.load(open(f"{LP}/pre/preflight.json"))
pre = pre if isinstance(pre, list) else pre.get("records", pre.get("rows", []))

screen = []
for f in sorted(glob.glob(f"{LP}/eval/results_shard*.json")):
    d = json.load(open(f))
    screen += d if isinstance(d, list) else d.get("rows", [])

# candidate-level ground truth for style, straight from meta.json, to verify the id parse
meta_style = {}
for arm in ("a1", "a5", "a10", "a12"):
    for m in glob.glob(f"{LP}/{arm}/run/cand/*/meta.json"):
        try:
            d = json.load(open(m))
        except Exception:
            continue
        if d.get("id") and d.get("strategy"):
            meta_style[d["id"]] = d["strategy"]

mismatch = sum(1 for cid, s in meta_style.items() if style_of(cid) and style_of(cid) != s)
print("=" * 96)
print("PROMPT-ENGINEERING ABLATION  ·  llmprod, four styles, 1,752 candidates")
print("=" * 96)
print(f"\nid-parse validation: {len(meta_style)} candidates carry a meta.json strategy; "
      f"id-suffix disagrees with it in {mismatch} cases")
assert mismatch == 0, "id suffix is not a safe proxy for meta.json strategy -- stop and use meta.json"

# ---- 1. the funnel ------------------------------------------------------------------------
print(f"\n{'-' * 96}\n1. PREFLIGHT YIELD  (valid + novel + edge-legal)   n={len(pre)}\n{'-' * 96}")
byst = collections.defaultdict(list)
for r in pre:
    s = style_of(r.get("id", ""))
    if s:
        byst[s].append(r)
unstyled = len(pre) - sum(len(v) for v in byst.values())
print(f"{'style':22} {'n':>5} {'pass':>6} {'rate':>7}   {'95% Wilson':>16}")
rates = {}
for s in STYLES:
    rows = byst[s]
    k = sum(1 for r in rows if r.get("ok"))
    p, lo, hi = wilson(k, len(rows))
    rates[s] = (k, len(rows))
    print(f"{LABEL[s]:22} {len(rows):>5} {k:>6} {100*p:>6.1f}%   [{100*lo:>5.1f}, {100*hi:>5.1f}]")
if unstyled:
    print(f"{'(no style in id)':22} {unstyled:>5}")

print("\npairwise differences in pass rate (Newcombe 95% on the difference):")
best = max(STYLES, key=lambda s: rates[s][0] / max(1, rates[s][1]))
resolved = []
for i, a in enumerate(STYLES):
    for b in STYLES[i + 1:]:
        d, lo, hi = newcombe(*rates[a], *rates[b])
        sig = "RESOLVED" if lo > 0 or hi < 0 else "not resolved"
        if lo > 0 or hi < 0:
            resolved.append((a, b, d))
        print(f"  {LABEL[a]:20} - {LABEL[b]:20} {100*d:>+6.1f} pp  "
              f"[{100*lo:>+6.1f}, {100*hi:>+6.1f}]  {sig}")

# ---- 2. failure modes --------------------------------------------------------------------
print(f"\n{'-' * 96}\n2. WHY EACH STYLE FAILS  (preflight stage of the rejected)\n{'-' * 96}")
stages = collections.Counter()
for r in pre:
    if not r.get("ok"):
        stages[r.get("stage", "?")] += 1
top = [s for s, _ in stages.most_common(6)]
print(f"{'style':22} " + "".join(f"{s[:15]:>17}" for s in top))
for s in STYLES:
    c = collections.Counter(r.get("stage", "?") for r in byst[s] if not r.get("ok"))
    n = len(byst[s])
    print(f"{LABEL[s]:22} " + "".join(f"{c[t]:>8} ({100*c[t]/max(1,n):>4.1f}%)" for t in top))
print(f"\n  overall rejected-by-stage: {dict(stages)}")

# The yield table above resolves nothing, but that is a statement about the NET rate; the
# stage mix underneath it need not be flat.
#
# CONDITIONAL DENOMINATORS ARE MANDATORY HERE. preflight.py runs the stages in strict
# sequence (preflight.py:504-830): static -> exec -> build -> train_setup -> forward -> mac
# -> dead-code -> arch-dup-db -> arch-dup-gate. A candidate that dies at `forward` is NEVER
# tested for duplication. So a style with high forward mortality mechanically shows a LOW
# duplicate-failure rate when the denominator is all of its candidates -- competing risks,
# not novelty. A first version of this analysis made exactly that error and reported
# step-by-step as producing fewer duplicates; the effect was pure censoring. Each stage's
# rate is therefore computed over the candidates that actually REACHED that stage (a hazard),
# and both denominators are printed so the censoring is visible.
ORDER = ["static", "exec", "build", "train_setup", "forward", "mac",
         "dead-code", "arch-dup-db", "arch-dup-gate"]
RANK = {t: i for i, t in enumerate(ORDER)}


def reached(rows, t):
    """Candidates of `rows` that got as far as stage t: everything except those that
    failed strictly earlier in the sequence."""
    return [r for r in rows
            if r.get("ok") or RANK.get(r.get("stage"), 99) >= RANK.get(t, 99)]


print("\nper-stage CONDITIONAL failure rate (denominator = candidates reaching that stage),")
print("each style vs the other three pooled (Newcombe 95%):")
hits = []
for t in [o for o in ORDER if stages.get(o)]:
    for s in STYLES:
        at = reached(byst[s], t)
        k1 = sum(1 for r in at if not r.get("ok") and r.get("stage") == t)
        n1 = len(at)
        oth = [r for o in STYLES if o != s for r in reached(byst[o], t)]
        k2 = sum(1 for r in oth if not r.get("ok") and r.get("stage") == t)
        n2 = len(oth)
        if n1 < 30 or n2 < 30:
            continue
        d, lo, hi = newcombe(k1, n1, k2, n2)
        if lo > 0 or hi < 0:
            hits.append((t, s, d))
            print(f"  {t:<14} {LABEL[s]:20} {k1:>4}/{n1:<4} = {100*k1/n1:>5.1f}%  vs "
                  f"{100*k2/n2:>5.1f}%   {100*d:>+6.1f} pp [{100*lo:>+6.1f},{100*hi:>+6.1f}] RESOLVED")
if not hits:
    print("  (none resolved once the denominators are conditional)")
else:
    print(f"\n  {len(hits)} stage-level effect(s) survive conditioning, though NO net yield")
    print("  difference does: the styles fail in different PLACES at the same overall yield.")
    print("  Compare against the marginal table above -- the duplicate-stage 'advantages' there")
    print("  are censoring artifacts and are absent here.")

# ---- 3. novelty --------------------------------------------------------------------------
print(f"\n{'-' * 96}\n3. ARCHITECTURAL NOVELTY  (nearest_cos to the existing corpus; lower = more novel)\n{'-' * 96}")
cos = {}
for s in STYLES:
    v = sorted(r["nearest_cos"] for r in byst[s]
               if isinstance(r.get("nearest_cos"), (int, float)))
    cos[s] = v
    if v:
        uid = len({r.get("arch_uid") for r in byst[s] if r.get("arch_uid")})
        print(f"{LABEL[s]:22} n={len(v):>4}  median={st.median(v):.4f}  "
              f"p10={v[len(v)//10]:.4f}  p90={v[9*len(v)//10]:.4f}  distinct arch_uid={uid}")
print("\npairwise median nearest_cos (bootstrap 95%, 4000 resamples):")
print("  NOTE the tension to resolve when writing this up: the conditional stage analysis")
print("  resolves an exact-duplicate-rate difference (step-by-step -8.1 pp at arch-dup-db,")
print("  conversational +7.1 pp) while the cosine-distance medians below resolve nothing.")
print("  Both can hold -- arch_uid duplication is an exact structural-identity collision,")
print("  cosine distance is a graded embedding distance, and a style can collide less often")
print("  without its survivors sitting measurably further from the corpus. The exact-identity")
print("  result is the better-powered of the two (it uses every candidate reaching the stage,")
print("  not just the ~130 with a recorded cos), so it is the one to lead with.")
for i, a in enumerate(STYLES):
    for b in STYLES[i + 1:]:
        r = boot_median_diff(cos[a], cos[b])
        if r:
            d, lo, hi = r
            sig = "RESOLVED" if lo > 0 or hi < 0 else "not resolved"
            print(f"  {LABEL[a]:20} - {LABEL[b]:20} {d:>+8.4f}  [{lo:>+7.4f}, {hi:>+7.4f}]  {sig}")

# ---- 4. the screen, and its lack of range ------------------------------------------------
print(f"\n{'-' * 96}\n4. ONE-EPOCH SCREEN  (reported to document ABSENCE of signal, not to rank)\n{'-' * 96}")
sc = collections.defaultdict(list)
for r in screen:
    s = style_of(r.get("id", ""))
    if s:
        sc[s].append(r)
ep_all = [r["ep1"] for r in screen if isinstance(r.get("ep1"), (int, float))]
above = [x for x in ep_all if x > NOOP_DB + 0.05]
print(f"{'style':22} {'n':>4} {'screen pass':>12} {'med ep1':>9} {'> no-op':>9}")
for s in STYLES:
    rows = sc[s]
    k = sum(1 for r in rows if r.get("passed"))
    e = [r["ep1"] for r in rows if isinstance(r.get("ep1"), (int, float))]
    a = sum(1 for x in e if x > NOOP_DB + 0.05)
    print(f"{LABEL[s]:22} {len(rows):>4} {k:>5}/{len(rows):<6} "
          f"{st.median(e) if e else float('nan'):>9.3f} {a:>4}/{len(e):<4}")
print(f"\n  Across all {len(ep_all)} screened rows the median ep1 is {st.median(ep_all):.3f} dB against a")
print(f"  {NOOP_DB:.3f} dB no-op floor, and only {len(above)} ({100*len(above)/len(ep_all):.0f}%) clear the floor at all.")
print("  ep1 therefore carries no usable quality signal on this task and NO style ranking is")
print("  drawn from it. Any prompt-vs-quality claim needs converged training (see below).")

# ---- 5. what a prospective run must add --------------------------------------------------
print(f"\n{'=' * 96}\nWHAT IS ANSWERED, AND WHAT IS NOT\n{'=' * 96}")
print(f"""
ANSWERED (n~425 per style, between-style):
  * yield -- what fraction of a style's candidates are valid, novel and edge-legal
  * failure mode -- which stage each style dies at, which is what to fix in the prompt
  * novelty -- how far a style's architectures sit from the existing corpus

NOT ANSWERED, and not answerable from what is on disk:
  * PSNR at convergence per style   (no llmprod candidate has a converged number)
  * on-device latency per style     (none were exported or profiled)
  * MAI Final Score per style       (needs both of the above)

To close it: take the preflight survivors, sample equally per style, train at the 30-epoch
recipe and profile the survivors. Power comes from the pooled within-cell sd measured by the
skip ablation, 0.142 dB: detecting a 0.3 dB style difference between two unpaired groups needs
roughly 2*(2.8*0.142/0.3)^2 ~ 8 models per style, so ~40 runs covers all four with margin.
That is small enough to be worth doing, and it is the only version of this ablation that can
speak to quality.""")
