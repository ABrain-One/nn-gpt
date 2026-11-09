"""Prompt-engineering ablation on the DEPLOYABILITY block: does telling the LLM about
unconvertible operators actually stop it emitting them?

WHY A SECOND PROMPT ABLATION EXISTS. ablation_prompt.py analyses the four prompt STYLES over the
1,752 llmprod candidates, and it is explicit about what it cannot answer: no llmprod candidate
reached the DB with a name, so there is no converged PSNR and no latency to join to, and the one
quality number that does exist (ep1) has a median of 32.72 dB against a 33.021 dB no-op floor --
one epoch of denoising barely beats the identity function. That ablation is therefore a funnel
study, and it is observational: the four styles were assigned, never crossed with anything.

This one is different in the two ways that matter. It is CONTROLLED -- the treatment is a single
contiguous block of prompt text that the generator can switch on or off (--deploy-guidance) -- and
its endpoint needs NO TRAINING AND NO GPU, because convertibility is a property of the emitted
source, not of trained weights. That removes both of the blockers above at once.

THE ENDPOINT, AND WHY IT IS WORTH AN EXPERIMENT. 97 of the 1,582 corpus models carry
AdaptiveMaxPool2d and can never be converted to TFLite, so they can never be profiled and can
never ship, however they rank on PSNR. 18 of those are LLM-generated. The prompt never mentioned
the constraint, so the generator had no way to know. The question is whether saying so helps, and
by how much -- not whether it sounds like it should.

THREE ARMS, NOT TWO. The real block adds ~1,250 characters to a ~7,100-character prompt, and
prompt length alone changes LLM behaviour. An on-vs-off contrast therefore confounds CONTENT with
LENGTH. The placebo arm substitutes a length-matched block (1,203 vs 1,254 chars, 96%) of true but
non-actionable text about the training harness, which mentions no operator, no API and no
architectural choice -- verified by --check-design. Read the arms as:

    on  vs  placebo   the effect of the CONTENT of the guidance      <- the scientific claim
    on  vs  off       the effect of ADDING the guidance              <- the engineering decision
    placebo vs off    whether prompt length alone moves the endpoint <- the confound, measured

FACTORIAL WITH STYLE, BECAUSE THE STYLES ARE ALREADY THERE. Each candidate carries its style in
its id, so crossing 3 arms x 4 styles costs nothing extra and answers a question the style-only
ablation could not: whether guidance works uniformly or only survives certain framings. Cells are
small, so the style breakdown is reported as a secondary, descriptive result with intervals, and
the primary endpoint is pooled over styles.

STATISTICS. Rates get Wilson intervals and differences get Newcombe intervals, both reused from
ablation_prompt.py rather than reimplemented, so the two ablations cannot drift apart in method.
There is no multiplicity correction on the three pairwise arm contrasts and that is stated in the
output; with one pre-specified primary contrast (on vs placebo) it is not needed, and the other two
are read as supporting.

  python ablation_prompt_deploy.py --check-design              # verify the arms before generating
  python ablation_prompt_deploy.py --power --baseline 0.06     # how many candidates per cell
  python ablation_prompt_deploy.py --run-dirs D1 D2 D3         # analyse (static endpoint only)
  python ablation_prompt_deploy.py --run-dirs D1 --convert     # + real TFLite conversion
"""
import argparse
import collections
import glob
import json
import math
import os
import re
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from npu_gate import SWEEP, unconvertible, unconvertible_warnings  # noqa: E402


def _borrow_stats_helpers():
    """Take wilson/newcombe/style_of/STYLES/LABEL from ablation_prompt.py without running it.

    WHY NOT A PLAIN IMPORT. ablation_prompt.py is a straight-through script: its helper
    definitions end at the `# ---- load` marker and everything after that is module-level analysis
    which reads llmprod's preflight.json and prints a full report. `import ablation_prompt` would
    therefore execute that entire report as a side effect of loading this file.

    WHY NOT COPY THE FUNCTIONS. Two ablations quoting differently-computed Wilson intervals is
    exactly the kind of silent divergence that makes two tables incomparable, and these are
    intended to be read side by side.

    WHY NOT REFACTOR THAT FILE. It is a working analysis script whose output has already been
    quoted; wrapping ~190 lines in a main() to enable an import is a large edit with a real chance
    of changing a published number, for no analytical gain.

    So: read the source, cut at the marker, exec only the prefix. The cut is asserted rather than
    assumed -- if the marker moves or a helper disappears, this fails loudly at startup instead of
    silently supplying something else. The numeric self-check guards against the prefix executing
    but yielding different maths than expected.
    """
    src_path = os.path.join(_HERE, "ablation_prompt.py")
    if not os.path.exists(src_path):
        sys.exit(f"FATAL {src_path} missing -- it is the source of the shared statistics helpers")
    src = open(src_path).read()
    marker = "# ---- load"
    if marker not in src:
        sys.exit(f"FATAL '{marker}' marker not found in ablation_prompt.py; the helper prefix "
                 f"cannot be isolated. Check the file before trusting any interval here.")
    ns = {"__name__": "_ablation_prompt_helpers"}
    exec(compile(src[:src.index(marker)], "ablation_prompt.py(prefix)", "exec"), ns)
    need = ("wilson", "newcombe", "style_of", "STYLES", "LABEL")
    missing = [n for n in need if n not in ns]
    if missing:
        sys.exit(f"FATAL ablation_prompt.py no longer defines {missing} before '{marker}'")
    # Both helpers return a 3-tuple -- wilson is (p, lo, hi) and newcombe is (diff, lo, hi), not
    # the bare (lo, hi) they read like. This assertion is here because assuming two caught it.
    got = ns["wilson"](5, 100)
    if len(got) != 3:
        sys.exit(f"FATAL borrowed wilson() returned {len(got)} values, expected (p, lo, hi)")
    p, lo, hi = got
    if not (abs(p - 0.05) < 1e-9 and 0.0 < lo < 0.05 < hi < 0.15):
        sys.exit(f"FATAL borrowed wilson() implausible for 5/100: {got}")
    return (ns["LABEL"], ns["STYLES"], ns["newcombe"], ns["style_of"], ns["wilson"])


LABEL, STYLES, newcombe, style_of, wilson = _borrow_stats_helpers()

# Known arm names in report order. v1 uses on/placebo/off; v2 uses v2/v2-placebo/off. `present`
# is derived from what is actually in the data rather than from this tuple -- hardcoding it made
# the v2 run report a single arm and print "ONLY ONE ARM PRESENT" while all three were loaded.
ARMS = ("on", "v2", "placebo", "v2-placebo", "off")
# _HERE is .../nn-gpt/ab/gpt/brute/ga/denoise; the prompt configs are at .../nn-gpt/ab/gpt/conf/...
# (three levels up, matching llm_denoise.py's CONF_TEST_DIR).
SHARED_CONF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(_HERE))),
    "conf", "prompt", "test", "Denoise_shared_blocks.json")


# ---------------------------------------------------------------- design check
def cmd_check_design(a):
    """Verify the three arms are comparable BEFORE spending generation on them.

    Two failure modes this catches, both of which would silently invalidate the experiment:
    a placebo that has drifted away from the real block's length (reintroducing the length
    confound it exists to remove), and a placebo that has acquired actionable content (making it a
    second treatment rather than a control).
    """
    if not os.path.exists(SHARED_CONF):
        sys.exit(f"FATAL shared prompt config not found at {SHARED_CONF}")
    d = json.load(open(SHARED_CONF))
    # v1 = the convertibility block (rules 10-12); v2 adds graph size and attention.
    rk, pk = (("deploy", "deploy_placebo") if a.variant == "v1"
              else ("deploy_v2", "deploy_v2_placebo"))
    for k in (rk, pk):
        if k not in d:
            sys.exit(f"FATAL shared config has no '{k}' block -- the ablation cannot run")
    print(f"variant {a.variant}: real='{rk}' placebo='{pk}'")
    real, plac = "\n".join(d[rk]), "\n".join(d[pk])
    ratio = len(plac) / len(real)
    print(f"real block    {len(real):5d} chars, {len(d['deploy']):2d} lines")
    print(f"placebo block {len(plac):5d} chars, {len(d['deploy_placebo']):2d} lines  "
          f"({ratio:.0%} of real)")
    bad = ratio < 0.85 or ratio > 1.15

    # A control must not carry the treatment. These are the terms the real block trades in; any of
    # them in the placebo means it is teaching something.
    #
    # MATCHED AS TOKENS, NOT SUBSTRINGS. The first version tested plain `in`, and "amin" duly
    # matched the word "naming" -- a false leak report on text that says nothing about operators.
    # A design checker that fails on innocent prose gets overridden by hand, which defeats it, so
    # the short operator names carry word boundaries. `convert` deliberately stays a prefix match
    # ("converted", "conversion") because the placebo must not discuss conversion at all.
    LEAK = (r"adaptive[_ ]?max", r"\bmaxpool", r"register_buffer", r"\bfft\b", r"\bamax\b",
            r"\bamin\b", r"\btflite\b", r"\bconvert", r"\blowering\b", r"\bself\.", r"forward\(")
    leaked = sorted({p for p in LEAK if re.search(p, plac, re.I)})
    print(f"leaked treatment terms in placebo: {leaked or 'none'}")
    if leaked:
        bad = True

    # The placebo must also not accidentally repeat guidance already present elsewhere, which
    # would reinforce rather than control.
    others = "\n".join(sum((d[k] for k in ("task", "constraints", "interface", "output")), []))
    dup = [ln.strip() for ln in d["deploy_placebo"] if ln.strip() and ln.strip() in others]
    print(f"placebo lines duplicated from other blocks: {len(dup)}")
    if dup:
        bad = True
    print("\nDESIGN " + ("REJECTED -- fix the placebo before generating" if bad else "OK"))
    sys.exit(1 if bad else 0)


# ---------------------------------------------------------------- power
def cmd_power(a):
    """Candidates per cell needed to detect a given reduction in the unconvertible rate.

    RUN THIS BEFORE GENERATING. The measured LLM unconvertible rate is about 5-6%, which is a low
    base rate, and low base rates need large n: at 6% -> 3% a two-sided 0.05 test with 80% power
    needs ~1,150 candidates PER ARM. Generating 100 per arm and reporting "no significant effect"
    would be a statement about the sample size, not about the guidance, and this project has
    already had one null result (the refab A/B) that had to be defended on exactly this ground.
    """
    z_a, z_b = 1.959963985, 0.8416212336          # two-sided 0.05, 80% power
    p0 = a.baseline
    print(f"baseline unconvertible rate p0 = {p0:.1%} (measured: 5.1% of 350 LLM corpus models, "
          f"6.1% of all 1,582)")
    print(f"\n{'target rate':>12}{'relative cut':>14}{'n per arm':>11}{'total 3 arms':>14}")
    for frac in (0.75, 0.5, 0.25, 0.0):
        p1 = p0 * frac
        if p1 == p0:
            continue
        pbar = (p0 + p1) / 2
        n = ((z_a * math.sqrt(2 * pbar * (1 - pbar))
              + z_b * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2) / ((p0 - p1) ** 2)
        n = int(math.ceil(n))
        print(f"{p1:>11.1%}{1 - frac:>13.0%}{n:>11,}{n * 3:>14,}")
    print("\nn is per ARM, pooled over styles. The 3x4 style breakdown will be descriptive at any")
    print("feasible run size -- do not power the experiment on the per-cell contrasts.")
    print("Generation is the only cost: the endpoint is static, so no GPU and no training.")


# ---------------------------------------------------------------- load
def load(run_dirs):
    """[{id, arm, style, extracted, code}] over every candidate in every run dir.

    The arm comes from meta.json (`deploy_guidance`), never from the directory name: a run may mix
    arms, and llmprod already demonstrated that a factor recoverable only from the producing run is
    a factor lost forever. Candidates whose meta.json predates the flag are reported as `unknown`
    and excluded from the contrasts rather than silently folded into a control.
    """
    rows = []
    for rd in run_dirs:
        for md in sorted(glob.glob(os.path.join(rd, "cand", "*", "meta.json"))):
            try:
                m = json.load(open(md))
            except Exception:
                continue
            cdir = os.path.dirname(md)
            cp = os.path.join(cdir, "candidate.py")
            code = open(cp).read() if os.path.exists(cp) else None
            arm = m.get("deploy_guidance", "unknown")
            if arm is True:
                arm = "on"          # tolerate the first revision, which stored a bool
            elif arm is False:
                arm = "off"
            rows.append({"id": m.get("id") or os.path.basename(cdir), "arm": str(arm),
                         "style": style_of(m.get("id") or os.path.basename(cdir)),
                         "strategy": m.get("strategy"), "extracted": bool(code), "code": code,
                         "run": rd, "dir": cdir})
    return rows


def classify(rows):
    """Attach the static endpoint to each row: parses, ban hits, advisory hits."""
    for r in rows:
        r["parses"] = False
        r["bans"], r["warns"] = [], []
        if not r["code"]:
            continue
        try:
            compile(r["code"], "cand", "exec")
            r["parses"] = True
        except SyntaxError:
            continue
        r["bans"] = [rid for rid, _, _ in unconvertible(r["code"])]
        r["warns"] = [rid for rid, _ in unconvertible_warnings(r["code"])]
    return rows


# ---------------------------------------------------------------- real conversion
def cmd_convert_stage(rows, work):
    """Actually convert every parsing candidate, via the existing npu_gate --convert path.

    STATIC IS THE PRIMARY ENDPOINT AND REAL CONVERSION IS THE CHECK ON IT. The bans explain only
    232 of the 350 known conversion failures (66%), so a static-clean candidate can still fail to
    convert for reasons no rule captures. Reporting only the static rate would flatter the
    treatment: guidance can only move what the rules can see, whereas the decision-relevant
    quantity is how many candidates actually produce a deployable graph.
    """
    src = os.path.join(work, "src")
    os.makedirs(src, exist_ok=True)
    # THE FILENAME MUST CARRY THE ARM, AND OMITTING IT SILENTLY DESTROYED A RESULT.
    # The candidate id is IDENTICAL across arms by design: --run-id is pinned so candidate i sees
    # the same references everywhere, which makes the comparison paired but also makes every arm
    # emit `0000_seeded_step_by_step` and friends. Staging by id alone dropped all three arms into
    # one directory where they overwrote each other -- 715 parsing candidates became 250 files,
    # and the analysis then reported an identical "75/250, 30.0% converts" for all three arms,
    # which is arithmetically impossible and was the tell. Prefixing with the arm keeps them
    # distinct, and `stem` is carried on the row so the join back is by construction rather than
    # by reconstructing the name.
    staged = 0
    for r in rows:
        if not r["parses"]:
            continue
        r["stem"] = f"{r['arm']}--{r['id']}"
        with open(os.path.join(src, r["stem"] + ".py"), "w") as fh:
            fh.write(r["code"])
        staged += 1
    stems = {r.get("stem") for r in rows if r.get("stem")}
    if staged != len(stems):
        sys.exit(f"FATAL staged {staged} files but only {len(stems)} distinct stems -- "
                 f"candidates are still colliding; the conversion column would be meaningless")
    print(f"staged {staged} parsing candidates -> {src}")
    if not staged:
        return {}
    rc = subprocess.call([sys.executable, os.path.join(_HERE, "npu_gate.py"),
                          "--convert", "--src", src, "--work", work])
    rep = os.path.join(work, "convert.json")
    if not os.path.exists(rep):
        print(f"WARNING no convert.json (npu_gate exit {rc}); real-conversion column unavailable")
        return {}
    return {x["model"]: x for x in json.load(open(rep))}


# ---------------------------------------------------------------- report
def rate(k, n):
    p, lo, hi = wilson(k, n)
    return f"{k:>4}/{n:<4} {p:>6.1%} [{lo:.1%},{hi:.1%}]"


def cmd_report(a):
    rows = classify(load(a.run_dirs))
    if not rows:
        sys.exit("no candidates found -- check --run-dirs")
    conv = {}
    if a.convert:
        conv = cmd_convert_stage(rows, a.work)
        for r in rows:
            # Join on the arm-qualified stem, never the bare id -- see cmd_convert_stage.
            c = conv.get(r.get("stem"))
            r["converted"] = bool(c and c.get("ok")) if c else None

    byarm = collections.Counter(r["arm"] for r in rows)
    print(f"{len(rows)} candidates | arms: {dict(byarm)}")
    unknown = byarm.get("unknown", 0)
    if unknown:
        print(f"  {unknown} candidates predate the --deploy-guidance flag and are EXCLUDED "
              f"from every contrast below")
    # order by ARMS where known, then anything unexpected, so a new arm name is still reported
    present = [x for x in ARMS if byarm.get(x)]
    present += [x for x in byarm if x not in ARMS and x != "unknown"]
    if len(present) < 2:
        print("\nONLY ONE ARM PRESENT -- nothing to contrast. Generate the control arms with "
              "`--deploy-guidance off` and `--deploy-guidance placebo`, then re-run.")

    # ---- funnel, pooled over styles (the primary endpoint)
    print("\n=== FUNNEL, pooled over styles (Wilson 95% CI) ===")
    hdr = f"{'arm':<9}{'n':>6}{'extracted':>22}{'parses':>22}{'ban-clean':>22}"
    if a.convert:
        hdr += f"{'converts':>22}"
    print(hdr)
    stats = {}
    for arm in present:
        rs = [r for r in rows if r["arm"] == arm]
        n = len(rs)
        ex = sum(1 for r in rs if r["extracted"])
        pa = sum(1 for r in rs if r["parses"])
        # DENOMINATOR IS THE PARSING CANDIDATES, not all of them: a candidate that produced no code
        # cannot carry a banned operator, so pooling it in would let an arm look cleaner simply by
        # failing to emit anything.
        clean = sum(1 for r in rs if r["parses"] and not r["bans"])
        line = f"{arm:<9}{n:>6}{rate(ex, n):>22}{rate(pa, n):>22}{rate(clean, pa):>22}"
        stats[arm] = {"n": n, "extracted": ex, "parses": pa, "clean": clean, "denom": pa}
        if a.convert:
            cn = [r for r in rs if r.get("converted") is not None]
            ck = sum(1 for r in cn if r["converted"])
            line += f"{rate(ck, len(cn)):>22}"
            stats[arm].update(conv_k=ck, conv_n=len(cn))
        print(line)

    # ---- the contrasts
    if len(present) >= 2:
        print("\n=== ARM CONTRASTS on the unconvertible rate (Newcombe 95% CI on the difference) ===")
        print("negative difference = the first arm emits FEWER unconvertible models = guidance helped")
        treat = "v2" if "v2" in present else "on"
        plac = "v2-placebo" if "v2-placebo" in present else "placebo"
        pairs = [(treat, plac, "PRIMARY: effect of guidance CONTENT"),
                 (treat, "off", "effect of ADDING the block"),
                 (plac, "off", "confound check: prompt LENGTH alone")]
        for x, y, why in pairs:
            if x not in stats or y not in stats:
                continue
            sx, sy = stats[x], stats[y]
            kx, ky = sx["denom"] - sx["clean"], sy["denom"] - sy["clean"]   # unconvertible counts
            if not (sx["denom"] and sy["denom"]):
                continue
            px, py = kx / sx["denom"], ky / sy["denom"]
            d, lo, hi = newcombe(kx, sx["denom"], ky, sy["denom"])
            sig = "RESOLVED" if (lo > 0 or hi < 0) else "not resolved"
            print(f"  {x:>8} - {y:<8} {px:>6.1%} - {py:<6.1%} = {d:>+6.1%} "
                  f"[{lo:+.1%},{hi:+.1%}]  {sig:<13} {why}")
        print("  No multiplicity correction: one pre-specified primary contrast (on vs placebo);")
        print("  the other two are supporting. Read them that way.")

    # ---- which rule moved
    print("\n=== WHICH BAN, by arm (share of parsing candidates) ===")
    rules = sorted({b for r in rows for b in r["bans"]})
    if not rules:
        print("  no banned construct appeared in any arm -- either the guidance is fully effective")
        print("  or the sample is too small to contain one; check n against --power")
    for rid in rules:
        cells = []
        for arm in present:
            rs = [r for r in rows if r["arm"] == arm and r["parses"]]
            k = sum(1 for r in rs if rid in r["bans"])
            cells.append(f"{arm} {k}/{len(rs)} ({k/len(rs) if rs else 0:.1%})")
        print(f"  {rid:<26} " + "   ".join(cells))
    print("\n=== ADVISORY constructs (fft, amax -- not banned, poor conversion odds) ===")
    for rid in sorted({w for r in rows for w in r["warns"]}) or []:
        cells = []
        for arm in present:
            rs = [r for r in rows if r["arm"] == arm and r["parses"]]
            k = sum(1 for r in rs if rid in r["warns"])
            cells.append(f"{arm} {k}/{len(rs)} ({k/len(rs) if rs else 0:.1%})")
        print(f"  {rid:<26} " + "   ".join(cells))

    # ---- style breakdown, descriptive
    print("\n=== ARM x STYLE, unconvertible rate (descriptive -- cells are small) ===")
    print(f"{'style':<22}" + "".join(f"{a_:>20}" for a_ in present))
    for st in STYLES:
        cells = ""
        for arm in present:
            rs = [r for r in rows if r["arm"] == arm and r["style"] == st and r["parses"]]
            k = sum(1 for r in rs if r["bans"])
            cells += f"{(f'{k}/{len(rs)}' if rs else '-'):>20}"
        print(f"{LABEL.get(st, st):<22}{cells}")

    out = os.path.join(a.work, "prompt_deploy_ablation.json")
    os.makedirs(a.work, exist_ok=True)
    json.dump({"n": len(rows), "arms": dict(byarm), "stats": stats,
               "rows": [{k: v for k, v in r.items() if k != "code"} for r in rows]},
              open(out, "w"), indent=1)
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="*", default=[])
    ap.add_argument("--variant", choices=["v1", "v2"], default="v1",
                    help="which guidance block pair to check/analyse")
    ap.add_argument("--check-design", action="store_true",
                    help="verify the placebo is length-matched and content-free, then exit")
    ap.add_argument("--power", action="store_true", help="n per arm for a detectable effect")
    ap.add_argument("--baseline", type=float, default=0.06,
                    help="assumed unconvertible rate without guidance (default 0.06)")
    ap.add_argument("--convert", action="store_true",
                    help="also run real TFLite conversion (CPU, ~3.4 s/model)")
    ap.add_argument("--work", default=os.path.join(SWEEP, "prompt_deploy_abl"))
    a = ap.parse_args()
    if a.check_design:
        cmd_check_design(a)
    elif a.power:
        cmd_power(a)
    elif a.run_dirs:
        cmd_report(a)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
