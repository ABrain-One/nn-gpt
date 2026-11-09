"""Turn one generation's rejections into the next generation's prompt block.

THE GAP THIS CLOSES. The diversity gate refuses ~32 of 147 candidates every generation and
tells the generator nothing. The LLM proposes, gets silently refused, and proposes the same
kind of thing again -- across sixteen generations of the first run that produced 812
near-duplicate rejections and no adaptation whatsoever. The gate had the number all along:
it computes the cosine to the nearest accepted model and, until now, discarded it.

WHAT MAKES A USEFUL BLOCK, and it is not a list of ids. "g4_0001 was rejected at 0.972
against DenoiseGA-c307912c" is unusable -- the model has never seen either file. What it can
act on is the EDIT PATTERN: on this corpus, near-duplicates are overwhelmingly the same layer
sequence with different kernel sizes, activations and pooling types, while genuine novelty
comes from changed topology. That is measurable from the corpus, so the block states the
pattern and backs it with this generation's own numbers rather than a general exhortation.

TOKENS ARE NOT FREE. The block is capped: aggregate counts, the few worst offenders, and at
most --examples truncated snippets. A prompt that spends 2,000 tokens on failure analysis is
2,000 tokens not spent on reference architectures, and the references are what the wide-pool
fix proved actually drives diversity.

  python3 rejection_digest.py --pre DIR/gen4/pre --out DIR/gen5/rejections.txt
"""
import argparse
import collections
import json
import os
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True, help="a generation's pre/ directory")
    ap.add_argument("--run", default=None, help="that generation's run/ dir, for snippets")
    ap.add_argument("--examples", type=int, default=2)
    # MUST match the gate. This was hardcoded to 0.95 in the message text while the gate ran
    # at 0.90, so the feedback told the model a candidate at 0.93 was safe when it would be
    # rejected -- actively wrong instruction, worse than no feedback at all.
    ap.add_argument("--cos-max", type=float, default=0.90)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    p = os.path.join(a.pre, "preflight.json")
    if not os.path.exists(p):
        print(f"no preflight.json at {p}; writing empty digest")
        open(a.out, "w").write("")
        return 0
    recs = json.load(open(p))

    dup = [r for r in recs if r.get("stage") == "arch-dup-gate"]
    db = [r for r in recs if r.get("stage") == "arch-dup-db"]
    ok = [r for r in recs if r.get("ok")]
    fwd = [r for r in recs if r.get("stage") == "forward"]
    n = len(recs)
    if not n:
        open(a.out, "w").write("")
        return 0

    cos = [r["nearest_cos"] for r in dup if isinstance(r.get("nearest_cos"), (int, float))]
    lines = []
    lines.append("FEEDBACK FROM THE PREVIOUS GENERATION — read this before designing.")
    lines.append("")
    lines.append(f"Of {n} architectures proposed last round, {len(ok)} were accepted.")
    if dup:
        lines.append(f"{len(dup)} were REJECTED as near-duplicates of models that already "
                     f"exist in the dataset.")
        if cos:
            cos_sorted = sorted(cos, reverse=True)
            lines.append(f"Their similarity to the closest existing model ranged "
                         f"{min(cos):.3f}–{max(cos):.3f} (median {cos_sorted[len(cos)//2]:.3f}). "
                         f"Anything at or above {a.cos_max:.2f} is discarded without "
                         f"training.")
    if db:
        lines.append(f"{len(db)} more were exact architectural duplicates of dataset entries.")
    if fwd:
        lines.append(f"{len(fwd)} failed to run at all (tensor shapes did not line up).")
    lines.append("")

    # The pattern, stated concretely. Measured from real rejected pairs on this corpus.
    lines.append("WHAT GETS REJECTED AS A DUPLICATE — measured, not hypothetical:")
    lines.append("  Keeping the same layer sequence and changing only")
    lines.append("    • kernel sizes (5→3, 1→7)")
    lines.append("    • activation functions (ELU→LeakyReLU)")
    lines.append("    • pooling type (AdaptiveAvgPool2d→AdaptiveMaxPool2d)")
    lines.append("    • channel widths")
    lines.append(f"  produces a model scoring >=0.98 against its source -- far above the "
                 f"{a.cos_max:.2f} limit. It will be discarded.")
    lines.append("")
    lines.append("WHAT COUNTS AS GENUINELY NEW — change the TOPOLOGY:")
    lines.append("  • a different number of resolution stages")
    lines.append("  • a different downsampling mechanism (strided conv vs pixel-unshuffle vs pooling)")
    lines.append("  • different skip wiring (which stages connect to which, and how they merge)")
    lines.append("  • a different feature-merging operator (concat vs add vs attention-weighted)")
    lines.append("  • combining mechanisms from TWO of the reference models rather than "
                 "varying one")
    lines.append("")

    if a.examples and a.run:
        shown = 0
        worst = sorted([r for r in dup if isinstance(r.get("nearest_cos"), (int, float))],
                       key=lambda r: -r["nearest_cos"])
        for r in worst:
            if shown >= a.examples:
                break
            f = os.path.join(a.run, "cand", r["id"], "candidate.py")
            if not os.path.exists(f):
                continue
            try:
                src = open(f).read()
            except OSError:
                continue
            # __init__ only: it carries the architecture, and the training boilerplate is
            # identical across every candidate and would waste the budget.
            m = re.search(r"def __init__\(self.*?(?=\n    def )", src, re.S)
            body = (m.group(0) if m else src)[:900]
            shown += 1
            lines.append(f"REJECTED EXAMPLE {shown} — scored {r['nearest_cos']:.3f} against an "
                         f"existing model. Do not produce anything of this shape:")
            lines.append("```python")
            lines.append(body.rstrip())
            lines.append("```")
            lines.append("")

    txt = "\n".join(lines).rstrip() + "\n"
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write(txt)
    print(f"=== rejection digest ===")
    print(f"  proposed {n} | accepted {len(ok)} | near-dup {len(dup)} | "
          f"exact-dup {len(db)} | shape-fail {len(fwd)}")
    if cos:
        print(f"  nearest-cosine range {min(cos):.3f}–{max(cos):.3f}")
    named = collections.Counter(r.get("nearest") for r in dup if r.get("nearest"))
    if named:
        print(f"  most-collided-with models:")
        for k, v in named.most_common(5):
            print(f"    {v:4d}  {k}")
    print(f"  wrote {a.out} ({len(txt)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
