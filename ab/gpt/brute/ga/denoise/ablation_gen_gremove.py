"""Regenerate the GLOBAL-skip arm as residual-REMOVED, because adding one was never possible.

WHAT WENT WRONG THE FIRST TIME. The G1 arm asked "does an input->output global skip help?" by
ADDING one with `skip_operators.add_global_residual`. All 96 of its training runs finished at
exactly 33.021 dB -- the no-op, the PSNR of returning the input untouched. Cause: every one of the
99 generated variant sources ALREADY has a global residual, written as

    identity = x
    ...
    return torch.clamp(self.out_conv(d0) + identity, 0.0, 1.0)

`add_global_residual`'s guard (`_return_adds_input`) only recognised a TOP-LEVEL `x + expr` against
the bare parameter name, so it saw neither the alias (`identity`) nor the nesting (inside
`torch.clamp(...)`). It therefore wrapped an existing residual into `x + clamp(... + identity ...)`,
doubling it, which makes the identity function an attractor of the optimisation.

WHY THE ARM IS INVERTED RATHER THAN REPAIRED. Since every base already carries the skip, "add one"
has no valid treatment to apply -- with the guard fixed, `add_global_residual` correctly declines on
all 32 G0 sources. The question is still answerable, just from the other side: REMOVE the residual
and compare against the base as-is. So the factor becomes

    Gkeep  = the base as generated       (global residual present)   <- already trained, the G0 cells
    Grm    = residual stripped out       (global residual absent)    <- what this script creates

and the contrast is paired within (base, B, L, repeat), exactly like the other factors. Note the
sign flips relative to the original framing: a POSITIVE Gkeep-minus-Grm difference now means the
global skip helps.

The 2x2 of (B, L) is preserved so the new arm slots into the existing analysis: 8 bases x 4 (B,L)
cells x 3 repeats = 96 runs, the same budget the void arm consumed.

Output goes to a SEPARATE directory. The completed run's manifest and results are not touched.
"""
import argparse
import ast
import glob
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import skip_operators as SO   # noqa: E402

ROOT = "/a/mm" if os.path.isdir("/a/mm/nn-gpt") else "/shared/ssd/home/b-b-ahmed/thesis"
OLD = f"{ROOT}/sweep_out/ablation"
POD_ROOT = "/a/mm"            # paths inside the manifest must be pod paths


def macs(src):
    """MACs/params via the repo's own counter, so the number is comparable to the old manifest."""
    try:
        sys.path.insert(0, HERE)
        import edge_cost as EC
        return EC.mac_params(src) if hasattr(EC, "mac_params") else (None, None)
    except Exception:
        return (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/sweep_out/ablation_g2")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()

    vdir = os.path.join(a.out, "variants")
    os.makedirs(vdir, exist_ok=True)

    old_man = json.load(open(f"{OLD}/manifest.json"))
    g0 = [r for r in old_man if r.get("cell", "").startswith("G0") and r.get("valid")]
    print(f"G0 rows in the completed manifest: {len(g0)}")

    man, made, declined, unchanged = [], 0, 0, 0
    for r in g0:
        host_path = r["path"].replace(POD_ROOT, ROOT, 1)
        if not os.path.exists(host_path):
            print(f"  MISSING source {host_path}")
            continue
        src = open(host_path).read()

        # sanity: the operator that broke must now refuse, on every single row
        if SO.add_global_residual(src) is not None:
            sys.exit(f"FATAL add_global_residual still fires on {r['cell']} of {r['base'][:30]} -- "
                     "the deep guard is not working; stop before generating anything")

        rm = SO.remove_global_residual(src)
        if rm is None:
            declined += 1
            continue
        if rm == src:
            unchanged += 1
            continue

        cell = "Grm" + r["cell"][2:]          # G0B1L0 -> GrmB1L0
        name = f"{r['base']}__{cell}.py"
        open(os.path.join(vdir, name), "w").write(rm)
        mm, pm = macs(rm)
        man.append({
            "base": r["base"], "kind": "factorial_gremove", "cell": cell,
            "g": 0, "b": r["b"], "l": r["l"], "b_index": r.get("b_index"),
            "l_target": r.get("l_target"), "valid": True,
            "macs_MMac": mm if mm is not None else r.get("macs_MMac"),
            "params_M": pm if pm is not None else r.get("params_M"),
            "base_macs_MMac": r.get("base_macs_MMac"),
            "mac_delta_pct": r.get("mac_delta_pct"), "over_cap": False,
            "repeats": a.repeats,
            "path": os.path.join(POD_ROOT, "sweep_out",
                                 os.path.basename(a.out), "variants", name),
            "pairs_with": r["cell"],
            "note": "global residual REMOVED; contrast against the same base's "
                    f"{r['cell']} cell, which keeps it",
        })
        made += 1

    print(f"variants written: {made} | operator declined: {declined} | no-change: {unchanged}")
    if not made:
        sys.exit("FATAL produced nothing")

    # the sources must all still be importable and shape-correct
    bad = []
    for m in man:
        p = m["path"].replace(POD_ROOT, ROOT, 1)
        try:
            ast.parse(open(p).read())
        except Exception as e:
            bad.append((os.path.basename(p), str(e)[:60]))
    if bad:
        sys.exit(f"FATAL {len(bad)} unparseable: {bad[:3]}")

    mp = os.path.join(a.out, "manifest.json")
    json.dump(man, open(mp, "w"), indent=1)
    print(f"wrote {mp}  ({len(man)} rows x {a.repeats} repeats = {len(man)*a.repeats} runs)")

    # copy the val/train symlink expectations are handled by ablation_train.sh; nothing else needed
    bases = sorted({m["base"] for m in man})
    print(f"bases: {len(bases)}  cells: {sorted({m['cell'] for m in man})}")


if __name__ == "__main__":
    main()
