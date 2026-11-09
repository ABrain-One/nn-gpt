"""Repair the one-line harness-contract defect that killed 81% of the cold arm.

THE DEFECT WAS IN THE PROMPT, NOT THE MODELS. Denoise_shared_blocks.json's INTERFACE
CONTRACT told the model to USE `self.device` twice -- `self.to(self.device)` in __init__ and
"move both to self.device" in learn() -- and never once told it to ASSIGN it. `nn.Module`
has no `.device` attribute, so a model that writes the natural PyTorch thing,
`self.to(device)`, raises

    AttributeError: 'Net' object has no attribute 'device'

on the first batch and scores zero no matter how good the architecture is. Measured over the
516 generated candidates:

    cold    122/150 (81.3%) use self.device but never assign it   <- dead on arrival
    seeded    3/366 ( 0.8%)

The arms differ that starkly because the seeded arm sees reference code that happens to
contain `self.device = device`; the cold arm had only the contract, and the contract was
wrong. Screening as-is would have reported "cold prompts produce 5x more broken models" --
an artefact of my prompt bug, not a property of zero-reference prompting.

WHY REPAIR RATHER THAN REGENERATE. The missing line is harness boilerplate, not
architecture: inserting it changes no layer, no topology, no MAC count. Regenerating would
also work and is the more purist option, but it would discard 122 architectures to fix a
line the LLM was never asked to write. The repair is recorded per file (`_repaired` in
meta.json) so repaired candidates can be excluded from any claim about raw prompt yield.
The prompt itself is fixed too, so the next run needs no repair.

Idempotent: a candidate that already assigns self.device is left untouched.

  python repair_device_attr.py --run-dirs .../cold/run .../seeded/run [--dry-run]
"""
import argparse
import glob
import json
import os
import re

ASSIGN = re.compile(r'self\.device\s*=')
USE = re.compile(r'self\.device\b')
# The insertion point: the super().__init__() call inside Net.__init__. Matching super()
# rather than the def line means the assignment lands after initialisation (required -- you
# cannot set attributes on an nn.Module before super().__init__()) and inherits its indent.
SUPER = re.compile(r'^(\s*)super\(\s*\)\.__init__\(\s*\)\s*$|^(\s*)super\([^)]*\)\.__init__\([^)]*\)\s*$')


def repair(src):
    """Return (new_src, status)."""
    if ASSIGN.search(src):
        return src, "already-ok"
    if not USE.search(src):
        return src, "no-use"          # never references self.device; nothing to fix
    lines = src.splitlines(keepends=True)
    for i, ln in enumerate(lines):
        m = SUPER.match(ln.rstrip("\n"))
        if not m:
            continue
        indent = m.group(1) or m.group(2) or "        "
        lines.insert(i + 1, f"{indent}self.device = device\n")
        return "".join(lines), "repaired"
    return src, "no-super-call"        # cannot place it safely; leave alone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    counts, touched = {}, []
    for r in a.run_dirs:
        for f in sorted(glob.glob(os.path.join(r, "cand", "*", "candidate.py"))):
            src = open(f).read()
            new, status = repair(src)
            counts[status] = counts.get(status, 0) + 1
            if status != "repaired":
                continue
            touched.append(os.path.basename(os.path.dirname(f)))
            if a.dry_run:
                continue
            # Keep the untouched original next to it -- the raw response is already
            # archived, but this makes the diff inspectable without re-extracting.
            orig = f + ".orig"
            if not os.path.exists(orig):
                open(orig, "w").write(src)
            open(f, "w").write(new)
            mp = os.path.join(os.path.dirname(f), "meta.json")
            try:
                meta = json.load(open(mp)) if os.path.exists(mp) else {}
                meta["_repaired"] = "inserted self.device = device (prompt omitted it)"
                json.dump(meta, open(mp, "w"), indent=2)
            except Exception as e:
                print(f"  WARN meta {mp}: {e}")

    print(f"{'DRY RUN ' if a.dry_run else ''}device-attr repair")
    for k in sorted(counts):
        print(f"  {k:<16} {counts[k]:4d}")
    print(f"  repaired ids   : {len(touched)}")
    if counts.get("no-super-call"):
        print("  NOTE: 'no-super-call' candidates were left alone and will still fail; "
              "they need regeneration, not a patch.")


if __name__ == "__main__":
    raise SystemExit(main())
