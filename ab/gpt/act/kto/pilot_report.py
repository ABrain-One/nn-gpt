#!/usr/bin/env python3
"""Generation-1 summary of a v2 run, next to a reference run, with a go/no-go verdict.

Used before launching runs with a new generator model: the untrained model's first
generation must yield enough trainable networks for the loop to get a training signal.

  python -m ab.gpt.act.kto.pilot_report --run out_v2qwen_frozen_s43/nngpt/pref_v2 \
      --reference out_v2_kto_s43/nngpt/pref_v2
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def _summary(run: Path) -> dict:
    cdir = run / "cycle_1"
    m = json.loads((cdir / "metrics.json").read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in (cdir / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    accs = [r["acc"] for r in rows if r.get("acc") is not None]
    b = m.get("bucketing", {})
    s = b.get("screening", {})
    return {
        "samples": m.get("generated", 0),
        "pass the generator's forward check": m.get("extractable", 0),
        "lexical duplicates (not evaluated)": s.get("lex_dup", 0),
        "traceable to a graph": s.get("traced", 0),
        "WL structure already in LEMUR": s.get("struct_seen_lemur", 0),
        "trained successfully (1 epoch)": len(accs),
        "good: novel and >= threshold": b.get("new_desirable", 0),
        "median accuracy": round(statistics.median(accs), 4) if accs else None,
        "best accuracy": round(max(accs), 4) if accs else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generation-1 pilot report")
    ap.add_argument("--run", required=True, help="pref_v2 output dir of the pilot run")
    ap.add_argument("--reference", default=None, help="pref_v2 output dir of a reference run")
    ap.add_argument("--min_trainable", type=float, default=0.30,
                    help="Minimum share of samples that train successfully")
    ap.add_argument("--min_good", type=int, default=5)
    a = ap.parse_args()

    run = _summary(Path(a.run))
    ref = None
    if a.reference and (Path(a.reference) / "cycle_1" / "metrics.json").exists():
        ref = _summary(Path(a.reference))
    print(f"{'generation 1':38s} {'pilot':>10s}" + (f" {'reference':>10s}" if ref else ""))
    for k, v in run.items():
        print(f"{k:38s} {str(v):>10s}" + (f" {str(ref[k]):>10s}" if ref else ""))

    share = run["trained successfully (1 epoch)"] / max(1, run["samples"])
    ok = share >= a.min_trainable
    print(f"\ntrainable share {share:.0%} (needs >= {a.min_trainable:.0%}): {'PASS' if ok else 'FAIL'}")
    if ok and run["good: novel and >= threshold"] < a.min_good:
        print(f"WARNING: only {run['good: novel and >= threshold']} good candidates; runs without anchors "
              "will start with very little training signal")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
