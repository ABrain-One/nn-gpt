"""
Analyse the OOD generalization scale-up.

Joins the generation manifest (baseline accuracy per held-out model) with the
per-candidate eval artifacts NNEval writes to each B{i} dir (eval_summary.json —
present even under --no-save_to_db, so nothing here touches the shared DB).
Reports delta-application rate, evaluation success, and the accuracy delta
(LLR-modified vs the model's STORED baseline), overall and per family. The stored
baseline is not a matched control; see ood_epoch_recheck_*.py for the fair comparison.

Run one --tag per adapter (e.g. cyc3old, newbest) to build the before/after
contrast.

Usage:
    python -m ab.gpt.brute.llr.ood_analyze --tag newbest --out-epoch-dir 950
"""
import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from ab.gpt.util.Const import epoch_dir, synth_dir

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _llr_accuracy(bdir: Path):
    """Evaluated accuracy of the LLR-modified candidate, or None if it failed."""
    f = bdir / "eval_summary.json"
    if f.exists():
        try:
            rows = json.loads(f.read_text())
            if rows:
                return float(rows[-1]["accuracy"])
        except Exception:
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out-epoch-dir', type=int, default=950)
    args = ap.parse_args()

    man_path = PROJECT_ROOT / 'out' / 'nngpt' / f'ood_manifest_{args.tag}_A{args.out_epoch_dir}.csv'
    out_root = synth_dir(epoch_dir(args.out_epoch_dir))
    rows = list(csv.DictReader(open(man_path)))
    print(f"Manifest: {man_path} ({len(rows)} candidates, adapter tag={args.tag})")

    results, deltas, fam_deltas = [], [], defaultdict(list)
    n_applied = n_eval = n_improved = 0
    print(f"\n{'B':>3s} {'family':18s} {'strategy':22s} {'base':>7s} {'llr':>7s} {'delta':>8s} {'status'}")
    print('-' * 78)
    for r in rows:
        b = int(r['B'])
        base = float(r['baseline_acc'])
        applied = str(r['delta_applied']).lower() == 'true'
        n_applied += applied
        llr = _llr_accuracy(out_root / f"B{b}")
        if llr is None:
            status = 'gen-failed' if not applied else 'eval-failed'
            print(f"{b:3d} {r['family']:18s} {r['strategy']:22s} {base:7.4f} {'--':>7s} {'--':>8s} {status}")
            results.append({**r, 'llr_acc': '', 'delta': '', 'status': status})
            continue
        n_eval += 1
        d = llr - base
        deltas.append(d); fam_deltas[r['family']].append(d)
        improved = d > 0
        n_improved += improved
        status = 'IMPROVED' if improved else 'worse'
        print(f"{b:3d} {r['family']:18s} {r['strategy']:22s} {base:7.4f} {llr:7.4f} {d:+8.4f} {status}")
        results.append({**r, 'llr_acc': round(llr, 4), 'delta': round(d, 4), 'status': status})

    print(f"\n{'=' * 60}")
    print(f"adapter tag        : {args.tag}")
    print(f"candidates         : {len(rows)}")
    print(f"delta applied      : {n_applied}/{len(rows)}  ({100*n_applied/max(len(rows),1):.0f}%)")
    print(f"evaluated          : {n_eval}/{len(rows)}")
    if deltas:
        print(f"improved over base : {n_improved}/{n_eval}  ({100*n_improved/n_eval:.0f}%)")
        print(f"mean delta         : {st.mean(deltas):+.4f}   median: {st.median(deltas):+.4f}")
        print(f"best / worst delta : {max(deltas):+.4f} / {min(deltas):+.4f}")
        print("per-family mean delta:")
        for fam, ds in sorted(fam_deltas.items(), key=lambda x: -st.mean(x[1])):
            print(f"    {fam:20s} n={len(ds)}  mean {st.mean(ds):+.4f}")

    res_path = PROJECT_ROOT / 'out' / 'nngpt' / f'ood_results_{args.tag}.csv'
    with open(res_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys())); w.writeheader(); w.writerows(results)
    print(f"\nWrote {res_path}")


if __name__ == '__main__':
    main()
