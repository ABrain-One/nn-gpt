#!/usr/bin/env python3
"""Cross-experiment tables, as in the Run_11 metric reference, from the per-experiment
plot_kto_cycles outputs (<root>/<experiment>/) and the novelty comparison
(<root>/novelty_compare/novelty_comparison.csv).

  experiments_summary.csv  headline accuracy (t-based 95% CIs) + throughput / text-novelty totals
  novelty_summary.csv      base (Jaccard) and graph (WL) novelty over all generations,
                           Wilson 95% CIs, sorted by graph-novelty rate

  python -m ab.gpt.act.kto.collect_run_metrics --root metrics_v2_c21
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ab.gpt.act.kto.plot_kto_cycles import wilson_ci


def _r(v, nd=2):
    return "" if v is None else round(v, nd)


def experiments_table(root: Path):
    rows = []
    for d in sorted(p for p in root.iterdir() if (p / "kto_cycle_summary.csv").exists()):
        cyc = list(csv.DictReader(open(d / "kto_cycle_summary.csv", encoding="utf-8")))
        card_f = d / "kto_card_summary.json"
        card = json.loads(card_f.read_text(encoding="utf-8")) if card_f.exists() else {}
        nov = sum(int(r["new_desirable"]) for r in cyc)
        dup = sum(int(r["not_novel"]) for r in cyc)
        rows.append({
            "experiment": d.name,
            "valid": card.get("valid"), "denominator": card.get("denominator"),
            "valid_pct": _r(card.get("valid_pct"), 1),
            "mean_all_valid": _r(card.get("mean_all")),
            "mean_all_ci_lo": _r((card.get("mean_all_ci") or [None, None])[0]),
            "mean_all_ci_hi": _r((card.get("mean_all_ci") or [None, None])[1]),
            "mean_ge_threshold": _r(card.get("mean_ge")),
            "mean_ge_ci_lo": _r((card.get("mean_ge_ci") or [None, None])[0]),
            "mean_ge_ci_hi": _r((card.get("mean_ge_ci") or [None, None])[1]),
            "n_ge_threshold": card.get("n_ge"),
            "median": _r(card.get("median")), "best": _r(card.get("best")),
            "ge_threshold_rate": _r(card.get("ge_rate"), 1),
            "cycles": len(cyc),
            "generated": sum(int(r["generated"]) for r in cyc),
            "evaluated": sum(int(r["evaluated"]) for r in cyc),
            "novel_new_desirable": nov, "not_novel": dup,
            "unique": f"{nov}/{nov + dup}",
            "unique_pct": _r(100 * nov / (nov + dup), 1) if nov + dup else "",
            "desirable_accum": cyc[-1]["desirable_total"] if cyc else "",
            "cycles_trained": f"{sum(1 for r in cyc if r['trained'] == 'True')}/{len(cyc)}",
        })
    return rows


def novelty_table(root: Path):
    f = root / "novelty_compare" / "novelty_comparison.csv"
    if not f.exists():
        return []
    acc = {}
    for r in csv.DictReader(open(f, encoding="utf-8")):
        a = acc.setdefault(r["run"], [0, 0, 0, 0])
        a[0] += 1
        a[1] += int(r["total"])
        a[2] += int(r["base_novel"])
        a[3] += int(r["graph_novel"])
    rows = []
    for run, (cycles, total, base, graph) in acc.items():
        blo, bhi = wilson_ci(base, total)
        glo, ghi = wilson_ci(graph, total)
        rows.append({
            "experiment": run, "cycles": cycles, "total_gens": total,
            "base_novel": base, "base_pct": _r(100 * base / total, 1) if total else "",
            "base_ci_lo": _r(blo * 100, 1), "base_ci_hi": _r(bhi * 100, 1),
            "graph_novel": graph, "graph_pct": _r(100 * graph / total, 1) if total else "",
            "graph_ci_lo": _r(glo * 100, 1), "graph_ci_hi": _r(ghi * 100, 1),
        })
    rows.sort(key=lambda r: -(r["graph_pct"] or 0))
    return rows


def _write(rows, path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cols = list(rows[0])
    widths = [max(len(c), *(len(str(r[c])) for r in rows)) for c in cols]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print("  ".join(str(r[c]).ljust(w) for c, w in zip(cols, widths)))
    print(f"-> {path}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-experiment metric tables")
    ap.add_argument("--root", required=True)
    root = Path(ap.parse_args().root)
    _write(experiments_table(root), root / "experiments_summary.csv")
    _write(novelty_table(root), root / "novelty_summary.csv")


if __name__ == "__main__":
    main()
