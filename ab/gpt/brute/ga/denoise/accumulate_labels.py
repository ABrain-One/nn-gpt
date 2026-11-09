"""Accumulate a GROWING label store for the v2 accuracy predictor.

The v1 predictor targets ep5 PSNR, which on MAI is mostly undertraining noise (spread
0.92 dB, below the noisy input) -> it can't rank (live Spearman ~0.05). The fix is to
retrain the predictor on a HIGH-SIGNAL target: 50ep PSNR (4x the spread) or SIDD PSNR
(9 dB spread). Those labels are scarce right now (~15 at 50ep, ~4 SIDD), so this script
consolidates every label as it accrues, so a v2 retrain is one command once ~100-150 exist.

It scans all result sources and merges by checksum (with name fallback for SIDD):
  - ep5 PSNR   <- seed_ga_*_cache.json (ablation + production)
  - 50ep PSNR  <- fulltrain_results.json (Gate 1 + future promotion full-trains)
  - SIDD PSNR  <- sidd_sweep_results.json + generalize_sidd_results.json (+ future)

Re-run it anytime new results land (post-production full-train, SIDD validation, next run);
it's idempotent and append/refresh. Writes predictor_labels.json + predictor_labels.csv.

    cd /a/mm/nn-dataset
    PYTHONPATH=. python .../accumulate_labels.py
"""
import csv, glob, json, os

_HERE = os.path.dirname(os.path.abspath(__file__))
PSNR_SCALE = 48.0


def _load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def main():
    labels = {}   # key = checksum (or "name:<name>" fallback) -> record

    def rec(key):
        return labels.setdefault(key, dict(checksum=None, name=None, macs=None, params=None,
                                           ep5_psnr=None, psnr50=None, sidd_psnr=None,
                                           deployable=None))

    # --- ep5 from all caches (ablation + production) ---
    for cf in glob.glob(os.path.join(_HERE, "seed_ga_*_cache.json")):
        d = _load(cf) or {}
        for cks, v in d.items():
            if not isinstance(v, dict) or v.get("edge_reject"):
                continue
            p = v.get("psnr")
            if not p or p <= 0:
                continue
            r = rec(cks)
            r["checksum"] = cks
            r["name"] = r["name"] or v.get("name")
            r["macs"] = r["macs"] or v.get("macs")
            r["params"] = r["params"] or v.get("params")
            r["ep5_psnr"] = round(p, 3)

    # --- 50ep from fulltrain results (Gate 1 + future) ---
    ft = _load(os.path.join(_HERE, "fulltrain_results.json"))
    for x in (ft.get("runs", ft) if isinstance(ft, (dict, list)) else []) if ft else []:
        if not isinstance(x, dict) or not x.get("checksum"):
            continue
        p = x.get("psnr50")
        if not p or p <= 0:
            continue
        r = rec(x["checksum"])
        r["checksum"] = x["checksum"]
        r["name"] = r["name"] or x.get("name")
        r["macs"] = r["macs"] or x.get("macs")
        r["params"] = r["params"] or x.get("params")
        r["deployable"] = x.get("deployable")
        r["psnr50"] = round(p, 3)

    # --- SIDD: map name->checksum via the sweep manifest, then attach by name ---
    name2cks = {}
    man = _load(os.path.join(_HERE, "sidd_sweep_manifest.json")) or []
    for m in man:
        cks = os.path.basename(m.get("src", ""))[:-3] if m.get("src") else None
        if m.get("name") and cks:
            name2cks[m["name"]] = cks
    for sf in ("sidd_sweep_results.json", "generalize_sidd_results.json"):
        sd = _load(os.path.join(_HERE, sf))
        for x in (sd if isinstance(sd, list) else [sd]) if sd else []:
            if not isinstance(x, dict):
                continue
            p = x.get("sidd_psnr")
            if not p or p <= 0:
                continue
            nm = x.get("name")
            key = name2cks.get(nm) or (nm if nm in labels else f"name:{nm}")
            r = rec(key)
            r["name"] = r["name"] or nm
            r["sidd_psnr"] = round(p, 3)

    rows = list(labels.values())
    have50 = [r for r in rows if r["psnr50"]]
    haveS = [r for r in rows if r["sidd_psnr"]]
    both = [r for r in rows if r["psnr50"] and r["ep5_psnr"]]

    json.dump(rows, open(os.path.join(_HERE, "predictor_labels.json"), "w"), indent=2)
    with open(os.path.join(_HERE, "predictor_labels.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["checksum", "name", "macs", "params",
                                          "ep5_psnr", "psnr50", "sidd_psnr", "deployable"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("=== v2 predictor label store ===")
    print(f"  total architectures:      {len(rows)}")
    print(f"  with ep5 PSNR:            {len([r for r in rows if r['ep5_psnr']])}")
    print(f"  with 50ep PSNR (target):  {len(have50)}   <- need ~100-150 to retrain")
    print(f"  with SIDD PSNR (target):  {len(haveS)}")
    print(f"  with ep5 AND 50ep (paired for calibration): {len(both)}")
    print(f"  -> predictor_labels.json + predictor_labels.csv")
    print(f"  RETRAIN v2 when 50ep count reaches ~100-150 (or use SIDD target).")


if __name__ == "__main__":
    main()
