"""Run-3 acceptance from the 1-epoch screen (the 50-epoch stage was removed 2026-08-03).

Selection now reads the screen's results_shard*.json directly: candidates are ranked by
their measured 1-epoch PSNR, accepted at --min-db, topped up to --min-k down to
--fallback-db (the classification pipeline's anti-starvation fallback). Accepted records
carry the path to the candidate's (possibly preflight-repaired) source, because without
the training stage these models never enter the DB -- the SFT append step reads the code
straight from the run directory.

BAR SEMANTICS CHANGED WITH THE STAGE REMOVAL. The old 34/33 bars were measured at 50
epochs; at 1 epoch almost nothing crosses 33 (median epoch-1 for hand-written models is
32.6, and eventual-35+ families measure 29-33 there). Defaults here are the epoch-1
calibrated equivalents: 32.5 accept / 31.5 fallback.

stdlib only -- runs on the login node between jobs.
"""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True, help="screen output dir with results_shard*.json")
    ap.add_argument("--run-dir", required=True, help="generation dir holding cand/<id>/candidate.py")
    ap.add_argument("--emit-root", default=None,
                    help="prefix to use for code paths WRITTEN to the output (e.g. the pod "
                         "view /a/mm/... when this runs on the login node); files are checked "
                         "under --run-dir, emitted under this root")
    ap.add_argument("--min-db", type=float, default=32.5)
    ap.add_argument("--fallback-db", type=float, default=31.5)
    ap.add_argument("--min-k", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    recs = []
    for p in (sorted(glob.glob(os.path.join(a.eval_dir, "results_shard*.json")))
              + sorted(glob.glob(os.path.join(a.eval_dir, "results.json")))):
        try:
            recs += json.load(open(p))
        except Exception as e:
            print(f"  WARN unreadable {p}: {e}")

    scored = sorted(((r["id"], r["ep1"]) for r in recs if r.get("ep1") is not None),
                    key=lambda t: -t[1])
    accepted = [(i, d) for i, d in scored if d >= a.min_db]
    fallback = []
    for i, d in scored:
        if len(accepted) + len(fallback) >= a.min_k:
            break
        if a.fallback_db <= d < a.min_db:
            fallback.append((i, d))

    emit_root = a.emit_root or a.run_dir
    kept = missing = 0
    with open(a.out, "w") as f:
        for cid, db in accepted + fallback:
            code = os.path.join(a.run_dir, "cand", cid, "candidate.py")
            if not os.path.isfile(code):
                print(f"  WARN {cid}: no candidate.py at {code}; skipped")
                missing += 1
                continue
            emit = os.path.join(emit_root, "cand", cid, "candidate.py")
            f.write(json.dumps({"name": cid, "db": round(db, 2), "code": emit}) + "\n")
            kept += 1

    if scored:
        print(f"[accept] {len(scored)} screened, {len(accepted)} >= {a.min_db} dB, "
              f"{len(fallback)} fallback (>= {a.fallback_db}), {missing} missing code, "
              f"best {scored[0][1]:.2f} dB @1ep -> {kept} accepted")
    else:
        print("[accept] nothing screened this cycle")


if __name__ == "__main__":
    main()
