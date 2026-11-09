"""Two-stage evaluation of LLM candidates: 1-epoch screen, then full train into the DB.

STAGE 1 -- 1-EPOCH SCREEN (cheap, ~30 s/model)
    Train each candidate for a single epoch and keep only those clearing --gate dB.

    THE GATE VALUE IS MEASURED, NOT CHOSEN. Across the 18 canonical families the epoch-1
    range is 31.07-33.14 dB, and the six that finish >= 35.0 dB after 50 epochs all sit at
    ep1 >= 32.708. So 32.7 retains every eventual winner. A "34 dB at 1 epoch" rule -- the
    intuitive reading of "only approve 34 dB ones" -- would approve 1 of 18, and that one is
    LiteDenoiseNet, whose 36.5 ep1 comes from a different recipe and which FINISHES 14th at
    34.92. The intuitive gate rejects every winner and keeps the one anomaly.

    WHAT THE SCREEN CAN AND CANNOT DO. The bottom four by ep1 all finish in the bottom half,
    so it reliably kills the bad. But DenoiseSubPixel is 7th at ep1 and finishes 1st, while
    DenoiseLinearMix has the highest ep1 of the sub-35 group and finishes 10th. Use it as a
    FLOOR, never as a ranking. (The 5-epoch proxy has the same property: rho=+0.263 to
    epoch 50 -- weak for ranking, fine for triage. That low rho is partly range restriction:
    those 18 span 0.6 dB. Across the far wider spread of LLM output it discriminates better.)

STAGE 2 -- DATASET TRAIN, 5 EPOCHS, MATCHING THE GA
    Survivors are retrained for --epochs with save_to_db=True and prefix LLM-Denoise.

    WHY 5 AND NOT 50. Checked against the DB rather than assumed. LEMUR's stat.epoch column
    is epoch_max -- a CONFIG KEY -- not a per-epoch curve index, so:
        a GA model contributes ONE row (final dB at epoch_max=5) plus its real per-epoch
        curve as _epoch_stats/<hash>/{1..5}.json on disk;
        the 16 hand-written seeds carry 50 rows because they were run across 50 epoch_max
        configs, which is a different and much more expensive protocol.
    Training LLM survivors for 50 epochs would therefore cost ~9x more AND produce entries
    that do not line up with the 223 GA models already in the DB. At a measured 40.7 s per
    epoch (the GA's own duration field), 90 survivors is ~6 h at 5 epochs versus ~52 h at 50.

    5 epochs makes the LLM contribution directly comparable to the GA contribution, which is
    the entire point of running both arms.

STAGE 3 -- CHAMPIONS ONLY (separate script, not run here)
    The 5-epoch number is NOT a quality ranking: ep5 -> ep50 Spearman on the canonical corpus
    is +0.263. So a small top-K is re-trained to 50 epochs for any headline dB claim. That is
    a deliberate second pass over ~10-20 models, not something to spend on all survivors.

Nothing here scores a model that was never trained: a candidate with no PSNR is reported as
such rather than defaulting to 0.

Usage:
  python screen_and_train.py --run-dir /a/mm/sweep_out/llm_gen/run --gate 32.7 --epochs 5
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

TASK, DATASET, METRIC = "img-denoising", "denoise", "psnr"
PSNR_SCALE = 48.0
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, "/a/mm/nn-gpt/ab/gpt/brute/ast"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import naming  # noqa: E402  (needs HERE on sys.path first)


def curve_from(stat_dir, scale=PSNR_SCALE):
    """[(epoch, dB)] from the per-epoch JSONs check_nn writes."""
    out = []
    for p in glob.glob(os.path.join(stat_dir, "**", "*.json"), recursive=True):
        base = os.path.basename(p)[:-5]
        if not base.isdigit():
            continue
        try:
            with open(p) as f:
                d = json.load(f)
            # check_nn writes a LIST wrapping one dict, not a bare dict. Reading it as a
            # dict raised AttributeError straight into the bare except below, so every
            # curve came back empty and every last5 was nan while the DB rows were fine --
            # a silent failure that looked like "no per-epoch stats were produced".
            if isinstance(d, list):
                d = d[0] if d else {}
            a = d.get("accuracy")
            if a is not None:
                out.append((int(base), a * scale))
        except Exception as e:
            print(f"[curve] skipped {p}: {type(e).__name__}: {e}", flush=True)
    return sorted(out)


def arm_of(cid):
    """'cold' / 'seeded' from the candidate id; '' when it cannot be told.

    Reads either separator. Candidate ids are run-local and keep their
    underscores (`p10_0000_seeded_constraint_based`) -- they never become a model
    name -- but the model names we now mint use hyphens, and this is called on
    both. See naming.arm_of, which this delegates to.
    """
    return naming.arm_of(cid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="/a/mm/sweep_out/llm_gen/run")
    ap.add_argument("--out", default="/a/mm/sweep_out/llm_eval")
    ap.add_argument("--gate", type=float, default=32.7,
                    help="minimum epoch-1 dB to qualify for full training (measured: 32.7 "
                         "retains all 6 families that finish >=35 dB)")
    # 5, not 50: matches the GA's epoch_max=5 contribution exactly (see module docstring).
    # Raise it only for the deliberate champions pass over a small top-K.
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=6)
    # LLM-Denoise, not DenoiseLLM: provenance must be readable at a glance in the DB, and
    # the GA writes DenoiseGA-*. A shared "Denoise*" stem made both sort together and made
    # "which models came from where" a prefix-substring question. LLM-Denoise-* sorts apart
    # from every Denoise* family and generated-model name in the tree.
    # Hyphen, not underscore: '_' is LEMUR's stat-dir field separator, so an
    # underscored model name breaks the 4-way split in Util.py:61 and aborts the
    # import of the whole batch. This default used to be "LLM_Denoise"; the 372
    # models it produced had to be renamed on disk after review. See naming.py.
    ap.add_argument("--prefix", default="LLM-Denoise")
    # The DB name is the ONLY thing that survives once the run dirs age out, and both arms
    # would otherwise write the identical LLM-Denoise-* stem -- making "was this cold or
    # seeded?" unanswerable after the fact. With this on, the arm is baked into the name:
    #   LLM-Denoise-cold-<hash>   LLM-Denoise-seeded-<hash>
    ap.add_argument("--arm-in-prefix", action="store_true", default=True,
                    help="append -cold/-seeded to the prefix, from the candidate id")
    ap.add_argument("--no-arm-in-prefix", dest="arm_in_prefix", action="store_false")
    # Stage 1 has already run for these ids (results_shard*.json); re-screening 26 survivors
    # would cost another ~35 min of GPU to recompute a number we already have.
    ap.add_argument("--skip-screen", action="store_true",
                    help="go straight to stage 2 for --only-ids (stage 1 already done)")
    ap.add_argument("--screen-only", action="store_true",
                    help="stage 1 only; do not full-train survivors")
    ap.add_argument("--limit", type=int, default=0)
    # SHARDING is stage-1 only by design. Stage 2 writes to the LEMUR SQLite DB, which lives
    # on NFSv3 -- concurrent writers there corrupt it rather than blocking. So the parallel
    # pass must run --screen-only (no DB writes at all), and the survivors are full-trained
    # afterwards by a SINGLE pod. Never run a sharded job without --screen-only.
    ap.add_argument("--shard", type=int, default=0, help="this shard index (0-based)")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="total shards; candidate i is handled by shard i %% num-shards")
    ap.add_argument("--run-dirs", nargs="*", default=None,
                    help="multiple run dirs to pool (overrides --run-dir)")
    # Spending a GPU to discover a model cannot forward is the single largest waste in this
    # pipeline: ~70% of the 516 LLM candidates failed at build or first forward, at ~80 s of
    # GPU each. preflight.py answers the same question on CPU in under a second per model.
    # Pass its pass_ids.txt here and the GPU only ever sees candidates that can run.
    ap.add_argument("--only-ids", default="",
                    help="file of candidate ids (one per line), e.g. preflight pass_ids.txt")
    a = ap.parse_args()
    if a.num_shards > 1 and not a.screen_only:
        # Sharded TRAINING is allowed only against a pod-local database, which is then
        # folded back by merge_shard_dbs.py from a single writer. The check below is
        # deliberately not "did the caller promise": it resolves the DB LEMUR will actually
        # write and refuses if that file sits on the shared mount. A flag alone would make
        # the corrupting configuration one typo away, and NFSv3 does not fail loudly -- the
        # writers interleave and the damage is found later, in a DB that is the only copy of
        # ~400 uncommitted models.
        try:
            from ab.nn.util.Const import db_file
            resolved = os.path.realpath(str(db_file))
        except Exception as e:
            raise SystemExit(f"REFUSING: --num-shards>1 for training but the DB path could "
                             f"not be resolved to verify it is pod-local ({e})")
        shared_roots = ("/a/mm/", "/shared/ssd/")
        if any(resolved.startswith(r) for r in shared_roots):
            raise SystemExit(
                f"REFUSING: sharded training would write {resolved}, which is on shared "
                f"storage. Concurrent writers on an NFS-hosted SQLite file interleave "
                f"rather than block, and corrupt it. Give each shard a pod-local DB copy "
                f"(PRIVATE_DB=1) and merge with merge_shard_dbs.py.")
        print(f"[shard {a.shard}/{a.num_shards}] pod-local DB verified: {resolved}")

    # FAIL FAST ON THE WRONG ROOT. The first smoke run died 4/4 with
    #   "No module named 'ab.nn.loader.denoise'"  ... Evaluating /a/mm/nn-gpt/out/nn/tmp/...
    # because it ran with cd=<denoise dir> and PYTHONPATH=/a/mm/nn-gpt, so `ab.nn` bound to
    # nn-gpt's package (no denoise loader) and LEMUR rooted itself in nn-gpt. The GA gets
    # this right by running from /a/mm/nn-dataset. Check it here, where the message can say
    # what to do, instead of once per candidate 400 times.
    try:
        import ab.nn.loader.denoise  # noqa: F401
    except Exception as e:
        raise SystemExit(
            f"FATAL: ab.nn.loader.denoise not importable ({type(e).__name__}: {e}).\n"
            f"  cwd={os.getcwd()}\n  PYTHONPATH={os.environ.get('PYTHONPATH')}\n"
            f"Run from the nn-dataset tree, e.g.:\n"
            f"  cd /a/mm/nn-dataset && PYTHONPATH=/a/mm/nn-dataset python {__file__} ...")
    # The denoise loader is CWD-relative (os.path.join(os.getcwd(), 'train')), so the data
    # must be reachable from here too -- otherwise it silently tries to re-download.
    for d in ("train", "validation"):
        if not os.path.isdir(os.path.join(os.getcwd(), d)):
            raise SystemExit(f"FATAL: '{d}/' not found under cwd={os.getcwd()}; the denoise "
                             f"loader resolves data relative to CWD.")
    import ab.nn.api as api

    os.makedirs(a.out, exist_ok=True)
    roots = a.run_dirs if a.run_dirs else [a.run_dir]
    cands = []
    for r in roots:
        cands += glob.glob(os.path.join(r, "cand", "*", "candidate.py"))
    cands = sorted(cands)
    n_all = len(cands)
    if a.only_ids:
        keep = {l.strip() for l in open(a.only_ids) if l.strip()}
        cands = [c for c in cands
                 if os.path.basename(os.path.dirname(c)) in keep]
        print(f"[only-ids] {a.only_ids}: {len(keep)} ids -> {len(cands)} of {n_all} "
              f"candidates kept ({n_all - len(cands)} skipped before touching the GPU)",
              flush=True)
    if a.limit:
        cands = cands[:a.limit]
    total = len(cands)
    if a.num_shards > 1:
        # Strided, not blocked: candidate ids are generated in strategy round-robin order, so
        # a contiguous block would hand one shard a skewed strategy mix and make per-shard
        # pass rates incomparable.
        cands = [c for i, c in enumerate(cands) if i % a.num_shards == a.shard]
    print(f"=== screen+train | {len(cands)} of {total} candidates "
          f"(shard {a.shard}/{a.num_shards}) | gate ep1 >= {a.gate:.2f} dB "
          f"| full train {a.epochs} ep ===", flush=True)

    results, t0 = [], time.time()
    for i, src in enumerate(cands, 1):
        cid = os.path.basename(os.path.dirname(src))
        code = open(src).read()
        arm = arm_of(cid)
        # '-' joins the arm on, never '_': this line is what minted the 372
        # LLM_Denoise_cold-* names that broke the stat-dir split. naming.check
        # refuses to let a bad prefix through even if --prefix is passed one.
        prefix = f"{a.prefix}-{arm}" if (a.arm_in_prefix and arm) else a.prefix
        naming.check(prefix, "model prefix")
        rec = {"id": cid, "arm": arm, "prefix": prefix, "ep1": None, "passed": None,
               "db_name": None, "final_db": None, "curve": None, "error": None}

        # With --skip-screen the stage-1 verdict was already established by the sharded run
        # and the caller has passed only its survivors via --only-ids; re-running it would
        # spend ~80 s of GPU per model to recompute a number already on disk.
        if a.skip_screen:
            rec["passed"] = True

        # ---- stage 1: one epoch ----
        sdir = os.path.join(a.out, "screen", cid)
        os.makedirs(sdir, exist_ok=True)
        prm = {"lr": a.lr, "batch": a.batch, "transform": "denoise",
               "epoch": 1, "num_workers": a.num_workers}
        if not a.skip_screen:
            try:
                _, acc, _, _ = api.check_nn(code, TASK, DATASET, METRIC, prm,
                                            save_to_db=False, prefix=prefix, save_path=sdir)
                rec["ep1"] = acc * PSNR_SCALE
            except Exception as e:
                rec["error"] = f"screen: {type(e).__name__}: {e}"[:200]

        if not a.skip_screen and rec["ep1"] is None:
            print(f"[{i}/{len(cands)}] {cid:<28} SCREEN-FAIL  {rec['error']}", flush=True)
            results.append(rec); continue

        if not a.skip_screen:
            rec["passed"] = rec["ep1"] >= a.gate
            verdict = "PASS" if rec["passed"] else "below gate"
            print(f"[{i}/{len(cands)}] {cid:<28} ep1={rec['ep1']:6.3f}  {verdict}", flush=True)
        else:
            print(f"[{i}/{len(cands)}] {cid:<28} prefix={prefix}", flush=True)

        if not rec["passed"] or a.screen_only:
            results.append(rec); continue

        # ---- stage 2: full train, into the DB ----
        fdir = os.path.join(a.out, "full", cid)
        os.makedirs(fdir, exist_ok=True)
        prm = dict(prm, epoch=a.epochs)
        try:
            # save_pth_weights: a DB row records a SCORE, never the network that earned it.
            # Without this the entire generated corpus is architectures plus numbers, and any
            # model we later want to deploy, fine-tune or inspect has to be retrained from
            # scratch. Weights land in $LEMUR_CKPT_DIR/<db_name>/best_model.pth -- set that
            # env var to persistent storage, or the pod-local root takes them down with it.
            # Only on the real train; the 1-epoch screen below writes nothing.
            db_name, acc, _, _ = api.check_nn(code, TASK, DATASET, METRIC, prm,
                                              save_to_db=True, prefix=prefix,
                                              save_path=fdir, save_pth_weights=True)
            rec["db_name"] = db_name
            rec["final_db"] = acc * PSNR_SCALE
            c = curve_from(fdir)
            rec["curve"] = c
            if c:
                tail = [d for _, d in c[-5:]]
                rec["last5"] = sum(tail) / len(tail)
            print(f"      -> {db_name}  final={rec['final_db']:.3f} dB  "
                  f"last5={rec.get('last5', float('nan')):.3f}  n_ep={len(c)}", flush=True)
        except Exception as e:
            rec["error"] = f"train: {type(e).__name__}: {e}"[:200]
            print(f"      -> TRAIN-FAIL {rec['error']}", flush=True)
        results.append(rec)

    # ---- summary ----
    scr = [r for r in results if r["ep1"] is not None]
    passed = [r for r in scr if r["passed"]]
    trained = [r for r in results if r["final_db"] is not None]
    print(f"\n=== summary ({time.time() - t0:.0f}s) ===")
    print(f"  candidates          {len(results)}")
    print(f"  screened ok         {len(scr)}   (screen-fail {len(results) - len(scr)})")
    print(f"  passed gate {a.gate:<5}   {len(passed)}"
          + (f"   ({len(passed)/len(scr):.0%} of screened)" if scr else ""))
    print(f"  full-trained        {len(trained)}")
    if trained:
        best = max(trained, key=lambda r: r.get("last5") or r["final_db"])
        print(f"  best                {best['id']}  last5="
              f"{best.get('last5', best['final_db']):.3f} dB  ({best['db_name']})")
    name = "results.json" if a.num_shards == 1 else f"results_shard{a.shard}.json"
    out = os.path.join(a.out, name)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  written             {out}")


if __name__ == "__main__":
    raise SystemExit(main())
