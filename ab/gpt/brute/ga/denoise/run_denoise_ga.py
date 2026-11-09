"""Denoise genetic-algorithm search driver (skip-focused).

Wires the generic GA core (ga_core.py) to a PSNR fitness function that renders a
skip-configurable model (denoise_template.py) and scores it with the nn-dataset
training hook ``ab.nn.api.check_nn`` at a short proxy-epoch budget.

Fitness = accuracy returned by check_nn (real dB = accuracy * 48). Failed / NaN
candidates (AccuracyException etc.) get fitness 0.0 so the GA discards them.

Run from a CWD under the nn-dataset repo (or pass --nn-dataset) so
``import ab.nn.api`` resolves to LEMUR. Example smoke test:

    export PYTHONPATH=/a/mm/nn-gpt:/a/mm/nn-dataset
    cd /a/mm/nn-dataset
    python -m ab.gpt.brute.ga.denoise.run_denoise_ga \\
        --pop 6 --gens 2 --epoch 1 --elitism 1 --smoke
"""

import argparse
import json
import os
import sys
import time
import traceback

# --- resolve nn-dataset (LEMUR root derived from CWD at import time) ---------
_NN_DATASET = os.environ.get("NN_DATASET_ROOT", "/a/mm/nn-dataset")


def _bootstrap(nn_dataset):
    if nn_dataset and os.path.isdir(nn_dataset):
        os.chdir(nn_dataset)
        if nn_dataset not in sys.path:
            sys.path.insert(0, nn_dataset)


# Run as a plain script (not `-m ab.gpt...`): the editable nn-dataset `ab`
# package shadows nn-gpt's namespace `ab`, so we add our own dir to sys.path and
# import flat — the same pattern denoise_search.py uses to coexist with LEMUR.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from denoise_template import render_model  # noqa: E402
from ga_core import (  # noqa: E402
    GeneticAlgorithmEngine,
    RandomResettingMutation,
    SinglePointCrossover,
    TournamentSelection,
)

TASK, DATASET, METRIC = "img-denoising", "denoise", "psnr"
PSNR_SCALE = 48.0

# Skip-focused search space. global_residual + fusion are the skip genes;
# depth/width/act give crossover structural material.
SEARCH_SPACE = {
    "global_residual": [True, False],
    "fusion": ["add", "concat"],
    "depth": [2, 3, 4],
    "width": [16, 32, 48],
    "act": ["relu", "gelu"],
}


def make_fitness(epoch, batch, lr, num_workers, save_to_db, log):
    import ab.nn.api as api

    seen = {}

    def fitness(chromosome):
        key = json.dumps(chromosome, sort_keys=True)
        if key in seen:  # avoid re-training identical genotypes within a run
            return seen[key]
        code = render_model(chromosome)
        prm = {"lr": lr, "batch": batch, "transform": "denoise",
               "epoch": epoch, "num_workers": num_workers}
        t0 = time.time()
        try:
            # Weights are saved exactly when the model enters the DB: a stat row is a score,
            # not a network, so without this every generated model would have to be retrained
            # before it could be deployed or inspected. Tied to save_to_db so the surrogate's
            # throwaway evaluations do not fill the disk. Destination is $LEMUR_CKPT_DIR,
            # which must point at persistent storage -- the GA runs on a pod-local root.
            name, acc, acc_to_time, code_quality = api.check_nn(
                code, TASK, DATASET, METRIC, prm,
                save_to_db=save_to_db, prefix="DenoiseGA",
                save_pth_weights=save_to_db)
            psnr = acc * PSNR_SCALE
            # persist DB-saved generated model as ab/nn/nn/<name>.py (see run_seed_ga
            # ._export_nn_source): keeps generated models as git-trackable source.
            if save_to_db and psnr > 0 and name and str(name).startswith("Denoise"):
                try:
                    import ab.nn as _abnn
                    _p = os.path.join(os.path.dirname(os.path.abspath(_abnn.__file__)),
                                      "nn", f"{name}.py")
                    if not os.path.exists(_p):
                        open(_p, "w").write(code)
                except Exception as _e:
                    log(f"  (nn-file export skipped for {name}: {type(_e).__name__})")
                # also write per-epoch stat JSON from the DB (check_nn saves stat rows to the
                # DB but not to the stat/train tree); reverse of save_stat, canonical format.
                try:
                    import json as _json, sqlite3 as _sql
                    from ab.nn.util.Const import db_file as _dbf
                    from ab.nn.util.Util import model_stat_dir as _msd
                    from ab.nn.util.db.Calc import sanitize_for_json as _san
                    _c = _sql.connect(str(_dbf)); _c.row_factory = _sql.Row; _cu = _c.cursor()
                    _by = {}
                    for _r in _cu.execute("SELECT task,dataset,metric,transform,epoch,accuracy,"
                                          "duration,prm FROM stat WHERE nn=?", (name,)).fetchall():
                        _prm = {k: v for k, v in _cu.execute(
                            "SELECT name,value FROM prm WHERE uid=?", (_r["prm"],)).fetchall()}
                        _t = dict(_prm); _t["accuracy"] = _r["accuracy"]
                        _t["duration"] = _r["duration"]; _t["uid"] = _r["prm"]
                        _t.setdefault("transform", _r["transform"])
                        # canonical repo schema: epoch_max (not epoch), no num_workers
                        if "epoch_max" not in _t and "epoch" in _t:
                            _t["epoch_max"] = _t.pop("epoch")
                        else:
                            _t.pop("epoch", None)
                        _t.pop("num_workers", None)
                        _by.setdefault(((_r["task"], _r["dataset"], _r["metric"], name),
                                        _r["epoch"]), []).append(_t)
                    _c.close()
                    for (_cfg, _ep), _trs in _by.items():
                        _d = _msd(_cfg); _d.mkdir(parents=True, exist_ok=True)
                        _trs = sorted(_trs, key=lambda x: x["accuracy"], reverse=True)
                        with open(_d / f"{_ep}.json", "w") as _f:
                            _json.dump(_san(_trs), _f, indent=4)
                except Exception as _e:
                    log(f"  (nn-stat export skipped for {name}: {type(_e).__name__})")
            log(f"  OK  {chromosome} -> {psnr:.3f} dB "
                f"(acc {acc:.4f}, {time.time() - t0:.0f}s, db={name})")
            fit = float(acc)
        except Exception as e:  # AccuracyException/NaN/build error -> unfit
            log(f"  DEAD {chromosome} -> {type(e).__name__}: "
                f"{str(e)[:120]} ({time.time() - t0:.0f}s)")
            fit = 0.0
        seen[key] = fit
        return fit

    return fitness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=6)
    ap.add_argument("--gens", type=int, default=2)
    ap.add_argument("--epoch", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--elitism", type=int, default=1)
    ap.add_argument("--tournament", type=int, default=2)
    ap.add_argument("--mutation-rate", type=float, default=0.3)
    ap.add_argument("--checkpoint", default="denoise_ga_checkpoint.pkl")
    ap.add_argument("--results", default="denoise_ga_results.json")
    ap.add_argument("--nn-dataset", default=_NN_DATASET)
    ap.add_argument("--save-to-db", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="print the search space and one rendered model, then run")
    args = ap.parse_args()

    _bootstrap(args.nn_dataset)

    def log(msg):
        print(msg, flush=True)

    log(f"=== Denoise GA (skip-focused) | pop={args.pop} gens={args.gens} "
        f"epoch={args.epoch} ===")
    log(f"Search space: {SEARCH_SPACE}")

    if args.smoke:
        sample = {k: v[0] for k, v in SEARCH_SPACE.items()}
        rendered = render_model(sample)
        log(f"\n--- sample genotype {sample} renders {len(rendered)} chars ---")

    fitness = make_fitness(args.epoch, args.batch, args.lr, args.num_workers,
                           args.save_to_db, log)

    engine = GeneticAlgorithmEngine(
        population_size=args.pop,
        search_space=SEARCH_SPACE,
        elitism_count=args.elitism,
        selection_strategy=TournamentSelection(args.tournament),
        crossover_strategy=SinglePointCrossover(),
        mutation_strategy=RandomResettingMutation(args.mutation_rate),
        checkpoint_path=args.checkpoint,
    )

    t0 = time.time()
    best = engine.run(args.gens, fitness)
    dt = time.time() - t0

    result = {
        "best_chromosome": best.chromosome if best else None,
        "best_fitness": best.fitness if best else None,
        "best_psnr_db": (best.fitness * PSNR_SCALE) if best and best.fitness else None,
        "pop": args.pop, "gens": args.gens, "epoch": args.epoch,
        "elapsed_sec": round(dt, 1),
        "search_space": SEARCH_SPACE,
    }
    with open(args.results, "w") as f:
        json.dump(result, f, indent=2)

    log(f"\n=== DONE in {dt:.0f}s ===")
    if best and best.fitness:
        log(f"BEST: {best.fitness * PSNR_SCALE:.3f} dB  {best.chromosome}")
    log(f"Results -> {args.results}")


if __name__ == "__main__":
    main()
