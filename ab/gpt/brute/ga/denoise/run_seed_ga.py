"""Genetic-algorithm search over REAL DenoiseUnet seed architectures.

Unlike run_denoise_ga.py (parametric template), this evolves the actual seed
*source code*: population = real seeds, mutation = one random brute AST op
(skip/activation/kernel/dimension/scale via mutate_source), crossover = atomic
parent inheritance (real-source splicing is future work), fitness = PSNR proxy
via check_nn. Reuses the generic engine in ga_core.py.

Resumable under ~hourly Slurm preemption via (a) per-generation engine
checkpoint AND (b) a persistent checksum->fitness disk cache, so a restart
re-evaluates only the in-flight candidate, not the whole run.

Run as a plain script (not -m) with CWD under nn-dataset, mutator + nn-dataset
importable. Skip emphasis via env, e.g.:

    export PYTHONPATH=/a/mm/nn-dataset
    export BRUTE_WEIGHTS="skip:1.0,activation:0.5,kernel_size:0.3,dimension:0.3,scale:0.4"
    export BRUTE_DEBUG=0
    cd /a/mm/nn-dataset
    python .../ga/denoise/run_seed_ga.py --gens 2 --epoch 2
"""
import argparse
import glob
import hashlib
import json
import os
import random
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                             # ga_core, seed_mutation
sys.path.insert(0, "/a/mm/nn-gpt/ab/gpt/brute/ast")   # mutator package

from ga_core import (  # noqa: E402
    GeneticAlgorithmEngine, Individual, TournamentSelection,
)

TASK, DATASET, METRIC = "img-denoising", "denoise", "psnr"
PSNR_SCALE = 48.0

# Edge-legal seeds only (≤9053 MMac). The old default listed DenoiseUnet2F/7F/10F,
# which are 1.5-3.3x over the edge cap (14005/30152/29054 MMac) and poison the GA;
# load_seeds now also filters over-cap seeds defensively (--seed-mac-cap).
DEFAULT_SEEDS = ["DenoiseUnet9F", "DenoiseUnet8F", "DenoiseUnet5F",
                 "DenoiseUnet1F", "LiteDenoiseNet"]


def _checksum(code):
    return hashlib.sha1(code.encode("utf-8")).hexdigest()[:16]


class SeedCrossover:
    """Recombination operator with two modes.

    mode='block' (arms C/D): REAL component-block-swap — child inherits parent
    A's U-Net skeleton with parent B's conv block internals (block_crossover).
    Validated (build+forward) before use; on incompatibility/invalidity falls
    back to the atomic path so a bad recombination never wastes a training slot.

    mode='none' (arms A/B): atomic — child inherits one parent's chromosome
    wholesale (recombination is parent selection; mutation drives the search).
    This is the baseline the 'block' arm is ablated against."""

    def __init__(self, mode="none", log=print, validate=True):
        self.mode = mode
        self.log = log
        self.validate = validate

    def _atomic(self, parent1, parent2):
        return dict(random.choice([parent1, parent2]).chromosome)

    def crossover(self, parent1, parent2):
        if self.mode != "block":
            return self._atomic(parent1, parent2)
        from block_crossover import crossover as block_xover
        a, b = (parent1, parent2) if random.random() < 0.5 else (parent2, parent1)
        na, nb = a.chromosome.get("name", "A"), b.chromosome.get("name", "B")
        for src_a, src_b, la, lb in ((a, b, na, nb), (b, a, nb, na)):
            child = block_xover(src_a.chromosome["code"], src_b.chromosome["code"])
            if not child:
                continue
            if self.validate:
                from seed_mutation import _valid_denoise_source
                if not _valid_denoise_source(child):
                    continue
            # CAP the name: f"{la}X{lb}" DOUBLES each crossover -> exponential blowup to
            # thousands of chars over generations (part of the Run-2 gen-29 deadlock). Keep
            # a readable head + a short stable hash suffix so distinct lineages stay distinct.
            raw = f"{la}X{lb}"
            name = raw if len(raw) <= 48 else \
                raw[:40] + "~" + hashlib.blake2b(raw.encode(), digest_size=3).hexdigest()
            return {"code": child, "name": name, "parent": f"{la}x{lb}"[:64]}
        return self._atomic(parent1, parent2)


class DiverseTournamentSelection(TournamentSelection):
    """Tournament for the FIRST parent (unchanged fitness pressure); the SECOND
    parent is the sampled competitor MOST DISTANT from the first in continuous-WL
    embedding space (arch_embed cosine). Measured motivation: random pairing under
    tournament selection mates within-family ~50% of the time; distance-aware
    pairing raises the distinct-child rate of block crossover to ~80% because
    cross-family parents produce children the graded gate actually accepts.
    Embeddings are cached per code checksum, so each unique architecture is
    traced once. Falls back to plain tournament when embeddings are unavailable."""

    def __init__(self, tournament_size=2, mate_pool=6, log=print):
        super().__init__(tournament_size)
        self.mate_pool = mate_pool
        self.log = log
        self._vecs = {}  # code checksum -> embedding (None = un-embeddable)

    def _vec(self, chromo):
        code = chromo.get("code")
        if not code:
            return None
        key = _checksum(code)
        if key not in self._vecs:
            try:
                import arch_embed as ae
                self._vecs[key] = ae.embed_source(code)
            except Exception:
                self._vecs[key] = None
        return self._vecs[key]

    def select_mate(self, population, first):
        k = min(max(self.mate_pool, self.tournament_size), len(population))
        competitors = random.sample(population.individuals, k)
        fv = self._vec(first.chromosome)
        scored = []
        if fv is not None:
            for c in competitors:
                cv = self._vec(c.chromosome)
                if cv is not None:
                    scored.append((1.0 - float(fv @ cv), c))
        if not scored:  # nothing embeddable -> plain fitness tournament
            return max(competitors,
                       key=lambda x: x.fitness if x.fitness is not None else -float("inf"))
        scored.sort(key=lambda t: (t[0], t[1].fitness if t[1].fitness is not None
                                   else -float("inf")), reverse=True)
        return scored[0][1]


class SeedMutation:
    """Mutate a chromosome's source with one random brute AST op.

    Retries until it produces a genotype that is BOTH distinct from the parent
    AND not already evaluated (checksum absent from the shared `seen` set). This
    kills the dominant waste mode observed in the first run: offspring that were
    identical to their parent (mutate_source returned parent / a duplicate),
    which burn a population slot on a cached fitness. Falls back gracefully to a
    distinct-but-seen candidate, then to the parent unchanged."""

    def __init__(self, mutation_rate=1.0, mutation_type=None, log=print,
                 seen=None, max_tries=8, rich_skip=False, skip_prob=0.5):
        self.mutation_rate = mutation_rate
        self.mutation_type = mutation_type
        self.log = log
        self.seen = seen if seen is not None else set()
        self.max_tries = max_tries
        # rich_skip: route a fraction (skip_prob) of mutations through the new
        # skip_operators module (global/block residual, ...), which the base
        # mutate_source planner does not cover. Default OFF -> unchanged behaviour.
        self.rich_skip = rich_skip
        self.skip_prob = skip_prob

    def mutate(self, chromosome, search_space=None):
        from seed_mutation import mutate_source
        if random.random() > self.mutation_rate:
            return chromosome
        name = chromosome.get("name", "seed")
        parent_key = _checksum(chromosome["code"])
        fallback = None  # distinct-from-parent but already-seen candidate
        for _ in range(self.max_tries):
            used_skip = self.rich_skip and random.random() < self.skip_prob
            if used_skip:
                from skip_operators import apply_random_skip_op
                new_code, _op = apply_random_skip_op(chromosome["code"], random)
            else:
                new_code = mutate_source(name, chromosome["code"], self.mutation_type)
            if not new_code:
                continue
            key = _checksum(new_code)
            if key == parent_key:
                continue  # no-op mutation -> retry
            if key in self.seen:
                fallback = new_code  # distinct but already evaluated -> keep as backup
                continue
            self.seen.add(key)  # reserve so siblings this gen don't collide
            suffix = "+s" if used_skip else "+m"  # +s = skip-operator child
            return {"code": new_code, "name": f"{name}{suffix}", "parent": name}
        if fallback is not None:
            self.log(f"  mutate {name}: only seen-before variants after "
                     f"{self.max_tries} tries -> reusing one")
            return {"code": fallback, "name": f"{name}+m", "parent": name}
        self.log(f"  mutate {name}: no distinct variant in {self.max_tries} "
                 f"tries -> parent unchanged")
        return chromosome


def _inject_nan_guard(code, log=None, name=""):
    """Insert a NaN-guard into a model's learn() loop to protect generation YIELD.

    Every seed already clips gradients (clip_grad_norm_ max_norm=0.5), yet 9F-lineage
    models still NaN-diverge to 0.0 at train time (Gate 1: 3/15; arms A0 84% / C0 88%
    yield). Clipping does NOT help those: the loss goes non-finite in the FORWARD pass
    (bad activation/BatchNorm stat), and clipping a NaN norm is a no-op, so optimizer.step()
    then poisons every weight permanently -> dead model.

    We insert `if not torch.isfinite(<loss>): continue` just BEFORE `<loss>.backward()`,
    so a single bad batch is skipped while weights are still finite, and training survives.
    Fires ONLY on non-finite loss -> healthy models are byte-for-byte unchanged (can only
    rescue divergences, never alter a good result). learn() is copied seed->child verbatim,
    so injecting here propagates the guard to every generated candidate."""
    import re
    lines = code.splitlines(keepends=True)
    out, injected = [], 0
    for ln in lines:
        m = re.match(r'^(\s*)(\w+)\.backward\(\)\s*$', ln)
        if m and 'isfinite' not in (out[-1] if out else ''):
            indent, var = m.group(1), m.group(2)
            out.append(f"{indent}if not torch.isfinite({var}): continue  # NaN-guard (yield>90%)\n")
            injected += 1
        out.append(ln)
    new = "".join(out)
    if injected and log:
        log(f"  + NaN-guard injected into {name} ({injected} backward site(s))")
    elif not injected and log:
        log(f"  ! NaN-guard NOT injected into {name} (no `<var>.backward()` match) — check learn()")
    return new


def _seed_is_edge_legal(code, name, mac_cap, log):
    """True if the seed's MACs are within the edge cap (or cost can't be measured —
    fail-open, so a thop hiccup never silently empties the pool). An OVER-cap seed
    poisons the GA: every mutant it fathers is also over-budget, so the candidate
    edge filter discards the whole lineage (wasted evals) or, unfiltered, it pollutes
    the distinct-denoiser dataset with illegal models. Drop such seeds up front."""
    if not mac_cap or mac_cap <= 0:
        return True
    try:
        import edge_cost as EC
        c = EC.measure_cost(code)
    except Exception as e:
        log(f"  (seed {name}: cost check errored [{type(e).__name__}] — keeping, fail-open)")
        return True
    if not c.get("ok"):
        log(f"  (seed {name}: cost unmeasurable — keeping, fail-open)")
        return True
    macs = c["macs"] / 1e6
    if macs > mac_cap:
        log(f"  DROP over-cap seed {name}: {macs:.0f} MMac > {mac_cap:.0f} cap "
            f"(would poison the GA with illegal descendants)")
        return False
    return True


def load_seeds(names, log, seed_files=None, mac_cap=9053.0):
    """Seeds from the DB by name (the F-family), PLUS optional source FILES —
    promoted winners from a previous production run (promote.py --emit-dir). The
    file path lets the seed-pool growth loop feed generated architectures back in
    without needing them in the DB (save_to_db stays OFF pre-greenlight). A file's
    name comes from its `# ... name=<X>` header if present, else the basename.

    Seeds whose MACs exceed `mac_cap` (default = the 9053 edge budget; 0 disables)
    are filtered OUT here so they can never poison the population. See
    _seed_is_edge_legal.

    Every seed's learn() is passed through _inject_nan_guard so the whole run keeps
    generation yield >90% (guard propagates seed->child)."""
    import ab.nn.api as api
    df = api.data(only_best_accuracy=False)
    out = []
    for n in names:
        rows = df[df["nn"] == n]
        if rows.empty:
            log(f"  (seed {n} not in DB — skipped)")
            continue
        code = _inject_nan_guard(rows.iloc[0]["nn_code"], log, n)
        if not _seed_is_edge_legal(code, n, mac_cap, log):
            continue
        out.append(Individual({"code": code, "name": n}))
    for fp in (seed_files or []):
        if not os.path.exists(fp):
            log(f"  (seed-file {fp} missing — skipped)")
            continue
        code = _inject_nan_guard(open(fp).read(), log, os.path.basename(fp))
        name = os.path.basename(fp)[:-3]
        first = code.splitlines()[0] if code else ""
        if "name=" in first:
            name = first.split("name=", 1)[1].split()[0]
        if not _seed_is_edge_legal(code, name, mac_cap, log):
            continue
        # promoted seeds carry no 'parent' -> exempt from the predictor screen (baseline)
        out.append(Individual({"code": code, "name": name}))
        log(f"  + promoted seed from file: {name}")
    return out


def _export_nn_source(db_name, code, log):
    """Write a DB-saved generated model out as ab/nn/nn/<db_name>.py so it becomes a
    git-trackable source file like the rest of the model zoo (19k+ tracked .py, incl.
    generated <Base>-<uuid>.py). The main LEMUR/Optuna pipeline persists one per trial;
    brute-GA's check_nn(is_code) writes only a temp file + DB row, so generated denoise
    models were DB-only until this hook. Idempotent; failures are non-fatal (logged)."""
    try:
        if not db_name or not str(db_name).startswith("Denoise"):
            return
        import ab.nn as _abnn
        nn_dir = os.path.join(os.path.dirname(os.path.abspath(_abnn.__file__)), "nn")
        path = os.path.join(nn_dir, f"{db_name}.py")
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write(code)
    except Exception as e:
        log(f"  (nn-file export skipped for {db_name}: {type(e).__name__}: {str(e)[:60]})")


def _export_nn_stats(db_name, log, stat_dir=None):
    """Write a DB-saved generated model's training stats out as canonical
    ab/nn/stat/train/<task>_<dataset>_<metric>_<db_name>/<epoch>.json files, so a kept model
    carries its per-epoch stats in-repo like the rest of the zoo. check_nn(is_code) writes the
    stat rows to the DB but NOT to the JSON tree (Train.train_n_eval only auto-writes JSON for
    non-code training), so kept generated models were DB-only for stats until this hook.
    Reconstructs each trial faithfully from the DB (stat row + prm table, the reverse of
    save_stat) and writes via the framework's canonical format. Idempotent (overwrites from the
    DB, the source of truth); failures are non-fatal (logged). Call only for kept models."""
    try:
        if not db_name or not str(db_name).startswith("Denoise"):
            return
        import sqlite3, shutil
        from ab.nn.util.Const import db_file
        from ab.nn.util.Util import model_stat_dir
        from ab.nn.util.db.Calc import sanitize_for_json

        # PREFERRED: the per-epoch JSONs check_nn(save_path=...) just wrote. They are
        # already in the canonical save_results trial format, so they transfer verbatim.
        # The DB path below can only ever yield ONE epoch for a GA model -- train_n_eval
        # writes per-epoch DB rows solely on its non-code branch, and train_new then saves
        # a single row at the final epoch. Rebuilding from the DB would therefore export a
        # 1-epoch stat file and silently lose the curve that was just measured.
        if stat_dir and os.path.isdir(stat_dir):
            eps = [f for f in os.listdir(stat_dir)
                   if f.endswith(".json") and f[:-5].isdigit()]
            if eps:
                d = model_stat_dir((TASK, DATASET, METRIC, db_name))
                d.mkdir(parents=True, exist_ok=True)
                for f in eps:
                    shutil.copyfile(os.path.join(stat_dir, f), d / f)
                log(f"        exported {len(eps)} per-epoch stat file(s) for {db_name}")
                return
        con = sqlite3.connect(str(db_file)); con.row_factory = sqlite3.Row
        cur = con.cursor()
        rows = cur.execute(
            "SELECT task,dataset,metric,transform,epoch,accuracy,duration,prm "
            "FROM stat WHERE nn=?", (db_name,)).fetchall()
        by_epoch = {}
        for r in rows:
            prm = {k: v for k, v in cur.execute(
                "SELECT name,value FROM prm WHERE uid=?", (r["prm"],)).fetchall()}
            trial = dict(prm)                       # reverse of save_stat
            trial["accuracy"] = r["accuracy"]
            trial["duration"] = r["duration"]
            trial["uid"] = r["prm"]
            trial.setdefault("transform", r["transform"])
            # normalise to the repo's canonical stat schema: epoch_max (not epoch),
            # and no num_workers (no tracked stat file carries it).
            if "epoch_max" not in trial and "epoch" in trial:
                trial["epoch_max"] = trial.pop("epoch")
            else:
                trial.pop("epoch", None)
            trial.pop("num_workers", None)
            cfg = (r["task"], r["dataset"], r["metric"], db_name)
            by_epoch.setdefault((cfg, r["epoch"]), []).append(trial)
        con.close()
        for (cfg, epoch), trs in by_epoch.items():
            d = model_stat_dir(cfg); d.mkdir(parents=True, exist_ok=True)
            trs = sorted(trs, key=lambda x: x["accuracy"], reverse=True)
            with open(d / f"{epoch}.json", "w") as f:
                json.dump(sanitize_for_json(trs), f, indent=4)
    except Exception as e:
        log(f"  (nn-stat export skipped for {db_name}: {type(e).__name__}: {str(e)[:60]})")


def _summarise_curve(by_epoch, scale):
    """Aggregate {epoch: accuracy} into the reported statistics block."""
    import statistics
    eps = sorted(by_epoch)
    vals = [by_epoch[e] * (scale or 1.0) for e in eps]
    bi = max(range(len(vals)), key=lambda i: vals[i])
    return {
        "n_epochs": len(vals), "max_epoch": max(eps),
        "best": round(vals[bi], 4), "best_epoch": eps[bi],
        "final": round(vals[-1], 4), "avg": round(statistics.mean(vals), 4),
        "std": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
        "min": round(min(vals), 4),
        "still_climbing": eps[bi] == max(eps),
        "curve": [[e, round(v, 4)] for e, v in zip(eps, vals)],
    }


def epoch_stats(db_name, scale=None, stat_dir=None):
    """Full per-epoch training statistics for an evaluated model.

    WHY THIS IS NOT JUST A DB QUERY. check_nn() returns only the single BEST accuracy, and
    the GA path does not persist a learning curve at all: Train.train_n_eval writes a
    per-epoch DB row only on its non-code branch, while the GA trains with is_code=True.
    On that branch the epoch row is written only if `save_path` is set -- and check_nn
    defaulted it to None, so the loop hit the "[WARN] ... statics will not be saved" arm
    and dropped every epoch. train_new then wrote ONE row at the final epoch. That is why
    all 131 existing DenoiseGA-* models carry a single stat row at epoch=5.

    So the curve is recovered from the per-epoch JSON that check_nn(save_path=...) now
    emits, with the DB as fallback for models trained through the normal LEMUR path
    (seeds), which do get per-epoch rows.

    The distinction matters because the GA scores on a short proxy budget: at 5 epochs
    nearly every denoise family peaks on its LAST epoch, so a lone `best` scalar cannot
    tell a converged model from one cut off mid-climb. `still_climbing` flags exactly that.

      n_epochs  epoch rows found          max_epoch  highest epoch reached
      best/best_epoch  peak and where     final      value at the last epoch
      avg/std/min      moments over ALL epochs       curve  [[epoch, value], ...]
      still_climbing   best_epoch == max_epoch -> budget was binding, not convergence

    Values are scaled by `scale` (PSNR_SCALE -> real dB). Returns None on any failure:
    these are diagnostics and must never break an evaluation."""
    # --- preferred: per-epoch JSON written by this run ---
    try:
        if stat_dir and os.path.isdir(stat_dir):
            by_epoch = {}
            for fn in os.listdir(stat_dir):
                if not fn.endswith(".json"):
                    continue
                try:
                    ep = int(fn[:-5])
                except ValueError:
                    continue                      # e.g. a training-summary file
                with open(os.path.join(stat_dir, fn)) as f:
                    trials = json.load(f)
                if isinstance(trials, dict):
                    trials = [trials]
                accs = [t["accuracy"] for t in trials
                        if isinstance(t, dict) and t.get("accuracy") is not None]
                if accs:
                    by_epoch[ep] = max(accs)
            if by_epoch:
                return _summarise_curve(by_epoch, scale)
    except Exception:
        pass
    # --- fallback: the DB (populated per-epoch only for non-code training) ---
    try:
        if not db_name:
            return None
        import sqlite3
        from ab.nn.util.Const import db_file
        con = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
        rows = con.execute(
            "SELECT epoch, accuracy FROM stat WHERE nn=? AND task=? AND dataset=? "
            "AND metric=? ORDER BY epoch", (db_name, TASK, DATASET, METRIC)).fetchall()
        con.close()
        if not rows:
            return None
        by_epoch = {}
        for e, a in rows:
            if a is not None and (e not in by_epoch or a > by_epoch[e]):
                by_epoch[e] = a
        return _summarise_curve(by_epoch, scale) if by_epoch else None
    except Exception:
        return None



def _deploy_ok(code, name, max_ops, log):
    """True if this candidate could actually ship: convertible AND small enough to be fast.

    A TEMPORARY, DELIBERATELY STRICT EXPORT GATE. It does not touch selection -- rejected models
    are still scored and still steer the search -- it only decides what enters the corpus. That
    separation matters: gating the SEARCH on deployability would shrink the explorable space and
    is not what was asked for.

    THREE CHECKS, CHEAPEST FIRST:
      1. the audited unconvertible-op rules (AST, microseconds, zero false positives measured)
      2. a real TFLite conversion (~3.4 s) -- the static rules explain only 66% of failures
      3. TFLite graph ops <= max_ops -- 92% precision against "latency > 10 ms" at 203

    WHAT IT CANNOT PROMISE. Graph size predicts slowness at 43% recall, so a survivor is not
    certified fast -- only "not obviously slow". A DenoiseUnet7 variant ran 2,621 ms while sitting
    entirely on the NPU, and no static check would have caught that. Real latency still needs the
    AI Hub profile, which is far too slow to run per candidate inside a GA generation.

    FAILS CLOSED. If the converter is unavailable this returns False rather than waving models
    through, because a silently-disabled deployability gate is exactly how the MAC cap got
    switched off for a whole run without anyone noticing.
    """
    import shutil
    import tempfile
    try:
        import npu_gate as NG
    except Exception as e:
        log(f"        DEPLOY-GATE unavailable ({type(e).__name__}: {e}); REJECTING")
        return False
    bad = NG.unconvertible(code)
    if bad:
        log(f"        DEPLOY-REJECT {name}: {','.join(b[0] for b in bad)}")
        return False
    tmp = tempfile.mkdtemp(prefix="dg_")
    try:
        src = os.path.join(tmp, "src")
        os.makedirs(src, exist_ok=True)
        with open(os.path.join(src, "m.py"), "w") as fh:
            fh.write(code)
        tfl = os.path.join(tmp, "tflite")
        os.makedirs(tfl, exist_ok=True)
        conv = os.path.join(_HERE, "reexport_convert.py")
        rc = subprocess.call([NG.VENV_PY, conv, "--src", src, "--out", tfl,
                              "--h", "256", "--w", "256",
                              "--report", os.path.join(tmp, "convert.json")],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        arts = glob.glob(os.path.join(tfl, "*.tflite"))
        if not arts:
            log(f"        DEPLOY-REJECT {name}: does not convert to TFLite (rc={rc})")
            return False
        counts = NG.count_ops(tfl, os.path.join(tmp, "layers.json"))
        n = next((v for v in counts.values() if v), None)
        if n is None:
            log(f"        DEPLOY-REJECT {name}: graph size unreadable")
            return False
        if n > max_ops:
            log(f"        DEPLOY-REJECT {name}: {n} graph ops > {max_ops}")
            return False
        log(f"        DEPLOY-OK {name}: converts, {n} ops")
        return True
    except Exception as e:
        log(f"        DEPLOY-REJECT {name}: {type(e).__name__}: {str(e)[:100]}")
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _deploy_predict_ok(code, name, thr, log):
    """Source-only eviction gate. FAILS CLOSED, exactly as _deploy_ok does: a gate that waves
    candidates through when its model is missing is indistinguishable from no gate, which is
    how the MAC cap was once disabled for an entire run while printing a success banner."""
    try:
        import deploy_gate as DPG
        passes, p = DPG.ok(code, thr)
    except Exception as e:
        log(f"        DEPLOY-PREDICT unavailable ({type(e).__name__}: {e}); REJECTING")
        return False
    if p is None:
        log(f"        DEPLOY-PREDICT {name}: unparseable, passing to preflight's judgement")
        return True
    if not passes:
        log(f"        DEPLOY-PREDICT-REJECT {name}: evict_prob {p:.3f} >= {thr:.2f}")
        return False
    log(f"        DEPLOY-PREDICT-OK {name}: evict_prob {p:.3f}")
    return True


def make_fitness(epoch, batch, lr, num_workers, save_to_db, cache_path, log,
                 cache, seen, cost_metric, budget, cost_weight, cost_mode,
                 models_dir, edge_filter=False, mac_cap=0.0, edge_strict=False,
                 diversity_gate=None, prof_predict=False, prof_epochs=3,
                 prof_target=50, noop_floor=33.021, noop_margin=0.207,
                 fullres_fitness=False, fullres_val=None, fullres_limit=0,
                 deploy_gate=False, deploy_max_ops=203, deploy_predict_thr=0.0):
    """Fitness = edge-aware effective PSNR.

    PSNR (the expensive part) is trained once and cached; edge cost is measured
    fresh each call (cheap, no retrain) and folded in via edge_cost.effective_db,
    so the objective (metric / budget / weight) can change between runs without
    invalidating cached PSNR — old PSNR-only cache entries stay reusable.

    Every evaluated genotype's SOURCE is persisted to models_dir/<checksum>.py
    (the same checksum keying the cache), so the full population of mutants is
    recoverable for the post-hoc similarity / diversity analysis. Without this
    the code is discarded after training and only scores survive."""
    import ab.nn.api as api
    import fullres_score
    from predictor import prof_acc
    from edge_cost import measure_cost, effective_db
    if edge_filter:
        from edge_deploy import deployability, macs_ok

    os.makedirs(models_dir, exist_ok=True)
    # per-candidate per-epoch stat JSONs (see epoch_stats); sibling of the mutant sources
    # so a run's full record -- source, score, learning curve -- lives together.
    stats_dir = os.path.join(models_dir, "_epoch_stats")
    os.makedirs(stats_dir, exist_ok=True)

    def _persist():
        tmp = cache_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cache, f)
        os.replace(tmp, cache_path)

    def _save_source(key, code, name, parent, psnr, cost, eff):
        """Write a mutant's source keyed by checksum (idempotent)."""
        path = os.path.join(models_dir, key + ".py")
        if os.path.exists(path):
            return
        macs = cost.get("macs") if cost.get("ok") else None
        params = cost.get("params") if cost.get("ok") else None
        header = (f"# name={name} parent={parent} checksum={key}\n"
                  f"# psnr_db={psnr:.4f} eff_db={eff:.4f} "
                  f"macs={macs} params={params}\n")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write(header + code)
        os.replace(tmp, path)

    def _cost_value(cost):
        if cost_metric == "none" or not cost.get("ok"):
            return None
        return cost.get(cost_metric)

    def fitness(chromosome):
        code = chromosome["code"]
        name = chromosome.get("name", "?")
        key = _checksum(code)

        cost = ({"ok": False, "params": 0, "macs": 0}
                if cost_metric == "none" else measure_cost(code))
        cval = _cost_value(cost)

        # EDGE GATE (generation-time hard filter): reject GPU-delegate-hostile or
        # over-MAC-cap models BEFORE the ~10-min train, so undeployable candidates
        # never win a population slot and no compute is wasted on them. Only for
        # freshly-generated genotypes (cached ones already passed / are seeds).
        cached = cache.get(key)
        if edge_filter and not (cached and cached.get("psnr") is not None):
            dep = deployability(code, strict=edge_strict)
            over = not macs_ok(cost.get("macs"), mac_cap)
            if not dep["deployable"] or over:
                why = (dep["hard"] or dep["soft"] or [])[:1]
                if over:
                    why = why + [f"macs>{mac_cap:.0f}MMac"]
                log(f"  EDGE-REJECT {name} [{key}] -> {', '.join(why) or 'undeployable'}")
                cache[key] = {"fitness": 0.0, "psnr": 0.0, "eff_db": 0.0, "name": name,
                              "parent": chromosome.get("parent"),
                              "params": cost.get("params"), "macs": cost.get("macs"),
                              "edge_reject": True}
                seen.add(key)
                _persist()
                return 0.0

        # DIVERSITY GATE (WL graph hash): reject a fresh GENERATED candidate whose
        # computation GRAPH already exists in the accepted set (exact graph-clone,
        # incl. renamed / width-rescaled duplicates), BEFORE the ~train cost — exactly
        # like the edge filter, and counted the same way in inclusive yield. Seeds
        # (parent None) are never rejected but DO seed the reference set. Fights Run-1
        # population collapse onto the Lite family. (Replaced residual-CKA, which is
        # weight-noise-dominated at init; functional CKA/NTK don't transfer to denoising.)
        div_fp = None
        if diversity_gate is not None and not (cached and cached.get("psnr") is not None):
            accept, fp = diversity_gate.check(code)
            div_fp = (key, fp)
            wl_hash = fp[0]
            is_seed = chromosome.get("parent") is None
            if not is_seed and not accept:
                diversity_gate.n_reject += 1
                reason = "graph-clone" if wl_hash in diversity_gate.accepted else "near-dup"
                log(f"  DIVERSITY-REJECT {name} [{key}] -> {reason} "
                    f"({wl_hash[:8] if wl_hash else '?'})")
                cache[key] = {"fitness": 0.0, "psnr": 0.0, "eff_db": 0.0, "name": name,
                              "parent": chromosome.get("parent"),
                              "params": cost.get("params"), "macs": cost.get("macs"),
                              "diversity_reject": True, "wl_hash": wl_hash, "reject": reason}
                seen.add(key)
                _persist()
                return 0.0

        # PSNR: reuse cached training result if present, else train.
        estats = None
        psnr_pred_meta = None
        if cached and cached.get("psnr") is not None:
            psnr, src, dt = cached["psnr"], "CACHE", 0.0
            estats = cached.get("epoch_stats")   # carry forward on a cache hit
            psnr_pred_meta = cached.get("psnr_predicted")
        else:
            # PROFESSOR'S PREDICTOR (opt-in, --prof-predictor): train only `prof_epochs`
            # and extrapolate to `prof_target`, instead of paying for the full budget.
            # Worth it only when prof_target >> prof_epochs: at the 5-epoch proxy budget
            # it saves 2 of 5 epochs and is not worth the added error, so the saving is
            # reported and the caller is expected to raise --epoch when enabling it.
            train_epochs = prof_epochs if prof_predict else epoch
            prm = {"lr": lr, "batch": batch, "transform": "denoise",
                   "epoch": train_epochs, "num_workers": num_workers}
            t0 = time.time()
            # save_path is what makes Train.train_n_eval emit a per-epoch <epoch>.json on
            # the is_code branch; left None (the check_nn default) it warns and discards
            # the curve. Keyed by the GA checksum because db_name is only known after the
            # call returns.
            cand_stat_dir = os.path.join(stats_dir, key)
            os.makedirs(cand_stat_dir, exist_ok=True)
            try:
                db_name, acc, acc_to_time, code_quality = api.check_nn(
                    code, TASK, DATASET, METRIC, prm,
                    save_to_db=save_to_db, prefix=os.environ.get("GA_PREFIX", "DenoiseGA"),
                    # A stat row records a SCORE, not the network that earned it. Without
                    # this every GA model would have to be retrained before it could be
                    # deployed, inspected or fine-tuned from. Tied to save_to_db so the
                    # predictor's throwaway evaluations do not fill the disk. Destination
                    # is $LEMUR_CKPT_DIR.
                    save_pth_weights=save_to_db,
                    save_path=cand_stat_dir)
                psnr, src = acc * PSNR_SCALE, "OK"
                # Keep the MEASURED value separately: `psnr` is overwritten by the
                # predictor below, and the corpus gate must never fire on a prediction.
                psnr_measured = psnr
                psnr_crop = psnr

                # FITNESS ON THE PUBLISHED METRIC, NOT THE CROP. See fullres_score.py: over
                # 1,578 models the crop metric has Spearman -0.025 against full-resolution RGB
                # among the top 100, so a search that maximises it goes blind precisely at the
                # frontier. Four arms and 1,287 new architectures produced nothing better than a
                # champion that predates them; this is the change that addresses why.
                # Fail-soft: on any scoring failure the crop score stands and the reason is
                # logged, because a scoring outage must not look like a quality collapse.
                if fullres_fitness:
                    ck = os.path.join(os.environ.get("LEMUR_CKPT_DIR", "/a/mm/sweep_out/ckpt"),
                                      db_name or "", "best_model.pth")
                    fr = fullres_score.score_source(code, ck, fullres_val, limit=fullres_limit,
                                                    log=log)
                    if fr is not None:
                        log(f"        FULLRES {name}: crop {psnr_crop:.3f} -> full-res "
                            f"{fr:.3f} dB ({fr - psnr_crop:+.3f})")
                        psnr, psnr_measured, src = fr, fr, "FULLRES"
                # capture the FULL learning curve while the DB rows are fresh --
                # check_nn returned only `acc` (the peak). Diagnostics only: never
                # feeds fitness, so a stats failure cannot alter selection.
                estats = epoch_stats(db_name, PSNR_SCALE, cand_stat_dir)
                # extrapolate ep1-3 -> ep`prof_target`. Feeds dB (the only scale the
                # predictor ranks on; raw LEMUR accuracy is anti-correlated, rho=-0.68).
                if prof_predict and estats and estats["n_epochs"] >= 3:
                    c = dict(estats["curve"])
                    pred = prof_acc.predict_db(
                        code, c[1], c[2], c[3], TASK, DATASET, METRIC, prof_target)
                    if pred is not None:
                        psnr_meas, psnr, src = psnr, pred, "PRED"
                        psnr_pred_meta = {"measured_at_ep": estats["n_epochs"],
                                          "measured_db": round(psnr_meas, 4),
                                          "predicted_db": round(pred, 4),
                                          "target_epoch": prof_target}
                        log(f"        PREDICT {name}: measured {psnr_meas:.3f} dB "
                            f"@ep{estats['n_epochs']} -> predicted {pred:.3f} dB "
                            f"@ep{prof_target}")
                    else:
                        # fail-soft: keep the measured value rather than drop the
                        # candidate, and say so -- a silent fallback would make a
                        # predictor outage look like a population-wide quality drop.
                        log(f"        PREDICT {name}: unavailable, keeping measured "
                            f"{psnr:.3f} dB @ep{estats['n_epochs']}")
                # Persist the DB-saved model as a .py in ab/nn/nn/ (the main LEMUR
                # pipeline does this per trial; brute-GA's check_nn writes only a temp
                # file + DB row). Keeps generated models as git-trackable source, like
                # the rest of the model zoo. Gated on psnr>0 == good()/saved (denoise
                # minimum_accuracy is 0.0).
                # THE NO-OP GATE. `psnr > 0` alone lets anything that trains at all into the
                # uploadable corpus, and "trains at all" is a much lower bar than "denoises":
                # returning the noisy input unchanged scores 33.021 dB on this benchmark, and
                # 84 models already in the corpus score BELOW that. They are not weak models,
                # they are models that make the image worse, and each one is a record a reader
                # has to be warned about rather than one they can use.
                #
                # The bar is the floor PLUS the measured run-to-run noise (0.207 dB, from the
                # confirmation run; the 0.14 dB used earlier in the project was an assumption).
                # Without the margin a model beating the no-op by 0.001 dB qualifies, which is a
                # coin flip rather than a result -- exactly how 90 of the 96 LLM models imported
                # on 2026-08-12 got in.
                #
                # Gated on the MEASURED value, never the predicted one, and it gates the EXPORT
                # only -- fitness is untouched, so selection still sees the real score and can
                # rank against it. Zeroing a sub-floor model's fitness would make it
                # indistinguishable from a DEAD one and throw away that gradient.
                # floor <= 0 disables the gate outright rather than collapsing the bar to the
                # bare margin, which would still silently reject near-zero scores.
                bar = (noop_floor + noop_margin) if noop_floor > 0 else float("-inf")
                if save_to_db and psnr > 0 and psnr_measured < bar:
                    log(f"  SUB-FLOOR {name} [{key}] -> {psnr_measured:.3f} dB < {bar:.3f} "
                        f"(no-op {noop_floor:.3f} + noise {noop_margin:.3f}); "
                        f"scored and kept in the search, NOT exported to the corpus")
                elif save_to_db and psnr > 0 and deploy_predict_thr > 0 and not _deploy_predict_ok(
                        code, name, deploy_predict_thr, log):
                    # SOURCE-ONLY DEPLOYABILITY GATE. Same role and same point in the pipeline
                    # as --deploy-gate, but it reads the AST instead of paying a ~3.4 s TFLite
                    # conversion, and it predicts full CPU EVICTION rather than graph size.
                    # Measured out-of-fold at AUC 0.986, and 0.921 on a generator held out
                    # entirely, against 3,713 handset profiles. Scored and kept in the search;
                    # just not admitted to the corpus, so the explorable space is unchanged.
                    pass
                elif save_to_db and psnr > 0 and deploy_gate and not _deploy_ok(
                        code, name, deploy_max_ops, log):
                    # Scored and kept in the search; just not admitted to the corpus.
                    pass
                elif save_to_db and psnr > 0:
                    _export_nn_source(db_name, code, log)
                    _export_nn_stats(db_name, log, cand_stat_dir)
            except Exception as e:
                psnr, src = 0.0, "DEAD"
                log(f"  DEAD {name} [{key}] -> {type(e).__name__}: {str(e)[:90]}")
            dt = time.time() - t0

        # Objective: dead models score 0 (no cost bonus can rescue them).
        eff = 0.0 if psnr <= 0 else effective_db(psnr, cval, budget,
                                                 cost_weight, cost_mode)
        fit = eff / PSNR_SCALE

        macs_m = (cost["macs"] / 1e6) if cost.get("ok") else float("nan")
        prm_m = (cost["params"] / 1e6) if cost.get("ok") else float("nan")
        log(f"  {src:5s}{name} [{key}] -> psnr {psnr:6.3f} | "
            f"{macs_m:8.1f} MMac {prm_m:6.2f} Mp | eff {eff:6.3f} dB"
            + (f" ({dt:.0f}s)" if dt else ""))
        if estats:
            # '^' marks best_epoch == max_epoch: the model was still improving when the
            # epoch cap cut it off, so this PSNR is a floor, not a converged value.
            log(f"        epochs n={estats['n_epochs']} max_ep={estats['max_epoch']} "
                f"best {estats['best']:.3f}@ep{estats['best_epoch']}"
                f"{'^' if estats['still_climbing'] else ' '} "
                f"final {estats['final']:.3f} avg {estats['avg']:.3f} "
                f"std {estats['std']:.3f} min {estats['min']:.3f}")

        parent = chromosome.get("parent")
        cache[key] = {"fitness": fit, "psnr": psnr, "eff_db": eff, "name": name,
                      "parent": parent,
                      "params": cost.get("params"), "macs": cost.get("macs"),
                      "epoch_stats": estats,
                      # provenance: when set, `psnr` is EXTRAPOLATED, not measured.
                      # Anything reporting champion numbers must check this field --
                      # a predicted dB is not a result.
                      "psnr_predicted": psnr_pred_meta}
        _save_source(key, code, name, parent, psnr, cost, eff)
        seen.add(key)
        # DIVERSITY: a model that actually WORKS joins the reference set (greedy),
        # so later children must differ from it too. Seeds included (baselines).
        if diversity_gate is not None and div_fp is not None and psnr > 0:
            diversity_gate.add(*div_fp)
        _persist()
        return fit

    return fitness


def make_prescreen(keep_frac, min_train, cache, cost_metric, budget,
                   cost_weight, cost_mode, log):
    """Accuracy-predictor funnel = GA stage 1 (professor-required).

    Before a generation is trained, rank its FRESH candidates (not seeds, not
    cache-hits) with the learned surrogate and TRAIN ONLY the top `keep_frac`.
    The bottom fraction is assigned a cheap PREDICTED surrogate fitness and its
    ~10-min ep5 training is skipped — that is the compute saving. The predictor
    NEVER crowns: skipped candidates are the LOWEST-ranked of their generation
    (so they can't be elites), and main() real-trains the final best if it was
    only ever predicted. Returns a hook(individuals, gen) or None if disabled."""
    from predictor.predict import score as pred_score
    from edge_cost import measure_cost, effective_db

    def hook(individuals, gen):
        fresh = []
        for ind in individuals:
            if ind.fitness is not None:
                continue                      # already scored (elite carry-over)
            # SEED EXEMPTION: never screen a raw seed (chromosome has no parent) —
            # it must get a real MEASURED baseline, not a predicted surrogate. This
            # covers gen 0 (population = the seeds) and any promoted seed reappearing
            # later in the pool-growth loop; only GENERATED candidates are screened.
            if ind.chromosome.get("parent") is None:
                continue                      # -> fitness stays None -> trained
            key = _checksum(ind.chromosome["code"])
            c = cache.get(key)
            if c and c.get("psnr") is not None:
                continue                      # cache-hit -> "training" is free
            fresh.append((ind, key))
        if len(fresh) <= min_train:
            return                            # too few to bother screening

        # Rank by predicted EFF (the SAME edge-aware objective the GA selects on),
        # not raw predicted PSNR — otherwise the funnel would keep expensive
        # high-PSNR models the eff objective penalises and drop cheap ones it
        # favours. eff folds the model's real (measured) cost into the predicted
        # PSNR; a likely-dead model (p_dead>=0.5) is floored to 0.
        scored = []
        for ind, key in fresh:
            code = ind.chromosome["code"]
            cost = measure_cost(code) if cost_metric != "none" else {"ok": False}
            macs = cost.get("macs") if cost.get("ok") else None
            params = cost.get("params") if cost.get("ok") else None
            try:
                p = pred_score(code, macs, params, ind.chromosome.get("name"))
            except Exception as e:
                log(f"  PREDICT-ERR {ind.chromosome.get('name')} -> {type(e).__name__}; training it")
                continue                      # on error, fall through to real train
            cval = cost.get(cost_metric) if cost.get("ok") else None
            pred_eff = 0.0 if p["p_dead"] >= 0.5 else effective_db(
                p["psnr_pred"], cval, budget, cost_weight, cost_mode)
            scored.append((pred_eff, ind, p))

        scored.sort(key=lambda t: t[0], reverse=True)
        n_train = max(min_train, int(round(len(scored) * keep_frac)))
        n_skip = 0
        for rank, (pred_eff, ind, p) in enumerate(scored):
            if rank < n_train:
                continue                      # keep: leave fitness None -> trained
            # skip: assign the predicted-eff surrogate fitness, tag predicted-only
            ind.fitness = pred_eff / PSNR_SCALE
            ind.chromosome["predicted"] = True
            n_skip += 1
        log(f"  [predictor] gen {gen + 1}: {len(scored)} fresh -> train "
            f"top {n_train}, skip {n_skip} (pred surrogate)")

    return hook


def _promote_after_run(args, log):
    """Integrated seed-growth stage: full-train this run's top-K, then promote the
    psnr50 survivors (WL-distinct, >=bar, deployable) into a grown seed pool. Reuses
    fulltrain_topk.py + promote.py so there is ONE promotion implementation."""
    import subprocess
    # arm tag is the run label embedded in the cache filename: seed_ga_<arm>_cache.json
    arm = os.path.basename(args.cache).replace("seed_ga_", "").replace("_cache.json", "")
    res = os.path.join(_HERE, f"fulltrain_{arm}.json")
    emit = args.promote_dir or os.path.join(_HERE, f"promoted_pool_{arm.lower()}")
    if os.path.isdir(emit) and any(f.endswith(".py") for f in os.listdir(emit)):
        log(f"[promote] {emit} already populated — skipping (idempotent)")
        return
    log(f"\n=== PROMOTE STAGE (arm {arm}): full-train top-{args.promote_k} "
        f"@{args.promote_epoch}ep, then promote (bar {args.promote_bar} dB) ===")
    subprocess.run([sys.executable, os.path.join(_HERE, "fulltrain_topk.py"),
                    "--arms", arm, "--k", str(args.promote_k),
                    "--epoch", str(args.promote_epoch),
                    "--num-workers", str(args.num_workers), "--results", res], check=False)
    pool = list(args.seeds or []) + list(args.seed_files or [])
    subprocess.run([sys.executable, os.path.join(_HERE, "promote.py"),
                    "--results", res, "--models-dirs", args.models_dir,
                    "--pool", *pool, "--pareto", "--bar", str(args.promote_bar),
                    "--strict-mac", "--mac-cap", str(args.mac_cap or 9053.0),
                    "--graded", "--graded-max", str(args.graded_max),
                    "--emit-dir", emit,
                    "--out", os.path.join(_HERE, f"promote_{arm}.json")], check=False)
    n = len([f for f in os.listdir(emit) if f.endswith(".py")]) if os.path.isdir(emit) else 0
    log(f"=== PROMOTE DONE: {n} new seed(s) -> {emit} ===")


def _retrain_predictor(args, log):
    """Refresh the label store + refit the accuracy predictor on the FULL accumulated
    corpus (ablation + every production run + new families). Keeps the funnel calibrated
    to newly-introduced architectures instead of the stale ablation-only model."""
    import subprocess
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}   # CPU: don't contend with training
    tp = os.path.join(_HERE, "predictor", "train_predictor.py")
    log("\n=== RETRAIN PREDICTOR (accumulate labels + refit V1 & V2, activate winner) ===")
    subprocess.run([sys.executable, os.path.join(_HERE, "accumulate_labels.py")],
                   check=False, env=env)
    # V1: ep5 target (plentiful) — the incumbent funnel model
    subprocess.run([sys.executable, tp, "--zero-cost"], check=False, env=env)
    # V2: 50-epoch target (high-signal, scarce) — candidate, saved separately
    subprocess.run([sys.executable, tp, "--zero-cost", "--target", "psnr50"],
                   check=False, env=env)
    # promote whichever generalises better (V2 only once it earns it); calibrates on activate
    subprocess.run([sys.executable, os.path.join(_HERE, "predictor", "activate_predictor.py")],
                   check=False, env=env)
    log("=== PREDICTOR RETRAIN DONE ===")


def main():
    global DATASET, METRIC  # overridden from --dataset/--metric after parse_args
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=None,
                    help="model-name prefix for DB inserts (default DenoiseGA). The random-"
                         "search baseline MUST set this to something else -- otherwise its "
                         "models are named DenoiseGA-* and become indistinguishable from GA "
                         "output in the corpus, silently contaminating every producer-split "
                         "analysis (the on-device tables, the arm comparison, the manifest).")
    ap.add_argument("--seeds", nargs="*", default=DEFAULT_SEEDS)
    ap.add_argument("--seed-files", nargs="*", default=None,
                    help="extra seed SOURCE files (promote.py --emit-dir output) added "
                         "to the pool — the cross-run seed-pool growth loop")
    ap.add_argument("--pop-size", type=int, default=0,
                    help="target population (0 = use #seeds); when > #seeds the "
                         "population is pre-seeded with the real seeds and grows "
                         "to this size (via crossover+mutation) after gen 0")
    ap.add_argument("--gens", type=int, default=2)
    ap.add_argument("--epoch", type=int, default=2)
    ap.add_argument("--prof-predictor", action="store_true",
                    help="use the remote ABrain/Accuracy-Prediction model to extrapolate "
                         "--prof-epochs -> --prof-target instead of training the full "
                         "budget. Validated on the 6 full-curve seeds at Spearman 0.84 / "
                         "MAE 1.20 dB WHEN FED dB (raw LEMUR accuracy is anti-correlated, "
                         "rho=-0.68), n=6 so provisional. Fails soft to the measured value "
                         "if the server is unreachable.")
    ap.add_argument("--prof-epochs", type=int, default=3,
                    help="epochs actually trained before extrapolating (the predictor "
                         "prompt is built around exactly 3 early epochs)")
    ap.add_argument("--prof-target", type=int, default=50,
                    help="epoch budget to extrapolate TO; the saving is "
                         "prof_target/prof_epochs, so keep this well above --prof-epochs")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--elitism", type=int, default=1)
    ap.add_argument("--tournament", type=int, default=2)
    ap.add_argument("--mutation-rate", type=float, default=1.0)
    ap.add_argument("--mutation-type", default=None,
                    help="force a single op, e.g. 'skip' (default: weighted mix)")
    ap.add_argument("--mutation-tries", type=int, default=8,
                    help="retries per mutation to find a distinct, non-cached genotype")
    ap.add_argument("--skip-ops", action="store_true",
                    help="route a fraction of mutations through skip_operators "
                         "(global/block residual etc.) — the SKIP arm (default off)")
    ap.add_argument("--skip-prob", type=float, default=0.5,
                    help="with --skip-ops, P(a mutation uses a skip operator) (default 0.5)")
    ap.add_argument("--edge-filter", action="store_true",
                    help="HARD-reject non-edge-deployable models at generation time "
                         "(GPU-delegate-hostile ops / over MAC cap) BEFORE training")
    ap.add_argument("--edge-strict", action="store_true",
                    help="with --edge-filter, also reject CPU-fallback norms "
                         "(LayerNorm/GroupNorm/InstanceNorm), not just hard failures")
    ap.add_argument("--mac-cap", type=float, default=0.0,
                    help="hard MAC ceiling in MMac tied to target device (0 = no cap)")
    ap.add_argument("--diversity-gate", action="store_true",
                    help="WL-graph-hash gate: reject a generated child whose computation "
                         "GRAPH is an exact clone (incl. renamed/width-rescaled) of an already-"
                         "accepted model, BEFORE training. Free, no GPU. Fights Run-1 Lite collapse.")
    # EXACT identity from the AST of every method rather than an fx trace of forward.
    # Measured on 1,468 denoise models: 1,411 distinct against 1,348, 100% coverage against
    # 97.5%, and it separates 38 groups whose forward graphs match but whose optimizer, loss
    # or update loop differ -- e.g. DenoiseUnet5 vs DenoiseUnet5F, 2.68 dB apart.
    # REQUIRES A FRESH --diversity-store: hashes from the two backends are not comparable,
    # and a mixed store silently behaves as an empty one.
    ap.add_argument("--ast-exact", action="store_true",
                    help="exact hash from the AST of all methods (see prof/uid_full.py)")
    ap.add_argument("--diversity-graded", action="store_true",
                    help="ALSO reject near-duplicates (continuous-WL embedding cosine >= "
                         "--graded-max), not just exact clones. OPT-IN: for a mutation-only "
                         "generator this forces cross-FAMILY diversity or starves (same-family "
                         "mutants cluster >=0.985) — enable only with topology operators.")
    ap.add_argument("--graded-max", type=float, default=0.95,
                    help="with --diversity-graded, reject if continuous-WL cosine to any accepted "
                         "model >= this. 0.95 calibrated (trivial variation >=0.985, distinct <=0.921).")
    ap.add_argument("--cka-max", type=float, default=0.80,
                    help="(legacy, unused by the WL gate; kept for launcher compatibility)")
    ap.add_argument("--diversity-n", type=int, default=16,
                    help="(legacy, unused by the WL gate; kept for launcher compatibility)")
    ap.add_argument("--diversity-store", default=None,
                    help="path to persist the accepted-graph reference set (default: "
                         "<checkpoint>_wlhashes.pt) — enables resume of the gate's reference set")
    ap.add_argument("--diverse-pairing", action="store_true",
                    help="choose the SECOND crossover parent as the most-DISTANT "
                         "(continuous-WL cosine) of a sampled mate pool instead of an "
                         "independent tournament. Measured: raises block-crossover "
                         "distinct-child rate ~50%%->~80%% (cross-family pairs)")
    ap.add_argument("--mate-pool", type=int, default=6,
                    help="with --diverse-pairing, size of the sampled mate pool the "
                         "most-distant second parent is chosen from")
    ap.add_argument("--predictor", action="store_true",
                    help="funnel STAGE 1: rank each generation with the learned "
                         "accuracy predictor and train only the top fraction "
                         "(--predict-keep); skip the rest with a predicted score")
    ap.add_argument("--predict-keep", type=float, default=0.5,
                    help="with --predictor, fraction of each generation to actually "
                         "train (default 0.5 = drop the bottom half)")
    ap.add_argument("--predict-min-train", type=int, default=4,
                    help="with --predictor, always train at least this many per gen "
                         "so a generation can't collapse to predictions only")
    ap.add_argument("--crossover", choices=["none", "block"], default="none",
                    help="none=atomic inheritance (arms A/B); block=real component-block swap (arms C/D)")
    ap.add_argument("--cost-metric", choices=["none", "params", "macs"],
                    default="macs",
                    help="edge-cost proxy folded into fitness (default: macs)")
    ap.add_argument("--cost-weight", type=float, default=1.0,
                    help="lambda on the log-cost penalty, in dB per e-fold (default 1.0)")
    ap.add_argument("--cost-mode", choices=["log", "soft"], default="log",
                    help="log = two-sided bonus/penalty; soft = one-sided budget")
    ap.add_argument("--cost-budget", type=float, default=0.0,
                    help="cost budget in the metric's units; 0 = auto (median seed cost)")
    ap.add_argument("--checkpoint", default=os.path.join(_HERE, "seed_ga_checkpoint.pkl"))
    ap.add_argument("--cache", default=os.path.join(_HERE, "seed_ga_fitness_cache.json"))
    ap.add_argument("--models-dir", default=os.path.join(_HERE, "seed_ga_models"),
                    help="dir where every evaluated genotype's source is saved (checksum.py)")
    ap.add_argument("--results", default=os.path.join(_HERE, "seed_ga_results.json"))
    ap.add_argument("--save-to-db", action="store_true")
    # --- integrated seed-growth (promotion) stage: after the GA finishes, full-train
    #     this run's top-K and promote the psnr50 survivors into a grown seed pool.
    #     Makes promotion a first-class part of the pipeline, not an external script. ---
    ap.add_argument("--promote", action="store_true",
                    help="after the run, full-train top-K + promote survivors to --promote-dir")
    ap.add_argument("--promote-k", type=int, default=10,
                    help="top-K offspring (by eff-dB) to full-train for promotion")
    ap.add_argument("--promote-epoch", type=int, default=50,
                    help="epochs for the promotion full-train (real psnr50)")
    ap.add_argument("--promote-bar", type=float, default=34.0,
                    help="psnr50 floor a model must clear to earn a seed slot")
    ap.add_argument("--promote-dir", default=None,
                    help="emit dir for promoted seeds (default promoted_pool_<arm>)")
    ap.add_argument("--retrain-predictor", action="store_true",
                    help="after the run (and promotion), refresh labels + refit the "
                         "accuracy predictor on the FULL corpus so the next run's funnel "
                         "is calibrated to newly-introduced families")
    # The no-op is a MEASURED property of the benchmark, not a tuning knob: 33.021 dB is what
    # returning the noisy 256x256 centre crop unchanged scores, verified against stored DB values
    # to 0.00 dB. Change it only if the benchmark changes. Set --noop-floor 0 to disable the gate
    # entirely (e.g. a sweep at a different sigma, where the floor moves).
    ap.add_argument("--deploy-gate", action="store_true",
                    help="EXPORT gate only: admit a model to the corpus only if it "
                         "converts to TFLite and its graph is <= --deploy-max-ops. Does "
                         "NOT affect selection -- rejected models still score and still "
                         "steer the search. Fails CLOSED if the converter is missing.")
    ap.add_argument("--deploy-predict-thr", type=float, default=0.0,
                    help="EXPORT gate only, source-only: admit a model to the corpus only if "
                         "its predicted full-CPU-eviction probability is below this "
                         "threshold (0 disables; 0.5 is the measured operating point at "
                         "precision 0.90 / recall 0.83). Needs no TFLite conversion and no "
                         "device. Like --deploy-gate it does NOT affect selection, and it "
                         "fails CLOSED if the frozen model is missing.")
    ap.add_argument("--deploy-max-ops", type=int, default=203,
                    help="graph-op ceiling for --deploy-gate (203 = 92%% precision "
                         "against latency>10ms; see npu_gate --calibrate-ops)")
    ap.add_argument("--fullres-fitness", action="store_true",
                    help="score candidates on FULL-RESOLUTION RGB PSNR (the published "
                         "protocol) instead of the 256x256 Y crop. Costs ~35 s per "
                         "candidate on top of ~210 s training (+17%%). The crop metric "
                         "has Spearman -0.025 vs full-res among the top 100 models, so a "
                         "search that maximises it is blind at the frontier -- see "
                         "fullres_score.py")
    ap.add_argument("--fullres-val", default="/a/mm/nn-dataset/validation",
                    help="validation root holding original/ and denoised/")
    ap.add_argument("--fullres-limit", type=int, default=0,
                    help="frames to use, 0 = all 50. Anything else is FASTER BUT NOT "
                         "comparable to the stored corpus numbers")
    ap.add_argument("--noop-floor", type=float, default=33.021,
                    help="PSNR of returning the input unchanged; children measuring below "
                         "floor+margin are scored but not exported to the corpus (0 disables)")
    ap.add_argument("--noop-margin", type=float, default=0.207,
                    help="run-to-run noise added to the floor (measured in the confirmation "
                         "run; 0.14 was the earlier ASSUMED value and is too low by ~1.5x)")
    ap.add_argument("--seed-mac-cap", type=float, default=9053.0,
                    help="drop input seeds whose MACs (MMac @256) exceed this edge cap "
                         "so over-cap heavyweights (e.g. DenoiseUnet7F/10F/2F) can't "
                         "poison the GA with illegal descendants; 0 disables the filter")
    ap.add_argument("--dataset", default=DATASET,
                    help="dataset name passed to check_nn (default 'denoise'/MAI; "
                         "use 'midd' for the paper's MIDD set)")
    ap.add_argument("--metric", default=METRIC,
                    help="metric name passed to check_nn (default 'psnr' Y-channel; "
                         "use 'psnr_rgb' for paper-comparable MIDD RGB PSNR)")
    args = ap.parse_args()
    if args.prefix:
        os.environ['GA_PREFIX'] = args.prefix


    # The fitness eval reads module-level DATASET/METRIC (see check_nn call above);
    # override them from the CLI so a single script can target MAI or MIDD.
    DATASET, METRIC = args.dataset, args.metric

    def log(m):
        print(m, flush=True)

    log(f"=== Seed GA (real DenoiseUnet architectures) | gens={args.gens} "
        f"epoch={args.epoch} | dataset={DATASET} metric={METRIC} ===")
    log(f"Mutation weights (env BRUTE_WEIGHTS): {os.environ.get('BRUTE_WEIGHTS', '(config default)')}")
    log(f"Forced op: {args.mutation_type or '(weighted mix)'}")

    seeds = load_seeds(args.seeds, log, args.seed_files, mac_cap=args.seed_mac_cap)
    if not seeds:
        log("No seeds loaded — abort.")
        return
    pop = args.pop_size if args.pop_size > 0 else len(seeds)
    log(f"Population target = {pop} (pre-seeded with {len(seeds)} real seeds: "
        f"{[i.chromosome['name'] for i in seeds]}; grows to {pop} after gen 0)")

    # Shared cache + seen set: the fitness cache survives preemption on disk,
    # and `seen` (its keys) lets mutation avoid regenerating anything already
    # evaluated — the two GA components share one view of what's been tried.
    cache = {}
    if os.path.exists(args.cache):
        try:
            cache = json.load(open(args.cache))
            log(f"  loaded fitness cache: {len(cache)} entries")
        except Exception:
            cache = {}
    seen = set(cache.keys())

    # Edge-cost budget: auto-calibrate to the median seed cost so the search is
    # pushed to match/beat the BASELINES' efficiency, not an arbitrary target.
    budget = None
    if args.cost_metric != "none":
        from edge_cost import measure_cost
        costs = []
        log(f"Edge cost: metric={args.cost_metric} mode={args.cost_mode} "
            f"weight={args.cost_weight}")
        for ind in seeds:
            c = measure_cost(ind.chromosome["code"])
            v = c.get(args.cost_metric) if c.get("ok") else None
            costs.append(v)
            nm = ind.chromosome["name"]
            if v:
                log(f"  seed {nm:16s} {c['macs']/1e6:8.1f} MMac  {c['params']/1e6:6.2f} Mp")
            else:
                log(f"  seed {nm:16s} (cost unmeasurable)")
        valid = sorted(v for v in costs if v)
        if args.cost_budget > 0:
            budget = args.cost_budget
        elif valid:
            budget = valid[len(valid) // 2]  # median
        unit = "MMac" if args.cost_metric == "macs" else "Mp"
        log(f"  budget = {(budget or 0)/1e6:.2f} {unit} "
            f"({'set' if args.cost_budget > 0 else 'auto=median seed'})")

    diversity_gate = None
    if args.diversity_gate:
        from diversity_gate import DiversityGate
        store = args.diversity_store or (os.path.splitext(args.checkpoint)[0] + "_wlhashes.pt")
        diversity_gate = DiversityGate(store_path=store, log=log,
                                       graded=args.diversity_graded, cos_max=args.graded_max,
                                       ast_exact=args.ast_exact)
        log(f"Diversity gate ON: WL graph-hash (exact clones)"
            + (f" + GRADED near-dup (cos>={args.graded_max})" if args.diversity_graded else "")
            + f" (store={os.path.basename(store)})")

    fitness = make_fitness(args.epoch, args.batch, args.lr, args.num_workers,
                           args.save_to_db, args.cache, log, cache, seen,
                           args.cost_metric, budget, args.cost_weight,
                           args.cost_mode, args.models_dir,
                           edge_filter=args.edge_filter, mac_cap=args.mac_cap,
                           edge_strict=args.edge_strict,
                           diversity_gate=diversity_gate,
                           prof_predict=args.prof_predictor,
                           prof_epochs=args.prof_epochs,
                           prof_target=args.prof_target,
                           noop_floor=args.noop_floor,
                           noop_margin=args.noop_margin,
                           fullres_fitness=args.fullres_fitness,
                           fullres_val=args.fullres_val,
                           fullres_limit=args.fullres_limit,
                           deploy_gate=args.deploy_gate,
                           deploy_max_ops=args.deploy_max_ops,
                           deploy_predict_thr=args.deploy_predict_thr)

    engine = GeneticAlgorithmEngine(
        population_size=pop,
        search_space={},  # unused: population is pre-seeded, mutation is source-based
        elitism_count=args.elitism,
        selection_strategy=(DiverseTournamentSelection(args.tournament,
                                                       args.mate_pool, log)
                            if args.diverse_pairing
                            else TournamentSelection(args.tournament)),
        crossover_strategy=SeedCrossover(args.crossover, log),
        mutation_strategy=SeedMutation(args.mutation_rate, args.mutation_type,
                                       log, seen, args.mutation_tries,
                                       rich_skip=args.skip_ops, skip_prob=args.skip_prob),
        checkpoint_path=args.checkpoint,
    )
    # Pre-seed (only used when there is no checkpoint to resume from).
    engine.population.individuals = seeds

    prescreen = None
    if args.predictor:
        pj = os.path.join(_HERE, "predictor", "predictor.joblib")
        if not os.path.exists(pj):
            log(f"  --predictor set but {pj} missing — run predictor/train_predictor.py "
                f"first; proceeding WITHOUT the funnel.")
        else:
            log(f"Predictor funnel ON: train top {args.predict_keep:.0%} of each gen "
                f"(min {args.predict_min_train}), skip rest with predicted score.")
            prescreen = make_prescreen(args.predict_keep, args.predict_min_train,
                                       cache, args.cost_metric, budget,
                                       args.cost_weight, args.cost_mode, log)

    t0 = time.time()
    best = engine.run(args.gens, fitness, pre_generation_hook=prescreen)

    # NEVER-CROWN guard: if the winner was only ever predicted (never trained),
    # train it for real now so the reported best is a measured result.
    if best and best.chromosome.get("predicted"):
        log("Best was predictor-only -> real-training it before reporting.")
        best.chromosome.pop("predicted", None)
        best.fitness = fitness(best.chromosome)
    dt = time.time() - t0

    # best.fitness is now EFFECTIVE (edge-aware) dB/scale; pull true PSNR + cost
    # from the cache for an honest report.
    be = cache.get(_checksum(best.chromosome["code"]), {}) if best else {}
    result = {
        "objective": {"cost_metric": args.cost_metric, "cost_mode": args.cost_mode,
                      "cost_weight": args.cost_weight, "budget": budget},
        "best_name": best.chromosome.get("name") if best else None,
        "best_eff_db": (best.fitness * PSNR_SCALE) if best and best.fitness else None,
        "best_psnr_db": be.get("psnr"),
        "best_macs": be.get("macs"),
        "best_params": be.get("params"),
        "best_parent": best.chromosome.get("parent") if best else None,
        "seeds": [i.chromosome["name"] for i in seeds],
        "gens": args.gens, "epoch": args.epoch, "elapsed_sec": round(dt, 1),
    }
    with open(args.results, "w") as f:
        json.dump(result, f, indent=2)
    # Save the winning source next to the results for inspection.
    if best and best.fitness:
        with open(os.path.join(_HERE, "seed_ga_best.py"), "w") as f:
            f.write(best.chromosome["code"])

    log(f"\n=== DONE in {dt:.0f}s ===")
    if best and best.fitness:
        macs_m = (be.get("macs") or 0) / 1e6
        log(f"BEST (edge-aware): {best.fitness * PSNR_SCALE:.3f} eff-dB  "
            f"name={best.chromosome.get('name')} "
            f"(from {best.chromosome.get('parent', 'seed')}) | "
            f"psnr {be.get('psnr', 0):.3f} dB  {macs_m:.1f} MMac")
    log(f"Results -> {args.results}")

    if args.promote:
        try:
            _promote_after_run(args, log)
        except Exception as e:
            log(f"[promote] stage failed: {type(e).__name__}: {e}")

    if args.retrain_predictor:
        try:
            _retrain_predictor(args, log)
        except Exception as e:
            log(f"[retrain] predictor retrain failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
