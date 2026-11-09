"""Run-3 SFT dataset: score-conditioned generation examples from the measured corpus.

The classification pipeline's recipe (iterative_finetune.py) adapted to denoise: the LLM is
fine-tuned on models whose MEASURED score clears a bar, with the score in the prompt, and the
dataset grows each cycle as newly generated models qualify.

Every example is rendered through llm_denoise.compose() -- the SAME template, strategy
round-robin and reference-selection the generation stage uses -- so the fine-tune
distribution matches the inference distribution exactly. The completion is the model's code
in the <nn>...</nn> contract that extract_nn() parses at generation time.

The target model NEVER appears among its own prompt's references (that would teach copying);
references are tournament-picked from the measured corpus with the target excluded.

  # cycle-0 base set (everything in the DB at or above the bar, deduped by arch class):
  python run3_sft_data.py --min-db 34.0 --uid-map uid_map.json --out sft_data/

  # append this cycle's accepted models:
  python run3_sft_data.py --append accepted.jsonl --out sft_data/
"""
import argparse
import hashlib
import json
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import llm_denoise as L                                            # noqa: E402

COMPLETION_FMT = "<nn>\n{code}\n</nn>"
DEV_FRACTION = 0.05


def _dev_split(name):
    """Deterministic membership by name hash -- stable across cycles, so an appended model
    never migrates between train and dev."""
    h = int(hashlib.sha256(name.encode()).hexdigest(), 16)
    return (h % 1000) < int(DEV_FRACTION * 1000)


def corpus_targets(db, min_db, uid_map):
    """Best-scored row per model at/above the bar; one representative per arch class."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT s.nn, MAX(s.accuracy), n.code FROM stat s JOIN nn n ON n.name = s.nn "
        "WHERE s.task='img-denoising' AND n.code IS NOT NULL GROUP BY s.nn").fetchall()
    con.close()
    bar = min_db / 48.0
    good = [(nn, acc, code) for nn, acc, code in rows if acc and acc >= bar]
    if not uid_map:
        return good
    best = {}
    for nn, acc, code in good:
        u = uid_map.get(nn, nn)          # unknown names count as their own class
        if u not in best or acc > best[u][1]:
            best[u] = (nn, acc, code)
    return sorted(best.values())


def deploy_targets(uid_map, w, min_psnr, device, min_score=None, psnr_map=None):
    """Targets ranked by the JOINT objective (full-res RGB PSNR + measured latency).

    Replaces corpus_targets for --objective deploy. See deploy_objective.py for why the crop-PSNR
    selection it supersedes could not work: its own no-op floor sat above every model the 20-cycle
    loop ever accepted.

    ONE REPRESENTATIVE PER ARCH CLASS, as before, but chosen by joint score rather than by crop
    accuracy -- otherwise the dedup step would quietly reintroduce the old objective as a
    tiebreak inside each class.

    CONVERTIBILITY IS NOT AN EXTRA FILTER HERE AND SHOULD NOT BE ADVERTISED AS ONE. Every model in
    this set has a measured latency, and a latency measurement requires a TFLite artifact -- so
    these models converted by construction. The convertibility term in the objective earns its
    keep on GENERATED candidates (npu_gate's pre-screen, seed_mutation's monotonicity rule), not
    on selecting targets from already-profiled models.
    """
    import deploy_objective as D
    codes = D.load_codes_from_db()
    rows, cov = D.joint(codes, w=w, min_psnr=min_psnr, device=device, psnr_map=psnr_map)
    print(f"[deploy] coverage: quality {cov['fullres']}, latency({device}) {cov['latency']}, "
          f"both {cov['joined']} -> kept {cov['kept']} "
          f"({cov['below_floor']} below the {min_psnr} dB no-op floor)")
    if not rows:
        return []
    if min_score is not None:
        n0 = len(rows)
        rows = [r for r in rows if r["score"] >= min_score]
        print(f"[deploy] {len(rows)}/{n0} at or above score {min_score}")
    best = {}
    for r in rows:
        if r["model"] not in codes:
            continue
        u = (uid_map or {}).get(r["model"], r["model"])
        if u not in best or r["score"] > best[u]["score"]:
            best[u] = r
    out = [(r["model"], r, codes[r["model"]]) for r in best.values()]
    out.sort(key=lambda t: -t[1]["score"])
    print(f"[deploy] {len(out)} targets after arch-class dedup"
          f"{' (no uid map -- each name is its own class)' if not uid_map else ''}")
    return out



# ---------------------------------------------------------------- target-set shaping
def _shingles(code, n=5):
    """Token n-grams. Comments and blank lines are dropped so that reformatting two copies of the
    same architecture does not make them look novel."""
    toks = []
    for line in code.splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            toks += line.replace('(', ' ( ').replace(')', ' ) ').replace(',', ' , ').split()
    return {hash(tuple(toks[i:i+n])) & 0xFFFFFFFF for i in range(max(0, len(toks)-n+1))}


def _minhash(sh, num_perm=128):
    """Cheap MinHash signature: num_perm independent hash permutations of the shingle set."""
    if not sh:
        return tuple([0xFFFFFFFF] * num_perm)
    sig = []
    for k in range(num_perm):
        a = (k * 2654435761 + 1) & 0xFFFFFFFF
        sig.append(min(((h * a) & 0xFFFFFFFF) for h in sh))
    return tuple(sig)


def _jac(a, b):
    return sum(1 for x, y in zip(a, b) if x == y) / float(len(a))


def _producer(name):
    """Producer family, e.g. DenoiseGA / DnSeed2Fast / LLM_Pg1_seeded -> LLM."""
    return 'LLM' if name.startswith('LLM') else name.split('-')[0]


def shape_targets(out, per_producer=0, jaccard_max=0.0, log=print):
    """Producer-balanced, near-duplicate-filtered target selection.

    WHY NOT JUST TAKE THE TOP N BY SCORE. Two independent reasons, both measured.

    1. PRODUCER IMBALANCE. The joint-objective pool is 86.5% GA output and 13.5% LLM. Fine-tuning
       on it unshaped teaches the model to reproduce GA edits, which is the one thing this arm does
       not need -- we already have a GA, and its whole advantage over the LLM is ceiling, not
       novelty. `per_producer` caps how many targets any one family contributes so the LLM's own
       best work is not drowned by an arm that outnumbers it 6:1.

    2. INTERNAL REDUNDANCY. The graded distinctness audit retained 182 of 2,670 models at
       cosine 0.90 -- the corpus is full of near-duplicates, and they concentrate in exactly the
       mutation-driven GA arms that dominate this pool. Score-ranked selection therefore returns
       many copies of one architecture and spends the fine-tune budget re-teaching it. The
       Jaccard filter takes targets in score order and skips any that is more than `jaccard_max`
       similar to something already taken -- the same criterion arXiv:2601.02997 uses at
       tau ~= 0.90, so our selection is comparable to theirs.

    Both are off by default (0 = disabled): this only shapes the set when asked.
    """
    if per_producer <= 0 and jaccard_max <= 0:
        return out
    kept, sigs, per = [], [], {}
    n_prod = n_dup = 0
    for name, row, code in out:                       # already sorted by descending score
        fam = _producer(name)
        if per_producer > 0 and per.get(fam, 0) >= per_producer:
            n_prod += 1
            continue
        if jaccard_max > 0:
            sig = _minhash(_shingles(code))
            if any(_jac(sig, s) > jaccard_max for s in sigs):
                n_dup += 1
                continue
            sigs.append(sig)
        per[fam] = per.get(fam, 0) + 1
        kept.append((name, row, code))
    log('[shape] %d -> %d targets (dropped %d over the per-producer cap, %d near-duplicates)'
        % (len(out), len(kept), n_prod, n_dup))
    mix = sorted(per.items(), key=lambda kv: -kv[1])
    log('[shape] producer mix: ' + ', '.join('%s=%d' % (k, v) for k, v in mix[:8]))
    return kept


# The joint-objective prompt. It differs from USER_TMPL in three ways that all matter:
#   * it names the metric as FULL-RESOLUTION RGB, because the number is not the crop PSNR the
#     model was historically asked to improve and conflating them would teach the wrong target;
#   * it states a latency budget, which the old template had no concept of;
#   * it tells the model the exchange rate, so "improve it" has a defined meaning when the two
#     goals conflict -- without that, a candidate that buys 0.2 dB for 4x the runtime looks like
#     an improvement.
DEFAULT_PSNR_DESC = ("PSNR, measured on RGB channels over full-resolution "
                     "validation images (2432x3200)")

USER_TMPL_DEPLOY = """You are a machine learning model designer.
Your task is to generate a LEMUR Dataset neural network model that beats the reference model below on BOTH of these measured quantities for the 'denoise' dataset and 'img-denoising' task:
  1. {psnr_desc}, higher is better. The reference achieves {base_psnr:.3f} dB. Aim for at least {target_psnr:.3f} dB.
  2. On-device inference latency at 256x256 on a mobile NPU, lower is better. The reference runs in {base_ms:.2f} ms. Stay at or below {target_ms:.2f} ms.
Returning the input unchanged scores {noop:.2f} dB, so any design must clearly exceed that to be useful at all.
When the two goals conflict, {tradeoff:.2f} dB of PSNR is worth a doubling of latency -- so a change that doubles runtime must gain more than {tradeoff:.2f} dB to be an improvement.
Improve the reference by making fundamental changes to the model design, not by scaling it up: a wider or deeper version of the same network will gain PSNR and lose more than it gains on latency.
Respond with a single <nn> XML tag containing the complete PyTorch model code (a standalone 'supported_hyperparameters' function, class 'Net' with __init__, forward, train_setup and learn) — no additional explanation or text.
Reference ({base_psnr:.3f} dB, {base_ms:.2f} ms):
<nn>
{base_code}
</nn>"""


def render_deploy(targets, tournament, seed_salt, w, noop, psnr_desc=None):
    """One example per target: a jointly-WORSE reference in, the better model out.

    The baseline must be worse on the JOINT score, not merely on PSNR. Picking a
    lower-PSNR-but-much-faster model as the "baseline to improve on" would train the model to
    treat a latency regression as progress, which is precisely the failure the old objective had
    no way to express.
    """
    import random
    shared, _ = L.load_shared()
    system = "\n".join(shared["system_base"])
    out = []
    pool = [(n, r) for n, r, _c in targets]
    for i, (nn, r, code) in enumerate(targets):
        rng = random.Random(f"{seed_salt}|deploy|{i}|{nn}")
        worse = [(n2, r2) for n2, r2 in pool if n2 != nn and r2["score"] < r["score"]]
        if not worse:
            continue
        # tournament on the joint score, so mid-strength baselines dominate rather than the worst
        cand = rng.sample(worse, min(tournament, len(worse)))
        bn, br = max(cand, key=lambda t: t[1]["score"])
        bcode = next(c for n2, _r2, c in targets if n2 == bn)
        user = USER_TMPL_DEPLOY.format(
            base_psnr=br["psnr"], target_psnr=r["psnr"], base_ms=br["ms"], target_ms=r["ms"],
            noop=noop, tradeoff=w / 2.0, base_code=bcode.strip(),
            psnr_desc=psnr_desc or DEFAULT_PSNR_DESC)
        out.append({"name": nn, "db": r["psnr"], "ms": r["ms"], "score": r["score"],
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": COMPLETION_FMT.format(code=code.strip())},
                    ]})
    return out


USER_TMPL = """You are a machine learning model designer.
Your task is to generate a LEMUR Dataset neural network model that increases the 'psnr' metric value to at least {target_db:.2f} dB PSNR at training epoch 50 on the 'denoise' dataset for an 'img-denoising' task.
Use the following reference model, which achieved {base_db:.2f} dB PSNR under identical evaluation conditions (metric, dataset, task, and epoch), as the baseline for architectural and training inspiration. Improve it by making fundamental changes to the model design.
Respond with a single <nn> XML tag containing the complete PyTorch model code (a standalone 'supported_hyperparameters' function, class 'Net' with __init__, forward, train_setup and learn) — no additional explanation or text.
Baseline:
<nn>
{base_code}
</nn>"""


def render(targets, supporting, tournament, seed_salt):
    """One SFT example per target, in the REPO's improvement format (conf/prompt/train/
    NN_gen.json): ONE baseline reference + a target score, completion = the better model.
    ~2k tokens per example against ~10k for the reference-block format -- the repo sizing.
    The `supporting` arg is kept for CLI compatibility but unused here."""
    import random
    shared, _ = L.load_shared()
    system = "\n".join(shared["system_base"])
    corpus = L.db_corpus()

    out = []
    for i, (nn, acc, code) in enumerate(targets):
        tgt_db = acc * 48.0
        rng = random.Random(f"{seed_salt}|{i}|{nn}")
        # baseline = a genuinely WORSE model, tournament-picked so mid-strength baselines
        # dominate (repo semantics: baseline -> improved pair)
        worse = [c for c in corpus if c["name"] != nn and c["db"] < tgt_db]
        pool = worse if worse else [c for c in corpus if c["name"] != nn]
        if not pool:
            continue
        base = max(rng.sample(pool, min(tournament, len(pool))),
                   key=lambda r: r.get("rank", 0.0))
        user = USER_TMPL.format(target_db=tgt_db, base_db=base["db"],
                                base_code=base["code"].strip())
        out.append({
            "name": nn,
            "db": tgt_db,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": COMPLETION_FMT.format(code=code.strip())},
            ],
        })
    return out


def write_split(examples, out_dir, append=False):
    os.makedirs(out_dir, exist_ok=True)
    mode = "a" if append else "w"
    n_tr = n_dev = 0
    with open(os.path.join(out_dir, "train.jsonl"), mode) as ftr, \
         open(os.path.join(out_dir, "dev.jsonl"), mode) as fdv:
        for ex in examples:
            rec = json.dumps({"messages": ex["messages"]})
            if _dev_split(ex["name"]):
                fdv.write(rec + "\n"); n_dev += 1
            else:
                ftr.write(rec + "\n"); n_tr += 1
    return n_tr, n_dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None, help="LEMUR DB (default: llm_denoise.db_path())")
    ap.add_argument("--objective", choices=["psnr-crop", "deploy"], default="psnr-crop",
                    help="psnr-crop = the original 256x256 Y-channel selection (kept so the "
                         "20-cycle run3 dataset stays reproducible). deploy = full-resolution RGB "
                         "PSNR + measured on-device latency; see deploy_objective.py")
    ap.add_argument("--latency-weight", type=float, default=1.0,
                    help="deploy only: w in score = 2*psnr - w*log2(ms). w=1 is the published MAI "
                         "trade-off (0.5 dB per 2x runtime); raise it to push harder on latency")
    ap.add_argument("--psnr-source", default="",
                    help="json {model: dB} replacing the full-res scan. Use the 50-EPOCH table: "
                         "5-epoch quality does not rank converged models. Pass --min-psnr and "
                         "--psnr-desc to match whatever metric it holds.")
    ap.add_argument("--psnr-desc", default="",
                    help="how the prompt describes the quality metric; must match --psnr-source")
    ap.add_argument("--min-psnr", type=float, default=30.88,
                    help="deploy only: full-res RGB floor; default is the published no-op")
    ap.add_argument("--min-score", type=float, default=None,
                    help="deploy only: additional cut on the joint score")
    ap.add_argument("--device", default="Samsung Galaxy S26 Ultra",
                    help="deploy only: latencies are not comparable across devices, so exactly "
                         "one is used")
    ap.add_argument("--min-db", type=float, default=34.0)
    ap.add_argument("--uid-map", default=None,
                    help="JSON with {'uids': {model: arch_uid}} for arch-class dedup")
    ap.add_argument("--per-producer", type=int, default=0,
                    help="cap targets per producer family (0 = no cap). The deploy pool is 86.5%% "
                         "GA; without a cap the fine-tune learns to imitate GA edits.")
    ap.add_argument("--jaccard-max", type=float, default=0.0,
                    help="drop a target more than this MinHash-Jaccard similar to one already "
                         "selected (0 = off; 0.90 matches arXiv:2601.02997)")
    ap.add_argument("--supporting", type=int, default=3)
    ap.add_argument("--tournament", type=int, default=3)
    ap.add_argument("--seed-salt", default="run3", help="determinism salt for reference picks")
    ap.add_argument("--append", default=None,
                    help="jsonl of {name, db} accepted this cycle; code read from the DB")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    db = a.db or L.db_path()
    uid_map = None
    if a.uid_map:
        with open(a.uid_map) as f:
            j = json.load(f)
        uid_map = j.get("uids", j)

    if a.append:
        # Records may carry the code inline via a "code" path (screen-accepted candidates
        # never enter the DB since the 50-epoch stage was removed); DB lookup is the
        # fallback for records that are corpus rows.
        targets, names = [], set()
        db_lookup = []
        with open(a.append) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                names.add(r["name"])
                if r.get("code") and os.path.isfile(r["code"]):
                    targets.append((r["name"], r["db"] / 48.0, open(r["code"]).read()))
                else:
                    db_lookup.append(r["name"])
        if db_lookup:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            rows = con.execute(
                "SELECT s.nn, MAX(s.accuracy), n.code FROM stat s JOIN nn n ON n.name = s.nn "
                "WHERE s.nn IN (%s) GROUP BY s.nn" % ",".join("?" * len(db_lookup)),
                sorted(db_lookup)).fetchall()
            con.close()
            targets += [(nn, acc, code) for nn, acc, code in rows if acc and code]
        if not targets:
            print("[append] nothing accepted this cycle; dataset unchanged")
            return
        ex = render(targets, a.supporting, a.tournament, a.seed_salt)
        n_tr, n_dev = write_split(ex, a.out, append=True)
        print(f"[append] {len(ex)} examples ({n_tr} train / {n_dev} dev) for {len(names)} models")
        return

    if a.objective == "deploy":
        import deploy_objective as D
        pm = None
        if a.psnr_source:
            raw = json.load(open(a.psnr_source))
            # accepts {model: dB} or the scored-list form written by pareto/gold tooling
            if isinstance(raw, list):
                pm = {r["n"]: float(r["db"]) for r in raw if r.get("n") and r.get("db")}
            else:
                pm = {k: float(v) for k, v in raw.items() if v is not None}
            print(f"[deploy] quality from {a.psnr_source}: {len(pm)} models "
                  f"(floor {a.min_psnr} dB)")
        tg = deploy_targets(uid_map, a.latency_weight, a.min_psnr, a.device, a.min_score,
                            psnr_map=pm)
        tg = shape_targets(tg, a.per_producer, a.jaccard_max)
        if not tg:
            print("[deploy] NO TARGETS. This objective needs both the full-res re-rank and the "
                  "AI Hub latency corpus; run `deploy_objective.py --report` to see which half "
                  "is missing. Refusing to fall back to the crop metric.")
            return 1
        ex = render_deploy(tg, a.tournament, a.seed_salt, a.latency_weight, a.min_psnr,
                           a.psnr_desc or None)
        n_tr, n_dev = write_split(ex, a.out, append=False)
        print(f"[deploy] wrote {n_tr} train / {n_dev} dev examples to {a.out}")
        best = max(ex, key=lambda e: e["score"]) if ex else None
        if best:
            print(f"[deploy] best target: {best['name'][:44]} "
                  f"{best['db']:.3f} dB @ {best['ms']:.2f} ms (score {best['score']:.2f})")
        return 0

    targets = corpus_targets(db, a.min_db, uid_map)
    print(f"[base] {len(targets)} target models at >= {a.min_db} dB"
          f"{' after arch dedup' if uid_map else ''}")
    ex = render(targets, a.supporting, a.tournament, a.seed_salt)
    n_tr, n_dev = write_split(ex, a.out, append=False)
    print(f"[base] wrote {n_tr} train / {n_dev} dev examples to {a.out}")


if __name__ == "__main__":
    main()
