"""
Export the TNNLS layerwise-LR experiment records into nn-dataset layout.

Two kinds of records go out:

  1. Per-seed runs (epoch/seed ablation, 100-epoch extension, rate-matched and ULMFiT
     controls, base-rate-0.1 runs, regression-to-the-mean arms, out-of-distribution
     fresh-control re-checks). Source: the model dirs under out/nngpt/llm/epoch/A<slot>/
     synth_nn, each holding new_nn.py and one <epoch>.json per trained epoch.

     Each seed copy differs from its siblings only by an inert trailing assignment
     (`_replicate_seed = k`, `_base_lr_tag = x`) added so the evaluator would not
     deduplicate it. We strip that tag, so all seeds of one condition share one model
     file, and write every seed as a separate trial in that model's <epoch>.json list.
     This is the layout nn-dataset already uses for repeated trials, and it survives
     the graph-based model hashing that nn-dataset adopted after commit be8657e5.

  2. One-epoch sweep records (strategy-injected models and their llr_uniform controls)
     that exist in the local LEMUR DB but not in nn-dataset. Source: the stat table of
     db/ab.nn.db. Model files for these are already in nn-dataset (PRs #225/#234/#236).

Output: <out>/ab/nn/nn/<name>.py and <out>/ab/nn/stat/train/<task>_<dataset>_<metric>_<name>/
<epoch>.json, ready to copy over an nn-dataset checkout, plus <out>/export_manifest.csv
mapping every exported trial back to its source dir or DB row.

Usage:
    python -m ab.gpt.brute.llr.export_nn_dataset --out /path/to/export [--repo /path/to/nn-dataset]
With --repo, existing <epoch>.json lists in the checkout are merged (trials already present,
matched on uid + seed, are not duplicated).
"""

import argparse
import csv
import hashlib
import json
import re
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUT_NNGPT = PROJECT_ROOT / 'out' / 'nngpt'
EPOCH_DIR = OUT_NNGPT / 'llm' / 'epoch'
DB_PATH = PROJECT_ROOT / 'db' / 'ab.nn.db'
TASK, METRIC = 'img-classification', 'acc'

# slot -> (experiment label, manifest file)
SLOTS = {
    900: ('ablation_10ep', 'epoch_ablation_manifest_A900.csv'),
    960: ('ood_recheck_6targets', 'ood_epoch_recheck_manifest_A960.csv'),
    961: ('ood_recheck_6targets', 'ood_epoch_recheck_manifest_A961.csv'),
    971: ('ablation_repl_100ep', 'tnnls_manifest_A971.csv'),
    972: ('ablation_repl_100ep', 'tnnls_manifest_A972.csv'),
    973: ('ablation_repl_100ep', 'tnnls_manifest_A973.csv'),
    974: ('ablation_repl_100ep', 'tnnls_manifest_A974.csv'),
    975: ('structure_lr0.01', 'tnnls_manifest_A975.csv'),
    976: ('baserate_lr0.1', 'tnnls_manifest_A976.csv'),
    977: ('ablation_repl_100ep', 'tnnls_manifest_A977.csv'),
    978: ('ablation_repl_100ep', 'tnnls_manifest_A978.csv'),
    979: ('baserate_lr0.1', 'tnnls_manifest_A979.csv'),
    980: ('baserate_lr0.1', 'tnnls_manifest_A980.csv'),
    981: ('ratematched_100ep', 'tnnls_manifest_A981.csv'),
    982: ('ratematched_100ep', 'tnnls_manifest_A982.csv'),
    983: ('ablation_10ep', 'tnnls_manifest_A983.csv'),
    984: ('regmean', 'tnnls_manifest_A984.csv'),
    985: ('regmean', 'tnnls_manifest_A985.csv'),
    986: ('regmean', 'tnnls_manifest_A986.csv'),
    987: ('regmean', 'tnnls_manifest_A987.csv'),
    988: ('regmean', 'tnnls_manifest_A988.csv'),
    989: ('regmean', 'tnnls_manifest_A989.csv'),
    991: ('ratematched_100ep', 'tnnls_manifest_A991.csv'),
    992: ('ood_recheck_14targets', 'ood_epoch_recheck_manifest_A992.csv'),
    993: ('ratematched_100ep', 'tnnls_manifest_A993.csv'),
    994: ('ablation_10ep+regmean', 'tnnls_manifest_A994.csv'),
    995: ('ood_recheck_14targets', 'ood_epoch_recheck_manifest_A995.csv'),
    996: ('heldout_10ep', 'tnnls_manifest_A996.csv'),
    997: ('heldout_10ep', 'tnnls_manifest_A997.csv'),
    999: ('heldout_10ep_ratematched', 'tnnls_manifest_A999.csv'),
    # Round 2: tuned baselines (cosine schedule, swept base rate; AdamW for ConvNeXt)
    963: ('tuned_ratesweep_100ep', 'tnnls_manifest_A963.csv'),
    964: ('tuned_ratesweep_100ep', 'tnnls_manifest_A964.csv'),
    965: ('tuned_ratesweep_100ep', 'tnnls_manifest_A965.csv'),
    966: ('tuned_ratesweep_100ep', 'tnnls_manifest_A966.csv'),
    949: ('tuned_ratesweep_100ep', 'tnnls_manifest_A949.csv'),
    951: ('tuned_ratesweep_100ep', 'tnnls_manifest_A951.csv'),
    953: ('tuned_ratesweep_100ep', 'tnnls_manifest_A953.csv'),
    954: ('tuned_ratesweep_100ep', 'tnnls_manifest_A954.csv'),
    955: ('tuned_ratesweep_100ep', 'tnnls_manifest_A955.csv'),
    **{s: ('tuned_100ep', f'tnnls_manifest_A{s}.csv') for s in (*range(943, 949), *range(920, 928), 956, 957)},
    # Round 2: mean-preserving strategies (group multipliers rescaled to a mean of 1.0)
    967: ('meanpres_1ep', 'tnnls_manifest_A967.csv'),
    968: ('meanpres_1ep', 'tnnls_manifest_A968.csv'),
    969: ('meanpres_1ep', 'tnnls_manifest_A969.csv'),
    959: ('meanpres_1ep', 'tnnls_manifest_A959.csv'),
    998: ('meanpres_2x2_10ep', 'tnnls_manifest_A998.csv'),
    970: ('meanpres_2x2_10ep', 'tnnls_manifest_A970.csv'),
    958: ('meanpres_2x2_10ep', 'tnnls_manifest_A958.csv'),
}
ROUND2_TUNED = {963, 964, 965, 966, 949, 951, 953, 954, 955, *range(943, 949), *range(920, 928), 956, 957}
ROUND2_MEANPRES = {967, 968, 969, 959, 998, 970, 958}

# Trailing replicate tags written by epoch_ablation_gen / tnnls_gen / ood_*_gen.
_TAG_RE = re.compile(r'\n*#[^\n]*(replicate|inert)[^\n]*\n(?:_[a-z_]+ = [^\n]*\n?)+\s*$')


def strip_tags(code: str) -> str:
    return _TAG_RE.sub('\n', code).rstrip() + '\n'


def model_name(prefix: str, code: str) -> str:
    # nn-dataset @ be8657e5 ab.nn.util.Util.uuid4: md5 of the source with whitespace removed
    digest = hashlib.md5(re.sub(r'\s', '', code).encode()).hexdigest()
    return f"{prefix}-{digest}"


def _read_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, newline='') as f:
        return {r['model_dir']: r for r in csv.DictReader(f)}


def collect_per_seed(slots=None):
    models, trials, manifest = {}, {}, []
    for slot, (exp, mf) in SLOTS.items():
        if slots and slot not in slots:
            continue
        synth = EPOCH_DIR / f'A{slot}' / 'synth_nn'
        man = _read_manifest(OUT_NNGPT / mf)
        if not synth.exists():
            continue
        for mdir in sorted(p for p in synth.iterdir() if p.is_dir()):
            meta = man.get(mdir.name)
            if meta is None:
                continue
            jdir = mdir / meta.get('dataset', 'cifar-10') if (mdir / meta.get('dataset', '_')).is_dir() else mdir
            epochs = sorted(int(p.stem) for p in jdir.glob('*.json') if p.stem.isdigit())
            if not epochs:
                continue        # stopped at chance, or deduplicated by the evaluator
            code = strip_tags((mdir / 'new_nn.py').read_text(encoding='utf-8'))
            prefix = 'llrood' if 'target_nn' in meta else 'llrtn'
            name = model_name(prefix, code)
            models[name] = code
            ds = meta.get('dataset', 'cifar-10')
            for ep in epochs:
                for t in json.loads((jdir / f'{ep}.json').read_text()):
                    t = dict(t)
                    t['epoch'] = ep                       # some evaluator versions wrote epoch_max here
                    t['seed'] = int(meta['seed'])           # replicate index; see export_manifest.csv
                    trials.setdefault((ds, name, ep), []).append(t)
            manifest.append({'source': f'A{slot}/{mdir.name}', 'experiment': exp, 'nn': name,
                             'dataset': ds, 'architecture': meta.get('architecture') or meta.get('target_nn'),
                             'condition': meta['condition'], 'strategy': meta['strategy'],
                             'base_lr': meta.get('base_lr', '0.01'), 'seed': meta['seed'],
                             'epochs': max(epochs)})
    return models, trials, manifest


def collect_sweep():
    """One-epoch sweep trials (strategies + their llr_uniform controls) from the local DB.

    Only the (model, dataset, transform, epoch) records the paper analyses are exported:
    every row of llr_vs_vanilla.csv, plus the architecture's llr_uniform control at the
    same dataset and transform. The local DB's prm table reuses one uid for many trials,
    so only hyperparameters with a single value under a uid are carried over.
    """
    with open(OUT_NNGPT / 'llr_vs_vanilla.csv', newline='') as f:
        sweep = list(csv.DictReader(f))
    with open(OUT_NNGPT / 'llr_evaluated_with_meta.csv', newline='') as f:
        meta = {r['nn']: r for r in csv.DictReader(f)}
    control = {r['architecture']: n for n, r in meta.items() if r.get('strategy') == 'llr_uniform'}
    keys = {}
    for r in sweep:
        keys[(r['nn'], r['dataset'], r['transform'])] = (r['architecture'], r['strategy'])
        if r['architecture'] in control:
            keys[(control[r['architecture']], r['dataset'], r['transform'])] = (r['architecture'], 'llr_uniform')
    con = sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True)
    names = sorted({k[0] for k in keys})
    q = ','.join('?' * len(names))
    rows = con.execute(
        f"SELECT id, nn, dataset, epoch, transform, accuracy, prm, duration FROM stat "
        f"WHERE nn IN ({q}) AND task=? AND metric=? AND epoch=1", (*names, TASK, METRIC)).fetchall()
    prm_cache, trials, manifest = {}, {}, []
    for sid, nn, ds, ep, tf, acc, uid, dur in rows:
        if (nn, ds, tf) not in keys:
            continue
        if uid not in prm_cache:
            vals = {}
            for name, value in con.execute("SELECT DISTINCT name, value FROM prm WHERE uid=?", (uid,)):
                vals.setdefault(name, set()).add(value)
            prm_cache[uid] = {k: next(iter(v)) for k, v in vals.items() if len(v) == 1}
        t = {**prm_cache[uid], 'accuracy': acc, 'metric_acc': acc, 'duration': dur, 'epoch': int(ep),
             'transform': tf, 'uid': uid}
        trials.setdefault((ds, nn, int(ep)), []).append(t)
        arch, strat = keys[(nn, ds, tf)]
        manifest.append({'source': f'db/ab.nn.db:stat:{sid}', 'experiment': 'sweep_1ep', 'nn': nn,
                         'dataset': ds, 'architecture': arch, 'condition': 'control' if strat == 'llr_uniform' else 'strategy',
                         'strategy': strat, 'base_lr': prm_cache[uid].get('lr', ''), 'seed': '', 'epochs': int(ep)})
    # Code for every exported model: nn-dataset skips any record whose ab/nn/nn/<name>.py is
    # missing (the llr_uniform controls were never part of the strategy PRs).
    code = {n: c for n, c in con.execute(f"SELECT DISTINCT name, code FROM nn WHERE name IN ({q})", names)}
    return trials, manifest, code


def _key(t):
    return (t.get('uid'), t.get('seed'), t.get('transform'), round(float(t.get('accuracy', -1)), 6))


def write(out: Path, models, trials, manifest, repo: Path | None):
    (out / 'ab' / 'nn' / 'nn').mkdir(parents=True, exist_ok=True)
    missing = sorted({nn for (_, nn, _) in trials} - set(models))
    if missing:
        raise SystemExit(f"{len(missing)} models have records but no code, e.g. {missing[:3]}")
    for name, code in models.items():
        if repo and (repo / 'ab' / 'nn' / 'nn' / f'{name}.py').exists():
            continue                              # already in nn-dataset
        (out / 'ab' / 'nn' / 'nn' / f'{name}.py').write_text(code, encoding='utf-8')
    new_files = merged = added = 0
    for (ds, nn, ep), ts in sorted(trials.items()):
        rel = Path('ab') / 'nn' / 'stat' / 'train' / f'{TASK}_{ds}_{METRIC}_{nn}' / f'{ep}.json'
        existing = []
        if repo and (repo / rel).exists():
            existing = json.loads((repo / rel).read_text())
            merged += 1
        else:
            new_files += 1
        seen = {_key(t) for t in existing}
        fresh = [t for t in ts if _key(t) not in seen]
        added += len(fresh)
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text(json.dumps(existing + fresh, indent=4, sort_keys=True) + '\n')
    with open(out / 'export_manifest.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)
    print(f"models written      : {len(models)}")
    print(f"epoch files         : {new_files} new, {merged} merged with the checkout")
    print(f"trials added        : {added}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--repo', default=None, help='nn-dataset checkout to merge existing records from')
    ap.add_argument('--skip-sweep', action='store_true')
    ap.add_argument('--slots', default='', help='Comma list of A<slot> numbers to export (default: all)')
    a = ap.parse_args()
    slots = {int(x) for x in a.slots.split(',') if x.strip()}
    models, trials, manifest = collect_per_seed(slots)
    print(f"per-seed: {len(manifest)} runs -> {len(models)} models")
    if not a.skip_sweep:
        st, sm, sc = collect_sweep()
        models.update(sc)
        print(f"sweep   : {len(sm)} trials over {len({m['nn'] for m in sm})} models")
        for k, v in st.items():
            trials.setdefault(k, []).extend(v)
        manifest += sm
    write(Path(a.out), models, trials, manifest, Path(a.repo) if a.repo else None)


if __name__ == '__main__':
    main()
