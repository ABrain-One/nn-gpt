"""Reuse results of an earlier evaluation run for identical loss/optimizer variants.

A result is reused only when the old variant has the same architecture ID
(``arch_uid``) as the new one, and was trained with the hyperparameters the new
run uses, so it describes exactly the training the new run would perform. For each
reused (variant, dataset) pair this writes ``eval_info_<dataset>.json`` and
``stat_<dataset>/<epoch>.json`` into the new variant folder. ``EvalVariants``
checks for that file right before training a pair, so the pair is skipped even by
a run that is already in progress. Results reaching ``MIN_ACCURACY`` are staged
into LEMUR as ``EvalVariants`` would stage them.

Pairs the new run has already started or finished are never touched. A dataset is
not reused at all when most of its old results share one identical accuracy: that
is a model always predicting the majority class, and when it happens to most nets
the old runs of that dataset were broken (data or loader), not the nets.

Usage:
    python -m ab.gpt.brute.loss_opt.ReuseResults --dry_run
    python -m ab.gpt.brute.loss_opt.ReuseResults --nn_train_epochs 5
    python -m ab.gpt.brute.loss_opt.ReuseResults --audit      # list and check reused results
    python -m ab.gpt.brute.loss_opt.ReuseResults --drop_degenerate   # take back results of broken datasets
"""
import argparse
import csv
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from ab.nn.util.ArchUID import arch_uid

from ab.gpt.util.Const import new_nn_file
from ab.gpt.brute.loss_opt.NNVariants import variants_synth_dir
from ab.gpt.brute.loss_opt.EvalVariants import (
    DEFAULT_PRM, METRIC, MIN_ACCURACY, TASK,
    copy_to_lemur_with_retry, extract_accuracy, resolve_prefix,
)

OLD_RUN = Path('out_forschung/nngpt')
# Hyperparameters that must match for an old result to stand in for a new one.
COMPARED_PRM = ('lr', 'batch', 'dropout', 'momentum', 'transform')
# Number of classes, to flag results at chance level (the net did not learn).
NUM_CLASSES = {'celeba-gender': 2, 'cifar-10': 10, 'cifar-100': 100,
               'imagenette': 10, 'mnist': 10, 'svhn': 10}


def old_results_by_arch(old_synth: Path) -> dict:
    """arch_uid -> old variant folder, for old variants with at least one result."""
    index = {}
    for b in sorted(old_synth.glob('B*')):
        if not any(b.glob('eval_info_*.json')) or not (b / new_nn_file).exists():
            continue
        try:
            index.setdefault(arch_uid((b / new_nn_file).read_text(encoding='utf-8')), b)
        except Exception:
            pass  # unparseable code cannot be matched, so its results are not reused
    return index


def degenerate_datasets(old_synth: Path, max_share: float = 0.5) -> dict:
    """dataset -> share of its old results having one identical accuracy, above ``max_share``."""
    accs = defaultdict(list)
    for f in old_synth.glob('B*/eval_info_*.json'):
        info = json.loads(f.read_text(encoding='utf-8'))
        a = extract_accuracy(info.get('eval_results'))
        if a is not None:
            accs[info.get('dataset', f.stem[len('eval_info_'):])].append(round(a, 4))
    shares = {ds: Counter(v).most_common(1)[0][1] / len(v) for ds, v in accs.items() if len(v) >= 5}
    return {ds: share for ds, share in shares.items() if share > max_share}


def drop_reused(new_synth: Path, datasets, dry_run: bool) -> Counter:
    """Remove results this tool reused for ``datasets``, so the evaluation trains those pairs."""
    tally = Counter()
    for info_path in sorted(new_synth.glob('B*/eval_info_*.json')):
        info = json.loads(info_path.read_text(encoding='utf-8'))
        if 'reused_from' not in info or info.get('dataset') not in datasets:
            continue
        tally[f"dropped {info['dataset']}"] += 1
        if not dry_run:
            info_path.unlink()
            shutil.rmtree(info_path.parent / f"stat_{info_path.stem[len('eval_info_'):]}", ignore_errors=True)
    return tally


def _write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=4, default=str), encoding='utf-8')
    os.replace(tmp, path)


def reuse(new_synth: Path, old_synth: Path, old_stat: Path, epochs: int, dry_run: bool) -> Counter:
    expected = {**DEFAULT_PRM, 'epoch': epochs}
    old_by_uid = old_results_by_arch(old_synth)
    print(f"Old variants with results: {len(old_by_uid)} (from {old_synth})")
    broken = degenerate_datasets(old_synth)
    for ds, share in broken.items():
        print(f"  Not reusing {ds}: {share:.0%} of its old results have one identical accuracy")
    tally = Counter()

    for b in sorted(new_synth.glob('B*'), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else -1):
        meta = json.loads((b / 'variant_meta.json').read_text(encoding='utf-8'))
        old = old_by_uid.get(meta.get('arch_uid'))
        if old is None:
            continue
        for info_path in sorted(old.glob('eval_info_*.json')):
            ds_safe = info_path.stem[len('eval_info_'):]
            info = json.loads(info_path.read_text(encoding='utf-8'))
            dataset = info.get('dataset', ds_safe)
            if dataset in broken:
                tally['old results degenerate'] += 1
                continue

            if any(p.exists() for p in (b / info_path.name, b / f'error_{ds_safe}.txt', b / f'stat_{ds_safe}')):
                tally['new run already has this pair'] += 1
                continue
            prm = info.get('eval_args', {}).get('prm', {})
            if prm.get('epoch') != epochs or any(prm.get(k) != expected[k] for k in COMPARED_PRM):
                tally['different hyperparameters'] += 1
                continue
            old_name = info['eval_results'][0]
            stats = old_stat / f'{TASK}_{dataset}_{METRIC}_{old_name}'
            epoch_files = sorted(stats.glob('[0-9]*.json')) if stats.is_dir() else []
            if not epoch_files:
                tally['old per-epoch stats missing'] += 1
                continue

            tally[f'reused {dataset}'] += 1
            if dry_run:
                continue
            # Statistics first, then the result file EvalVariants checks for.
            stat_tmp = b / f'.stat_{ds_safe}.tmp'
            shutil.rmtree(stat_tmp, ignore_errors=True)
            stat_tmp.mkdir()
            for f in epoch_files:
                shutil.copyfile(f, stat_tmp / f.name)
            os.replace(stat_tmp, b / f'stat_{ds_safe}')
            _write_json_atomic(b / info_path.name, {**info, 'reused_from': str(info_path)})

            accuracy = extract_accuracy(info['eval_results'])
            prefix = resolve_prefix(b, None)
            if prefix and accuracy is not None and accuracy >= MIN_ACCURACY:
                copy_to_lemur_with_retry(b, prefix, TASK, dataset, METRIC, b / f'stat_{ds_safe}')
                tally['staged to LEMUR'] += 1
    return tally


def audit_pair(info_path: Path, epochs: int, broken: dict) -> dict:
    """Check one reused result for signs of an incomplete or mixed-up training run."""
    b, ds_safe = info_path.parent, info_path.stem[len('eval_info_'):]
    info = json.loads(info_path.read_text(encoding='utf-8'))
    meta = json.loads((b / 'variant_meta.json').read_text(encoding='utf-8'))
    dataset = info.get('dataset', ds_safe)
    accuracy = extract_accuracy(info.get('eval_results'))
    problems, notes = [], []

    if accuracy is None or math.isnan(accuracy) or not 0.0 <= accuracy <= 1.0:
        problems.append(f'invalid accuracy {accuracy}')

    stat_dir = b / f'stat_{ds_safe}'
    present = sorted(int(f.stem) for f in stat_dir.glob('[0-9]*.json')) if stat_dir.is_dir() else []
    if present != list(range(1, epochs + 1)):
        problems.append(f'epoch files {present}, expected 1..{epochs}')
    records = {e: json.loads((stat_dir / f'{e}.json').read_text(encoding='utf-8')) for e in present}

    multi = [e for e, r in records.items() if len(r) > 1]
    if multi:
        problems.append(f'several runs recorded in epochs {multi}')
    final = records.get(epochs, [])
    if accuracy is not None and not any(abs(r.get('accuracy', -1) - accuracy) < 1e-9 for r in final):
        problems.append(f'final accuracy {accuracy} not in epoch-{epochs} stats')
    if not multi and len(records) == epochs:
        durations = [records[e][0].get('duration', 0) for e in range(1, epochs + 1)]
        if any(d2 <= d1 for d1, d2 in zip(durations, durations[1:])):
            problems.append('durations not increasing across epochs')
    if any(r.get('epoch') != epochs for rs in records.values() for r in rs):
        problems.append(f'records not from a {epochs}-epoch run')

    if dataset in broken:
        problems.append(f"old {dataset} results are degenerate ({broken[dataset]:.0%} identical accuracy)")
    old_error = Path(info.get('reused_from', '')).parent / f'error_{ds_safe}.txt'
    if old_error.exists():
        problems.append(f'old run also has {old_error.name}')

    n_cls = NUM_CLASSES.get(dataset)
    if accuracy is not None and n_cls and accuracy <= 1.2 / n_cls:
        notes.append(f'near chance level (1/{n_cls})')

    return {
        'variant': b.name, 'base_nn': meta['base_nn'], 'loss': meta['loss'], 'optimizer': meta['optimizer'],
        'dataset': dataset, 'accuracy': accuracy,
        'epoch_accuracies': ' '.join(f"{records[e][0]['accuracy']:.4f}" for e in sorted(records) if records[e]),
        'status': 'PROBLEM' if problems else 'ok',
        'details': '; '.join(problems + notes), 'reused_from': info.get('reused_from', ''),
    }


def audit(new_synth: Path, epochs: int, out_csv: Path, broken: dict) -> list[dict]:
    rows = [audit_pair(p, epochs, broken) for p in sorted(new_synth.glob('B*/eval_info_*.json'))
            if 'reused_from' in json.loads(p.read_text(encoding='utf-8'))]
    rows.sort(key=lambda r: (r['status'] != 'PROBLEM', r['dataset'], r['base_nn'], r['loss'], r['optimizer']))
    if rows:
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Reuse earlier results for identical loss/optimizer variants.")
    parser.add_argument('--synth_dir', default=str(variants_synth_dir(0)),
                        help="New variants (default: out_loss_opt/A0/synth_nn)")
    parser.add_argument('--old_synth_dir', default=str(OLD_RUN / 'llm/epoch/A0/synth_nn'),
                        help="Old variants with eval_info_<dataset>.json results")
    parser.add_argument('--old_stat_dir', default=str(OLD_RUN / 'new_lemur/train'),
                        help="Old per-epoch statistics, one <task>_<dataset>_<metric>_<name> folder per result")
    parser.add_argument('--nn_train_epochs', type=int, default=5,
                        help="Epochs of the new run; only old results with the same count are reused (default: 5)")
    parser.add_argument('--dry_run', action='store_true', help="Report what would be reused without writing")
    parser.add_argument('--audit', action='store_true',
                        help="Check every reused result and write <synth_dir>/../reuse_audit.csv")
    parser.add_argument('--drop_degenerate', action='store_true',
                        help="Remove reused results of datasets whose old results are degenerate")
    args = parser.parse_args()
    broken = degenerate_datasets(Path(args.old_synth_dir))

    if args.drop_degenerate:
        tally = drop_reused(Path(args.synth_dir), broken, args.dry_run)
        print(f"Degenerate datasets: {', '.join(f'{d} ({s:.0%})' for d, s in broken.items()) or 'none'}")
        for k, n in sorted(tally.items()):
            print(f"  {k:<35} {n}")
        return

    if args.audit:
        out_csv = Path(args.synth_dir).parent / 'reuse_audit.csv'
        rows = audit(Path(args.synth_dir), args.nn_train_epochs, out_csv, broken)
        status = Counter(r['status'] for r in rows)
        print(f"Reused results: {len(rows)}  ok: {status['ok']}  with problems: {status['PROBLEM']}")
        for r in rows:
            if r['status'] == 'PROBLEM' or r['details']:
                print(f"  {r['status']:<7} {r['variant']:>5} {r['base_nn']}/{r['loss']}/{r['optimizer']} "
                      f"{r['dataset']} acc={r['accuracy']:.4f}  {r['details']}")
        print(f"\nFull list: {out_csv}")
        return

    tally = reuse(Path(args.synth_dir), Path(args.old_synth_dir), Path(args.old_stat_dir),
                  args.nn_train_epochs, args.dry_run)
    print(f"\n{'Would reuse' if args.dry_run else 'Done'}:")
    for k, n in sorted(tally.items()):
        print(f"  {k:<35} {n}")
    print(f"  {'total reused':<35} {sum(n for k, n in tally.items() if k.startswith('reused '))}")


if __name__ == '__main__':
    main()
