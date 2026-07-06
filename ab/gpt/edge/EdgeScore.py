"""
Post-run edge scorer for the curriculum edge pipeline.

Walks the per-epoch generation outputs (out/nngpt/llm/epoch/A*/synth_nn/B*),
benchmarks every generated model with EdgeBench, applies novelty flags, and
writes out/edge/edge_tracker.json ranking epochs and models by
efficiency = accuracy / latency_ms.

IMPORTANT: run this AFTER a tune run finishes and BEFORE relaunching, because
Tune_Curriculum.tune() clears the epoch output directory at the start of the
next run. INT8 .tflite files for every convertible model are copied to
out/edge/tflite/ for the on-device benchmarking pipeline.

Usage:
    python -m ab.gpt.edge.EdgeScore
    python -m ab.gpt.edge.EdgeScore --dataset cifar-10 --param-limit 6000000
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ab.gpt.util.Const import epoch_dir, synth_dir, new_nn_file, hp_file
from ab.nn.util.Const import out_dir
from ab.gpt.edge.EdgeBench import benchmark, PARAM_LIMIT_DEFAULT, INPUT_SIZE_DEFAULT

EDGE_OUT_DIR = out_dir / 'edge'
REFERENCE_KEYS = ('nn_1', 'nn_2', 'nn_3', 'nn_4')
SIMILARITY_THRESHOLD_DEFAULT = 0.8       # vs in-prompt references (anti-copying)
SIBLING_SIMILARITY_THRESHOLD = 0.95      # vs other generations (anti-duplicate);
                                         # looser because the prompt mandates a rigid
                                         # code skeleton that inflates baseline similarity
_TOKEN_RE = re.compile(r"[A-Za-z_]\w*|[^\s]")
_SHINGLE_N = 7
_MINHASH_PERM = 128


def code_minhash(code: str):
    """MinHash over 7-token shingles — estimates Jaccard similarity of code."""
    try:
        from datasketch import MinHash
    except ImportError:
        return None
    tokens = _TOKEN_RE.findall(code)
    mh = MinHash(num_perm=_MINHASH_PERM)
    for i in range(max(1, len(tokens) - _SHINGLE_N + 1)):
        mh.update(' '.join(tokens[i:i + _SHINGLE_N]).encode('utf-8'))
    return mh


def max_similarity(mh, others: List[Tuple[str, Any]]) -> Tuple[Optional[float], Optional[str]]:
    """Highest estimated Jaccard similarity of mh against (label, minhash) pairs."""
    if mh is None or not others:
        return None, None
    best_sim, best_label = 0.0, None
    for label, other in others:
        if other is None:
            continue
        sim = mh.jaccard(other)
        if sim > best_sim:
            best_sim, best_label = sim, label
    return round(best_sim, 4), best_label


def _accuracy_from_payload(payload: Any) -> Optional[float]:
    # Trial files ({epoch}.json) hold a list of dicts with 'accuracy';
    # eval_info.json holds a dict with 'accuracy' or nested 'eval_results'.
    if isinstance(payload, list):
        accs = [t.get('accuracy') for t in payload if isinstance(t, dict) and t.get('accuracy') is not None]
        return max(float(a) for a in accs) if accs else None
    if isinstance(payload, dict):
        candidates = [
            payload.get('accuracy'),
            payload.get('eval_results', {}).get('accuracy') if isinstance(payload.get('eval_results'), dict) else None,
            payload.get('eval_results', {}).get('acc') if isinstance(payload.get('eval_results'), dict) else None,
        ]
        epochs = payload.get('epochs') or (payload.get('eval_results', {}) or {}).get('epochs')
        if isinstance(epochs, list) and epochs:
            candidates.append(epochs[-1].get('accuracy', epochs[-1].get('acc')))
        for c in candidates:
            if c is not None:
                try:
                    return float(c)
                except (TypeError, ValueError):
                    continue
    return None


def extract_accuracy(model_dir: Path) -> Optional[float]:
    """Accuracy of a generated model, from NNEval's trial files ({epoch}.json)
    or eval_info.json — whichever this NNEval version wrote."""
    candidates = sorted(model_dir.glob('[0-9]*.json')) + [model_dir / 'eval_info.json']
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        acc = _accuracy_from_payload(payload)
        if acc is not None:
            return acc
    return None


def load_prm(model_dir: Path) -> Optional[Dict[str, Any]]:
    hp_path = model_dir / hp_file
    if hp_path.is_file():
        try:
            return json.loads(hp_path.read_text())
        except Exception:
            return None
    return None


def register_references(checker, model_dir: Path,
                        ref_hashes: Dict[str, Any]) -> None:
    """Feed the in-prompt reference codes to the novelty checker and the
    MinHash reference pool (if recorded in dataframe.df)."""
    df_path = model_dir / 'dataframe.df'
    if not df_path.is_file():
        return
    try:
        row = pd.read_pickle(df_path)
    except Exception:
        return
    for key in REFERENCE_KEYS:
        try:
            ref = row.get(key) if hasattr(row, 'get') else None
        except Exception:
            ref = None
        if isinstance(ref, str) and ref.strip():
            dedup = hash(ref)
            if dedup not in ref_hashes:
                ref_hashes[dedup] = (f'reference:{key}', code_minhash(ref))
            if checker is not None:
                try:
                    checker.add_training_data(ref, source=f'reference:{key}')
                except TypeError:
                    checker.add_training_data(ref)


def collect_model_dirs(epoch_root: Path) -> List[Dict[str, Any]]:
    """All A*/synth_nn/B* dirs that contain a generated model file,
    including models rejected by the pre-eval novelty filter."""
    entries = []
    for a_dir in sorted(epoch_root.glob('A*')):
        models_dir = synth_dir(a_dir)
        if not models_dir.is_dir():
            continue
        epoch_num = a_dir.name.lstrip('A')
        for b_dir in sorted(models_dir.glob('B*')):
            if (b_dir / new_nn_file).is_file():
                entries.append({'epoch': epoch_num, 'model_id': b_dir.name, 'dir': b_dir,
                                'pre_eval_rejected': False})
            elif (b_dir / (new_nn_file + '.rejected')).is_file():
                entries.append({'epoch': epoch_num, 'model_id': b_dir.name, 'dir': b_dir,
                                'pre_eval_rejected': True})
    return entries


def score_run(dataset: str = 'cifar-10',
              input_size: int = INPUT_SIZE_DEFAULT,
              param_limit: int = PARAM_LIMIT_DEFAULT,
              epoch_root: Optional[Path] = None,
              out_root: Optional[Path] = None,
              save_tflite: bool = True,
              similarity_threshold: float = SIMILARITY_THRESHOLD_DEFAULT) -> Dict[str, Any]:
    epoch_root = epoch_root or epoch_dir()
    out_root = out_root or EDGE_OUT_DIR
    out_root.mkdir(parents=True, exist_ok=True)
    tflite_dir = out_root / 'tflite'

    try:
        from ab.gpt.iterative_pipeline.novelty_checker import NoveltyChecker
        checker = NoveltyChecker()
    except Exception as e:
        print(f'[WARN] NoveltyChecker unavailable ({e}) — novelty flags disabled')
        checker = None

    entries = collect_model_dirs(epoch_root)
    print(f'[EDGE SCORE] {len(entries)} generated models found under {epoch_root}')

    per_model: List[Dict[str, Any]] = []
    ref_hashes: Dict[str, Any] = {}          # dedup -> (label, minhash) for in-prompt references
    seen_minhashes: List[Tuple[str, Any]] = []  # (tag, minhash) of previously scored generations
    for entry in entries:
        b_dir: Path = entry['dir']
        tag = f"A{entry['epoch']}_{entry['model_id']}"

        if entry['pre_eval_rejected']:
            reason_file = b_dir / 'rejection_reason.txt'
            reason = reason_file.read_text().strip() if reason_file.is_file() else 'unknown'
            print(f'[EDGE SCORE] {tag}: rejected before eval ({reason})')
            per_model.append({
                'tag': tag, 'epoch': entry['epoch'], 'model_id': entry['model_id'],
                'pre_eval_rejected': True, 'rejection_reason': reason,
                'novel': False, 'eligible': False, 'accuracy': None, 'efficiency': None,
            })
            continue

        code = (b_dir / new_nn_file).read_text(encoding='utf-8', errors='replace')

        register_references(checker, b_dir, ref_hashes)

        structurally_novel = None
        if checker is not None:
            try:
                structurally_novel = checker.is_novel(code, model_id=tag)
                checker.mark_as_seen(code, model_id=tag, source='generated')
            except Exception:
                structurally_novel = None

        mh = code_minhash(code)
        ref_sim, ref_sim_to = max_similarity(mh, list(ref_hashes.values()))
        sib_sim, sib_sim_to = max_similarity(mh, seen_minhashes)
        seen_minhashes.append((tag, mh))
        sim, sim_to = max(
            [(ref_sim, ref_sim_to), (sib_sim, sib_sim_to)],
            key=lambda p: p[0] if p[0] is not None else -1.0,
        )

        # Eligibility gates on MinHash similarity only. The structural checker
        # (layer-sequence hash, persisted across runs) is recorded as advisory:
        # the mandated code skeleton makes structural signatures collide even
        # between meaningfully different models.
        novel = True
        if ref_sim is not None and ref_sim >= similarity_threshold:
            novel = False
        if sib_sim is not None and sib_sim >= SIBLING_SIMILARITY_THRESHOLD:
            novel = False

        accuracy = extract_accuracy(b_dir)
        prm = load_prm(b_dir)
        tflite_path = (tflite_dir / f'{tag}.tflite') if save_tflite else None

        print(f'[EDGE SCORE] {tag}: acc={accuracy} novel={novel} '
              f'max_sim={sim}{f" (vs {sim_to})" if sim_to else ""} — benchmarking...')
        bench = benchmark(
            b_dir / new_nn_file,
            prm=prm,
            accuracy=accuracy,
            dataset=dataset,
            input_size=input_size,
            param_limit=param_limit,
            tflite_path=tflite_path,
        )

        eligible = bool(
            accuracy is not None
            and bench.get('param_gate_ok')
            and bench.get('efficiency') is not None
            and novel is not False
        )
        per_model.append({
            'tag': tag,
            'epoch': entry['epoch'],
            'model_id': entry['model_id'],
            'novel': novel,
            'structurally_novel': structurally_novel,
            'max_similarity': sim,
            'similar_to': sim_to,
            'eligible': eligible,
            **bench,
        })

    per_epoch: Dict[str, Dict[str, Any]] = {}
    for m in per_model:
        ep = per_epoch.setdefault(m['epoch'], {
            'epoch': m['epoch'], 'models': 0, 'evaluated': 0, 'eligible': 0,
            'best_efficiency': None, 'best_model': None,
        })
        ep['models'] += 1
        if m['accuracy'] is not None:
            ep['evaluated'] += 1
        if m['eligible']:
            ep['eligible'] += 1
            if ep['best_efficiency'] is None or m['efficiency'] > ep['best_efficiency']:
                ep['best_efficiency'] = m['efficiency']
                ep['best_model'] = m['tag']

    best = None
    for m in per_model:
        if m['eligible'] and (best is None or m['efficiency'] > best['efficiency']):
            best = m

    tracker = {
        'timestamp': datetime.now().isoformat(),
        'dataset': dataset,
        'input_size': input_size,
        'param_limit': param_limit,
        'similarity_threshold': similarity_threshold,
        'epoch_root': str(epoch_root),
        'per_epoch': sorted(per_epoch.values(), key=lambda e: int(e['epoch'])),
        'per_model': per_model,
        'best_model': best['tag'] if best else None,
        'best_efficiency': best['efficiency'] if best else None,
        'best_epoch': best['epoch'] if best else None,
    }

    tracker_file = out_root / 'edge_tracker.json'
    tracker_file.write_text(json.dumps(tracker, indent=2))

    print('\n[EDGE SCORE] Per-epoch summary (efficiency = accuracy / latency_ms):')
    for ep in tracker['per_epoch']:
        print(f"  epoch A{ep['epoch']}: models={ep['models']} evaluated={ep['evaluated']} "
              f"eligible={ep['eligible']} best_eff={ep['best_efficiency']} best={ep['best_model']}")
    print(f"\n[EDGE SCORE] Best model: {tracker['best_model']} "
          f"(epoch A{tracker['best_epoch']}, efficiency={tracker['best_efficiency']})")
    print(f"[EDGE SCORE] Tracker written to {tracker_file}")
    if save_tflite:
        print(f"[EDGE SCORE] INT8 tflite files in {tflite_dir} — ready for on-device benchmarking")
    return tracker


def main() -> None:
    parser = argparse.ArgumentParser(description='Score generated models by edge efficiency (accuracy / latency).')
    parser.add_argument('--dataset', type=str, default='cifar-10')
    parser.add_argument('--input-size', type=int, default=INPUT_SIZE_DEFAULT)
    parser.add_argument('--param-limit', type=int, default=PARAM_LIMIT_DEFAULT)
    parser.add_argument('--epoch-root', type=str, default=None,
                        help='Root of per-epoch outputs (default: out/nngpt/llm/epoch)')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Output directory (default: out/edge)')
    parser.add_argument('--no-tflite', action='store_true', help='Skip TFLite export/benchmark')
    parser.add_argument('--similarity-threshold', type=float, default=SIMILARITY_THRESHOLD_DEFAULT,
                        help='Flag models whose MinHash Jaccard similarity to any reference or '
                             'earlier generation is at or above this value (default: 0.8)')
    args = parser.parse_args()

    score_run(
        dataset=args.dataset,
        input_size=args.input_size,
        param_limit=args.param_limit,
        epoch_root=Path(args.epoch_root) if args.epoch_root else None,
        out_root=Path(args.out_dir) if args.out_dir else None,
        save_tflite=not args.no_tflite,
        similarity_threshold=args.similarity_threshold,
    )


if __name__ == '__main__':
    main()
