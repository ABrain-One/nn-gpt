"""
Build the llr metadata CSV that maps each brute-force model's DB name
(prefix-uuid4(code)) to its architecture/strategy metadata.

This is the bridge consumed by:
  - ab.gpt.brute.llr.analyze_llr_vs_vanilla  (vanilla-vs-llr analysis)
  - ab.gpt.util.prompt.NNGenPrompt          (vanilla->llr training pairs)
both of which read out/nngpt/llr_evaluated_with_meta.csv keyed by nn name.

Approach (matches the single-dir + eval-time dataset/transform fan-out flow):
  1. Run all three generators into a temp dir (one dir per (arch, strategy)).
  2. For each dir, compute nn_name = f"{prefix}-{uuid4(code)}" exactly as LEMUR
     stores it, and read model_meta.txt for the architecture/strategy fields.
  3. Emit one row per generated combo. This is a superset of what gets evaluated
     (combos that fail evaluation simply never appear in the DB and are never
     looked up), so no matching against evaluated artifacts is needed.

The dataset/transform of a model is NOT taken from here — it is resolved per row
from the DB stat table at analysis/pairing time, since each model is fanned out
across multiple (dataset, transform) settings.

Run inside a pod (ab.nn must be installed):
    python -m ab.gpt.brute.llr.build_metadata_csv
"""

import csv
import re
import shutil
import collections
import tempfile
from pathlib import Path

from ab.nn.util.Util import uuid4

from ab.gpt.brute.llr.layerwise_lr  import generate_models as gen_llr
from ab.gpt.brute.llr.layerwise_lr2 import generate_models as gen_llr2
from ab.gpt.brute.llr.layerwise_lr3 import generate_models as gen_llr3

PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUT_CSV      = PROJECT_ROOT / 'out' / 'nngpt' / 'llr_evaluated_with_meta.csv'

FIELDS = ['nn', 'architecture', 'strategy', 'strategy_type', 'n_groups',
          'multipliers', 'split_ratios', 'dataset', 'task', 'metric',
          'transform', 'description', 'meta_source']


def parse_meta(path: Path) -> dict:
    meta = {}
    for line in path.read_text().splitlines():
        if ':' in line:
            k, _, v = line.partition(':')
            meta[k.strip()] = v.strip()
    return meta


def main():
    tmp = Path(tempfile.mkdtemp(prefix='llr_meta_'))
    try:
        print(f"Generating models into: {tmp}")
        gen_llr( str(tmp), prefix='llr')
        gen_llr2(str(tmp), prefix='llr2')
        gen_llr3(str(tmp), prefix='llr3')

        rows = []
        for batch_dir in sorted(tmp.iterdir()):
            if not batch_dir.is_dir():
                continue
            nn_file   = batch_dir / 'new_nn.py'
            meta_file = batch_dir / 'model_meta.txt'
            if not nn_file.exists() or not meta_file.exists():
                continue

            code   = nn_file.read_text(encoding='utf-8')
            meta   = parse_meta(meta_file)
            prefix = re.match(r'^(llr[23]?)', batch_dir.name).group(1)
            nn_name = f"{prefix}-{uuid4(code)}"
            rows.append({
                'nn':            nn_name,
                'architecture':  meta.get('architecture', ''),
                'strategy':      meta.get('strategy', ''),
                'strategy_type': meta.get('strategy_type', ''),
                'n_groups':      meta.get('n_groups', ''),
                'multipliers':   meta.get('multipliers', ''),
                'split_ratios':  meta.get('split_ratios', ''),
                # nominal only — actual per-row dataset/transform come from the DB
                'dataset':       meta.get('dataset', ''),
                'task':          meta.get('task', ''),
                'metric':        meta.get('metric', ''),
                'transform':     meta.get('transform', ''),
                'description':   meta.get('description', ''),
                'meta_source':   'brute_force',
            })

        # Deduplicate on nn (identical code across batches would collide; keep first)
        seen = set()
        unique = []
        for r in rows:
            if r['nn'] in seen:
                continue
            seen.add(r['nn'])
            unique.append(r)
        unique.sort(key=lambda r: r['nn'])

        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(unique)

        print(f"\nWrote {len(unique)} metadata rows -> {OUT_CSV}")
        print(f"By prefix   : {dict(collections.Counter(r['nn'].split('-')[0] for r in unique))}")
        print(f"By type     : {dict(collections.Counter(r['strategy_type'] for r in unique))}")
        print(f"Architectures: {len(set(r['architecture'] for r in unique))}")
        print(f"Uniform ctrl : {sum(1 for r in unique if 'uniform' in r['strategy'].lower())}")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
