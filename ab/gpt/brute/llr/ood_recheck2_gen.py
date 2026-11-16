"""
OOD matched re-baseline, batch 2: the remaining 14 of the 20 evaluable OOD scale-up
targets (the first 6 were covered by ood_epoch_recheck_gen.py's top_gain/null_control
set). Closes the "only 6 of 20 have a matched control" gap the paper's own limitations
section names.

Reuses ood_epoch_recheck_gen.generate() verbatim (same 'generated' vs fresh-'uniform'
design, same Route C seed tagging) with a different target list and, by default, a
1-epoch budget — matching the OOD scale-up's own protocol, since the question here is
whether the scale-up's per-target deltas hold up against a fair baseline AT THE SAME
BUDGET it used, not whether they later decay (batch 1 already proved that pattern on
6/6 targets; repeating the full epoch-1/3/10 sweep on all 14 more would cost roughly
10x more GPU time for a confirmatory result we already have in principle).

Usage:
    python -m ab.gpt.brute.llr.ood_recheck2_gen --epoch-idx 990 --seeds 3
    python -m ab.gpt.brute.llr.ood_recheck2_gen --epoch-idx 990 --seeds 3 --targets 10,11,12,13,14,16,18
Then (1 epoch, matching the scale-up's own budget):
    python -m ab.gpt.act.eval.Eval --only_epoch 990 --nn_train_epochs 1 \\
        --dataset cifar-100 --task img-classification --metric acc \\
        --nn_name_prefix llroodep2 --save_to_db
"""

import argparse
from pathlib import Path

from ab.gpt.brute.llr import ood_epoch_recheck_gen as base

# (B_idx in A950/mem_c5, target nn name, family, generator strategy, transform,
# 1-epoch baseline_acc, role) — the 14 evaluable ood_results_mem_c5.csv targets NOT
# used in ood_epoch_recheck_gen.py's OOD_RECHECK_TARGETS. role='remaining' throughout
# (no top_gain/null_control split was made for this batch).
REMAINING_TARGETS = [
    (0,  'rl-back-init-4ab8f21cb3e0d1abe9b42b844cc339bf', 'rl-back-init', 'llr_2grp_01x',
     'bf-v1-RandomAutocontrast_RandomAdjustSharpnes_256', 0.7893, 'remaining'),
    (10, 'rag-daaf39952431e04f58e5a9c4060438dc',          'rag',          'llr_4grp_lin',
     'norm_32', 0.3387, 'remaining'),
    (11, 'unq-11f06b5cb50d83128cfc3b3eb548c3cd',          'unq',          'llr3_3grp_step_head',
     'bf-v1-RandomPosterize_Pad', 0.3259, 'remaining'),
    (12, 'alt-nn3-6ac96a6751b0e44a94babfd580094d6b',      'alt-nn3',      'llr_2grp_01x',
     'echo_32', 0.3175, 'remaining'),
    (13, 'unq-11f06b5cb50d83128cfc3b3eb548c3cd',          'unq',          'llr2_2grp_inv_01x',
     'bf-v1-RandomRotation_RandomAutocontrast_3', 0.3093, 'remaining'),
    (14, 'unq-d634cfb6ac61c77cf4839664087b4612',          'unq',          'llr3_2grp_05x',
     'norm_32_flip', 0.3012, 'remaining'),
    (15, 'ga-mut-1662',                                    'ga-mut',       'llr_3grp_lin',
     'bf-v1-RandomHorizontalFlip_RandomPosterize_4', 0.2983, 'remaining'),
    (16, 'rag-4703fde7c8e148743fb1a9e5988bafe0',          'rag',          'llr3_6grp_cos_inv',
     'bf-v1-CenterCrop_Pad_RandomHorizontalFlip', 0.2910, 'remaining'),
    (18, 'rag-997efe7b660b3e411e38abd18b8ab5cc',          'rag',          'llr3_4grp_geo_splits',
     'bf-v1-RandomGrayscale_CenterCrop_2', 0.2849, 'remaining'),
    (19, 'ga-mut-1749',                                    'ga-mut',       'llr_2grp_75pct',
     'bf-v1-RandomAdjustSharpnes', 0.2773, 'remaining'),
    (20, 'ga-mut-1240',                                    'ga-mut',       'llr3_2grp_inv_05x',
     'bf-v1-RandomHorizontalFlip_RandomAdjustSharpnes_RandomPerspective', 0.2722, 'remaining'),
    (21, 'alt-nn1-72f9148d9e72b96c12643c64d46a52b2',      'alt-nn1',      'llr2_cyclic_01x',
     'bf-v1-RandomCrop_RandomAdjustSharpnes_RandomAutocontrast', 0.2708, 'remaining'),
    (23, 'ga-mut-2318',                                    'ga-mut',       'llr3_3grp_step_head',
     'echo_64_flip', 0.2595, 'remaining'),
    (24, 'ga-mut-1219',                                    'ga-mut',       'llr_2grp_01x',
     'bf-v1-RandomAdjustSharpnes_RandomRotation_3', 0.2582, 'remaining'),
]  # source: out/nngpt/ood_results_mem_c5.csv, all rows with status != 'eval-failed'
   # minus the 6 B-indices already in ood_epoch_recheck_gen.OOD_RECHECK_TARGETS.


def main():
    p = argparse.ArgumentParser(description="Generate the OOD matched re-baseline, batch 2 (remaining 14 targets).")
    p.add_argument('--epoch-idx', type=int, default=990)
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--targets', type=str, default='', help="Comma-separated B-indices subset, e.g. --targets 10,11")
    p.add_argument('--prefix', type=str, default='llroodep2')
    p.add_argument('--db', type=str, default=base.DB_PATH_DEFAULT)
    p.add_argument('--no-clean', dest='clean', action='store_false')
    args = p.parse_args()
    wanted_b = [int(x) for x in args.targets.split(',') if x.strip()] if args.targets else []

    base.OOD_RECHECK_TARGETS = REMAINING_TARGETS  # swap the target list; generate() is otherwise identical
    project_root = Path(__file__).resolve().parents[4]
    base.generate(args.epoch_idx, args.seeds, wanted_b, args.prefix, args.clean, args.db, project_root)


if __name__ == '__main__':
    main()
