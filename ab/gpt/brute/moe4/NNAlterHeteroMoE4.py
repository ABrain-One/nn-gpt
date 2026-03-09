"""
CLI entry point for 4-Expert Heterogeneous MoE generation.

Usage
-----
Generate with defaults (up to 500 models)::

    python -m ab.gpt.brute.moe4.NNAlterHeteroMoE4

Limit to 50 models::

    python -m ab.gpt.brute.moe4.NNAlterHeteroMoE4 -n 50

Custom output directory::

    python -m ab.gpt.brute.moe4.NNAlterHeteroMoE4 -o my_output_dir

Evaluate generated models (separate step, uses existing pipeline)::

    python -m ab.gpt.NNEval --custom_synth_dir out/nngpt/hetero_moe4/synth_nn \\
           --nn_name_prefix MoE4 --nn_alter_epochs 1 --only_epoch 0

Analyze results (reuses existing AnalyzeResults from moe/)::

    python -m ab.gpt.brute.moe.AnalyzeResults -d out/nngpt/hetero_moe4/synth_nn
"""

import argparse
from ab.gpt.brute.moe4.AlterHeteroMoE4 import alter, _DEFAULT_OUT_DIR


def main():
    parser = argparse.ArgumentParser(
        description='Generate 4-expert heterogeneous MoE models from cross-family UUID-variant quartets.')
    parser.add_argument('-n', '--max_variants', type=int, default=500,
                        help='Maximum number of MoE variants to generate (default: 500)')
    parser.add_argument('-o', '--out_dir', type=str, default=None,
                        help=f'Output directory (default: {_DEFAULT_OUT_DIR})')
    args = parser.parse_args()
    alter(max_variants=args.max_variants, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
