"""
Delta-based neural network generation wrapper.

This module provides a simple interface for generating improved neural networks
using delta-based approach, where the LLM generates code deltas (unified diffs)
instead of full code.

Usage:
    python -m ab.gpt.NNAlter_7B_delta --epochs 8
"""

import argparse

from ab.gpt.util.AlterNN import alter_delta


def main():
    """
    Main entry point for delta-based neural network generation.
    
    Uses alter_delta() function which:
    1. Loads delta-enabled config (NN_gen_delta.json)
    2. Generates code deltas from LLM
    3. Applies deltas to baseline code
    4. Saves improved code
    """
    parser = argparse.ArgumentParser(
        description="Generate improved neural networks using delta-based approach."
    )
    parser.add_argument(
        '-e', '--epochs', 
        type=int, 
        default=8, 
        help="Maximum number of generation epochs."
    )
    parser.add_argument(
        '-n', '--num-supporting-models', 
        type=int, 
        default=1, 
        help="Number of supporting models to fetch from database for more ideas."
    )
    args = parser.parse_args()
    
    alter_delta(
        args.epochs, 
        'NN_gen_delta.json', 
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', 
        n=args.num_supporting_models, 
        temperature=0.8, 
        top_k=100
    )


if __name__ == "__main__":
    main()

