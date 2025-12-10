"""
Delta-based fine-tuning wrapper for TuneNNGen.

This module provides a simple interface for fine-tuning LLMs to generate
code deltas instead of full neural network code.

Usage:
    python -m ab.gpt.TuneNNGen_delta
"""

import ab.gpt.TuneNNGen as TuneNNGen


def main():
    """
    Main entry point for delta-based fine-tuning.
    
    Configures TuneNNGen to use delta-specific config files:
    - NN_gen_delta.json for training prompts
    - NN_gen_delta.json for generation prompts
    - improvement_classification_delta as the config key
    """
    TuneNNGen.main(
        llm_conf='ds_coder_7b_olympic.json',
        llm_tune_conf='NN_gen_delta.json',
        nn_gen_conf='NN_gen_delta.json',
        nn_gen_conf_id='improvement_classification_delta'
    )


if __name__ == '__main__':
    main()

