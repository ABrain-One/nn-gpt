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
    
    Parameters optimized for delta generation:
    - temperature=0.60: Higher than paper (0.20) for diversity in short delta outputs
                        Paper's 0.20 was for full code (longer, more deterministic)
                        Deltas are short → need higher temp to avoid mode collapse
    - top_k=50: Paper setting for focused sampling
    - top_p=0.9: Paper setting for nucleus sampling
    - max_new_tokens=2048: Paper uses 2048 (sufficient for delta, prevents rambling)
    - test_nn=10: Generate 10 models per epoch for evaluation
    """
    TuneNNGen.main(
        llm_conf='ds_coder_7b_instruct.json',  # DeepSeek-Coder-7B-Instruct-v1.5 (paper's model)
        llm_tune_conf='NN_gen_delta.json',
        nn_gen_conf='NN_gen_delta.json',
        nn_gen_conf_id='improvement_classification_delta',
        # Parameters optimized for delta generation:
        temperature=0.60,         # Higher for diversity (deltas are short)
        top_k=50,                 # Paper: top-k 50
        top_p=0.9,                # Paper: nucleus p 0.9
        max_new_tokens=2048,      # Paper: max new tokens 2,048
        test_nn=10,               # Models to generate per epoch for evaluation
    )


if __name__ == '__main__':
    main()

