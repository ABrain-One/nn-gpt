"""
Delta-based fine-tuning entry point (standalone mode).

Implements delta code generation using paper-aligned hyperparameters:
- Learning rate: 1e-5, Temperature: 0.35, Top-k: 50
- LoRA with lm_head included, 3 train epochs, gradient_accumulation=8
- DeepSeek-Coder-7B-Instruct (default) or Qwen2.5-Coder-7B-Instruct

Usage (standalone):
    python -m ab.gpt.TuneNNGen_delta
    python -m ab.gpt.TuneNNGen_delta --llm_conf qwen2.5_coder_7b_instruct.json
"""

import argparse

import ab.gpt.TuneNNGen as TuneNNGen

# Delta-only defaults (paper-aligned); kept out of TuneNNGen.py for upstream parity.
DELTA_LEARNING_RATE = 1e-5
DELTA_WEIGHT_DECAY = 0.01
DELTA_WARMUP_STEPS = 20
DELTA_NUM_TRAIN_EPOCHS = 3
DELTA_GRADIENT_ACCUMULATION_STEPS = 8
DELTA_TARGET_MODULES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj', 'lm_head')
DELTA_MAX_NEW_TOKENS = 1024
DELTA_TEMPERATURE = 0.35
DELTA_TOP_K = 50
DELTA_TOP_P = 0.9
DELTA_LLM_TUNE_CONF = 'NN_gen_delta.json'
DELTA_NN_GEN_CONF = 'NN_gen_delta.json'
DELTA_NN_GEN_CONF_ID = 'improvement_classification_delta'
DELTA_NN_NAME_PREFIX = 'delta'


def get_delta_pipeline_defaults():
    """Hyperparameters for delta fine-tuning / generation (calls TuneNNGen.main with these)."""
    return {
        'learning_rate': DELTA_LEARNING_RATE,
        'weight_decay': DELTA_WEIGHT_DECAY,
        'warmup_steps': DELTA_WARMUP_STEPS,
        'num_train_epochs': DELTA_NUM_TRAIN_EPOCHS,
        'gradient_accumulation_steps': DELTA_GRADIENT_ACCUMULATION_STEPS,
        'target_modules': ','.join(DELTA_TARGET_MODULES),
        'max_new_tokens': DELTA_MAX_NEW_TOKENS,
        'temperature': DELTA_TEMPERATURE,
        'top_k': DELTA_TOP_K,
        'top_p': DELTA_TOP_P,
        'llm_tune_conf': DELTA_LLM_TUNE_CONF,
        'nn_gen_conf': DELTA_NN_GEN_CONF,
        'nn_gen_conf_id': DELTA_NN_GEN_CONF_ID,
        'nn_name_prefix': DELTA_NN_NAME_PREFIX,
    }


def main(
    llm_conf: str = 'ds_coder_7b_instruct.json',
    test_nn: int = 50,
    skip_epoches: int = -1,
    **kwargs
):
    """
    Delta fine-tuning with paper-aligned hyperparameters.
    
    Args:
        llm_conf: LLM config file (default: ds_coder_7b_instruct.json)
        test_nn: Models to generate per epoch (default: 50, paper uses ~50-208)
        skip_epoches: Resume from this epoch (-1 = fresh start, N = skip A0..A{N-1})
        **kwargs: Override any default parameter
    """
    defaults = get_delta_pipeline_defaults()

    TuneNNGen.main(
        llm_conf=llm_conf,
        llm_tune_conf=kwargs.get('llm_tune_conf', defaults['llm_tune_conf']),
        nn_gen_conf=kwargs.get('nn_gen_conf', defaults['nn_gen_conf']),
        nn_gen_conf_id=kwargs.get('nn_gen_conf_id', defaults['nn_gen_conf_id']),
        temperature=kwargs.get('temperature', defaults['temperature']),
        top_k=kwargs.get('top_k', defaults['top_k']),
        top_p=kwargs.get('top_p', defaults['top_p']),
        max_new_tokens=kwargs.get('max_new_tokens', defaults['max_new_tokens']),
        learning_rate=kwargs.get('learning_rate', defaults['learning_rate']),
        num_train_epochs=kwargs.get('num_train_epochs', defaults['num_train_epochs']),
        gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps', defaults['gradient_accumulation_steps']),
        weight_decay=kwargs.get('weight_decay', defaults['weight_decay']),
        warmup_steps=kwargs.get('warmup_steps', defaults['warmup_steps']),
        target_modules=kwargs.get('target_modules', defaults['target_modules']).split(',') if isinstance(kwargs.get('target_modules', defaults['target_modules']), str) else kwargs.get('target_modules', defaults['target_modules']),
        test_nn=test_nn,
        skip_epoches=skip_epoches,
        nn_name_prefix=kwargs.get('nn_name_prefix', defaults['nn_name_prefix']),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Delta-based fine-tuning with paper-aligned hyperparameters (standalone mode)'
    )
    
    parser.add_argument('--llm_conf', type=str, default='ds_coder_7b_instruct.json',
                        help='LLM config file (default: ds_coder_7b_instruct.json)')
    parser.add_argument('--test_nn', type=int, default=50,
                        help='Models to generate per epoch (default: 50)')
    parser.add_argument('--skip_epoches', '-k', type=int, default=-1,
                        help='Resume from this epoch (-1 = fresh start, 17 = skip A0-A16 and load A16 adapter)')
    
    args = parser.parse_args()
    
    main(
        llm_conf=args.llm_conf,
        test_nn=args.test_nn,
        skip_epoches=args.skip_epoches,
    )
