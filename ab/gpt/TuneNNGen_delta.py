"""
Delta-based fine-tuning entry point.

Defaults to deepseek-coder-7b-instruct-v1.5 (paper baseline).
To use a different model, change llm_conf to any config in ab/gpt/conf/llm/.

Usage:
    python -m ab.gpt.TuneNNGen_delta
"""

import ab.gpt.TuneNNGen as TuneNNGen


def main():
    TuneNNGen.main(
        llm_conf='ds_coder_7b_instruct.json',
        llm_tune_conf='NN_gen_delta.json',
        nn_gen_conf='NN_gen_delta.json',
        nn_gen_conf_id='improvement_classification_delta',
        temperature=0.60,
        top_k=50,
        top_p=0.9,
        max_new_tokens=2048,
        test_nn=10,
        nn_name_prefix='delta',
    )


if __name__ == '__main__':
    main()
