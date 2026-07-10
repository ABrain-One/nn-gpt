import os
from ab.gpt.util.Const import ab_root_path

REPO_ROOT = ab_root_path

os.environ.setdefault(
    'AB_GPT_NNGPT_DIR',
    str(REPO_ROOT / 'out' / 'benchmarks' / 'tunenngen_cifar10_focus' / 'baseline_edit_run_16_fair_3epoch'),
)
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import ab.gpt.analog.TuneNNGenAnalog as TuneNNGen


def main():
    TuneNNGen.main(
        llm_conf='ds_coder_7b_olympic.json',
        llm_tune_conf='analog/NN_gen_cifar10_edit.json',
        nn_gen_conf='analog/NN_gen_cifar10_edit.json',
        nn_gen_conf_id='improve_classification_only_cifar10_edit',
        num_train_epochs=3,
        test_nn=16,
        nn_train_epochs=1,
        max_prompts=16,
        max_new_tokens=1536,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=5e-6,
        logging_steps=5,
        temperature=0.6,
        top_k=50,
        top_p=0.9,
        prompt_batch=1,
        save_llm_output=True,
        nn_name_prefix='edit',
        eval_save_to_db=False,
    )


if __name__ == '__main__':
    main()
