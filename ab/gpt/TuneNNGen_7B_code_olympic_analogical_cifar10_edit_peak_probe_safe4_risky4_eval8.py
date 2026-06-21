import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = Path(os.environ.get('AB_NN_DATASET_ROOT', '/home/kabir/newws/nn-dataset')).resolve()

for path in (str(REPO_ROOT), str(DATASET_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault(
    'AB_GPT_NNGPT_DIR',
    str(REPO_ROOT / 'out' / 'benchmarks' / 'tunenngen_cifar10_adaptive_science' / 'analogical_probe_safe4_risky4_eval8'),
)
os.environ.setdefault('AB_GPT_ENABLE_EDIT_SAFETY_GATE', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import ab.gpt.TuneNNGen as TuneNNGen


def main():
    TuneNNGen.main(
        llm_conf='ds_coder_7b_olympic_cifar10_smoke.json',
        llm_tune_conf='NN_gen_analogical_cifar10_edit_peak.json',
        nn_gen_conf='NN_gen_analogical_cifar10_edit_peak.json',
        nn_gen_conf_id=(
            'improve_classification_only_analogical_cifar10_edit_peak_safe',
            'improve_classification_only_analogical_cifar10_edit_peak_risky',
        ),
        num_train_epochs=1,
        test_nn=4,
        nn_train_epochs=1,
        max_prompts=8,
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
