import sys
import ab.gpt.TuneNNGen as TuneNNGen


def main(dry_run=False):
    TuneNNGen.main(
        llm_conf='ds_coder_7b_olympic.json',
        llm_tune_conf='NN_dataset_compare_code_based.json',
        nn_gen_conf='NN_dataset_compare_code_based.json',
        nn_gen_conf_id='dataset_comparison',
        max_new_tokens=2048,  # OlympicCoder emits <think>...</think> before answering
        max_prompts=3 if dry_run else 1024,
        onnx_run=False,
        classification_mode=True,
        test_nn=30
    )


if __name__ == '__main__':
    if '--dry-run' in sys.argv:
        main(dry_run=True)
    else:
        main()
