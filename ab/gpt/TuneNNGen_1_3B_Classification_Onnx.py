import sys
import ab.gpt.TuneNNGen as TuneNNGen


def main(dry_run=False):
    TuneNNGen.main(
        llm_conf='ds_coder_1.3b_instruct.json',
        llm_tune_conf='NN_dataset_compare_code_based.json',
        nn_gen_conf='NN_dataset_compare_code_based.json',
        nn_gen_conf_id='dataset_comparison',
        max_new_tokens=150,
        prompt_batch=1,         # ONNX batch-padding causes empty outputs for shorter prompts
        num_train_epochs=1,     # 1.3B degrades with > 1 inner epoch per outer cycle
        max_prompts=3 if dry_run else 1024,
        onnx_run=True,
        classification_mode=True,
        test_nn=30
    )


if __name__ == '__main__':
    if '--dry-run' in sys.argv:
        main(dry_run=True)
    else:
        main()
