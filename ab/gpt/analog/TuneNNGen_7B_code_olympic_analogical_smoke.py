import sys

import ab.gpt.analog.TuneNNGenAnalog as TuneNNGen


def main():
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(
            "Run the OlympicCoder analogical smoke experiment. "
            "This wrapper delegates to ab.gpt.analog.TuneNNGenAnalog "
            "with analog prompt configs; run without arguments to launch it."
        )
        return

    TuneNNGen.main(
        llm_conf='ds_coder_7b_olympic_analogical_smoke.json',
        llm_tune_conf='analog/NN_gen_analogical.json',
        nn_gen_conf='analog/NN_gen_analogical.json',
        nn_gen_conf_id='improve_classification_only_analogical',
        num_train_epochs=1,
        test_nn=2,
        nn_train_epochs=1,
        max_prompts=16,
        max_new_tokens=4096,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        logging_steps=5,
        temperature=0.6,
        top_k=50,
        top_p=0.9,
        prompt_batch=1,
        save_llm_output=True,
    )


if __name__ == '__main__':
    main()
