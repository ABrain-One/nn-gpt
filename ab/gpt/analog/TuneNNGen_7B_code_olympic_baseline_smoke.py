import ab.gpt.analog.TuneNNGenAnalog as TuneNNGen


def main():
    TuneNNGen.main(
        llm_conf='ds_coder_7b_olympic.json',
        llm_tune_conf='NN_gen.json',
        nn_gen_conf='NN_gen.json',
        nn_gen_conf_id='improve_classification_only',
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
