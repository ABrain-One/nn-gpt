import ab.gpt.act.tune.Tune as TuneNNGen


def main():

    TuneNNGen.main(
        llm_conf='qwen3.5_9b.json',
        llm_tune_conf='NN_gen_train_nn_improve.json',
        nn_gen_conf='NN_gen_test_nn_improve.json',
        nn_gen_conf_id='optimal_channel_configuration_test',
        test_nn=30,
        skip_epoches=-1,  # generate at epoch 0 => pre-fine-tuning baseline
        nn_name_prefix='llm',
        context_length=8 * 1024,
        sft_max_length=4 * 1024,  # packing block size; keep <= context_length
        max_new_tokens=8 * 1024,
        cycles=22,
        num_train_epochs=1,
        max_prompts=128,
        learning_rate=5e-5,
        lora_alpha=32,
        suppress_thinking=True,
        use_unsloth=True,
        use_vllm=True
)


if __name__ == "__main__":
    main()