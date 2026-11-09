import ab.gpt.act.tune.Tune as TuneNNGen


def main():

    TuneNNGen.main(
        llm_conf='ds_coder_7b_olympic.json',
        llm_tune_conf='Augment.json', # 'Transform_gen.json' and 'Augment.json' for transform and augment fine-tune
        nn_gen_conf='Augment.json',
        test_nn=5,
        trans_mode=False,
        aug_mode=True,
        use_agents = False,
        use_vllm=True
)


if __name__ == "__main__":
    main()