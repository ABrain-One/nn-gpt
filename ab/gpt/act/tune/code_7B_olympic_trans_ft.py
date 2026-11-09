import ab.gpt.act.tune.Tune as TuneNNGen


def main():

    TuneNNGen.main(
        llm_conf='ds_coder_7b_olympic.json',
        llm_tune_conf='Transform_gen.json', # 'Transform_gen.json' and 'Augment.json' for transform and augment fine-tune
        nn_gen_conf='Transform_gen.json',
        test_nn=10,
        trans_mode=True,
        aug_mode=False,
        use_agents = False,
        use_vllm=True
)


if __name__ == "__main__":
    main()