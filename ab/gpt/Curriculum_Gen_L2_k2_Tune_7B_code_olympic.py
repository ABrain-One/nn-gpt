import ab.gpt.CurriculumGen_L2 as CurriculumGen_L2


def main():

    CurriculumGen_L2.main(
        llm_conf='ds_coder_7b_olympic_ft.json',
        llm_tune_conf='Curriculum_L2_medium_k2_train.json',
        nn_gen_conf='Curriculum_L2_medium_k2.json',
        nn_gen_conf_id='curriculum_L2_medium_k2',
        test_nn=10,
        skip_epoches=1,
        nn_name_prefix='l2',
        use_agents=False
    )


if __name__ == "__main__":   # ← CORRECT, outside main()
    main()