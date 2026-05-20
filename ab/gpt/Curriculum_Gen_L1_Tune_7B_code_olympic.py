import ab.gpt.CurriculumGen_L1 as CurriculumGen_L1


def main():

    CurriculumGen_L1.main(
        llm_conf='ds_coder_7b_olympic.json',
        llm_tune_conf='Curriculum_L1_high_k2_train.json',
        nn_gen_conf='Curriculum_L1_high_k2.json',
        nn_gen_conf_id='curriculum_L1_high_k2',
        test_nn=10,
        skip_epoches=1,
        nn_name_prefix='l1',
        use_agents=False
    )


if __name__ == "__main__":   # ← CORRECT, outside main()
    main()