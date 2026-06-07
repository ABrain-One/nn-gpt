import ab.gpt.CurriculumGen_L3_k4 as CurriculumGen_L3_k4


def main():

    CurriculumGen_L3_k4.main(
        llm_conf='ds_coder_7b_olympic_ft.json',
        llm_tune_conf='Curriculum_L3_very_low_near_k4_train.json',
        nn_gen_conf='Curriculum_L3_very_low_near_k4.json',
        nn_gen_conf_id='curriculum_L3_very_low_near_k4',
        test_nn=10,
        skip_epoches=1,
        nn_name_prefix='l3_k4',
        use_agents=False
    )


if __name__ == "__main__":   # ← CORRECT, outside main()
    main()