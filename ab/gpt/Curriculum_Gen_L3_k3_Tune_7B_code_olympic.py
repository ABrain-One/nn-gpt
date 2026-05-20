import ab.gpt.CurriculumGen_L3_k3 as CurriculumGen_L3_k3


def main():

    CurriculumGen_L3_k3.main(
        llm_conf='ds_coder_7b_olympic_ft.json',
        llm_tune_conf='Curriculum_L3_very_low_near_k3_train.json',
        nn_gen_conf='Curriculum_L3_very_low_near_k3.json',
        nn_gen_conf_id='curriculum_L3_very_low_near_k3',
        test_nn=10,
        skip_epoches=1,
        nn_name_prefix='l3_k3',
        use_agents=False
    )


if __name__ == "__main__":   # ← CORRECT, outside main()
    main()