import ab.gpt.TuneNNGen as TuneNNGen


def main():
    TuneNNGen.main(llm_conf='ds_coder_7b_instruct.json',
                   max_prompts=128,
                   test_nn=2,
                   skip_epoches=0,
                   nn_name_prefix='check_results',
                   unsloth_opt=False
                   )


if __name__ == '__main__':
    main()
