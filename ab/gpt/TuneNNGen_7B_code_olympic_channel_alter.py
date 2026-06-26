import ab.gpt.TuneNNGen as TuneNNGen


def main():

    TuneNNGen.main(
        llm_conf='ds_coder_7b_olympic.json',
        llm_tune_conf='NN_gen_train_channel.json',
        nn_gen_conf='NN_gen_test_channel.json',
        nn_gen_conf_id='optimal_channel_configuration_test',
        test_nn=10,
        skip_epoches=1,
        nn_name_prefix='chn'
    )


if __name__ == "__main__":   # ← CORRECT, outside main()
    main()
