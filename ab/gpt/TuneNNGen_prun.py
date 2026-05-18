import ab.gpt.TuneNNGen as TuneNNGen

def main(llm_conf=None, llm_tune_conf=None, nn_gen_conf=None, 
         nn_gen_conf_id=None, test_nn=None, skip_epoches=None, 
         nn_name_prefix=None, use_agents=None, use_predictor=None, 
         num_train_epochs=None):
    
    TuneNNGen.main(
        llm_conf=llm_conf or 'ds_coder_7b_olympic_prun.json',
        llm_tune_conf=llm_tune_conf or 'NN_gen_train_prun.json',
        nn_gen_conf=nn_gen_conf or 'NN_gen_test_prun.json',
        nn_gen_conf_id=nn_gen_conf_id or 'efficiency_calculation_test',
        test_nn=test_nn or 10,
        skip_epoches=skip_epoches or 10,
        nn_name_prefix=nn_name_prefix or 'pruned',
        use_agents=use_agents or False,
        use_predictor=use_predictor or False,
        num_train_epochs=num_train_epochs or 2,
    )

if __name__ == "__main__":
    main()