# Add path fix at the VERY TOP
import sys
import os

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, repo_root)
os.chdir(repo_root)

import ab.gpt.TuneNNGen_prun as TuneNNGenPrun

def main():
   
    TuneNNGenPrun.main(
        llm_conf='ds_coder_7b_olympic_prun.json',
        llm_tune_conf='NN_gen_train_prun.json',
        nn_gen_conf='NN_gen_test_prun.json',
        nn_gen_conf_id='efficiency_calculation_test',
        test_nn=10,
        skip_epoches=10,
        nn_name_prefix='pruned',
        use_agents=False,
        use_predictor=False,
        num_train_epochs=2,
    )

if __name__ == "__main__":
    main()