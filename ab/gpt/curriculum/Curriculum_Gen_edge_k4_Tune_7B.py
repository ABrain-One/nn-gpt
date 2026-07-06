import os

import ab.gpt.edge.EdgeGen_k4 as EdgeGen


def _env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value else default


def main():
    EdgeGen.main(
        llm_conf='ds_coder_7b_olympic.json',
        llm_tune_conf='Curriculum_edge_k4_train.json',
        nn_gen_conf='Curriculum_edge_k4.json',
        nn_gen_conf_id='curriculum_edge_k4',
        nn_name_prefix='edge',
        # Environment overrides allow the same entry point to run on 24GB
        # nodes (EDGE_SFT_MAX_LENGTH=4096) and 80GB nodes (16384) without
        # code changes — set them in the K8s job manifest.
        test_nn=_env_int('EDGE_TEST_NN', 2),
        num_cycles=_env_int('EDGE_NUM_CYCLES', 10),
        skip_epoches=_env_int('EDGE_SKIP_EPOCHES', 0),
        context_length=_env_int('EDGE_CONTEXT_LENGTH', 16384),
        sft_max_length=_env_int('EDGE_SFT_MAX_LENGTH', 4096),
        max_new_tokens=_env_int('EDGE_MAX_NEW_TOKENS', 16384),
    )


if __name__ == "__main__":
    main()
