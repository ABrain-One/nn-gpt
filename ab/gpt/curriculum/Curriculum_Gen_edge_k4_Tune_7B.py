import os

import ab.gpt.edge.EdgeGen_k4 as EdgeGen


def _env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value else default


def main():
    # EDGE_STAGE selects the curriculum stage (prompt difficulty):
    #   k2 = easy   (2 similar references, format rules only)
    #   k3 = medium (3 references, medium band, param budget + backbone whitelist)
    #   k4 = hard   (4 dissimilar references, full edge constraints) — default
    stage = os.environ.get('EDGE_STAGE', 'k4')
    EdgeGen.main(
        llm_conf='ds_coder_7b_olympic.json',
        llm_tune_conf=f'edge/curriculum_{stage}_train.json',
        nn_gen_conf=f'edge/curriculum_{stage}.json',
        nn_gen_conf_id=f'curriculum_edge_{stage}',
        nn_name_prefix=os.environ.get('EDGE_NN_PREFIX') or f'edge-{stage}',
        # Environment overrides allow the same entry point to run on 24GB
        # nodes (EDGE_SFT_MAX_LENGTH=4096) and 80GB nodes (16384) without
        # code changes — set them in the K8s job manifest.
        # Chain curriculum stages: point EDGE_PEFT at the previous stage's
        # saved adapter directory to continue fine-tuning from it.
        peft=os.environ.get('EDGE_PEFT') or None,
        test_nn=_env_int('EDGE_TEST_NN', 2),
        num_cycles=_env_int('EDGE_NUM_CYCLES', 10),
        skip_epoches=_env_int('EDGE_SKIP_EPOCHES', 0),
        context_length=_env_int('EDGE_CONTEXT_LENGTH', 16384),
        sft_max_length=_env_int('EDGE_SFT_MAX_LENGTH', 4096),
        max_new_tokens=_env_int('EDGE_MAX_NEW_TOKENS', 16384),
        # Reference pool filter (advisor-approved): params AND accuracy floor.
        # Set EDGE_REF_MAX_PARAMS=0 to disable.
        ref_max_params=_env_int('EDGE_REF_MAX_PARAMS', 6_000_000),
        ref_min_acc=float(os.environ.get('EDGE_REF_MIN_ACC') or 0.85),
        # Curated reference regime: comma-separated model-name prefixes,
        # e.g. EDGE_REF_PREFIXES="EfficientNet,MobileNetV2,MobileNetV3,RegNet"
        ref_prefixes=tuple(p for p in (os.environ.get('EDGE_REF_PREFIXES') or '').split(',') if p),
    )


if __name__ == "__main__":
    main()
