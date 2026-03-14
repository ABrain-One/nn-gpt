"""
TuneNNGen Pipeline Wrapper
==========================
Simple wrapper to trigger the iterative fine-tuning pipeline
via TuneNNGen_v2.main() with run_iterative_pipeline=True.
"""

import ab.gpt.TuneNNGen as TuneNNGen


def main():
    TuneNNGen.main(
        llm_conf='ds_coder_7b_instruct.json',
        run_iterative_pipeline=True,
        cycles=22,
        resume_from_cycle=None,
        nn_name_prefix='unq'
        # models_per_cycle=20,
    )


if __name__ == '__main__':
    main()
