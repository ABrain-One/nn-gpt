Example:
```bash
export NNGPT_CARRY_ADAPTER=1 NNGPT_CORPUS_FIX=1 NNGPT_SEED_SFT_SPLIT=1 NNGPT_CYCLE_DEDUP=1 NNGPT_EVAL_PRM_MODE=fixed NNGPT_FIXED_EVAL_TRANSFORM=norm_32 NNGPT_SFT_EVAL_STRATEGY=epoch NNGPT_SEED_SHUFFLE=1002 NNGPT_STUB_FIXERS=1 LLMATIC_ADMIT_GENERATED=1 LLMATIC_FITNESS_EPOCH=1
python3 -m ab.gpt.act.tune.Tune --run_iterative_pipeline --llm_conf mistral_7b_instruct_v03.json --cycles 25 --models_per_cycle 25 --nn_train_epochs 1 --dataset cifar-10 --nn_prefixes ga-,GenFractalNet --nn_gen_conf NN_gen_shortctx_n32.json --max_new_tokens 2048 --num_train_epochs 1 --max_retries 0 --llmatic --llmatic_crossover_every 3 --no-use_vllm
```           
