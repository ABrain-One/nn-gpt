# Closed-loop self-improvement of an LLM architecture generator (Qwen3.5-9B)

Code, generated networks, statistics and complete logs of the experiments in the paper
*Closed-Loop Self-Improvement of an LLM Architecture Generator: Reward Scaling, Clean Training, and Convergence to One Design*.

The generator, `Qwen/Qwen3.5-9B`, writes PyTorch `Net` classes for CIFAR-10. Each cycle consists of a PPO phase
(learned value baseline) and a GRPO phase (group-mean baseline), and every phase's LoRA adapter is merged into the
weights. The run has 4 cycles, i.e. 8 phases of 16 steps × 2 prompts × 8 samples. Every generated network is executed
without repair (strict check) and trained twice for 10 epochs; its reward is `0.5 + 3 (acc - 0.6)`. After every phase,
the merged model is evaluated on 100 frozen prompts × 3 sampling runs.

## Contents

| Path | What it is |
|---|---|
| `qwen_cycle.py` | One training phase (PPO or GRPO): generation with the NNGPT vLLM engine, strict reward in an isolated worker, token-level update. Version v5, md5 `8e118c2c10824660ea2c45598f0f48fb`. |
| `run_final.sh` | Main experiment: seeds 0, 1 and 2 × 4 cycles (PPO → merge → GRPO → merge), frozen eval after every phase. md5 `bf1511d1745e550f5025811660f68c49`. |
| `run_ablation.sh` | Ablation with one update type only: 8 PPO phases, or 8 GRPO phases, with the same settings and seeds. |
| `run_pilot.sh` | One-cycle pilot used to adopt the scaled reward. |
| `gen_qwen.py`, `merge_qwen.py` | Frozen-eval generation; merging an adapter into its base model. |
| `check_strict_safe.py`, `train_valid.py` | Strict check of eval outputs (each network in a worker with RAM and time limits); 10-epoch training of the valid eval networks. |
| `reward/` | The reward functions the scripts load: `ab/gpt/TunePPO.py` (`_strict_extract`, `_strict_build`, `_train_eval`, `CIFARRewardEvaluator`) and `analyze_collapse.py` (repeat signature). |
| `prompts/frozen_prompts_v3.json` | The 100 frozen evaluation prompts. |
| `test_memory.py`, `test_v5_gpu.py` | Tests of the memory-bounded update and of the isolated reward worker. |
| `status.sh`, `make_release.sh` | Progress overview (read-only); collection of all results for this folder. |
| `lemur_export.py` | Export of the generated networks to the NN Dataset (LEMUR) format; see below. |
| `analysis/` | Scripts that recompute the paper's tables (`make_tables.py`) and figures (`make_fig_*.py`) from `results/data`. |
| `results/data/` | Tables used in the paper (CSV). `frozen_eval_seed0.csv`: every frozen-eval output of the base model and the 8 phases of seed 0, with strict status, accuracy, parameters, MACs, convolution layers and layer layout. `train_steps_seed0.csv`: every update step. The other files are the development run (`frozen_eval.csv`, `train_steps.csv`, `train_samples.csv`, phase logs). |
| `results/runs/` | Raw records and complete logs (gzip): `final/seed<S>/<phase>/generations.jsonl` (every rewarded generation, with its code, status, accuracies, training times and peak RAM), `steps.jsonl`, `best.json`, `config.json`; `final/seed<S>/eval_<phase>/run<k>.jsonl`, `run<k>_strict.jsonl`, `run<k>_acc.jsonl` (every frozen-eval output and its result) and the gen/strict/train logs; `log_<phase>.log` (full training log, with timing); `run_final.log`; `base/` (frozen eval of the base model); `stopped_v4c/` (first start of the final run, stopped by a hang; see the paper's run history). |

Included so far: seed 0 complete and the finished phases of seed 1. Seeds 1 and 2 and the ablation are added when
they are finished.

## Running

The scripts are the exact versions used, including our workstation's paths (an RTX 4090 with 24 GB, shared). They
expect:

- this NNGPT checkout with its environment in `~/nn-gpt-main/.venv` (`run_final.sh` runs from there);
- the reward code at `~/nn-gpt/ab/gpt/TunePPO.py` and `~/nn-gpt/analyze_collapse.py`, with CIFAR-10 in `~/nn-gpt/data`.
  `qwen_cycle.py` takes this location from `OLD_CODE` (default `~/nn-gpt`), so you can point `OLD_CODE` at `reward/`
  and place CIFAR-10 in `reward/data`. `check_strict_safe.py` reads `~/nn-gpt/ab/gpt/TunePPO.py` directly;
- for `train_valid.py`, an environment in which `ab.gpt.TunePPO` can be imported. That module imports the earlier
  `ab.gpt.rl_pipeline` layout of NNGPT. `qwen_cycle.py` and `check_strict_safe.py` load only the needed functions from
  the file, so they do not need that layout;
- `~/frozen_prompts_v3.json` (copy it from `prompts/`).

```bash
cd ~/nn-gpt-main
nohup ./run_final.sh > run_final.log 2>&1 &          # main experiment (resumable)
./status.sh                                          # progress
nohup ./run_ablation.sh > run_ablation.log 2>&1 &    # starts after the main run has finished
```

## Export to the NN Dataset (LEMUR)

`lemur_export.py prepare <release dir> <out dir>` selects the strict-valid, trained frozen-eval networks, best first
and unique by syntax tree. Each network is converted to the NN Dataset interface without changing the network: the
program is extracted exactly as in the strict reward, and missing `torch`, `nn` and `F` imports are added (the strict
check provided them). Top-level calls are removed, and `self.device` is set. `supported_hyperparameters`,
`train_setup` and `learn` are added with NNGPT's `ab.gpt.util.nn.PostprocessNN` (`_fix_hyperparams`,
`_fix_missing_train_setup`, `_fix_missing_learn`). Every network is then checked on CPU: it builds, runs, and trains
one batch.

`lemur_export.py register <out dir> --limit N` trains the first N prepared networks with `ab.nn.api.check_nn`
(`img-classification`, `cifar-10`, `acc`, transform `norm`, batch 128, SGD lr 0.01, momentum 0.9, dropout 0.2,
5 epochs). It saves them in the NN Dataset with the prefix `qwcyc`. In seed 0 and the finished phases of seed 1,
2,747 of 2,786 distinct networks pass the conversion check.

Note: the 10-epoch accuracies in this folder come from the experiment's protocol (Adam 1e-3, mean/std 0.5
normalization). The NN Dataset statistics come from the NN Dataset pipeline and are therefore not identical.
