# Reproducing: Source-Guided Candidate Generation for Improving Weak Neural Networks with LLMs

## Environment

- Python: Python 3.10.12
- PyTorch: 2.9.1+cu128
- CUDA: 12.8
- GPU: NVIDIA GeForce RTX 4090
- OS: Linux 6.8.0-111-generic #111~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Tue Apr 14 17:13:45 UTC x86_64 x86_64 x86_64 GNU/Linux

## Setup

*Note: Replace `/home/kabir/newws` with your preferred clone location and update `NN_GPT_ROOT`, and `VENV` accordingly in the later scripts.*

### 1. Clone both repositories

```bash
git clone https://github.com/ABrain-One/nn-gpt.git ${NN_GPT_ROOT}
```

### 2. Create and activate the virtual environment

```bash
python -m venv ${VENV}
source ${VENV}/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r ${NN_GPT_ROOT}/requirements.txt
python -m pip install \
  accelerate==1.13.0 \
  bitsandbytes==0.49.2 \
  numpy==2.2.6 \
  peft==0.18.1 \
  pillow==12.2.0 \
  pytest==9.1.1 \
  requests==2.33.1 \
  scipy==1.15.3 \
  torch==2.9.1 \
  torchvision==0.24.1 \
  transformers==5.4.0
```

### 4. Set PYTHONPATH

```bash
export PYTHONPATH=${NN_GPT_ROOT}:${PYTHONPATH:-}
```

## Repository structure

- `nn-gpt/ab/gpt/`: Existing public TuneNNGen entrypoints and upstream-compatible utilities.
- `nn-gpt/ab/gpt/analog/`: Isolated paper experiment runners and analog-specific pipeline code.
- `nn-gpt/ab/gpt/analog/TuneAnalog.py`: Source-guided arm dispatch for `hp_default`, `baseline_edit`, `hp_transfer`, `analogical_edit`, and `hp_copy`.
- `nn-gpt/ab/gpt/analog/TuneNNGenAnalog.py`: CLI-compatible analog experiment entrypoint used by the commands below.
- `nn-gpt/ab/gpt/conf/prompt/test/analog/`: Test prompt configs for the source-guided paper experiments.
- `nn-gpt/ab/gpt/conf/prompt/train/analog/`: Training prompt configs for the source-guided paper experiments.
- `nn-gpt/ab/gpt/conf/prompt/test/` and `nn-gpt/ab/gpt/conf/prompt/train/`: Existing flat prompt configs retained for old entrypoints.
- `nn-gpt/ab/gpt/util/EditUtil.py`: Additive edit prompt construction and LLM response parsing helpers.
- `nn-gpt/results_registry/`: Curated CSV/JSON result registries used to generate paper tables.
- `nn-dataset/ab/nn/util/`: Training and evaluation harness, including `Train.py`, `CodeEval.py`, and `Const.py`.
- `nn-dataset/ab/nn/transform/`: Dataset transform implementations such as `echo_224`, `echo_28`, and `norm_299_flip`.

## Running existing baseline functionality

The existing public entrypoints remain in their original locations and continue to resolve flat prompt configs such as `NN_gen.json`.

```bash
${VENV}/bin/python -m ab.gpt.TuneNNGen --help
${VENV}/bin/python -m ab.gpt.TuneNNGen_delta --help
${VENV}/bin/python ab/gpt/NNEval.py --help
```

The analog experiments below intentionally use `ab.gpt.analog.TuneNNGenAnalog` and prompt config names under `analog/` so the paper-specific runs do not override those baseline entrypoints.

## Running the main experiments

All commands run from the `nn-gpt` repository root with `PYTHONPATH` set as above. The helper below is the exact row runner used to express the table commands. Each `run_row` call launches all four comparable arms: `hp_default`, `baseline_edit`, `hp_transfer`, and `analogical_edit`.

```bash
# Adjust these three paths to match your clone location
export NN_GPT_ROOT=/home/kabir/newws/nn-gpt
export VENV=/home/kabir/newws/.venv

cd ${NN_GPT_ROOT}

export PYTHONPATH=${NN_GPT_ROOT}:${PYTHONPATH:-}
export AB_GPT_SKIP_POST_FINETUNE=1
export AB_GPT_STRICT_NO_REPAIR=0
export AB_GPT_REPAIR_MODE=minimal
export AB_GPT_ENABLE_EDIT_SAFETY_GATE=0
export AB_GPT_ABORT_ON_STARTUP_FAILURE=1
export AB_GPT_FIXED_TEST_SEED=20260427
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

run_one_arm() {
  local suite="$1"
  local slug="$2"
  local llm_conf="$3"
  local target_id="$4"
  local source_id="$5"
  local repeat_n="$6"
  local seed_offset="$7"
  local geometry_guard="$8"
  local arm="$9"
  local prompt_conf="${10}"
  local prompt_key="${11}"
  local max_new_tokens="${12}"

  export AB_GPT_NNGPT_DIR="${NN_GPT_ROOT}/out/benchmarks/${suite}_${arm}"
  export AB_GPT_TARGET_IDS="$target_id"
  export AB_GPT_SOURCE_IDS="$source_id"
  export AB_GPT_REPEAT_TARGET_N="$repeat_n"
  export AB_GPT_CANDIDATE_SEED_OFFSET="$seed_offset"
  export AB_GPT_GEOMETRY_GUARD="$geometry_guard"

  ${VENV}/bin/python -m ab.gpt.analog.TuneNNGenAnalog \
    --llm_conf "$llm_conf" \
    --llm_tune_conf "$prompt_conf" \
    --nn_gen_conf "$prompt_conf" \
    --nn_gen_conf_id "$prompt_key" \
    --num_train_epochs 1 \
    --test_nn 1 \
    --nn_train_epochs 1 \
    --max_prompts 16 \
    --max_new_tokens "$max_new_tokens" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-6 \
    --logging_steps 5 \
    --temperature 0.75 \
    --top_k 50 \
    --top_p 0.95 \
    --prompt_batch 1 \
    --nn_name_prefix edit \
    --no-eval_save_to_db
}

run_row() {
  local suite="$1"
  local slug="$2"
  local llm_conf="$3"
  local target_id="$4"
  local source_id="$5"
  local repeat_n="$6"
  local seed_offset="$7"
  local geometry_guard="$8"
  local baseline_conf="$9"
  local baseline_key="${10}"
  local analogical_conf="${11}"
  local analogical_key="${12}"

  run_one_arm "$suite" "$slug" "$llm_conf" "$target_id" "$source_id" "$repeat_n" "$seed_offset" "$geometry_guard" \
    hp_default analog/NN_gen_frozen_image_starter.json "improve_classification_only_${slug}_frozen_hp_default" 1024
  run_one_arm "$suite" "$slug" "$llm_conf" "$target_id" "$source_id" "$repeat_n" "$seed_offset" "$geometry_guard" \
    baseline_edit "$baseline_conf" "$baseline_key" 1536
  run_one_arm "$suite" "$slug" "$llm_conf" "$target_id" "$source_id" "$repeat_n" "$seed_offset" "$geometry_guard" \
    hp_transfer analog/NN_gen_frozen_image_starter.json "improve_classification_only_${slug}_frozen_hp_transfer" 1536
  run_one_arm "$suite" "$slug" "$llm_conf" "$target_id" "$source_id" "$repeat_n" "$seed_offset" "$geometry_guard" \
    analogical_edit "$analogical_conf" "$analogical_key" 2048
}
```

### CIFAR-10 main N=32 (Tables 3 and 5)

```bash
# ds67b archbest
run_row cifar10_clean_archbest_ds67b_n32 cifar10 ds_coder_6.7b_instruct_smoke.json ed1246fb7fb1bb50c9127eebcb6e05d2 ed44f284382e3593f982eeaa71996065 32 7500 1 analog/NN_gen_cifar10_edit_paper.json improve_classification_only_cifar10_edit_paper analog/NN_gen_analogical_cifar10_edit_paper.json improve_classification_only_analogical_cifar10_edit_paper

# ds67b highsource
run_row cifar10_clean_highsource_ds67b_n32 cifar10 ds_coder_6.7b_instruct_smoke.json ed1246fb7fb1bb50c9127eebcb6e05d2 f6923a5f94145abd0a362581ae0012e9 32 6200 1 analog/NN_gen_cifar10_edit_paper.json improve_classification_only_cifar10_edit_paper analog/NN_gen_analogical_cifar10_edit_paper.json improve_classification_only_analogical_cifar10_edit_paper

# qwen25coder7b archbest
run_row cifar10_clean_archbest_qwen25coder7b_n32 cifar10 qwen2_5_coder_7b_instruct_smoke.json ed1246fb7fb1bb50c9127eebcb6e05d2 ed44f284382e3593f982eeaa71996065 32 7200 1 analog/NN_gen_cifar10_edit_paper.json improve_classification_only_cifar10_edit_paper analog/NN_gen_analogical_cifar10_edit_paper.json improve_classification_only_analogical_cifar10_edit_paper

# qwen25coder7b highsource
run_row cifar10_clean_highsource_qwen25coder7b_n32 cifar10 qwen2_5_coder_7b_instruct_smoke.json ed1246fb7fb1bb50c9127eebcb6e05d2 f6923a5f94145abd0a362581ae0012e9 32 7100 1 analog/NN_gen_cifar10_edit_paper.json improve_classification_only_cifar10_edit_paper analog/NN_gen_analogical_cifar10_edit_paper.json improve_classification_only_analogical_cifar10_edit_paper

# olympic7b archbest
run_row cifar10_clean_archbest_olympic7b_n32 cifar10 ds_coder_7b_olympic_cifar10_smoke.json ed1246fb7fb1bb50c9127eebcb6e05d2 ed44f284382e3593f982eeaa71996065 32 7400 1 analog/NN_gen_cifar10_edit_paper.json improve_classification_only_cifar10_edit_paper analog/NN_gen_analogical_cifar10_edit_paper.json improve_classification_only_analogical_cifar10_edit_paper

# olympic7b highsource
run_row cifar10_clean_highsource_olympic7b_n32 cifar10 ds_coder_7b_olympic_cifar10_smoke.json ed1246fb7fb1bb50c9127eebcb6e05d2 f6923a5f94145abd0a362581ae0012e9 32 7300 1 analog/NN_gen_cifar10_edit_paper.json improve_classification_only_cifar10_edit_paper analog/NN_gen_analogical_cifar10_edit_paper.json improve_classification_only_analogical_cifar10_edit_paper
```

### SVHN AlexNet main N=32 (Tables 4 and 5)

```bash
# ds67b
run_row svhn_hpschema_alexnet_n32_ds67b svhn ds_coder_6.7b_instruct_smoke.json 82f102e1bd4884ac574a1543d9eac6fb 9da54aec32cd2aebd2374d086521e20f 32 10300 0 analog/NN_gen_svhn_edit_paper.json improve_classification_only_svhn_edit_paper analog/NN_gen_analogical_svhn_edit_paper.json improve_classification_only_analogical_svhn_edit_paper

# qwen25coder7b
run_row svhn_hpschema_alexnet_n32_qwen25coder7b svhn qwen2_5_coder_7b_instruct_smoke.json 82f102e1bd4884ac574a1543d9eac6fb 9da54aec32cd2aebd2374d086521e20f 32 10400 0 analog/NN_gen_svhn_edit_paper.json improve_classification_only_svhn_edit_paper analog/NN_gen_analogical_svhn_edit_paper.json improve_classification_only_analogical_svhn_edit_paper

# olympic7b
run_row svhn_hpschema_alexnet_n32_olympic7b svhn ds_coder_7b_olympic_cifar10_smoke.json 82f102e1bd4884ac574a1543d9eac6fb 9da54aec32cd2aebd2374d086521e20f 32 10500 0 analog/NN_gen_svhn_edit_paper.json improve_classification_only_svhn_edit_paper analog/NN_gen_analogical_svhn_edit_paper.json improve_classification_only_analogical_svhn_edit_paper
```

### hp_copy ablation (Table 14)

```bash
run_hp_copy() {
  local suite="$1"
  local slug="$2"
  local target_id="$3"
  local source_id="$4"
  local geometry_guard="$5"

  export AB_GPT_NNGPT_DIR="${NN_GPT_ROOT}/out/benchmarks/${suite}"
  export AB_GPT_TARGET_IDS="$target_id"
  export AB_GPT_SOURCE_IDS="$source_id"
  export AB_GPT_REPEAT_TARGET_N=1
  export AB_GPT_CANDIDATE_SEED_OFFSET=0
  export AB_GPT_GEOMETRY_GUARD="$geometry_guard"

  ${VENV}/bin/python -m ab.gpt.analog.TuneNNGenAnalog \
    --llm_conf ds_coder_6.7b_instruct_smoke.json \
    --llm_tune_conf analog/NN_gen_frozen_image_starter.json \
    --nn_gen_conf analog/NN_gen_frozen_image_starter.json \
    --nn_gen_conf_id "improve_classification_only_${slug}_frozen_hp_copy" \
    --num_train_epochs 1 \
    --test_nn 1 \
    --nn_train_epochs 1 \
    --max_prompts 16 \
    --max_new_tokens 128 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-6 \
    --logging_steps 5 \
    --temperature 0.75 \
    --top_k 50 \
    --top_p 0.95 \
    --prompt_batch 1 \
    --nn_name_prefix edit \
    --no-eval_save_to_db
}

# CIFAR-10 archbest
run_hp_copy hp_copy_cifar10_archbest cifar10 ed1246fb7fb1bb50c9127eebcb6e05d2 ed44f284382e3593f982eeaa71996065 1

# CIFAR-10 highsource
run_hp_copy hp_copy_cifar10_highsource cifar10 ed1246fb7fb1bb50c9127eebcb6e05d2 f6923a5f94145abd0a362581ae0012e9 1

# SVHN AlexNet
run_hp_copy hp_copy_svhn_alexnet svhn 82f102e1bd4884ac574a1543d9eac6fb 9da54aec32cd2aebd2374d086521e20f 1

# Imagenette AlexNet
run_hp_copy hp_copy_imagenette_alexnet imagenette bb88d97f5db9a331051f1695a5d81d1d 61897b06c8dabb270284572c01d2b229 0
```

### SVHN robustness probes N=8 (Table 8)

```bash
# BagNet across three LLMs
run_row svhn_hpschema_bagnet_ds67b_n8 svhn ds_coder_6.7b_instruct_smoke.json 039b97c112dad21861b5c89b44c9bb8e 94eade47ad96eedb4109c74cd939800f 8 15300 0 analog/NN_gen_svhn_edit_paper.json improve_classification_only_svhn_edit_paper analog/NN_gen_analogical_svhn_edit_paper.json improve_classification_only_analogical_svhn_edit_paper
run_row svhn_hpschema_bagnet_qwen25coder7b_n8 svhn qwen2_5_coder_7b_instruct_smoke.json 039b97c112dad21861b5c89b44c9bb8e 94eade47ad96eedb4109c74cd939800f 8 15400 0 analog/NN_gen_svhn_edit_paper.json improve_classification_only_svhn_edit_paper analog/NN_gen_analogical_svhn_edit_paper.json improve_classification_only_analogical_svhn_edit_paper
run_row svhn_hpschema_bagnet_olympic7b_n8 svhn ds_coder_7b_olympic_cifar10_smoke.json 039b97c112dad21861b5c89b44c9bb8e 94eade47ad96eedb4109c74cd939800f 8 15500 0 analog/NN_gen_svhn_edit_paper.json improve_classification_only_svhn_edit_paper analog/NN_gen_analogical_svhn_edit_paper.json improve_classification_only_analogical_svhn_edit_paper

# AirNext across three LLMs
run_row svhn_hpschema_airnext_ds67b_n8 svhn ds_coder_6.7b_instruct_smoke.json 28df9f99ef7a99c438773e9b42f582c2 05e1773a0b197bf63c5788302d6052d8 8 15600 0 analog/NN_gen_svhn_edit_paper.json improve_classification_only_svhn_edit_paper analog/NN_gen_analogical_svhn_edit_paper.json improve_classification_only_analogical_svhn_edit_paper
run_row svhn_hpschema_airnext_qwen25coder7b_n8 svhn qwen2_5_coder_7b_instruct_smoke.json 28df9f99ef7a99c438773e9b42f582c2 05e1773a0b197bf63c5788302d6052d8 8 15700 0 analog/NN_gen_svhn_edit_paper.json improve_classification_only_svhn_edit_paper analog/NN_gen_analogical_svhn_edit_paper.json improve_classification_only_analogical_svhn_edit_paper
run_row svhn_hpschema_airnext_olympic7b_n8 svhn ds_coder_7b_olympic_cifar10_smoke.json 28df9f99ef7a99c438773e9b42f582c2 05e1773a0b197bf63c5788302d6052d8 8 15800 0 analog/NN_gen_svhn_edit_paper.json improve_classification_only_svhn_edit_paper analog/NN_gen_analogical_svhn_edit_paper.json improve_classification_only_analogical_svhn_edit_paper

# DPN68 across three LLMs
run_row svhn_hpschema_dpn68_ds67b_n8 svhn ds_coder_6.7b_instruct_smoke.json 48ebb84d913cc8ffad0b9379f25006bc d7e91737df547831f65a082b391c28d1 8 15900 0 analog/NN_gen_svhn_edit_paper.json improve_classification_only_svhn_edit_paper analog/NN_gen_analogical_svhn_edit_paper.json improve_classification_only_analogical_svhn_edit_paper
run_row svhn_hpschema_dpn68_qwen25coder7b_n8 svhn qwen2_5_coder_7b_instruct_smoke.json 48ebb84d913cc8ffad0b9379f25006bc d7e91737df547831f65a082b391c28d1 8 16000 0 analog/NN_gen_svhn_edit_paper.json improve_classification_only_svhn_edit_paper analog/NN_gen_analogical_svhn_edit_paper.json improve_classification_only_analogical_svhn_edit_paper
run_row svhn_hpschema_dpn68_olympic7b_n8 svhn ds_coder_7b_olympic_cifar10_smoke.json 48ebb84d913cc8ffad0b9379f25006bc d7e91737df547831f65a082b391c28d1 8 16100 0 analog/NN_gen_svhn_edit_paper.json improve_classification_only_svhn_edit_paper analog/NN_gen_analogical_svhn_edit_paper.json improve_classification_only_analogical_svhn_edit_paper
```

### AlexNet cross-dataset probes (Table 10)

```bash
# Imagenette ds13b
run_row imagenette_hpschema_alexnet_n8_ds13b imagenette ds_coder_1.3b_instruct_smoke.json bb88d97f5db9a331051f1695a5d81d1d 61897b06c8dabb270284572c01d2b229 8 11200 0 analog/NN_gen_imagenette_edit_paper.json improve_classification_only_imagenette_edit_paper analog/NN_gen_analogical_imagenette_edit_paper.json improve_classification_only_analogical_imagenette_edit_paper

# Imagenette ds67b
run_row imagenette_hpschema_alexnet_n8_ds67b imagenette ds_coder_6.7b_instruct_smoke.json bb88d97f5db9a331051f1695a5d81d1d 61897b06c8dabb270284572c01d2b229 8 11300 0 analog/NN_gen_imagenette_edit_paper.json improve_classification_only_imagenette_edit_paper analog/NN_gen_analogical_imagenette_edit_paper.json improve_classification_only_analogical_imagenette_edit_paper

# Imagenette olympic7b
run_row imagenette_hpschema_alexnet_n8_olympic7b imagenette ds_coder_7b_olympic_cifar10_smoke.json bb88d97f5db9a331051f1695a5d81d1d 61897b06c8dabb270284572c01d2b229 8 11500 0 analog/NN_gen_imagenette_edit_paper.json improve_classification_only_imagenette_edit_paper analog/NN_gen_analogical_imagenette_edit_paper.json improve_classification_only_analogical_imagenette_edit_paper

# CelebA-Gender ds67b
run_row celebagender_geomguard_v2_alexnet_ds67b_n8 celebagender ds_coder_6.7b_instruct_smoke.json cb3b158684048414d60999434f84cec5 eb8e2a71de857766413f04c3f81f0a26 8 14300 1 analog/NN_gen_celebagender_edit_paper.json improve_classification_only_celebagender_edit_paper analog/NN_gen_analogical_celebagender_edit_paper.json improve_classification_only_analogical_celebagender_edit_paper
```

## Reading results

Each run writes a run-local `cycle_results.json` and per-candidate `eval_info.json` under `out/benchmarks/<suite>_<arm>/`. The curated `results_registry/` directory contains derived result files used by the paper. `paper_expected_outputs_20260621.csv` records the values reported in the paper tables. `acc_gap_advantage_pairs_20260606.csv` is a separate all-logs pair analysis used for architecture-family win rates; it can select a different logged run for a dataset/LLM/source pair and does not contain the `hp_copy` ablation rows.

To reproduce a table number, filter `paper_expected_outputs_20260621.csv` by dataset, architecture family, target ID, source ID, LLM, and method group. For a fresh run, compare the run-local max valid accuracy for non-source arms (`hp_default`, `baseline_edit`) against source-guided arms (`hp_transfer`, `analogical_edit`).

CSV column names in `results_registry/acc_gap_advantage_pairs_20260606.csv`: `dataset`, `target_family`, `target_acc`, `source_acc`, `acc_gap`, `llm`, `source_guided_best`, `non_source_best`, `advantage`, `result`, `target_id`, `source_id`, `source_guided_method`, `non_source_method`.

## Expected outputs

| Experiment | Arm | Expected accuracy |
|---|---|---|
| CIFAR-10 ds67b archbest | src-guided best | 0.5049 |
| CIFAR-10 ds67b archbest | non-src best | 0.2398 |
| CIFAR-10 ds67b highsource | src-guided best | 0.4213 |
| CIFAR-10 ds67b highsource | non-src best | 0.2354 |
| CIFAR-10 qwen25coder7b archbest | src-guided best | 0.4757 |
| CIFAR-10 qwen25coder7b archbest | non-src best | 0.2156 |
| CIFAR-10 qwen25coder7b highsource | src-guided best | 0.4814 |
| CIFAR-10 qwen25coder7b highsource | non-src best | 0.2386 |
| CIFAR-10 olympic7b archbest | src-guided best | 0.4806 |
| CIFAR-10 olympic7b archbest | non-src best | 0.4971 |
| CIFAR-10 olympic7b highsource | src-guided best | 0.4033 |
| CIFAR-10 olympic7b highsource | non-src best | 0.4876 |
| SVHN AlexNet ds67b | src-guided best | 0.7880 |
| SVHN AlexNet ds67b | non-src best | 0.1959 |
| SVHN AlexNet qwen25coder7b | src-guided best | 0.1959 |
| SVHN AlexNet qwen25coder7b | non-src best | 0.1959 |
| SVHN AlexNet olympic7b | src-guided best | 0.1965 |
| SVHN AlexNet olympic7b | non-src best | 0.1959 |
| Imagenette AlexNet ds13b | src-guided best | 0.3534 |
| Imagenette AlexNet ds67b | src-guided best | 0.2492 |
| Imagenette AlexNet olympic7b | src-guided best | 0.3656 |
| hp_copy CIFAR-10 archbest | hp_copy | 0.4267 |
| hp_copy CIFAR-10 highsource | hp_copy | 0.3506 |
| hp_copy SVHN AlexNet | hp_copy | 0.1959 |
| hp_copy Imagenette AlexNet | hp_copy | 0.2892 |

## Local checks performed after analog isolation

These checks were run after moving the experiment scripts into `ab/gpt/analog/` and moving prompt configs into `ab/gpt/conf/prompt/{train,test}/analog/`.

| Check | Result |
|---|---|
| `python -m compileall ab/gpt/analog ab/gpt/util/EditUtil.py ab/gpt/util/Mergedecision.py ab/nn/util/CodeEval.py ab/nn/util/Train.py ab/nn/transform/echo_224.py ab/nn/transform/echo_28.py` | pass |
| `python ab/gpt/NNEval.py --help` | pass |
| `python -m ab.gpt.TuneNNGen --help` | pass |
| `python -m ab.gpt.TuneNNGen_delta --help` | pass |
| `python -m ab.gpt.analog.TuneNNGenAnalog --help` | pass |
| `python -m ab.gpt.analog.TuneNNGen_7B_code_olympic_analogical_smoke --help` | pass |
| JSON validation for every file under `ab/gpt/conf/` | pass |
| Existing flat prompt lookup and new `analog/` prompt lookup | pass |

Full `python -m compileall ab/gpt ab/nn` was also attempted. It fails on pre-existing template/generated files outside the analog PR path, including `ab/gpt/brute/fract/backbone/FractalFusion_template.py` (`$$`) and `ab/gpt/brute/fract/pure/Fractal_template.py` (`N = ?1`). The analog package and additive helper files compile successfully.

`pytest -q` was attempted and fails during collection because `test/test_pipeline.py` imports `ab.gpt.util.prompt.NNGenPromptPrun`, which is absent from `origin/main`. `python -m unittest discover -v` was attempted and fails on existing `ab.gpt.brute.ast.mutator` package imports that expect a top-level `mutator` module.

The full one-epoch generation/evaluation commands were not rerun after the isolation refactor because the local environment lacks `deepspeed` and the commands require GPU/model/dataset execution. The expected-output values above are the logged paper values from `results_registry/`.

## Reproducibility note

PyTorch deterministic mode is not used; exact bitwise replay is not claimed. Reproducibility is defined at the protocol level: the same frozen prompts, same source/target IDs, same candidate budget, same LLM, same seed offset, and same validation settings should produce results within normal stochastic variation of the reported values. Analog experiment output paths are configured with `AB_GPT_NNGPT_DIR`, which is resolved through `ab.gpt.util.Const` relative to the repository/output root conventions. The `results_registry/` directory contains the logged outputs used to generate the paper tables; fresh runs should be checked through their run-local `cycle_results.json` and `eval_info.json` files before being merged into the curated registry.
