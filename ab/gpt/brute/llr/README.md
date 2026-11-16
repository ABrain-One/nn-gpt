# Layerwise Learning-Rate (LLR) Strategies

Code for generating layerwise learning-rate strategies, injecting them into
LEMUR models, and running the experiments of "Layerwise Learning-Rate Scaling
in Vision Networks: Effective-Rate Identification, Power-Mean Aggregation, and
Budget Dependence".

This folder is separate from [`ab/gpt/brute/lr/`](../lr/), which holds an
unrelated learning-rate scheduler search.

## Strategy generators and screening sweep

| File | Purpose |
|------|---------|
| `layerwise_lr.py`, `layerwise_lr2.py`, `layerwise_lr3.py` | The 38 strategies in three batches (`llr_`, `llr2_`, `llr3_`), injected into each model's `train_setup()` |
| `build_metadata_csv.py` | Rebuilds strategy metadata by matching generated code to evaluated model names |
| `analyze_llr_vs_vanilla.py` | Compares each strategy model with its vanilla baseline and builds leaderboards |
| `analyze_llr.py` | Collects evaluated model directories into results and strategy-ranking tables |
| `llr_baselines.py` | Shared helpers: locating the LEMUR database, strategy metadata, vanilla baselines |
| `export_vanilla_baselines.py` | Baseline-accuracy lookup per architecture, dataset, epoch and transform |
| `fdr_correction.py` | Benjamini-Hochberg correction for the sweep |

## Controlled experiments

| File | Purpose |
|------|---------|
| `epoch_ablation_gen.py`, `epoch_ablation_analyze.py` | Epoch and seed ablation (selected strategy against `llr_uniform`, 5 seeds) |
| `tnnls_gen.py`, `tnnls_collect.py` | All later controlled runs: 100-epoch extensions, rate-matched and mean-preserving controls, second base rate, ULMFiT, tuned baselines, held-out architectures |
| `build_per_seed_csv.py` | Merges the collector outputs into one per-seed CSV |
| `export_nn_dataset.py` | Exports the runs into the NN Dataset layout |

The `tnnls_*` names come from an earlier revision round and are kept on
purpose: `tnnls_gen.py` writes a replicate tag into each generated model, so
renaming would change the code and checksums of the deposited `llrtn-` models.

## LLM-generated recipes

| File | Purpose |
|------|---------|
| `llr_prompt_pairs.py` | Builds the baseline-to-strategy training pairs for the LLM fine-tune |
| `llr_generation.py` | Samples baseline candidates for each generation cycle of the fine-tune |
| `ood_scale_up.py`, `ood_analyze.py` | Applies generated edits to held-out CIFAR-100 models and evaluates them |
| `ood_epoch_recheck_gen.py`, `ood_epoch_recheck_analyze.py`, `ood_recheck2_gen.py`, `ood_recheck2_collect.py` | Re-check of the generated recipes against freshly trained controls |

Generated edits are applied with AST anchor patching,
`apply_delta_to_optimizer_assignment` in `ab/gpt/util/nn/DeltaUtil.py`, which
replaces the optimizer assignment in the target file's syntax tree instead of
matching the diff's context lines.

## Models in the NN Dataset

Generated models carry these name prefixes:

- `llr-`, `llr2-`, `llr3-`: screening sweep
- `llrtn-`: controlled experiments
- `llrood-`: re-check of the generated recipes

All controlled runs used nn-dataset at commit `be8657e5`. Later commits
identify a model by its architecture graph, so seed replicates of one
architecture collapse into one record.

## Main result

Against freshly trained uniform-rate controls, layerwise strategies win 136 of
738 matched screening trials (18.4%). Most of the effect is explained by the
effective learning rate the strategy implies: a parameter-weighted power mean
of the multipliers with exponent 0.35 accounts for 42.6% of the
within-architecture variance. See the paper for the full analysis.
