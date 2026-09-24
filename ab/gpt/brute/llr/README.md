# Layerwise Learning-Rate (LLR): generators, validation and fine-tuning tooling

Generators for layerwise learning-rate (LLR) training recipes, their evaluation, the
analyses that check whether an apparent gain is real, and the tooling that supports
LLM fine-tuning on LLR edits.

> This folder is separate from [`ab/gpt/brute/lr/`](../lr/), which holds an unrelated
> **LR-scheduler** search built on `schedulers.py`, not the LLR work.

## Generators and sweep analysis

| File | Purpose |
|------|---------|
| `layerwise_lr.py`  | Brute-force generator, batch 1 (`llr_`, 12 strategies: uniform control + monotone decay) |
| `layerwise_lr2.py` | Brute-force generator, batch 2 (`llr2_`, 13: inverted, asymmetric, cosine, 6-group, cyclic) |
| `layerwise_lr3.py` | Brute-force generator, batch 3 (`llr3_`, 14: geometric, V-shaped, plateau, step, skewed) |
| `build_metadata_csv.py` | Rebuild strategy metadata by checksum-matching generated code to evaluated hash names |
| `analyze_llr_vs_vanilla.py` | Compare each llr model to its baseline at matched (dataset, epoch); emit leaderboards |
| `fdr_correction.py` | Benjamini-Hochberg FDR correction over architectures and strategies |

## Validation experiments

A one-epoch, one-seed sweep cannot separate a real effect from run-to-run noise, and cannot
say whether a short-budget gain persists. These scripts check that.

| File | Purpose |
|------|---------|
| `epoch_ablation_gen.py` | Winning strategy vs the `llr_uniform` control, several seeds, trained for 10 epochs; the per-epoch checkpoints come from a single run |
| `epoch_ablation_analyze.py` | Per (architecture, condition, epoch) means with confidence intervals and the winning-vs-control difference |
| `ood_scale_up.py` | Run a fine-tuned adapter on held-out targets; apply the generated edit and evaluate it (`--no-save_to_db`, so the shared DB is only read) |
| `ood_analyze.py` | Delta-application rate and accuracy deltas of an OOD run, overall and per family |
| `ood_epoch_recheck_gen.py` / `ood_epoch_recheck_analyze.py` | Re-evaluate selected OOD targets against a fresh `llr_uniform` control at epochs 1, 3 and 10 |

Notes that matter when reading results:

- **Compare against a matched control.** The `ood_scale_up.py` delta is measured against the
  target's *stored* database accuracy, which was not trained under this protocol. On six
  re-checked targets the stored accuracy was 0.02 to 0.21 below a matched `llr_uniform`
  control, and the apparent gains disappeared against the control. Use
  `ood_epoch_recheck_*` for a fair comparison.
- **Seed replicates without touching the evaluator.** NNEval deduplicates on the code
  checksum, so each replicate carries an inert trailing assignment (`_replicate_seed = k`).
  It changes the checksum, not the behaviour, so replicates differ only by SGD
  nondeterminism.
- **One directory slot per job.** NNEval evaluates *every* directory under
  `A{epoch-idx}/synth_nn`, so make sure the slot holds only the models you generated.
- **Per-epoch results** are also written as `<model_dir>/N.json` (N = epoch), so they can be
  read without querying the DB.

## Quick start

```bash
# 1. Generate model variants (writes to out/nngpt/.../synth_nn)
python -m ab.gpt.brute.llr.layerwise_lr
python -m ab.gpt.brute.llr.layerwise_lr2
python -m ab.gpt.brute.llr.layerwise_lr3

# 2. (after LEMUR evaluation) rebuild the strategy metadata CSV
python -m ab.gpt.brute.llr.build_metadata_csv

# 3. analyse which strategies beat the baselines, then correct for multiplicity
python -m ab.gpt.brute.llr.analyze_llr_vs_vanilla
python -m ab.gpt.brute.llr.fdr_correction

# 4. epoch/seed ablation (6 architectures x {winning, control} x 5 seeds, 10 epochs)
python -m ab.gpt.brute.llr.epoch_ablation_gen --epoch-idx 900 --seeds 5
python -m ab.gpt.act.eval.Eval --only_epoch 900 --nn_train_epochs 10 --dataset cifar-10 \
    --transform norm_64_flip --task img-classification --metric acc \
    --nn_name_prefix llrep --save_to_db
python -m ab.gpt.brute.llr.epoch_ablation_analyze --epoch-idx 900 --prefix llrep
```

## Outputs (`out/nngpt/`)

- `llr_evaluated_with_meta.csv`: evaluated models + strategy metadata (the bridge from llr
  hash names to architecture/strategy).
- `llr_vs_vanilla.csv`: per-model improvement vs the baseline; `baseline_source` says
  whether the baseline was a matched `llr_uniform` run (`uniform_control`) or a stored
  database accuracy (`db_vanilla`). Do not pool the two.
- `best_strategy_per_arch_dataset.csv`, `strategy_leaderboard.csv`: winners per
  (architecture, dataset) and per-strategy win rates.
- `epoch_ablation_results.csv`: per (architecture, epoch) means and the winning-vs-control
  difference.

## Headline result

Against a matched `llr_uniform` control, 136 of 738 one-epoch trials (18.4%) beat their
baseline and the mean accuracy difference is -0.061; no strategy has a positive mean. The
benefit is architecture-specific, and in the epoch/seed ablation it does not persist:
GoogLeNet and ConvNeXt are ahead at epoch 1, and at epoch 10 no architecture is ahead of
the control while ConvNeXt, ICNet and UNet2D are behind it. The trials share their
baselines, so treat per-trial p-values with care and report architecture-level results.
