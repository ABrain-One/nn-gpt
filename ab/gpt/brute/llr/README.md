# Layerwise Learning-Rate (LLR) — Generators & Pipeline

This folder's `layerwise_lr*.py` generators, their evaluation, the
vanilla-vs-LLR analysis, and the LLM fine-tuning pipeline that consumes the
results.

> 📄 **Full reference:** [docs/LLR_PIPELINE.md](../../../../docs/LLR_PIPELINE.md) —
> goal, strategies, evaluation, analysis, fine-tuning, and lessons learned.
>
> This folder is separate from [`ab/gpt/brute/lr/`](../lr/), which holds an
> unrelated **LR-scheduler** search built on `schedulers.py` — not the LLR work.

## Files in this folder (LLR)

| File | Purpose |
|------|---------|
| `layerwise_lr.py`  | Brute-force generator — batch 1 (`llr_`, 12 strategies: uniform control + monotone decay) |
| `layerwise_lr2.py` | Brute-force generator — batch 2 (`llr2_`, 13: inverted, asymmetric, cosine, 6-group, cyclic) |
| `layerwise_lr3.py` | Brute-force generator — batch 3 (`llr3_`, 14: geometric, V-shaped, plateau, step, skewed) |
| `build_metadata_csv.py` | Rebuild strategy metadata by checksum-matching generated code to evaluated hash names |
| `analyze_llr_vs_vanilla.py` | Compare each llr model to its vanilla parent at matched (dataset, epoch); emit leaderboards |

## Quick start

```bash
# 1. Generate model variants (writes to out/nngpt/.../synth_nn)
python -m ab.gpt.brute.llr.layerwise_lr
python -m ab.gpt.brute.llr.layerwise_lr2
python -m ab.gpt.brute.llr.layerwise_lr3

# 2. (after LEMUR evaluation) rebuild the strategy metadata CSV
python -m ab.gpt.brute.llr.build_metadata_csv

# 3. analyse which strategies beat the vanilla baselines
python -m ab.gpt.brute.llr.analyze_llr_vs_vanilla
```

## Outputs (`out/nngpt/`)

- `llr_evaluated_with_meta.csv` — evaluated models + strategy metadata (the bridge
  from llr hash names to architecture/strategy).
- `llr_vs_vanilla.csv` — per-model improvement vs the vanilla baseline.
- `best_strategy_per_arch_dataset.csv` — winning strategy per (architecture, dataset).
- `strategy_leaderboard.csv` — per-strategy win-rate and mean/median/best Δ.

## Headline result

126 of 931 evaluated LLR models (**13.5%**) beat their vanilla architecture at the
same (dataset, epoch). There is no universally good strategy — the benefit is
architecture-specific (robust responders: GoogLeNet, BagNet, ResNet, DPN107).
See the full document for details.
