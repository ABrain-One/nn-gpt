# The Denoise Accuracy Predictor (V1 / V2) — how it works and why it matters

## Why it exists

Neural architecture search is bottlenecked by **evaluation cost**: every candidate must be
trained to be judged. The accuracy predictor is a **learned surrogate** that estimates a
candidate's PSNR *from its source code alone*, without training it — so the GA can cheaply
**drop the least-promising candidates before spending GPU** on them. It is a professor-required
deliverable and the main compute lever in the pipeline.

Crucially, the predictor **never crowns a winner** — full-training remains the sole promotion
authority (late-blooming architectures must still reach it). It only *screens the bottom*.

## How the funnel uses it (GA stage 1)

Each generation, in `run_seed_ga.py` (`make_prescreen`):

1. Collect the generation's **fresh** candidates (skip raw seeds and cache-hits).
2. Score each with the predictor → rank by predicted **edge-aware** fitness
   `eff = PSNR_pred − 0.3·ln(MACs/budget)` (the same objective the GA selects on).
3. **Train the top `--predict-keep` fraction** at the ep5 proxy; **skip the rest**, assigning
   them the predicted surrogate as fitness (they are the lowest-ranked, so they can never
   become elites, and the final best is always real-trained before it counts).

Net effect: ~50% of ep5 proxy-trainings are skipped each generation — roughly halving the
search's GPU cost — while the genuinely promising candidates still get trained.

## How the model works

Two stages, mirroring how the funnel needs it (`predictor/train_predictor.py`):

- **Stage 1 — deadness gate** (`HistGradientBoostingClassifier`): P(PSNR < 20 dB), i.e. a
  training that collapses. Cheaply drops dead-on-arrival candidates. Currently AUC ≈ 0.71 —
  genuinely useful.
- **Stage 2 — viable regressor** (`HistGradientBoostingRegressor`): predicts PSNR on the
  viable (≥20 dB) band — the ranking that actually decides which models get trained.

**Features** (`predictor/features.py` + `predictor/zero_cost.py`):
- Static AST features: op-type counts, depth, width, skip/residual structure, norm/activation
  types, etc. (no training needed — parsed from source).
- Cost features: MACs, params.
- Lineage: base-seed family.
- Zero-cost proxies (`--zero-cost`): one forward+backward pass statistics (e.g. activation
  std) — the single most predictive feature group.

**Validation** the professor asked for: hold-out **Spearman/Kendall** rank fidelity +
predicted-vs-actual scatter, under **two splits** — KFold (deployment-realistic: production
screens children of the same seeds) and **LOGO** (leave-one-base-seed-out: generalisation to
unseen families). Trains on all discovered arms (ablation A0–D0 **and** every production run
R3/R4/R5… — auto-discovered, so new families are always included).

## V1 vs V2 — the target-fidelity problem

| | **V1 (live)** | **V2 (candidate)** |
|---|---|---|
| Target | ep5 proxy PSNR | 50-epoch PSNR |
| Labels | plentiful (~350) | scarce (accruing from promotions) |
| Signal | **low** — ep5 spread is ~0.9 dB (mostly undertraining noise) | **high** — real final quality |
| Live rank fidelity | **~0.05 Spearman** (barely ranks in production) | expected much higher once trained |

**The core issue:** ep5 is a poor *target* — its spread is dominated by undertraining noise, so
even a well-fit V1 ranks production candidates near-randomly (cross-validated Spearman looks OK
at 0.54–0.69 because CV is on the ablation distribution, but *live* it is ~0.05). The real fix
is to predict the signal we actually care about: **50-epoch PSNR**. Those labels are a **free
byproduct** of the full-train-gated promotion stage the pipeline already runs — no extra GPU;
they just accumulate (`accumulate_labels.py`).

## Scale-safety: why a live V2 is calibrated

The funnel assigns skipped candidates a *predicted fitness that competes with the measured ep5
fitness* of trained candidates, on the same scale. A raw V2 predicts ep50 (~1–2 dB higher), so
dropping it in unchanged would make skipped candidates spuriously outrank trained ones and
**corrupt selection**. Fix: on activation, V2's output is **calibrated back to the ep5 scale**
(`ep5 ≈ a·ep50 + b`, fit from paired labels) and stored in the model
(`predict.py` applies it). Ranking is unchanged (the map is monotone); only the absolute scale
is matched, so a live V2 is a safe drop-in.

## Better ⇒ live: the activation gate

`predictor/activate_predictor.py` makes promotion **data-driven and automatic**. Each retrain
trains both V1 and V2, then:

```
activate V2  iff  n_v2 ≥ 60  AND  LOGO_Spearman(V2) ≥ max(0.45, LOGO(V1) + 0.05)
```

If V2 clears the bar it is calibrated and copied to the live `predictor.joblib` (V1 backed up
to `predictor_v1_prev.joblib`); otherwise V1 stays live and nothing regresses. Because ep50
labels accrue with every promotion, **V2 automatically goes live the moment it is genuinely
better** — no manual switch.

## Auto-retrain integration

Promotion and retraining are first-class pipeline stages, not side scripts. `run_seed_ga.py`
supports `--promote` (full-train top-K → grow seed pool) and `--retrain-predictor`, which runs:

```
accumulate_labels.py  →  train V1 (ep5)  →  train V2 (psnr50)  →  activate_predictor.py
```

so the next run's funnel is always calibrated to the current corpus and running the best
available predictor.

## Why it matters (thesis framing)

1. **Compute** — the predictor roughly halves proxy-training cost; on a single GPU that is the
   difference between weeks and months of search.
2. **Selection quality** — moving to the 50-epoch target (V2) makes the funnel optimise *final
   quality* instead of a noisy proxy, which is what actually determines whether the search
   surfaces champion architectures.
3. **A named deliverable** — a validated learned PSNR surrogate with honest rank-fidelity
   reporting (KFold + LOGO, scatter) is an explicit thesis requirement.

## Current status (2026-07-25)

- V1 retrained on the full corpus (all arms + new NAFNet/Restormer families).
- V2 capability built; a preliminary V2 is trained on the ~29 paired 50-epoch labels available
  so far. It is **not yet activated** — needs ≥60 labels to be trusted; the growth loop's
  full-trains are accruing them. V1 remains live until V2 earns promotion.
- Files: `predictor/{train_predictor,predict,activate_predictor}.py`,
  `predictor.joblib` (live), `predictor_v2.joblib` (candidate), `metrics*.json`,
  `accumulate_labels.py`, `predictor_labels.csv`.
