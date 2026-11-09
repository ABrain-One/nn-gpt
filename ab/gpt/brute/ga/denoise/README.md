# Denoise NAS — GA architecture search for mobile image denoising

Genetic-algorithm Neural Architecture Search that evolves **image-denoising** architectures
for **resource-constrained edge (mobile) devices**. It mutates and recombines real
`DenoiseUnet*` seed architectures, screens candidates with a short training proxy, filters for
on-device deployability, and reports the winners against quality (PSNR) and cost (MACs / TFLite
latency) axes.

Built on the [ABrain-One](https://github.com/ABrain-One) **LEMUR** stack — it reuses
`nn-dataset` (datasets, training, metrics, the `DenoiseUnet*` seeds) and the `nn-gpt`
`ab/gpt/brute` mutation engine. This directory (`ab/gpt/brute/ga/denoise/`) is the denoise
GA layer on top of them.

> Research code accompanying the thesis *"Towards Automated Image Denoising Model Generation
> for Resource-Constrained Edge Systems."* Reproducible at the architecture/result level;
> the GA and training are not bit-seeded (see `thesis/REPRODUCIBILITY.md`).

---

## 1. Repository layout

The pipeline needs **two sibling repos** under one root:

```
<THESIS_ROOT>/
├── nn-dataset/        # LEMUR: datasets, Train.py, metrics, DenoiseUnet* seeds, ab.nn.db
│   └── ab/nn/
│       ├── nn/                 DenoiseUnet1F..10F, LiteDenoiseNet  (the seeds)
│       ├── loader/             denoise.py (MAI), sidd.py (SIDD)   (datasets)
│       ├── metric/psnr.py      PSNR metric (stored ×48 = real dB)
│       └── api.py              check_nn(...)  ← the training/eval entry point
└── nn-gpt/            # generation engine
    └── ab/gpt/brute/
        ├── ast/mutator/        the AST mutation engine (skip/conv/kernel/... operators)
        └── ga/denoise/         ← THIS pipeline
```

`<THESIS_ROOT>` is `/home/s493053/thesis` in the current scripts. **Anyone else must re-point
that** — see [§6 Portability](#6-making-it-run-on-your-machine-important).

---

## 2. Prerequisites

- **Python 3.13** (3.11+ works), Linux, a CUDA GPU (developed on an NVIDIA L40; ~48 GB not
  required — the models are small, batch 16 @ 256×256).
- **PyTorch 2.8** + CUDA. Core deps: `numpy`, `pandas`, `scikit-learn`, `joblib`,
  `matplotlib`, `datasketch` (similarity), and for the TFLite step `litert_torch` +
  `ai_edge_litert`.
- The `nn-dataset` package importable (its editable install / `PYTHONPATH`).

## 3. Setup

```bash
# 1. clone both repos side by side under YOUR root
export THESIS_ROOT=$HOME/thesis          # <-- pick your own
mkdir -p "$THESIS_ROOT" && cd "$THESIS_ROOT"
git clone https://github.com/ABrain-One/nn-dataset
git clone https://github.com/ABrain-One/nn-gpt

# 2. one virtualenv (nn-dataset's) drives everything
cd "$THESIS_ROOT/nn-dataset"
python -m venv .venv && source .venv/bin/activate
pip install -e .                          # installs ab.nn + core deps
pip install datasketch litert_torch ai_edge_litert scikit-learn joblib matplotlib

# 3. sanity check
PYTHONPATH="$THESIS_ROOT/nn-dataset" python -c "import ab.nn.api as a; print('nn-dataset OK')"
```

### Datasets
- **MAI (search dataset)** — auto-loaded by `ab/nn/loader/denoise.py` via LEMUR's dataset
  fetch on first use. This is what the GA searches on.
- **SIDD (validation dataset, optional)** — download SIDD-Medium sRGB and place it at
  `nn-dataset/sidd/` (loader: `ab/nn/loader/sidd.py`, expects `original/` noisy + `denoised/`
  clean; it makes a deterministic 90/10 split). Only needed for the winner-validation step.

---

## 4. Run the search

The full production configuration is one script. From `nn-dataset` with the venv active:

```bash
cd "$THESIS_ROOT/nn-dataset"
bash "$THESIS_ROOT/nn-gpt/ab/gpt/brute/ga/denoise/launch_production.sh"
```

What it runs (`run_seed_ga.py`): a **pop-50 / 30-generation** GA over 9 seeds, block crossover
+ skip/scale mutation operators, an accuracy-predictor pre-screen funnel, and an edge
deployability filter (rejects non-TFLite-convertible or over-MAC-cap children). Fitness =
`eff = PSNR − 0.3·ln(MACs / budget)` at a 5-epoch proxy. Key knobs (all CLI flags on
`run_seed_ga.py`):

| Flag | Meaning |
|---|---|
| `--seeds ...` | starting architectures (from `ab/nn/nn/`) |
| `--pop-size / --gens` | population / generations |
| `--crossover block` | block-level source recombination |
| `--skip-ops --skip-prob` | enable skip-connection operators |
| `--predictor --predict-keep 0.5` | train only the top-ranked half each gen, surrogate the rest |
| `--edge-filter --mac-cap 9053` | deployability + compute-budget gate (9053 MMac = 9F's footprint) |
| `--epoch 5` | proxy training epochs per candidate |
| `--save-to-db` | persist evaluated models to the local LEMUR SQLite DB |

**Smaller smoke test** (minutes, proves the wiring end-to-end):
```bash
PYTHONPATH="$THESIS_ROOT/nn-dataset" python \
  "$THESIS_ROOT/nn-gpt/ab/gpt/brute/ga/denoise/run_seed_ga.py" \
  --seeds DenoiseUnet7F LiteDenoiseNet --pop-size 6 --gens 2 --epoch 2 \
  --checkpoint /tmp/smoke_ckpt.pkl --cache /tmp/smoke_cache.json \
  --models-dir /tmp/smoke_models --results /tmp/smoke_results.json
```

### Resumability (long runs)
The GA checkpoints after **every generation** (`*_checkpoint.pkl`) and caches every evaluated
model (`*_cache.json`), so a restart **continues from the last completed generation, never from
gen 0**. For unattended runs use the self-healing supervisor, which re-launches on crash until
the run prints `=== DONE`:
```bash
nohup bash "$THESIS_ROOT/nn-gpt/ab/gpt/brute/ga/denoise/production_supervisor.sh" \
  >> production_supervisor.log 2>&1 &
```
> **Do not delete** `*_checkpoint.pkl` / `*_cache.json` — that forces a gen-0 restart.
> Do not run two production processes against the same namespace at once.

---

## 5. After the search — validate & report the winners (in order)

| Step | Script | Produces |
|---|---|---|
| 1. Full-train top ~15–20 @50ep | `fulltrain_topk.py` / `launch_fulltrain.sh` | real PSNR (the promotion authority; the ep5 proxy is not) |
| 2. Validate on SIDD | `sidd_sweep.py` | cross-dataset PSNR (credibility: MAI has little headroom, SIDD ~+9 dB) |
| 3. **TFLite / measured latency** | `to_tflite.py` | convert? · real MACs · tflite_KB · **latency_ms** · torch-vs-tflite parity_dB |
| 4. Refresh label store | `accumulate_labels.py` | `predictor_labels.{json,csv}` for retraining the predictor |
| 5. Grow the seed pool | `promote.py` | promotes distinct deployable winners as next-run seeds |

**TFLite latency caveat:** `to_tflite.py` measures real XNNPACK inference time on the **desktop
CPU**, not a phone — it *ranks* models correctly and correlates with phone cost, but the
absolute ms are not phone ms. True on-device numbers need a physical device (ADB
`benchmark_model`) or a hardware-calibrated predictor (nn-Meter).

---

## 6. Making it run on YOUR machine (important)

The scripts currently **hardcode** `/home/s493053/thesis` in ~60 places. Until the paths are
parameterized (queued cleanup), re-point them to your own root on your clone:

```bash
cd "$THESIS_ROOT/nn-gpt/ab/gpt/brute/ga/denoise"
grep -rl '/home/s493053/thesis' . | grep -v _temp_model \
  | xargs sed -i "s#/home/s493053/thesis#$THESIS_ROOT#g"
```

This edits only *your* checkout and makes every script portable. (The planned proper fix is a
single `THESIS_ROOT` env var read at the top of each script; tracked in the Run 2 punch-list.)

---

## 7. Directory map (key files)

**Engine / generation**
- `ga_core.py` — generic GA engine (selection, crossover, elitism, checkpointing)
- `run_seed_ga.py` — denoise entry point: seeds → mutate/recombine → `check_nn` proxy → rank
- `seed_mutation.py`, `block_crossover.py`, `skip_operators.py` — the operators
- `edge_deploy.py`, `edge_cost.py`, `to_tflite.py` — deployability + cost + TFLite conversion

**Analysis / selection**
- `predictor/` + `track_predictor.py` — the accuracy predictor (surrogate) and its live tracking
- `functional_similarity.py` / `functional_diversity.py`, `skip_inventory.py` — model-diversity analysis
- `fulltrain_topk.py`, `sidd_sweep.py`, `generalize_sidd.py`, `promote.py`, `accumulate_labels.py` — the wrap-up tools

**Ops / launchers**
- `launch_production.sh` — the RUN 1 configuration (canonical run command)
- `production_supervisor.sh` — crash-resilient wrapper; `launch_{A,B,C,D,E}0.sh` — the ablation arms
- `RESUME_PRODUCTION.txt` — one-command resume + status checks

Outputs land next to the scripts under the run namespace (default `seed_ga_P1_*`):
`*_models/` (candidate `.py`), `*_cache.json`, `*_checkpoint.pkl`, `*_results.json`,
and `production_run.log`.

---

## 8. Notes & limitations

- **PSNR units:** the metric stores PSNR × 48; divide by 48 for real dB (`PSNR_SCALE = 48.0`).
- **Proxy vs. final:** ep5-on-MAI compresses architecture differences (~1 dB spread); full
  training spreads them ~4×, SIDD ~10×. Rank on the proxy, *report* on full-train / SIDD.
- **MAC cap is a proxy budget**, not a device-measured latency limit (see §5 TFLite caveat).
- **Reproducibility:** artifact/result-level yes; not bit-exact (GA + training unseeded).
- `_temp_model_*.py` are scratch conversion artifacts — safe to ignore / clean up.
