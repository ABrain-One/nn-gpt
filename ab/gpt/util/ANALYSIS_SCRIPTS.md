# Analysis Scripts — Usage Guide

These scripts analyse the output directories produced by NNGPT-CL experiments.
All commands are run from the **repo root** (`nn-gpt/`).  They use only the
`out*/` directories and write results to a local sub-folder; they never modify
any experiment data.

Scripts live in `ab/gpt/util/analize/` (analysis) and `ab/gpt/util/vis/` (visualization).

---

## 1. `analyze_output_sizes.py` — output token / character statistics

Measures characters and token counts per full_output.txt across epochs.
Useful for confirming the token-budget effect shown in paper Table 2.

```bash
# latest epoch only
python ab/gpt/util/analize/analyze_output_sizes.py

# specific epoch
python ab/gpt/util/analize/analyze_output_sizes.py --epoch 3

# compare two epochs side-by-side
python ab/gpt/util/analize/analyze_output_sizes.py --compare-epochs 0 5

# detailed per-probe table for epochs 0, 1, 2
python ab/gpt/util/analize/analyze_output_sizes.py --compare-epochs-detailed 0 1 2

# point to a different out/ directory
python ab/gpt/util/analize/analyze_output_sizes.py --base-path ./out_109_expB
```

Default `--base-path` is `./out` (the `out/` directory under the repo root).

---

## 2. `statistical_analysis.py` — per-epoch validity and accuracy stats

Loads `full_output.txt` files and computes:
- extraction rate (fraction of probes that produced parseable code)
- mean / median LEMUR accuracy per epoch
- epoch-over-epoch deltas

```bash
# all epochs, table format
python ab/gpt/util/analize/statistical_analysis.py

# specify experiment out/ dir
python ab/gpt/util/analize/statistical_analysis.py --base-path ./out_109_expA

# CSV output
python ab/gpt/util/analize/statistical_analysis.py --format csv > stats.csv
```

Default `--base-path` is `./out`.

---

## 3. `analyze_phase0.py` — baseline (A0) sweep analysis

Used for Phase 1 (token budget sweep, paper Table 2).  Reads `out_phaseA_*`
directories under the repo root and writes a CSV + stats markdown to
`phase0_output/`.

```bash
# run from repo root — finds out_phaseA_* automatically
python ab/gpt/util/analize/analyze_phase0.py

# custom base and output dirs
python ab/gpt/util/analize/analyze_phase0.py \
    --base_dir . \
    --out_dir my_phase0_output

# skip plot generation (faster)
python ab/gpt/util/analize/analyze_phase0.py --no_plots
```

Output files (in `phase0_output/` by default):
- `phase0_master.csv`  — per-probe raw rows
- `phase0_stats.md`    — summary statistics
- `phase0_plots/`      — matplotlib figures (unless `--no_plots`)

---

## 4. `analyze_phase1.py` — closed-loop sweep analysis

Reads `out_phaseA_*` directories (same layout as phase0) and produces
accuracy-vs-epoch tables.  Also used for Phase 1 multi-budget comparisons.

```bash
python ab/gpt/util/analize/analyze_phase1.py

# no plots
python ab/gpt/util/analize/analyze_phase1.py --no_plots

# custom dirs
python ab/gpt/util/analize/analyze_phase1.py \
    --base_dir . \
    --out_dir my_phase1_output
```

Output files (in `phase1_output/` by default):
- `phase1_master.csv`
- `phase1_summary.csv`
- `phase1_stats.md`
- `phase1_plots/`

---

## 5. `compare_outputs.py` — diff two full_output.txt files

Shows differences in code structure between two generated `new_nn.py` outputs.

```bash
python ab/gpt/util/analize/compare_outputs.py \
    out_109_expA/nngpt/llm/epoch/A0/synth_nn/B0/full_output.txt \
    out_109_expA/nngpt/llm/epoch/A5/synth_nn/B0/full_output.txt
```

Takes two positional path arguments; no path-default configuration needed.

---

## 6. `plot_epoch_analysis.py` — epoch trajectory plots

Generates matplotlib charts showing accuracy and extraction rate across epochs.
Useful for producing the Experiment B/C trajectory plots in the paper.

```bash
# auto-detect all epochs in ./out
python ab/gpt/util/vis/plot_epoch_analysis.py

# explicit epoch list
python ab/gpt/util/vis/plot_epoch_analysis.py --epochs 0 1 2 3 4 5

# epoch range
python ab/gpt/util/vis/plot_epoch_analysis.py --start-epoch 0 --end-epoch 10

# point to a specific experiment directory
python ab/gpt/util/vis/plot_epoch_analysis.py --base-path ./out_109_expC
```

Figures are written next to the script unless `--output-dir` is specified.

---

## Typical workflow after a run

```bash
# 1. Check output sizes (confirm token budget effect)
python ab/gpt/util/analize/analyze_output_sizes.py

# 2. Validity + accuracy summary
python ab/gpt/util/analize/statistical_analysis.py

# 3. Full epoch trajectories
python ab/gpt/util/vis/plot_epoch_analysis.py

# 4. Phase 1 budget sweep (run after multiple budget experiments)
python ab/gpt/util/analize/analyze_phase0.py && python ab/gpt/util/analize/analyze_phase1.py
```

---

## See also

- `ab/gpt/NNGPT_CL_USAGE.md` — experiment entrypoints and flag reference.
- `ab/gpt/util/eval/static_validity.py` — AST-only structural checker (no execution).
- `ab/gpt/util/eval/nn_cost.py` — param count and FLOPs for any generated `new_nn.py`.
