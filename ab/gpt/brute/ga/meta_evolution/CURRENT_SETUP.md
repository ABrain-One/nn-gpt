# CloneScience Meta-Evolution Pipeline — Current Setup

> **Last Updated:** 2026-09-27  
> **Status:** All 8 jobs (Baseline CIFAR-10/100, Meta CIFAR-10/100 × Qwen/Mistral/DeepSeek) can run simultaneously on dgx-h100.

---

## Active Monkeypatches (applied in runner scripts)

### 1. Database Download Bypass
- **What:** `nn_dataset.data = lambda *args, **kwargs: pd.DataFrame(columns=['nn_id'])`
- **Why:** Prevents Eval.py from downloading the massive remote SQLite database on every evaluation.
- **Where:** Top of `baseline_run_ga.py` and `run_fractal_evolution.py`

### 2. PID Race Condition Fix
- **What:** `train_runtime.out = f"out_{DATASET}_{uuid.uuid4().hex[:8]}"`
- **Why:** Kubernetes pods often share PID 99, causing `Eval.py` to collide on temp directories. UUID makes each evaluation's directory unique.
- **Where:** Top of `baseline_run_ga.py` and `run_fractal_evolution.py`

### 3. DB Write Bypass
- **What:** `DB_Write.save_results = lambda *args, **kwargs: 1`
- **Why:** Prevents SQLite locking errors when `save_to_db=True` is needed for per-epoch JSON output.
- **Where:** Inside `fitness_function()` in both runner scripts

### 4. LLM Predictor Temperature Override
- **What:** `httpx.Client.post` is monkeypatched to force `temperature=0.8` in the request payload.
- **Why:** The upstream `acc_client.py` (which we cannot edit) hardcodes `temperature=0.0`, causing repetitive `.12` decimal outputs. The monkeypatch intercepts the HTTP request and overrides the temperature before it reaches the vLLM server. **Changed from 0.9 to 0.8 on 2026-09-18** — 0.9 was causing 390+ syntax errors in generated code; 0.8 is the established working default used everywhere else (`modular/llm_loader.py`, `modular/rl_mutation.py`).
- **Where:** Top of `baseline_run_ga.py` and `run_fractal_evolution.py`

### 5. LLM Predictor Elitism Disabled
- **What:** GA `ultimate_fitness` is hard-locked to strictly use `epoch_accs[3]` (3rd-epoch true accuracy).
- **Why:** The LLM Predictor was causing a broken fitness landscape. When it succeeded it scored out of 100%, but when it failed it fell back to 1-epoch accuracy (out of 50%), causing the GA to evolve random mediocre models instead of the true elites. The predictor is still queried for observation, but ignored for GA selection.
- **Where:** `baseline_run_ga.py` and `run_fractal_evolution.py`

---

## Evaluation Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Epochs | 3 | 3-epoch proxy for LLM predictor input |
| Batch Size | 64 | Increased from 32 for more signal per step |
| Transform | `norm_32_flip` | Native CIFAR resolution (no 256 resize) |
| Max Batches | None | Full dataset (782 batches for CIFAR-10) |
| LLM Predictor | Qwen via vLLM at `132.187.14.67:30031` | Temperature forced to 0.8 via monkeypatch |
| Predictor Max Epochs | 50 | The LLM predicts final accuracy assuming 50 epoch budget |

---

## Reward Function & Evaluation (Fitness)

In this evolutionary pipeline, the "Reward" (or fitness score) assigned to each architecture is calculated through a hybrid proxy-and-prediction system:

1. **Base Signal (3-Epoch Proxy):** The architecture is trained for 3 epochs. This provides a fast, noisy signal of its learning capability.
2. **LLM as a Reward Estimator:** Instead of using the raw 3-epoch accuracy (which heavily biases toward fast-learners rather than deep-capacity models), the 3-epoch trajectory is fed into an LLM (Qwen). The LLM acts as a zero-shot regression model (value network), predicting the **asymptotic 50-epoch accuracy**.
3. **Ultimate Reward:** The predicted 50-epoch accuracy is returned to the Genetic Algorithm as the final fitness score. The GA then uses tournament selection based on this reward to breed the next generation.
4. **Memoization (Cache Hits):** Because evaluating the reward function is extremely expensive (GPU training), the system computes a SHA256 checksum of the generated PyTorch code. If the exact architecture is proposed again, it instantly receives its cached reward, bypassing the training and LLM steps.

### Meta-Evolution Reward (recommomandation #6, 2026-09-18)

The meta-evolution loop uses `rl_rewards.py::calculate_meta_reward()` to score GA mutations. The reward structure was changed to prioritize **fitness improvement** over novelty:

- **Primary signal:** `fitness_improvement = max(0, current_score - best_ever_score)` — SOTA-relative delta, scaled by 2.0
- **Secondary bonus:** `archive_novelty * 0.5` — Reduced from dominant (weight 2.0) to minor bonus
- **Tertiary bonus:** `top3_mean / 100 * 0.5` — Reduced from weight 1.5 to minor bonus
- **Stagnation penalty:** `-2.0` if no improvement and no novelty

Previously, `archive_novelty` was the dominant signal and `current_score`, `best_ever_score`, `baseline_score` were dead parameters (passed but never used). Now all parameters are properly utilized.

---

## Hallucination Guardrails

If the LLM Predictor returns an accuracy > 3x the **true 3-epoch accuracy**, the prediction is discarded and the true accuracy is used as fitness instead. This prevents the GA from being misled by wildly inaccurate predictions.

**Bug Fix (2026-09-18, recommomandation #8):** The guardrail was comparing against `epoch_accs[1]` (1-epoch accuracy) instead of `epoch_accs[3]` (3-epoch accuracy). After a plan change on 2026-09-15 set `true_fitness = epoch_accs[1]`, the guardrail threshold was trivially exceeded (since 1-epoch accuracy is only 2-10%), causing a **97-99% hallucination rate**. Fixed by using `epoch_accs[3]` for the guardrail comparison while keeping `epoch_accs[1]` for `_true.jsonl` logging.

---

## Logging System

### Dual JSONL Output
Each run produces two JSONL files side-by-side:
- `*_evaluations_DATASET_TIMESTAMP.jsonl` — Contains the **ultimate fitness** used by the GA (predicted accuracy if predictor succeeded, true accuracy otherwise)
- `*_evaluations_DATASET_TIMESTAMP_true.jsonl` — Contains the **raw 1-epoch accuracy** (ground-truth baseline for hallucination guardrail)

### Failed Evaluations
Evaluations that crash or return `0.0` accuracy are **filtered out and silently ignored**. They are not written to the JSONL logs to keep the visualization and statistics clean.

### Fitness Cache
The `fitness_cache` dictionary stores tuples `(ultimate_fitness, true_fitness)` so that when a duplicate architecture is encountered within the **same run**, the cached fitness is reused instead of re-evaluating.

**Key behavior (recommomandation #2, 2026-09-17):** `fitness_cache` starts **empty** at each run. The `_load_existing_checksums()` pre-loading function has been **disabled** — it no longer scans the stats/ directory at module load time. This prevents stale cached fitness from previous runs (with different GA configurations) from contaminating the current run's selection pressure. The cache still builds incrementally during the run as architectures are evaluated.

### Per-Attempt Fitness History (recommomandation #4, 2026-09-17)
Each meta-attempt's `fitness_history` (one best fitness per GA generation) is now collected and merged into `meta_attempt_histories_{DATASET}.json` after all attempts complete. The original `fitness_history` in `evolved_results_*.json` remains a flat list for backward compatibility with `compare_ga_intelligence.py`. This enables tracking GA progress generation-by-generation across all meta-attempts, making the per-generation evolutionary trajectory visible for scientific analysis.

### Population-Level Dedup (recommomandation #7, 2026-09-18)
A `_generation_seen = set()` tracks which architecture checksums have been evaluated within the current generation. If a duplicate chromosome is encountered, it receives a `-1.0` fitness penalty instead of being evaluated again. This prevents the GA from wasting compute on identical architectures within a single generation. The existing `fitness_cache` continues handling cross-generation dedup normally.

**Applies to:** `run_fractal_evolution.py` (via `fitness_with_archive` wrapper) and `baseline_run_ga.py` (inside `fitness_function`).

---

## Meta-Evolution Prompting Strategy (Mutator LLM)

Instead of traditional model fine-tuning (which requires thousands of gradient updates), the meta-evolution pipeline uses **In-Context Learning (Prompt Engineering)** to dynamically "fine-tune" the Genetic Algorithm's behavior at runtime. 

When the `meta_evolver.py` decides to mutate the GA engine itself (e.g., rewriting the crossover or selection logic), it sends a comprehensive context prompt to the LLM. 

### 5. Decoupled Fine-Tuning (Recommendation #9) + Bug Fix
- **Mechanism**: The Meta Job now successfully triggers LoRA fine-tuning whenever the Mutator LLM produces code that either (a) beats the baseline score, OR (b) discovers novel architectures (archive expansion).
- **Bug Fix**: A previous ordering bug in `archive_expanded` falsely evaluated to False, skipping all fine-tuning. This has been resolved so the Mutator LLM will properly capture and fine-tune on exploratory successes.

The LLM is now rewarded and fine-tuned if it either beats the baseline accuracy OR discovers novel/diverse architectures that expand the Map of Elites archive. This encourages exploration and jumpstarts the LoRA loop much earlier. The EMA `baseline_score` is only dragged up if the SOTA was actually beaten, preventing exploratory runs from ruining the baseline.

Additionally, `BENCH_GENS` and `BENCH_POP` dynamically read from the Kubernetes `GENERATIONS` and `POPULATION_SIZE` environment variables (defaulting to 5 and 20 respectively) so the computational budget for the LLM evaluation benchmarks matches the Baseline evaluation.

### The Prompt Structure
The `BASE_PROMPT_TEMPLATE` constructs a highly structured prompt containing:
1. **Feedback from Recent Failures:** The LLM is provided with the error traces and low fitness scores of its previous unsuccessful code mutations, forcing it to learn from its mistakes.
2. **Search Space & Baseline:** The full architectural `SEARCH_SPACE` dictionary and the current best baseline chromosome.
3. **Skeletonized GA Context:** To save token space, the AST parser extracts only the relevant parts of the GA script (skeletonizing heavy methods like checkpointing) so the LLM understands the class structure (`self.population`, etc.).
4. **Target Function:** The specific GA operator to rewrite (e.g., `combine_genes`, `mutate_gene`, or `select_competitor`), alongside targeted instructions (e.g., "force the GA operators to break out of local minima").

### Code Injection
The LLM generates pure Python code. The pipeline extracts the function via AST parsing, verifies it is syntactically valid Python, and dynamically injects it into the running GA engine to guide the evolution of the next generation of architectures.

---

## Pipeline Directory Structure

```
meta_evolution/
├── baseline_run_ga.py          # Baseline GA runner (no LLM mutation)
├── run_fractal_evolution.py    # Meta-evolution runner (LLM-guided mutation)
├── baseline_visualization.py   # Generates accuracy/time/diversity plots
├── meta_visualization.py       # Generates meta-evolution plots
├── FractalNet_evolvable_backbone.py  # Search space definition
├── genetic_algorithm_baseline.py     # Baseline GA engine
├── meta_evolver.py                   # LLM-guided mutation engine
├── cifar10_pipeline/
│   ├── architectures/          # Saved .py model files
│   ├── stats/                  # Per-model per-epoch JSON stats
│   ├── logs_cifar10/
│   │   ├── Baseline/           # Baseline JSONL logs + pod logs
│   │   └── qwen/               # Meta-evolution JSONL logs + pod logs
│   └── visualizations/         # Generated plots
├── cifar100_pipeline/
│   ├── (same structure as cifar10_pipeline)
├── base_evol_tune_nngpt_cifar10.json   # K8s job spec
├── base_evol_tune_nngpt_cifar100.json  # K8s job spec
├── meta_evol_tune_nngpt_cifar10.json   # K8s job spec
├── meta_evol_tune_nngpt_cifar100.json  # K8s job spec
├── CURRENT_SETUP.md            # THIS FILE (always up to date)
└── CHANGELOG/                  # Sequential change logs
```

---

## Visualization Safeguards

`baseline_visualization.py` gracefully handles:
- **Single generation runs:** X-axis set to `(0, 2)` instead of crashing on `(1, 1)`
- **Flat accuracy lines:** Y-axis dynamically padded by ±5% when accuracy range < 1.0%

---

## Kubernetes Job Concurrency & Node Affinity

### Node Affinity (Updated 2026-09-27)
All 8 job manifests use **strict** node affinity (`requiredDuringSchedulingIgnoredDuringExecution`) targeting the `dgx-h100` node exclusively. Jobs will **never** fall back to non-H100 nodes — they wait in `Pending` state until an H100 GPU becomes available.

Each job is an independent single-replica Kubernetes `Job` requesting `nvidia.com/gpu: 1`. Scheduling is first-come-first-served on an individual basis — whichever job gets a free GPU starts immediately; others queue independently.

### Concurrency Safety
All 8 jobs can safely run simultaneously because:
1. UUID monkeypatch guarantees unique temp directories per evaluation
2. CIFAR-10 and CIFAR-100 pipelines write to separate `cifar10_pipeline/` and `cifar100_pipeline/` directories
3. Baseline and Meta pipelines write to **separate stats subdirectories** within each pipeline: `stats/baseline/` and `stats/meta/`. This prevents cache cross-contamination where the meta-evo inherits the baseline's 4,570+ pre-loaded checksums into its `fitness_cache`, which would make the GA blind to actual fitness differences.
4. Pod startup uses `[ -f /a/mm/ab/nn/train.py ]` existence check to skip redundant `cp -r` operations
5. Model-specific log isolation: each model (qwen/mistral/deepseek/Baseline) writes to its own subdirectory under `logs_cifar10/` and `logs_cifar100/`

