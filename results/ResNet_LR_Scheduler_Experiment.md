# ResNet Learning Rate Scheduler Comparison Results

**Experiment Date:** March 2026  
**Architecture:** ResNet  
**Dataset:** CIFAR-10  
**Training Epochs:** 5  
**Models Tested:** 10

## Executive Summary

This experiment evaluates 10 ResNet models with different learning rate schedulers to demonstrate the LR scheduler generation system. Out of 10 models, **8 completed successfully** with accuracy ranging from 65.14% to 83.39%.

### Top 3 Performers

1. **CosineWarmRestarts_T5** - 83.39% accuracy
2. **CosineAnnealingLR_T20** - 81.01% accuracy  
3. **ExponentialLR_g09** - 79.27% accuracy

**Average Accuracy:** 76.48% (across 8 completed models)

## Detailed Results

| Rank | Model ID | Scheduler | Accuracy | Status |
|------|----------|-----------|----------|--------|
| 1 | PR_0009 | CosineWarmRestarts_T5 | 83.39% | ✅ Completed |
| 2 | PR_0006 | CosineAnnealingLR_T20 | 81.01% | ✅ Completed |
| 3 | PR_0004 | ExponentialLR_g09 | 79.27% | ✅ Completed |
| 4 | PR_0007 | MultiStepLR_m5_10_g05 | 77.61% | ✅ Completed |
| 5 | PR_0005 | CosineAnnealingLR_T10 | 77.50% | ✅ Completed |
| 6 | PR_0002 | StepLR_s5_g03 | 75.09% | ✅ Completed |
| 7 | PR_0001 | StepLR_s10_g05 | 72.85% | ✅ Completed |
| 8 | PR_0010 | LinearLR_sf01 | 65.14% | ✅ Completed |
| - | PR_0003 | ConstantLR_f03 | N/A | ❌ Failed (checksum duplicate) |
| - | PR_0008 | PolynomialLR_p2 | N/A | ❌ Failed (disk full crash) |

## PR Review Artifacts (Code + Standard Pipeline Outputs)

To make review complete, this PR includes the generated model code and evaluation artifacts directly in the repository:

- **Artifact root:** `ab/gpt/brute/lr/pr_models_artifacts/`
- **Per model folders:** `PR_0001` ... `PR_0010`
- **Included code/config files per model:** `new_nn.py`, `hp.txt`, `model_meta.txt`
- **Included pipeline outputs per completed model:** `eval_info.json`, `1.json`, `2.json`, `3.json`, `4.json`, `5.json`, `error.txt`
- **Global summaries:** `ab/gpt/brute/lr/pr_models_artifacts/summary.json` and `ab/gpt/brute/lr/pr_models_artifacts/summary.csv`

This keeps all generated code and NNGPT standard-pipeline statistics visible in a single PR for direct inspection.

## Key Findings

### 1. Cyclic Schedulers Excel
**CosineWarmRestarts_T5** achieved the best performance (83.39%), showing that cyclical learning rate patterns with warm restarts help escape local minima during training.

### 2. Cosine Annealing Effectiveness
Both **CosineAnnealingLR** variants (T20 and T10) performed well, ranking 2nd (81.01%) and 5th (77.50%) respectively. The longer period (T20) achieved higher accuracy.

### 3. Exponential Decay Remains Competitive
**ExponentialLR_g09** secured 3rd place (79.27%), demonstrating that gradual exponential decay continues to be a reliable scheduling strategy.

### 4. Linear Decay Underperforms
**LinearLR_sf01** ranked last among completed models (65.14%), suggesting linear learning rate reduction may be too aggressive for this architecture/dataset combination.

### 5. Step Schedulers Show Moderate Performance
**StepLR** variants showed middle-tier performance (72.85% - 75.09%), with more frequent decay steps (s5) outperforming less frequent ones (s10).

## Technical Details

### Model Configuration
- **Base Architecture:** ResNet (from nn-dataset/ab/nn/loader/cifar-10.py)
- **Weight Decay:** 0.0001 (consistent across all models)
- **Optimizer:** Adam
- **Input Size:** 32x32x3 (CIFAR-10 standard)
- **Batch Size:** 128
- **Training Duration:** 5 epochs

### Scheduler Parameters
Each scheduler was configured with architecture-specific hyperparameters generated via `schedulers.py`:

- **StepLR_s10_g05**: step_size=10, gamma=0.5
- **StepLR_s5_g03**: step_size=5, gamma=0.3
- **ConstantLR_f03**: factor=0.3 (failed)
- **ExponentialLR_g09**: gamma=0.9
- **CosineAnnealingLR_T10**: T_max=10
- **CosineAnnealingLR_T20**: T_max=20
- **MultiStepLR_m5_10_g05**: milestones=[5,10], gamma=0.5
- **PolynomialLR_p2**: power=2 (failed)
- **CosineWarmRestarts_T5**: T_0=5
- **LinearLR_sf01**: start_factor=0.1

### Evaluation Infrastructure
- **Evaluation Pipeline:** NNEval → Eval.evaluate() → Train.train_new()
- **Hardware:** NVIDIA RTX 4090 (24GB VRAM)
- **Framework:** PyTorch 2.10.0+cu126
- **Database:** SQLite (db/ab.nn.db)

## Generator System

This experiment demonstrates the automated LR scheduler generation system implemented in `ab/gpt/brute/lr/schedulers.py`. The system:

1. **Reads source code** from nn-dataset architectures
2. **Injects scheduler configurations** with parameterized hyperparameters
3. **Generates hp.txt files** with architecture-specific defaults
4. **Supports 30 architectures × 25 schedulers × 7 weight decay values** = 5,250 possible model variants

### Code Reference

The LR scheduler generation system is already merged in the main repository:

- **Generator Code:** [ab/gpt/brute/lr/schedulers.py](https://github.com/ABrain-One/nn-gpt/blob/main/ab/gpt/brute/lr/schedulers.py) (728 lines)
  - Source code injection engine
  - Architecture-specific hyperparameter generation
  - 25 scheduler variants with configurable parameters
  
- **Statistics Module:** [ab/gpt/brute/lr/stats.py](https://github.com/ABrain-One/nn-gpt/blob/main/ab/gpt/brute/lr/stats.py) (236 lines)
  - Post-evaluation analysis
  - Performance metrics aggregation

- **Merged via:** PR #87 and PR #100
- **Repository:** https://github.com/ABrain-One/nn-gpt

## Recommendations

Based on these results, for ResNet on CIFAR-10:

1. **Preferred:** CosineWarmRestarts with T_0=5 for maximum accuracy
2. **Alternative:** CosineAnnealingLR with T_max=20 for stable convergence
3. **Reliable Baseline:** ExponentialLR with gamma=0.9
4. **Avoid:** LinearLR schedulers for this architecture/dataset combination

## Failed Models Analysis

- **PR_0003 (ConstantLR_f03):** Training halted due to checksum duplicate detection
- **PR_0008 (PolynomialLR_p2):** Crashed during disk space exhaustion (partition 100% full, resolved by clearing 26GB cache)

## Future Work

- Extend training to 50+ epochs for convergence analysis
- Test scheduler combinations (e.g., warmup + cosine annealing)
- Expand to other architectures (VGG, DenseNet, MobileNet)
- Cross-dataset validation (CIFAR-100, ImageNet-subset)

---

**Generated by:** Learning Rate Scheduler Brute-Force Search System  
**Contact:** Hafsa Mateen (Hafsa70)  
**Last Updated:** March 8, 2026
