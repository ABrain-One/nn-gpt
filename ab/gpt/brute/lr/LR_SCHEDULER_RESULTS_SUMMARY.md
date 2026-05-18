# LR Scheduler Search Evaluation Results

## Summary
- **Date**: March 26, 2026
- **Total Models Evaluated**: 1,513 LR scheduler variants
- **Target**: 2,000 models (75.65% complete)
- **Status**: Successfully evaluated with high-quality results

## Performance Statistics
- **Average Accuracy**: 0.5253
- **Min Accuracy**: 0.1251
- **Max Accuracy**: 0.8617
- **Top Performers**: lr_2777 (85.69%), lr_4361 (81.95%), lr_3566 (80.71%)

## Model Configuration
- **Architecture Variants**: 30 base architectures
- **Scheduler Types**: 25 LR scheduler types (StepLR, ExponentialLR, CosineAnnealing*, LinearLR, PolynomialLR, etc.)
- **Weight Decay Values**: 7 variants
- **Training Epochs**: 5 epochs per model
- **Batch Size**: 64
- **Dataset**: CIFAR-10 (32×32 images)
- **Task**: Image Classification
- **Metric**: Accuracy

## Results Format
Results are stored in `1513_old_results.csv` with columns:
- `model`: Model identifier (lr_0001 to lr_5250)
- `accuracy`: Final validation accuracy after 5 epochs

## Issues & Fixes
### Previous Bug (FIXED April 23, 2026)
- **Issue**: LinearLR and PolynomialLR had incorrect total_iters scaling
- **Impact**: ~50% of models failed with "ValueError: Tried to step 2501 times"
- **Solution**: Fixed in commit 96fee9bd - use epoch count (5) instead of epoch_max scaling
- **Result**: Fresh evaluation restarted with corrected code

## Next Steps
1. Continue evaluation of remaining ~3,737 models (as of April 23)
2. Merge with fresh evaluation results for complete 5,250 model dataset
3. Analyze scheduler effectiveness across architectures
4. Generate publication-ready analysis and visualizations
