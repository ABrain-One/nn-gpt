# Three Ways to Compare Epochs in analyze_output_sizes.py

## Mode 1: Simple Per-Run Comparison (Within Single Epoch)
```bash
python3 analyze_output_sizes.py --epoch 0 --compare
```
**Shows:** For epoch A0, compares B0 vs B1 vs B2 vs B3 vs B4 vs B5
**Output:** Per-run metrics table, B0→Bn reduction percentage
**Use case:** Check if output improves within a single epoch

---

## Mode 2: Aggregate Comparison (Across Epochs - Means Only)
```bash
python3 analyze_output_sizes.py --compare-epochs 0 1 2
```
**Shows:** 
- Mean metrics for A0, A1, A2 (averaged across all probes)
- Trend analysis: A0 → A1 → A2 comparison
**Output:** Compact table, percent changes between epochs
**Use case:** High-level view of fine-tuning effectiveness

---

## Mode 3: Detailed Comparison (Per-Run + Aggregate) ⭐ NEW
```bash
python3 analyze_output_sizes.py --compare-epochs-detailed 0 1 2
```
**Shows:** 
1. **For each epoch** (A0, A1, A2):
   - Detailed per-run breakdown (B0 vs B1 vs ... B5)
   - B0→Bn reduction within that epoch
2. **Then aggregate comparison**:
   - Mean metrics across epochs
   - Trend analysis (A0 → A1 → A2)
**Output:** Full detailed analysis with all levels
**Use case:** Comprehensive analysis - see both per-run AND cross-epoch trends

---

## Quick Reference

| Command | Shows | Best For |
|---------|-------|----------|
| `--epoch 0 --compare` | B0→B5 within A0 | Individual epoch debugging |
| `--compare-epochs 0 1 2` | A0 vs A1 vs A2 means | Quick cross-epoch view |
| `--compare-epochs-detailed 0 1 2` | Both per-run + aggregate | **Complete analysis** ← Recommended |

---

## Example: Compare All 6 Epochs With Full Details
```bash
python3 analyze_output_sizes.py --compare-epochs-detailed 0 1 2 3 4 5
```

**Output structure:**
```
EPOCH A0 - DETAILED PER-RUN BREAKDOWN (6 runs)
  Run    Chars    Lines    Tokens    Reasoning%    Code%
  B0     35436    450      8859      48.2%         51.8%
  B1     34890    448      8722      47.5%         52.5%
  ...
  B0 → B5: Characters ✓ REDUCED +2.5% | Tokens ✓ REDUCED +1.5%

EPOCH A1 - DETAILED PER-RUN BREAKDOWN (6 runs)
  Run    Chars    Lines    Tokens    Reasoning%    Code%
  B0     32100    425      8025      32.1%         67.9%
  B1     31500    422      7875      31.8%         68.2%
  ...
  B0 → B5: Characters ✓ REDUCED +4.2% | Tokens ✓ REDUCED +3.1%

AGGREGATE COMPARISON - MEAN METRICS ACROSS EPOCHS
  Epoch    Runs    Avg Chars    Avg Tokens    Avg Lines    Reasoning%    Code%
  A0       6       35436        8859          450           48.2%         51.8%
  A1       6       32100        8025          425           32.1%         67.9%
  A2       6       28950        7237          410           25.5%         74.5%

TREND ANALYSIS
A0 → A1:
  Characters: ✓ REDUCED -9.4% (35,436 → 32,100)
  Tokens:     ✓ REDUCED -9.4% (8,859 → 8,025)
  Reasoning:  ✓ REDUCED -16.1% (48.2% → 32.1%)
  ✓ Solutions Working (>15% reduction)

A1 → A2:
  Characters: ✓ REDUCED -9.8% (32,100 → 28,950)
  Tokens:     ✓ REDUCED -9.8% (8,025 → 7,237)
  Reasoning:  ✓ REDUCED -6.5% (32.1% → 25.5%)
  ≈ Marginal Change (within ±15%)
```

---

## When to Use Each Mode

### Use Mode 1 (`--epoch --compare`) When:
- Debugging a single epoch
- Want detailed file-by-file breakdown
- Checking if individual probes consistent
- Not comparing with other epochs

### Use Mode 2 (`--compare-epochs`) When:
- Need quick comparison across epochs
- Want clean summary table
- Not interested in per-run details
- Creating presentation slides

### Use Mode 3 (`--compare-epochs-detailed`) When: ⭐ Recommended
- Writing thesis/paper
- Need complete analysis
- Want to show both levels of detail
- Analyzing fine-tuning effectiveness
- Presenting to advisor/committee

---

## Pro Tips

1. **Always use `--compare-epochs-detailed` for comprehensive analysis**
   ```bash
   python3 analyze_output_sizes.py --compare-epochs-detailed 0 1 2 3 4 5
   ```

2. **For quick snapshot of trend:**
   ```bash
   python3 analyze_output_sizes.py --compare-epochs 0 1 2 3 4 5
   ```

3. **For thesis figures, export and analyze:**
   ```bash
   python3 analyze_output_sizes.py --compare-epochs-detailed 0 1 2 | tee epoch_analysis.txt
   # Then create graphs from the mean values
   ```

4. **Compare specific epochs only:**
   ```bash
   python3 analyze_output_sizes.py --compare-epochs-detailed 0 2 5
   # Skip intermediate epochs
   ```

---

## What the Output Shows

### Per-Run Section (For each epoch):
- **Chars**: Total characters per probe
- **Lines**: Line count per probe
- **Tokens**: Token count per probe
- **Reasoning%**: How much is reasoning vs code
- **B0 → Bn change**: Shows if solutions working within epoch

### Aggregate Section:
- **Avg Chars**: Mean characters across all probes
- **Avg Tokens**: Mean tokens (primary metric)
- **Reasoning%**: Mean reasoning percentage
- **A0 → An change**: Shows fine-tuning effectiveness

### Verdict:
- ✓ **REDUCED >15%** = Solutions working
- ✗ **INCREASED >15%** = Solutions not working
- ≈ **±15% or less** = Marginal/unclear change

---

## Example Runs

```bash
# Option 1: See everything (recommended for thesis)
cd /shared/ssd/home/b-r-duvvuri/nngpt3/nn-gpt
python3 analyze_output_sizes.py --compare-epochs-detailed 0 1 2 3 4 5

# Option 2: Compare nngpt2 and nngpt3 separately
cd /shared/ssd/home/b-r-duvvuri/nngpt2/nn-gpt
python3 analyze_output_sizes.py --compare-epochs-detailed 0 1 2 3 4 5 --base-path /shared/ssd/home/b-r-duvvuri/nngpt2/nn-gpt/out

cd /shared/ssd/home/b-r-duvvuri/nngpt3/nn-gpt
python3 analyze_output_sizes.py --compare-epochs-detailed 0 1 2 3 4 5 --base-path /shared/ssd/home/b-r-duvvuri/nngpt3/nn-gpt/out
```

---

**Bottom line:** Use `--compare-epochs-detailed` to see both per-run AND cross-epoch comparisons at once! 🎯
