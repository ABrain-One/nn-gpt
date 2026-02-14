# TuneNNGen Header-Only Training Guide

## Overview

This guide explains how to run **TuneNNGen for header-only LoRA training** with your custom modifications (Solutions 3 & 5 for output optimization).

---

## Quick Start

### Basic Command
```bash
cd /shared/ssd/home/b-r-duvvuri/nngpt3/nn-gpt
python -m ab.gpt.TuneNNGen --tune_headers_only
```

### With All Current Parameters (From ---Batch-job---.json)
```bash
python -m ab.gpt.TuneNNGen \
  --tune_headers_only \
  --num_train_epochs 1 \
  --header_max_tokens 16384 \
  --temperature 0.6 \
  --top_k 30 \
  --top_p 0.8
```

---

## Parameter Reference

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--tune_headers_only` | N/A | flag | **Enables header-only LoRA training** |
| `--num_train_epochs` | 1 | 1-10 | Number of fine-tuning cycles |
| `--header_max_tokens` | 16384 | 8000-20000 | Maximum tokens for generation 
| `--temperature` | 0.6 | 0.0-1.0 | Sampling randomness (0=deterministic, 1=random) |
| `--top_k` | 30 | 10-100 | Consider top K tokens |
| `--top_p` | 0.8 | 0.5-0.95 | Cumulative probability threshold |
| `--learning_rate` | 5e-6 | 1e-7 to 1e-4 | Learning rate for LoRA training |
| `--per_device_train_batch_size` | 1 | 1-4 | Batch size per GPU |
| `--gradient_accumulation_steps` | 4 | 1-8 | Gradient accumulation steps |

---

## Common Usage Examples

### 1. Standard Training (Recommended)
Uses experiment3.json defaults with new conditions:
```bash
python -m ab.gpt.TuneNNGen \
  --tune_headers_only \
  --num_train_epochs 1 \
  --header_max_tokens 16384 \
  --temperature 0.6 \
  --top_k 30 \
  --top_p 0.8
```

### 2. Conservative (Less Creative)
```bash
python -m ab.gpt.TuneNNGen \
  --tune_headers_only \
  --temperature 0.3 \
  --top_k 20 \
  --top_p 0.7
```

### 3. Creative (More Diverse)
```bash
python -m ab.gpt.TuneNNGen \
  --tune_headers_only \
  --temperature 0.9 \
  --top_k 50 \
  --top_p 0.95
```

### 4. Smaller Output (Solutions 3 & 5 Work Better)
```bash
python -m ab.gpt.TuneNNGen \
  --tune_headers_only \
  --header_max_tokens 12000
```

### 5. Larger Output (More Generation)
```bash
python -m ab.gpt.TuneNNGen \
  --tune_headers_only \
  --header_max_tokens 20000
```

### 6. Multiple Epochs
```bash
python -m ab.gpt.TuneNNGen \
  --tune_headers_only \
  --num_train_epochs 3
```

### 7. Custom Learning Rate
```bash
python -m ab.gpt.TuneNNGen \
  --tune_headers_only \
  --learning_rate 1e-5
```

### 8. Full Custom Configuration
```bash
python -m ab.gpt.TuneNNGen \
  --tune_headers_only \
  --num_train_epochs 2 \
  --header_max_tokens 14000 \
  --temperature 0.65 \
  --top_k 35 \
  --top_p 0.85 \
  --learning_rate 3e-6 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2
```

---

## What Happens During Execution

1. **Model Loading**: Loads pre-trained LLM from HuggingFace
2. **LoRA Configuration**: Sets up LoRA with header-only focus
   - `target_modules = ('lm_head',)` (only output layer)
   - `r = 16` (LoRA rank)
   - `lora_alpha = 16` (LoRA scaling)
3. **Fine-tuning**: Trains for 1 epoch per cycle
4. **Generation** (With new conditions):
   - **Condition 1**: Early stopping when all XML tags properly paired
   - **Condition 2**: Removes unnecessary whitespace and normalizes indentation
5. **Output**: Generates 5 probes (B0-B4) per epoch
   - Saved to: `out/nngpt/llm/epoch/A{N}/synth_nn/B{0-4}/full_output.txt`

---

## Output Structure

After running, check the generated outputs:

```
out/
└── nngpt/
    └── llm/
        └── epoch/
            └── A0/
                └── synth_nn/
                    ├── B0/
                    │   └── full_output.txt
                    ├── B1/
                    │   └── full_output.txt
                    ├── B2/
                    │   └── full_output.txt
                    ├── B3/
                    │   └── full_output.txt
                    └── B4/
                        └── full_output.txt
```

---

## Verify Your Changes

### Check Output Size Reduction 
```bash
python3 analyze_output_sizes.py --epoch 0 --compare
```

Expected output:
```
B0 → B4: Characters ✓ REDUCED +15-20% | Tokens ✓ REDUCED +12-18%
```

### Compare Multiple Epochs
```bash
python3 analyze_output_sizes.py --compare-epochs-detailed 0 1 2
```

### Generate Visualization Graphs
```bash
python3 plot_epoch_analysis.py --epochs 0 1 2 3 4 5
```

---

## Parameter Tuning Guide

### To Reduce Output Size
- **Lower** `--header_max_tokens` (12000-14000)
- **Lower** `--temperature` (0.4-0.5) for more deterministic output
- **Lower** `--top_k` (15-25)
- **Lower** `--top_p` (0.6-0.7)

### To Increase Output Diversity
- **Raise** `--temperature` (0.8-0.9)
- **Raise** `--top_k` (40-50)
- **Raise** `--top_p` (0.85-0.95)

### To Train Faster
- **Lower** `--per_device_train_batch_size` (use 1)
- **Raise** `--gradient_accumulation_steps` (use 4 or 8)

### To Train Better (Slower)
- **Raise** `--per_device_train_batch_size` (2-4)
- **Lower** `--gradient_accumulation_steps` (1-2)
- **Lower** `--learning_rate` (1e-6 to 5e-7)

---

## Differences: Header-Only vs Full Model

### Header-Only Training (`--tune_headers_only`)
✅ **Pros:**
- 5-10x faster training
- Lower memory usage
- Good for fine-tuning on specific tasks

❌ **Cons:**
- Limited to output layer modifications
- Can't change model backbone

### Full Model Training (without flag)
✅ **Pros:**
- Can modify entire model
- Better for significant changes

❌ **Cons:**
- 5-10x slower
- Higher memory usage
- More computational cost

---

## Example: Reproduce experiment3.json Behavior

```bash
# Exactly replicate ---Batch-job---.json setup
python -m ab.gpt.TuneNNGen \
  --tune_headers_only \
  --num_train_epochs 1 \
  --header_max_tokens 16384 \
  --temperature 0.6 \
  --top_k 30 \
  --top_p 0.8
```

This will:
1. Train header-only with LoRA
2. Apply Solutions 3 & 5 during generation
3. Generate outputs with optimized size
4. Create epoch A0 directory with B0-B4 outputs

---

### Error: "CUDA out of memory"
```bash
# Reduce memory usage
python -m ab.gpt.TuneNNGen \
  --tune_headers_only \
  --per_device_train_batch_size 1 \
  --header_max_tokens 12000
```

## Analysis & Visualization

### View Solutions Effectiveness
```bash
# Compare baseline (A0) vs latest epoch
python3 plot_epoch_analysis.py

# See detailed metrics
python3 analyze_output_sizes.py --compare-epochs-detailed 0 26
```

Expected improvements with new conditions:
- **Token reduction**: 40-50% over 27 epochs
- **Reasoning reduction**: 50-60% (Solution 3 effectiveness)
- **Code increase**: 40-50% (more useful content)

---

## Integration with Kubernetes Jobs

Use experiment3.json with your custom parameters:
```json
{
  "command": "python -m ab.gpt.TuneNNGen --tune_headers_only --num_train_epochs 1 --header_max_tokens 16384 --temperature 0.6 --top_k 30 --top_p 0.8",
  "resources": {
    "gpu": 1,
    "cpu": 8,
    "memory": "30Gi"
  }
}
```

---

## Next Steps

1. **Run Training**:
   ```bash
   python -m ab.gpt.TuneNNGen --tune_headers_only
   ```

2. **Analyze Results**:
   ```bash
   python3 analyze_output_sizes.py --epoch 0 --compare
   ```

3. **Visualize**:
   ```bash
   python3 plot_epoch_analysis.py --epochs 0
   ```

4. **Compare Epochs** (after multiple runs):
   ```bash
   python3 plot_epoch_analysis.py --epochs 0 1 2 3 4 5
   ```

---

**For more information**, see:
- [EPOCH_COMPARISON_MODES.md](EPOCH_COMPARISON_MODES.md) - Analysis modes
