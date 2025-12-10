# Setup Verification Checklist ✅

## Quick Verification

### ✅ Core Setup:
1. **Virtual Environment**: `.venv/` directory created
2. **Pip Upgraded**: Latest pip version installed
3. **Dependencies Installed**: All packages from `requirements.txt`
   - PyTorch with CUDA support
   - nn-dataset
   - transformers, deepspeed, peft, etc.
4. **Delta Utilities**: `DeltaUtil` module imports successfully

### ⚠️ Optional:
- **flash_attn**: Requires CUDA 11.7+ and nvcc compiler
  - Install if needed: `pip install flash_attn --no-build-isolation`

## Quick Test Commands

```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Training test (minimal)
python -m ab.gpt.TuneNNGen_delta \
    --num_train_epochs 1 \
    --test_nn 2 \
    --nn_train_epochs 1 \
    --max_prompts 50 \
    --skip_epoches -1

# Generation test (minimal)
python -m ab.gpt.NNAlter_7B_delta \
    --epochs 1 \
    --num-supporting-models 0

# Evaluation test (minimal)
python -m ab.gpt.NNEval \
    --only_epoch 0 \
    --nn_train_epochs 1 \
    --nn_alter_epochs 1
```

## Verification

- CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
- Virtual environment location: `.venv/` (relative to project root)

For detailed setup instructions, see [README.md](README.md).

