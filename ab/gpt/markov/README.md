# Resource-Efficient Iterative LLM-Based NAS with Feedback Memory

Resource-Efficient Iterative prompt-improvement pipeline: uses an LLM to generate model code, then trains and evaluates it. Supports CIFAR-10, CIFAR-100, and ImageNette. You can turn off the improver or reference via flags for ablation runs.

## Environment and dependencies

- **Python**: 3.10 recommended
- **Dependencies**: 
  ```bash
  cd /path/to/nngpt
  pip install -e .
  ```

## How to run

```bash
python -m ab.gpt.markov.pipeline \
  --model <HuggingFace_model_name> \
  --dataset <cifar10|cifar100|imagenette> \
  --max-iterations <N> \
  --target-accuracy <0~1> \
  --output-dir <output_directory> \
```

Example:
```bash
python -m ab.gpt.markov.pipeline \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset cifar10 \
  --max-iterations 2000 \
  --target-accuracy 0.9 \
  --output-dir ./out/markov
```

## Output

Under `--output-dir` you get:

- **summary.json**: total iterations, best accuracy, whether target was reached, `results_history`
- **results.log**: per-iteration accuracy, success, and errors
- **generated_models/**: generated model code per iteration (e.g. `model_iter_*.py`)
