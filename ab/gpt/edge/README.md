# Edge Model Generation Pipeline

LLM-based generation of image-classification models optimized for edge/mobile
deployment. A fine-tuned code LLM (OlympicCoder-7B) synthesizes new
architectures from pretrained TorchVision backbones, guided by k=4 reference
models from the LEMUR database. Every generated model is trained for one epoch
and scored by the edge efficiency metric:

```
efficiency = accuracy / inference latency (ms)
```

subject to a hard parameter budget (< 6M) and TFLite INT8 convertibility
(a model that cannot be converted is not edge-deployable).

## Design constraints

This pipeline does not modify any shared repository file. It reuses the
curriculum tuner (`ab/gpt/util/Tune_Curriculum.py`), the LoRA trainer
(`ab/gpt/util/LoRA.py`), the evaluator (`ab/gpt/NNEval.py`) and the novelty
checker (`ab/gpt/iterative_pipeline/novelty_checker.py`) strictly via imports.
All edge-specific behavior is configured from this package:

- trl >= 0.24 rejects a custom data collator when BFD packing forces
  padding-free mode; this pipeline passes `packing_strategy='wrapped'` in its
  own `SFTConfig` to avoid the conflict without touching the shared trainer.
- trl's `SFTConfig.max_length` defaults to 1024, which silently truncates the
  ~10-14k-token k=4 training prompts; this pipeline sets it explicitly
  (`sft_max_length`, default 16384).

## Files

| File | Purpose |
|---|---|
| `EdgeGen_k4.py` | Builds `SFTConfig`/`LoraConfig` with the settings above and calls `Tune_Curriculum.tune()` |
| `EdgeBench.py` | Per-model edge metrics: parameter count + 6M gate, TFLite INT8 export, CPU latency (TFLite interpreter, PyTorch fallback), efficiency score |
| `EdgeScore.py` | Post-run scorer: walks all epoch outputs, benchmarks every generated model, applies novelty flags, writes `out/edge/edge_tracker.json` |
| `../curriculum/Curriculum_Gen_edge_k4_Tune_7B.py` | Entry script |
| `../conf/prompt/test/edge/curriculum_k4.json` | Generation prompt (edge constraints, mobile backbone whitelist) |
| `../conf/prompt/train/edge/curriculum_k4_train.json` | Fine-tuning prompt (same edge constraints, `is_generation: false`) |

## Usage

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. Run the generate/evaluate/finetune loop (10 outer epochs, 5 models each)
python -m ab.gpt.curriculum.Curriculum_Gen_edge_k4_Tune_7B

# 2. Score all generated models by efficiency — run BEFORE relaunching step 1,
#    because the tuner clears the epoch output directory at the start of a run
python -m ab.gpt.edge.EdgeScore

# Optional overrides
python -m ab.gpt.edge.EdgeScore --dataset cifar-10 --param-limit 6000000 --input-size 256
```

## Outputs

```
out/nngpt/llm/epoch/A{e}/synth_nn/B{i}/   generated model, hp, transform, eval_info.json
out/nngpt/epoch_tracker.json              per-epoch success rate / accuracy (written by the shared tuner)
out/edge/edge_tracker.json                per-model and per-epoch efficiency ranking (thesis metric)
out/edge/tflite/A{e}_B{i}.tflite          INT8 models for on-device benchmarking
```

`edge_tracker.json` records for every generated model: parameter count and
budget-gate result, TFLite convertibility, latency (with its measurement
source), 1-epoch accuracy, novelty flag, and the efficiency score. The best
epoch/adapter is selected by efficiency, not raw accuracy.

The `.tflite` files are ready for the Android benchmarking pipeline
(real-device CPU/GPU/NPU latency); on-device numbers can then replace the
host-CPU latency in the final evaluation.

## Latency measurement

In-loop latency is measured on the host CPU at batch 1 and the serving
resolution (256x256, matching the `norm_256_flip` transform):

1. preferred: INT8 TFLite model via the litert interpreter (same kernels as
   Android CPU execution)
2. fallback: PyTorch CPU forward pass (recorded as `latency_source:
   'torch_cpu'` so mixed runs stay comparable)

Median of 20 runs after 5 warmup runs.

## Hardware notes

- Generation and 1-epoch evaluation fit on a 24 GB GPU (4-bit 7B).
- Fine-tuning at `sft_max_length=16384` requires more memory than 24 GB
  (the ~152k-vocabulary logits alone are ~10 GB at 16k tokens); run full
  experiments on an 80 GB GPU, or lower `sft_max_length` for local smoke tests.
