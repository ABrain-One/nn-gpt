import os

# ---------------------------------------------------------------------------
# Qwen3.8-27B variant. READ THIS BEFORE CHANGING ANYTHING.
#
# Qwen3.8-27B is strictly harder on this card than 3.6-27B, measured:
#
#                      weights     KV per token
#   Qwen3.6-27B       16.08 GiB    ~104 KB (bf16) -> ~52 KB (fp8)
#   Qwen3.8-27B       19.63 GiB    ~184 KB (fp8)
#
# 3.5 GiB more weights AND ~3.5x more KV per token at the same dtype. At 6144
# tokens one sequence costs 1.08 GiB of KV here against 3.6's ~0.30 GiB, so
# expect roughly 2 concurrent sequences where 3.6 gets ~5. It may build with
# the settings below, but it will be SLOWER than 3.6, not faster.
#
# Two settings are known-fatal for this model and are deliberately absent:
#   NNGPT_VLLM_MODEL=...-INT4   -> auto_awq unpacks to 21.64 GiB (failed 3x)
#   NNGPT_VLLM_ENFORCE_EAGER=0  -> CUDA graphs take the whole KV cache
# ---------------------------------------------------------------------------

# Set before Tune is imported, so the engine sees them when it builds at
# generation time. setdefault, so a real environment variable still wins.

# ---------------------------------------------------------------------------
# NO NNGPT_VLLM_MODEL / NNGPT_VLLM_QUANTIZATION HERE -- ON PURPOSE.
#
# With neither set, the engine serves the trainer's own bitsandbytes nf4
# weights (16.08 GiB). That is the only configuration measured to build and
# generate on this card. It reaches ~25-35 tok/s aggregate.
#
# Serving a separate AWQ checkpoint was tried three times and failed
# identically every time:
#
#   auto_awq.py:119 _convert_awq_to_standard_format
#   21.64 GiB allocated, 271.94 MiB free, 340 MiB needed
#
# vLLM 0.32 *unpacks* AWQ int4 into a wider layout at load, so an AWQ copy of
# this model needs 21.64 GiB against bnb's 16.08 -- a 5.5 GiB regression. The
# failure did not move when gpu_memory_utilization went 0.936 -> 0.851 (the
# weights are allocated through vLLM's CuMemAllocator, outside that budget),
# and it did not move when the backend was forced to awq_marlin (the same
# auto_awq module handles both). It is not a tuning problem.
#
# One untried variant, a different loader module rather than the same one:
#   NNGPT_VLLM_MODEL=btbtyler09/Qwen3.8-27B-GPTQ-4bit
#   NNGPT_VLLM_QUANTIZATION=gptq_marlin
# gptq_marlin lives in its own file and repacks rather than unpacking, so it
# may avoid this. Worth one run; not worth a fourth round of margin tuning.
# ---------------------------------------------------------------------------

# Halve the KV cache. The weights take most of the engine's budget, so KV is
# what limits how many sequences run at once -- and concurrency is the only
# thing that makes vLLM faster than a plain HuggingFace loop.
os.environ.setdefault("NNGPT_VLLM_KV_CACHE_DTYPE", "fp8")

# The engine does not need the trainer's full 8192-token window: measured
# output is ~700-1400 tokens and the longest accepted prompt is ~3800, so 6144
# covers prompt + answer. KV per sequence is linear in this number.
os.environ.setdefault("NNGPT_VLLM_MAX_MODEL_LEN", "6144")

# n-gram speculation: proposes continuations from n-grams already in the prompt
# and the text generated so far. No draft model, no VRAM -- which is why it is
# used here rather than MTP, which does not fit in what is left of this card.
# This matters more than usual: with KV for only a few sequences, batching
# cannot help and per-token speed is all there is. Set to 0 to measure it.
os.environ.setdefault("NNGPT_VLLM_SPEC_TOKENS", "4")

# CUDA graphs OFF (enforce_eager=1). They raise GPU utilisation from 47% to
# 72%, but they are not affordable here -- they and the KV cache compete for
# the same ~1.8 GiB, and three runs settled it:
#
#   graphs off : ~1.5 GiB KV -> engine builds and generates
#   graphs on  : 0.75 GiB KV -> "0.81 GiB needed", failed
#   graphs on  : 0 GiB KV    -> "No available memory for the cache blocks"
#
# KV is mandatory, graphs are an optimisation, so graphs lose.
os.environ.setdefault("NNGPT_VLLM_ENFORCE_EAGER", "1")

# Quantize lm_head in the TRAINER. At this model's ~248k vocabulary that table
# is 2.37 GiB of bf16 that bitsandbytes skips by default. This shrinks what is
# parked to host RAM each cycle (14.67 -> ~13 GiB), which matters because
# parking 15 GiB into 32 GiB of RAM drove the machine 17 GiB into swap. It
# costs some logit precision; set to 0 and compare generated-NN success rate
# if that is a concern.
os.environ.setdefault("NNGPT_QUANTIZE_LM_HEAD", "1")

# The park/restore fingerprint check samples every tensor twice per cycle. Runs
# have now reported "Park/restore verified: weights unchanged", so it has done
# its job; across 22 cycles it is overhead. Set to 1 to re-enable if the
# unsloth version or the quantization config changes.
os.environ.setdefault("NNGPT_VLLM_PARK_VERIFY", "0")

import ab.gpt.act.tune.Tune as TuneNNGen


def main():
    TuneNNGen.main(
        llm_conf='qwen3.8_27b.json',      # unsloth/Qwen3.8-27B, nf4. This is
                                          # now BOTH the trainer's and the
                                          # engine's weights. Never put an
                                          # AWQ/GPTQ repo here -- that is what
                                          # produced the Quark ImportError.
        llm_tune_conf='NN_gen_train_channel.json',
        nn_gen_conf='NN_gen_test_channel.json',
        nn_gen_conf_id='optimal_channel_configuration_test',
        test_nn=30,
        skip_epoches=-1,          # generate at epoch 0 => pre-fine-tuning baseline
        nn_name_prefix='llm',
        context_length=8 * 1024,
        sft_max_length=4 * 1024,  # packing block size; keep <= context_length
        max_new_tokens=8 * 1024,
        cycles=22,
        num_train_epochs=1,
        max_prompts=128,
        learning_rate=5e-5,
        # r defaults to 32, and VLLMSleep's max_lora_rank default is also 32.
        # If you raise the LoRA rank, raise NNGPT_VLLM_MAX_LORA_RANK to match
        # or the engine will refuse the adapter.
        lora_alpha=32,
        suppress_thinking=True,
        use_unsloth=True,
        use_vllm=True
    )


if __name__ == "__main__":
    main()