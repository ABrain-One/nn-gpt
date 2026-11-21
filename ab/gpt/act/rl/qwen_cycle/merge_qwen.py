"""Merge a phase's LoRA adapter into its base model and save a full bf16 model for the next phase.
Usage: python merge_qwen.py <base model or dir> <adapter dir> <output dir> [cuda|cpu]"""
import sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base, adapter, out = sys.argv[1:4]
device = sys.argv[4] if len(sys.argv) > 4 else "cuda"
t = time.time()
m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map={"": device})
print(f"[MERGE] loaded base {base} as {type(m).__name__} on {device}", flush=True)
m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
m.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
AutoTokenizer.from_pretrained(base).save_pretrained(out)
print(f"[MERGE] saved {type(m).__name__} -> {out} in {time.time() - t:.0f}s | architectures={m.config.architectures}", flush=True)
