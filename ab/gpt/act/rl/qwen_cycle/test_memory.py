"""Test of the memory-bounded update before the clean runs (run on the workstation, GPU otherwise idle).
1. same log-probs and entropies as the previous implementation (300-token completion)
2. same LoRA gradients (200-token completion)
3. peak GPU memory of forward + backward on a 4096-token completion: new vs previous implementation
Usage: cd ~/nn-gpt-main && .venv/bin/python test_memory.py"""
import json, os, sys, time
import torch

GIB = 2 ** 30
sys.argv = sys.argv[:1]
os.environ.setdefault("ALGO", "ppo")
import qwen_cycle as Q                                   # the patched cycle script in the same folder


def peak(tag):
    torch.cuda.synchronize()
    print(f"[MEM] {tag}: peak {torch.cuda.max_memory_allocated() / GIB:.2f} GiB, now {torch.cuda.memory_allocated() / GIB:.2f} GiB", flush=True)


def main():
    from ab.gpt.act.rl.trainer_runtime import (load_tokenizer, load_quantized_causal_lm, build_lora_config,
                                               attach_or_resume_lora, enable_non_reentrant_gradient_checkpointing)
    free, total = torch.cuda.mem_get_info()
    print(f"[MEM] GPU free at start: {free / GIB:.1f} of {total / GIB:.1f} GiB (other users take the rest)", flush=True)
    base = os.environ.get("BASE", "Qwen/Qwen3.5-9B")
    tok = load_tokenizer(base)
    prec = {"dtype": torch.bfloat16, "bf16": True, "fp16": False, "label": "bf16"}
    model = load_quantized_causal_lm(model_source=base, precision=prec, train_device="cuda", use_deepspeed=False)
    model = attach_or_resume_lora(model, peft_config=build_lora_config(r=16, alpha=32, dropout=0.0),
                                  stage_adapter_dir=None, log_prefix="[MEM]")
    enable_non_reentrant_gradient_checkpointing(model, log_prefix="[MEM]")
    # give the LoRA B matrices small non-zero values so that the adapter changes the output (as after some updates)
    torch.manual_seed(0)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 1e-3)
    prompt = json.load(open(os.path.expanduser("~/frozen_prompts_v3.json")))[0]
    pids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                   enable_thinking=False, tokenize=True)
    if isinstance(pids, dict) or hasattr(pids, "input_ids"):
        pids = list(pids["input_ids"])
    V = model.get_base_model().get_output_embeddings().weight.shape[0]
    g = torch.Generator().manual_seed(1)
    rand = lambda n: torch.randint(1000, min(V, 150000), (n,), generator=g).tolist()
    print(f"[MEM] prompt {len(pids)} tokens, vocabulary {V}", flush=True)
    ok = True

    # 1. identical log-probs and entropies
    c = rand(300)
    model.eval()
    with torch.no_grad():
        a_lp, a_en = Q.completion_stats(model, pids, c)
        b_lp, b_en = Q.completion_stats_reference(model, pids, c)
    d_lp, d_en = (a_lp - b_lp).abs().max().item(), (a_en - b_en).abs().max().item()
    t1 = d_lp < 0.05 and d_en < 0.05
    ok &= t1
    print(f"[TEST 1] log-prob max diff {d_lp:.4f}, entropy max diff {d_en:.4f} (mean log-prob {a_lp.mean():.3f} vs {b_lp.mean():.3f}) -> {'PASS' if t1 else 'FAIL'}", flush=True)

    # 2. identical gradients
    c = rand(200)
    model.train()
    grads = []
    for fn in (Q.completion_stats, Q.completion_stats_reference):
        model.zero_grad(set_to_none=True)
        lp, en = fn(model, pids, c)
        (-lp.mean() - 0.01 * en.mean()).backward()
        grads.append(torch.cat([p.grad.flatten().float() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None]))
    rel = ((grads[0] - grads[1]).norm() / grads[1].norm()).item()
    cos = torch.nn.functional.cosine_similarity(grads[0], grads[1], dim=0).item()
    t2 = rel < 0.02 and cos > 0.999
    ok &= t2
    print(f"[TEST 2] gradient relative diff {rel:.4f}, cosine {cos:.5f} -> {'PASS' if t2 else 'FAIL'}", flush=True)
    model.zero_grad(set_to_none=True); del grads; torch.cuda.empty_cache()

    # 3. memory on the longest completion
    c = rand(4096)
    for name, fn in (("new", Q.completion_stats), ("previous", Q.completion_stats_reference)):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        base_mem = torch.cuda.memory_allocated()
        try:
            t = time.time()
            lp, en = fn(model, pids, c)
            (-lp.mean() - 0.01 * en.mean()).backward()
            torch.cuda.synchronize()
            extra = (torch.cuda.max_memory_allocated() - base_mem) / GIB
            print(f"[TEST 3] {name}: 4096-token forward+backward needs {extra:.2f} GiB above the model, {time.time() - t:.0f} s", flush=True)
            if name == "new":
                t3 = extra
        except torch.OutOfMemoryError:
            print(f"[TEST 3] {name}: OUT OF MEMORY on the 4096-token completion", flush=True)
            if name == "new":
                ok = False; t3 = None
        model.zero_grad(set_to_none=True)
        lp = en = None; torch.cuda.empty_cache()
    print(f"[MEM] RESULT: {'ALL TESTS PASSED' if ok else 'A TEST FAILED'}", flush=True)


if __name__ == "__main__":
    main()
