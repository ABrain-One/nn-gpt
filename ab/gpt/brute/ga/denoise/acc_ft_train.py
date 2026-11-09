"""Fine-tune the ABrain accuracy predictor on DENOISING data, then score it against the baseline.

WHAT IT HAS TO BEAT, AND WHY THAT BAR IS THE POINT. On 100 frozen holdout models (all genuine
50-epoch ft50 runs) the trivial rule "just use epoch 3" scores rho=+0.728, MAE 2.559 dB, 82% overlap
with the true top half. The stock predictor scores rho=+0.226 / MAE 1.62 / 57% on the full 280.
A fine-tune that does not beat the epoch-3 row is not worth serving, and this script reports the
comparison on the same 100 models rather than reporting a standalone number that sounds impressive.
That is exactly the check nobody ran on the in-house surrogate, which spent months at rho=-0.178
while its own k-fold CV said +0.693.

MIRRORS run3_sft.py's PROVEN PATH, not the unsloth path in AccPredictor.py. The ai-linux image runs
peft/trl 4-bit on the H100 today for run-3's SFT; unsloth is not known to be installed, and a
dependency gamble on a queued GPU job is a bad trade. Two conventions are copied deliberately:
  * merge the warm-start adapter, THEN attach a fresh LoRA (the repo's chaining convention)
  * gradient_checkpointing_enable() AND enable_input_require_grads() before the PEFT wrap --
    ab/gpt/util/KTO.py records that checkpointing silently retains every activation without the
    second call, which looks like a memory leak rather than a missing line.

WARM START FROM THE PUBLISHED ADAPTER. Training starts from ABrain/Accuracy-Prediction rather than
from scratch: it already emits the exact JSON contract and understands the prompt, so the fine-tune
only has to move it from classification to denoising, on ~890 examples. A from-scratch LoRA on that
much data would mostly relearn the output format.

EVERYTHING RUNS OFFLINE. HF_HUB_OFFLINE=1 in the pod; the base (7.5 GB) and adapter were pre-cached
from the login node, which is the only place with network.

  python acc_ft_train.py --data /a/mm/sweep_out/acc_ft --out /a/mm/sweep_out/acc_ft/adapter
"""
import argparse
import json
import math
import os
import re
import sys

BASE = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
WARM = "ABrain/Accuracy-Prediction"
SCALE = 48.0


def ranks(xs):
    o = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and xs[o[j + 1]] == xs[o[i]]:
            j += 1
        a = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[o[k]] = a
        i = j + 1
    return r


def pearson(a, b):
    n = len(a)
    if n < 3:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    va = math.sqrt(sum((x - ma) ** 2 for x in a))
    vb = math.sqrt(sum((y - mb) ** 2 for y in b))
    return None if va == 0 or vb == 0 else \
        sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb)


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def parse_json_out(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    s = text.find("{")
    if s < 0:
        return None
    d = 0
    for i in range(s, len(text)):
        if text[i] == "{":
            d += 1
        elif text[i] == "}":
            d -= 1
            if d == 0:
                try:
                    return json.loads(text[s:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/a/mm/sweep_out/acc_ft")
    ap.add_argument("--out", default="/a/mm/sweep_out/acc_ft/adapter")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max_seq_length", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval_only", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, "/a/mm/nn-gpt")
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if not a.eval_only:
        qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            BASE, quantization_config=qc, device_map="auto", trust_remote_code=True)
        # repo chaining convention: merge the published adapter, then attach a fresh LoRA
        model = PeftModel.from_pretrained(model, WARM, is_trainable=True).merge_and_unload()
        model = prepare_model_for_kbit_training(model)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()   # without this, checkpointing is a silent no-op
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
        model.print_trainable_parameters()

        def load(split):
            rows = [json.loads(l) for l in open(os.path.join(a.data, f"{split}.jsonl")) if l.strip()]
            return Dataset.from_list(
                [{"text": tok.apply_chat_template(r["messages"], tokenize=False)} for r in rows])

        tr, va = load("train"), load("val")
        print(f"train {len(tr)} | val {len(va)}", flush=True)
        trainer = SFTTrainer(
            model=model, train_dataset=tr, eval_dataset=va,
            args=SFTConfig(output_dir=os.path.join(a.out, "_ckpt"),
                           num_train_epochs=a.epochs,
                           per_device_train_batch_size=a.batch,
                           gradient_accumulation_steps=a.grad_accum,
                           learning_rate=a.lr, bf16=True, logging_steps=10,
                           eval_strategy="epoch", save_strategy="epoch",
                           save_total_limit=1, load_best_model_at_end=True,
                           max_seq_length=a.max_seq_length, gradient_checkpointing=True,
                           report_to=[], dataset_text_field="text"))
        trainer.train()
        trainer.model.save_pretrained(a.out)
        tok.save_pretrained(a.out)
        print(f"adapter -> {a.out}", flush=True)
        del trainer, model
        torch.cuda.empty_cache()

    # ---- evaluation on the frozen holdout, head to head with epoch 3 -------------------
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=qc, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, a.out)
    model.eval()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from acc_ft_dataset import SYSTEM_PROMPT, user_message, load_codes
    codes = load_codes()
    test = json.load(open(os.path.join(a.data, "test_rows.json")))
    preds = []
    for i, r in enumerate(test, 1):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message(
                    r["task"], r["dataset"], r["metric"], codes[r["model"]],
                    r["e1"] * 100, r["e2"] * 100, r["e3"] * 100, r["max_epochs"])}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=a.max_seq_length).to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=64, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        d = parse_json_out(gen if "}" in gen else gen + "}")
        preds.append({"model": r["model"], "actual_dB": r["best_acc"] * SCALE,
                      "e3_dB": r["e3"] * SCALE,
                      "pred_dB": (float(d["best_accuracy"]) / 100.0 * SCALE) if d else None,
                      "pred_epoch": int(d["best_epoch"]) if d else None,
                      "raw": None if d else gen[:120]})
        if i % 20 == 0:
            print(f"  eval {i}/{len(test)}", flush=True)
    json.dump(preds, open(os.path.join(a.data, "ft_predictions.json"), "w"), indent=1)

    ok = [p for p in preds if p["pred_dB"] is not None]
    y = [p["actual_dB"] for p in ok]
    pv = [p["pred_dB"] for p in ok]
    e3 = [p["e3_dB"] for p in ok]
    k = len(ok) // 2
    tt = {ok[i]["model"] for i in sorted(range(len(ok)), key=lambda i: -y[i])[:k]}
    tp = {ok[i]["model"] for i in sorted(range(len(ok)), key=lambda i: -pv[i])[:k]}
    tb = {ok[i]["model"] for i in sorted(range(len(ok)), key=lambda i: -e3[i])[:k]}
    res = {"n": len(ok), "n_unparsed": len(preds) - len(ok),
           "finetuned": {"spearman": spearman(pv, y),
                         "mae_dB": sum(abs(x - z) for x, z in zip(pv, y)) / len(ok),
                         "top50_overlap": len(tp & tt) / k},
           "baseline_ep3": {"spearman": spearman(e3, y),
                            "mae_dB": sum(abs(x - z) for x, z in zip(e3, y)) / len(ok),
                            "top50_overlap": len(tb & tt) / k},
           "stock_abrain_on_280": {"spearman": 0.226, "mae_dB": 1.616, "top50_overlap": 0.57},
           "inhouse_surrogate": {"spearman": -0.178}}
    res["beats_baseline"] = (res["finetuned"]["spearman"] > res["baseline_ep3"]["spearman"]
                             and res["finetuned"]["top50_overlap"] >= res["baseline_ep3"]["top50_overlap"])
    json.dump(res, open(os.path.join(a.data, "ft_report.json"), "w"), indent=1)
    print("\n=== HOLDOUT (n=%d, %d unparsed) ===" % (res["n"], res["n_unparsed"]))
    for kk in ("finetuned", "baseline_ep3"):
        v = res[kk]
        print(f"  {kk:<14} rho {v['spearman']:+.3f}  MAE {v['mae_dB']:.3f} dB  "
              f"top50 {v['top50_overlap']:.0%}")
    print(f"\n  BEATS THE BASELINE: {res['beats_baseline']}")
    print("  (a fine-tune that does not beat epoch-3-alone should not be served)")


if __name__ == "__main__":
    main()
