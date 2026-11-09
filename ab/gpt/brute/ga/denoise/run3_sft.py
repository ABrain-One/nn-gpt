"""Run-3 per-cycle SFT: score-conditioned fine-tune on the >=bar corpus.

One cycle of the classification recipe's fine-tune stage, denoise edition. Mirrors the
repo's KTO path (TuneNNGenKTO.py / ab/gpt/util/KTO.py) exactly where the two share
mechanics:

  * base model from the llm_conf json via ab.gpt.util.LLM (4-bit nf4)
  * warm start = load previous adapter, merge_and_unload into the base, attach a FRESH
    LoRA on top (the repo's chaining convention)
  * gradient_checkpointing_enable() + enable_input_require_grads() BEFORE the PEFT wrap
    (without the second call checkpointing silently retains every activation -- measured
    in the KTO run, see ab/gpt/util/KTO.py)
  * LoRA r=32/alpha=32, lr 1e-5, batch 2 x grad-accum 4 -- TuneNNGenKTO.py defaults

Loss is completion-only (the assistant turn), so the model is trained to PRODUCE code
given score-annotated references, not to reproduce the references.

  python run3_sft.py --data sft_data/ --out cycle_1/adapter --peft out/qlora-kto/final
"""
import argparse
import json
import os
import sys

NNGPT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
sys.path.insert(0, NNGPT_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with train.jsonl / dev.jsonl")
    ap.add_argument("--out", required=True, help="adapter output dir")
    ap.add_argument("--peft", default=None, help="previous adapter to merge as warm start")
    ap.add_argument("--llm_conf", default="qwen2.5_coder_7b_instruct.json")
    ap.add_argument("--num_train_epochs", type=int, default=1)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--max_seq_length", type=int, default=12288)  # measured: full examples run 9.0-11k tok (4 ref blocks + completion)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=32)
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
    from ab.gpt.util.LLM import LLM
    from ab.gpt.util.LLMUtil import quantization_config_4bit
    from ab.gpt.util.KTO import kto_lora_config, find_all_linear_names

    from ab.gpt.util.Const import conf_llm_dir
    with open(conf_llm_dir / a.llm_conf) as f:
        llm_conf = json.load(f)

    loader = LLM(llm_conf["base_model_name"], quantization_config_4bit,
                 context_length=llm_conf.get("context_length"))
    model, tok = loader.get_model(), loader.get_tokenizer()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    if a.peft:
        print(f"[SFT] warm start: merging previous adapter {a.peft}")
        model = PeftModel.from_pretrained(model, a.peft, is_trainable=True)
        model = model.merge_and_unload()

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()      # see ab/gpt/util/KTO.py -- silent no-op without this
    peft_config = kto_lora_config(find_all_linear_names(model),
                                  r=a.lora_r, lora_alpha=a.lora_alpha)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def load_jsonl(path):
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def to_text(rows, split):
        texts, dropped = [], 0
        for r in rows:
            t = tok.apply_chat_template(r["messages"], tokenize=False)
            n = len(tok(t, add_special_tokens=False)["input_ids"])
            if n > a.max_seq_length:
                dropped += 1        # dropping beats truncating: a cut completion breaks <nn>
                continue
            texts.append(t)
        print(f"[SFT] {split}: {len(texts)} examples ({dropped} dropped over {a.max_seq_length} tok)")
        return Dataset.from_dict({"text": texts})

    train_ds = to_text(load_jsonl(os.path.join(a.data, "train.jsonl")), "train")
    if len(train_ds) == 0:
        print("[SFT] no trainable examples; nothing to do")
        return
    dev_path = os.path.join(a.data, "dev.jsonl")
    eval_ds = to_text(load_jsonl(dev_path), "dev") if os.path.exists(dev_path) else None
    if eval_ds is not None and len(eval_ds) == 0:
        eval_ds = None                       # tiny per-cycle sets may split 0 dev examples

    resp_ids = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(response_template=resp_ids, tokenizer=tok)

    args = SFTConfig(
        output_dir=os.path.join(a.out, "trainer"),
        num_train_epochs=a.num_train_epochs,
        per_device_train_batch_size=a.per_device_train_batch_size,
        gradient_accumulation_steps=a.gradient_accumulation_steps,
        learning_rate=a.learning_rate,
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        eval_strategy="epoch" if eval_ds is not None else "no",
        report_to=[],
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        dataset_text_field="text",
        max_length=a.max_seq_length,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=tok,
    )
    trainer.train()

    os.makedirs(a.out, exist_ok=True)
    trainer.model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"[SFT] adapter saved to {a.out}")


if __name__ == "__main__":
    main()
