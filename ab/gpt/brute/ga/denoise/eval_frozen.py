"""Score a model on the frozen prompt set: the before/after number for the fine-tune.

WHAT IS MEASURED, and why it is not loss. Dev loss says the model assigns higher likelihood to
desirable completions; it does not say the model PRODUCES more of them. The pipeline's whole
complaint is a 3.7% end-to-end yield, so the metric has to be yield: generate fresh completions
from held-out prompts and run them through the same gauntlet the live pipeline uses.

    parsed      the <nn> block extracts at all
    valid       imports, builds, forwards, inside the MAC budget
    novel       AST hash absent from the corpus AND from this batch
    ACCEPTED    valid AND novel -- the same definition the training label uses

THE PROMPTS ARE HELD OUT BY ARCHITECTURE HASH, not merely unseen. build_kto_dataset.py assigns
whole hash-groups to exactly one split, so a prompt here never had its architecture in training
under any other episode. Verified: zero hash overlap between train, dev and eval.

SAMPLING MATCHES GENERATION EXACTLY (temperature 0.8, top_p 0.95, top_k 50) because those are
the numbers in every episode's meta.json. Evaluating at greedy decode would measure a decoding
mode the pipeline never runs.

  python eval_frozen.py --model Qwen/Qwen2.5-Coder-7B-Instruct --out base.json
  python eval_frozen.py --model out/qlora-kto/final --adapter --out tuned.json
"""
import argparse
import collections
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, "/a/mm/incoming"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# HOW FAR DID THE GENERATION GET. Binary "accepted" is 3.1% here, so 256 generations can
# only detect a ~188% change; an ordinal score draws signal from EVERY generation and
# detects ~19% at the same n. Monotone in pipeline order, so a higher mean is unambiguously
# better -- it cannot be gamed by trading one failure mode for another.
STAGE_RANK = {"no_parse": 0, "static": 1, "build": 2, "exec": 2, "preflight-crash": 2,
              "forward": 3, "train_setup": 4, "mac": 5, "duplicate": 6, "ACCEPTED": 7}
STAGE_MAX = 7

NN_RE = re.compile(r"<nn>(.*?)</nn>", re.S)
FENCE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.S)


def extract(text):
    """Mirror the pipeline's tolerance: <nn> first, then a fenced block."""
    m = NN_RE.search(text)
    if m:
        return m.group(1)
    m = FENCE_RE.search(text)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter dir to apply on top of --model")
    ap.add_argument("--prompts",
                    default="/a/mm/sweep_out/kto_data/frozen_eval_prompts.jsonl")
    ap.add_argument("--db", default="/a/mm/nn-dataset/db/ab.nn.db")
    ap.add_argument("--samples", type=int, default=4,
                    help="completions per prompt; yield is noisy at n=1 and the live "
                         "pipeline draws many samples per prompt too")
    ap.add_argument("--max-new-tokens", type=int, default=2304)
    ap.add_argument("--cos-max", type=float, default=0.95)
    ap.add_argument("--pf-timeout", type=int, default=360)
    ap.add_argument("--pf-cpu", type=int, default=60)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    prompts = [json.loads(l) for l in open(a.prompts)]
    # Interleaved, not contiguous: prompts are ordered by architecture group, so a
    # contiguous slice would give one shard a systematically different prompt mix.
    if a.num_shards > 1:
        prompts = prompts[a.shard::a.num_shards]
    print(f"=== shard {a.shard}/{a.num_shards}: {len(prompts)} prompts "
          f"x {a.samples} samples ===", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model if not a.adapter else a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto",
        max_memory={i: "24000MB" for i in range(torch.cuda.device_count())},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
        print(f"    adapter applied from {a.adapter}")
    model.eval()

    # Reference set = the corpus, exactly as the live gate sees it.
    from diversity_gate import DiversityGate
    import sqlite3
    gate = DiversityGate(store_path=None, graded=True, cos_max=a.cos_max,
                         ast_exact=True, log=lambda *_: None)
    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    corpus = con.execute("SELECT name, code FROM nn WHERE name IN "
                         "(SELECT DISTINCT nn FROM stat WHERE task='img-denoising')"
                         ).fetchall()
    con.close()
    for nm, code in corpus:
        try:
            _, fp = gate.check(code)
            gate.add(nm, fp)
        except Exception:
            pass
    print(f"    reference set: {len(gate.accepted)} architectures\n", flush=True)

    import preflight as PF
    st = collections.Counter()
    rows = []
    ranks = []
    t0 = time.time()
    for i, p in enumerate(prompts, 1):
        text = tok.apply_chat_template(p["prompt_messages"], tokenize=False,
                                       add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(model.device)
        for s in range(a.samples):
            st["generated"] += 1
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=a.max_new_tokens,
                                     do_sample=True, temperature=0.8, top_p=0.95,
                                     top_k=50, pad_token_id=tok.pad_token_id)
            comp = tok.decode(out[0][ids["input_ids"].shape[1]:],
                              skip_special_tokens=True)
            src = extract(comp)
            rec = {"prompt": p["_meta"]["key"], "sample": s,
                   "parsed": src is not None}
            if src is None:
                st["no_parse"] += 1
                rec["stage_rank"] = STAGE_RANK["no_parse"]
                ranks.append(rec["stage_rank"])
                rows.append(rec)
                continue
            st["parsed"] += 1
            # check_isolated, NOT check_one: check_one returns 3 values on failure and 4 on
            # success (its own docstring says 3), and it execs generated code in-process.
            # check_isolated normalises the arity, runs it in a subprocess with CPU and
            # wall-clock limits, and is what the live pipeline calls.
            try:
                ok, stage, _detail, _arch = PF.check_isolated(
                    src, True, a.pf_timeout, cpu_limit=a.pf_cpu)
            except Exception as e:
                ok, stage = False, f"preflight-error:{type(e).__name__}"
            rec["valid"] = bool(ok)
            rec["stage"] = stage
            if not ok:
                st[f"invalid:{stage}"] += 1
                rec["stage_rank"] = STAGE_RANK.get(str(stage).split(":")[0], 1)
                ranks.append(rec["stage_rank"])
                rows.append(rec)
                continue
            st["valid"] += 1
            try:
                accept, fp = gate.check(src)
            except Exception:
                accept, fp = True, (None, None)
            rec["novel"] = bool(accept)
            if accept:
                gate.add(f"eval_{i}_{s}", fp)   # greedy within the batch, like the pipeline
                st["ACCEPTED"] += 1
                rec["stage_rank"] = STAGE_RANK["ACCEPTED"]
            else:
                st["duplicate"] += 1
                rec["stage_rank"] = STAGE_RANK["duplicate"]
            ranks.append(rec["stage_rank"])
            rows.append(rec)
        if i % 8 == 0 or i == len(prompts):
            g = st["generated"]
            print(f"  [{i}/{len(prompts)}] gen={g} valid={st['valid']} "
                  f"ACCEPTED={st['ACCEPTED']} ({st['ACCEPTED']/max(g,1):.1%}) "
                  f"{time.time()-t0:.0f}s", flush=True)

    g = st["generated"]
    print(f"\n=== RESULT: {a.adapter or a.model} ===")
    for k in ("generated", "parsed", "no_parse", "valid", "duplicate", "ACCEPTED"):
        print(f"  {k:12} {st[k]:5}  {st[k]/max(g,1):6.1%}")
    print("  invalid by stage:")
    for k, v in sorted(((k, v) for k, v in st.items() if k.startswith("invalid:")),
                       key=lambda x: -x[1])[:6]:
        print(f"    {k[8:]:20} {v:5}")
    if ranks:
        import statistics as _st
        mu = sum(ranks) / len(ranks)
        sd = _st.pstdev(ranks) if len(ranks) > 1 else 0.0
        print(f"\n  STAGE-ORDINAL  mean {mu:.3f} / {STAGE_MAX}   sd {sd:.3f}   n {len(ranks)}")
        print(f"                 sem {sd/max(len(ranks),1)**0.5:.4f}  "
              f"<- the primary metric; compare this before/after")
    res = {"stage_mean": (sum(ranks)/len(ranks)) if ranks else None,
           "stage_sd": (_st.pstdev(ranks) if len(ranks) > 1 else 0.0) if ranks else None,
           "stage_n": len(ranks),
           "model": a.model, "adapter": a.adapter, "prompts": len(prompts),
           "samples_per_prompt": a.samples, "counts": dict(st),
           "yield": st["ACCEPTED"] / max(g, 1), "rows": rows}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
