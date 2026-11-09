"""Two predictors, two jobs. Rank with epoch 3; estimate the number with the fine-tune.

MEASURED HEAD-TO-HEAD on 100 held-out ft50 runs (50 epochs, per-epoch curves; the fine-tune never
saw them). Scored on rank correlation, absolute error, and top-50% overlap -- the last being the
decision a selection funnel actually makes:

    method                                    rho      MAE       top-50%
    in-house zero-cost surrogate            -0.178   1.80 dB       --      (n=326)
    stock ABrain/Accuracy-Prediction        +0.226   1.616 dB     57%      (n=280)
    that model fine-tuned on our denoising  +0.629   0.509 dB     74%      (n=100)
    just use epoch 3's PSNR                 +0.728   2.559 dB     82%      (n=100)

So: NOTHING LEARNED BEATS EPOCH 3 AT RANKING, and the fine-tune beats everything at calibration by
5x. This module hard-codes that split so the two never get confused again.

  rank_for_selection()   epoch 3 only. No model call, no GPU, no network, no failure mode.
  estimate_final_db()    the fine-tuned adapter, for reporting an expected score with an error bar.

WHY EPOCH 3 IS THIS STRONG. A denoiser's early PSNR is largely set by whether the architecture works
at all, so the signal arrives early: rho(ep1)=+0.491, rho(ep2)=+0.661, rho(ep3)=+0.698 over 280
models. That is an unusually high bar, and it is task-specific -- in classification the early epochs
are far noisier and a learned predictor has much more room. The stock model's weakness here was
DOMAIN TRANSFER (it was trained on LEMUR image classification), not the method: fine-tuning on
denoising tripled its rho, from +0.226 to +0.629. It still lost.

WHY THIS FILE EXISTS AT ALL. The in-house surrogate was enabled via `--predictor --predict-keep 0.5`
in four GA launchers and logged "Predictor funnel ON" in all ten production GA arms, so for months it
discarded the top half of every generation by a ranking that is ANTI-correlated with truth. Its own
k-fold CV reported +0.693 while production ran at -0.178, and nobody compared the two. Any predictor
proposed here gets held-out validation against the epoch-3 baseline before it is allowed near a
selection decision -- that is the lesson, not a style preference.
"""
import json
import os

THESIS = "/a/mm" if os.path.isdir("/a/mm/nn-gpt") else "/shared/ssd/home/b-b-ahmed/thesis"
ADAPTER = os.path.join(THESIS, "sweep_out", "acc_ft", "adapter")
REPORT = os.path.join(THESIS, "sweep_out", "acc_ft", "ft_report.json")
BASE = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
SCALE = 48.0

# From ft_report.json, so a caller can quote the error bar honestly rather than guess it.
CALIBRATION_MAE_DB = 0.509
SELECTION_RHO = 0.728
_model = _tok = None


def rank_for_selection(candidates):
    """Order candidates best-first for a keep/discard decision.

    `candidates` is an iterable of dicts carrying at least `dB_ep3` (or `curve`, from which epoch 3
    is read). Returns the list sorted best-first.

    THIS DELIBERATELY IGNORES THE LEARNED PREDICTOR. On the same 100 models epoch 3 ranks at
    rho=+0.728 with 82% top-half overlap against the fine-tune's +0.629 / 74%. It is also free,
    deterministic, and cannot fail -- no GPU, no adapter to load, no JSON to parse. Adding a model
    call here would cost accuracy AND add failure modes.
    """
    def ep3(c):
        if c.get("dB_ep3") is not None:
            return c["dB_ep3"]
        curve = c.get("curve") or []
        d = {e: v for e, v in curve}
        # fall back down the curve rather than to 0: a 2-epoch run still carries real signal
        for e in (3, 2, 1):
            if e in d:
                return d[e]
        return float("-inf")
    return sorted(candidates, key=ep3, reverse=True)


def _load():
    global _model, _tok
    if _model is not None:
        return _model, _tok
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if not os.path.isdir(ADAPTER):
        raise RuntimeError(f"no fine-tuned adapter at {ADAPTER}")
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=qc, device_map="auto",
                                             trust_remote_code=True)
    _model, _tok = PeftModel.from_pretrained(m, ADAPTER).eval(), tok
    return _model, _tok


def estimate_final_db(nn_code, dB_ep1, dB_ep2, dB_ep3, max_epochs=50,
                      task="img-denoising", dataset="denoise", metric="psnr"):
    """Expected final best PSNR in dB, plus the epoch it is expected at.

    Returns (dB, best_epoch, mae_dB) where mae_dB is this predictor's MEASURED held-out error, so
    callers report an interval instead of a bare point estimate. Needs a GPU and the adapter.

    max_epochs is a real argument, not a constant: 96% of the available training curves were 3-5
    epochs, and telling the model "50" for a 5-epoch run taught systematic under-prediction until
    every example carried its own budget. Pass the budget you will actually train to.
    """
    import torch
    from acc_ft_dataset import SYSTEM_PROMPT, user_message
    from acc_ft_train import parse_json_out
    model, tok = _load()
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message(
                task, dataset, metric, nn_code,
                dB_ep1 / SCALE * 100, dB_ep2 / SCALE * 100, dB_ep3 / SCALE * 100, max_epochs)}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=64, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    d = parse_json_out(gen if "}" in gen else gen + "}")
    if not d:
        raise ValueError(f"unparseable prediction: {gen[:120]!r}")
    return (float(d["best_accuracy"]) / 100.0 * SCALE, int(d["best_epoch"]), CALIBRATION_MAE_DB)


def provenance():
    """The measured numbers behind this split, for a report or a paper table."""
    r = json.load(open(REPORT)) if os.path.exists(REPORT) else {}
    return {"selection": {"method": "epoch-3 PSNR", "spearman": SELECTION_RHO,
                          "top50_overlap": (r.get("baseline_ep3") or {}).get("top50_overlap")},
            "calibration": {"method": "ABrain LoRA fine-tuned on denoising",
                            "mae_dB": CALIBRATION_MAE_DB,
                            "spearman": (r.get("finetuned") or {}).get("spearman")},
            "rejected": {"in_house_zero_cost": -0.178, "stock_abrain": 0.226},
            "n_holdout": r.get("n")}


if __name__ == "__main__":
    print(json.dumps(provenance(), indent=1))
