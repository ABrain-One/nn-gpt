"""Experiment 3 generator = base pipeline v3 (advisor meeting 13 Oct 2026).

Same design as Experiment 2 (whole-file edit -> smoke check -> up to 2 repair rounds in the same conversation,
identical in both conditions; web suggestions only in the first message, switched on by --extra_info).
New file; Experiment 2 code and results are not modified (its functions are imported read-only).

Changes against Experiment 2 (all apply to BOTH conditions):
  1. Output budget per request: max_tokens = min(--max_new_tokens, context - prompt tokens - margin), prompt tokens
     counted by the vLLM server (/tokenize). Exp2 used a fixed 4096, which cut off every answer for the three largest
     networks (InceptionV3-1, MaxVit, SwinTransformer: 30/30 jobs without code).
  2. A cut-off answer (finish_reason "length", no closing </nn>) gets its own repair message ("your answer was cut
     off, return only the code") instead of the misleading "no code block found".
  3. The task states the 1-epoch goal explicitly (advisor: optimising for one epoch differs from optimising for
     many epochs; quickly trainable models win).
  4. The repair message asks to fix the cause and forbids deleting error checks or disabling parts of the model
     (observed in a MaxVit test: the model removed a `raise ValueError` instead of fixing its own change);
     results record removed_checks = raise/assert statements lost against the original.
  OOM: the smoke check already treats CUDA out-of-memory as an infrastructure result (no repair round); the code is
  kept and goes to Eval.

Usage: python -m ab.gpt.util.knowledge.exp3.edit_alter --set SET.csv --vllm_url URL [--extra_info JSON] --name NAME
       test on CPU without touching the GPU:  CUDA_VISIBLE_DEVICES= python -m ... --only MaxVit --samples 1
"""
import argparse
import ast
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from ab.gpt.act.alter.web_alter import build_jobs, family, load_rows
from ab.gpt.util.Const import new_nn_file, new_out_file
from ab.gpt.util.knowledge.exp2 import delta_alter as d2

REPO = Path(__file__).resolve().parents[5]

PROMPT = """Task: improve the ARCHITECTURE of the baseline model below with a few targeted changes, \
to increase its accuracy on CIFAR-10 image classification after 1 training epoch.

Important: the model is trained for exactly ONE epoch and then evaluated. Optimise for accuracy after that single \
epoch, not after long training: a model that learns quickly (for example fewer parameters, simpler or shallower \
parts) can beat a larger model that would need many epochs.

=== BASELINE MODEL (current accuracy: @ACC@) ===
<nn>
@CODE@
</nn>

@EXTRA@=== RULES ===
@FMT_RULES@
3. Change only the architecture (layers, blocks, forward pass). Do not change supported_hyperparameters,
   train_setup or learn.
4. Use only hyperparameters the baseline already reads from prm; never add new prm keys.
5. If you use a new class, function or name, define or import it in your change (e.g. `from typing import Optional`).
6. Do not assume a fixed input resolution: images may be resized (e.g. to 299 px). Never hard-code spatial sizes.
7. Keep the change small and focused: a few targeted changes (at most @BUDGET@ added or removed lines), not a rewrite.

@FMT_EXAMPLE@
"""

REPAIR_RUN = """The model below fails when it is built and trained:

<nn>
@CODE@
</nn>

Error: @ERR@
@WHERE@
Fix the CAUSE of this error with a minimal change to THIS code exactly as shown. Never delete or weaken error \
checks (raise, assert) and never disable parts of the model just to make the error disappear; if one of your own \
changes caused the error, adjust or undo that change. Keep the other intended architecture changes; do not change \
supported_hyperparameters, train_setup or learn unless the error is there; never add new prm keys; define or import \
every name you use. @OUT@
"""

REPAIR_TRUNC = """Your answer was cut off because it was too long: the closing </nn> tag is missing. \
Reply again with ONLY the complete modified code inside <nn></nn> tags, with no explanation before or after it. \
Never omit or abbreviate any part of the code.
"""


def n_checks(code):
    """Number of raise/assert statements (to measure whether repairs remove error checks)."""
    try:
        return sum(isinstance(x, (ast.Raise, ast.Assert)) for x in ast.walk(ast.parse(code)))
    except SyntaxError:
        return 0


class ContextFull(Exception):
    pass


def _http(url, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Backend:
    """OpenAI-compatible vLLM chat endpoint with a per-request output budget."""

    def __init__(self, url, model, a):
        self.a = a
        self.base = url.rstrip("/")
        if not self.base.endswith("/v1"):
            self.base += "/v1"
        self.root = self.base[:-3]
        cards = _http(self.base + "/models")["data"]
        card = next((c for c in cards if model in (None, c["id"])), None)
        if card is None:
            sys.exit(f"[edit_alter] model {model!r} not served at {url}; available: {[c['id'] for c in cards]}")
        self.model = card["id"]
        self.ctx = a.ctx or int(card.get("max_model_len") or 32768)
        self.exact_count = True
        self.name = f"vllm:{self.model}@{url} ctx={self.ctx}"

    def count(self, msgs):
        try:
            return int(_http(self.root + "/tokenize", {"model": self.model, "messages": msgs})["count"])
        except Exception:  # older server: conservative estimate (code is ~4 characters per token)
            self.exact_count = False
            return sum(len(m["content"]) for m in msgs) // 3 + 16 * len(msgs)

    def generate(self, msgs, seed):
        n_in = self.count(msgs)
        max_out = min(self.a.max_new_tokens, self.ctx - n_in - self.a.ctx_margin)
        if max_out < self.a.min_new_tokens:
            raise ContextFull(f"context full: prompt {n_in} tokens, only {max_out} left for the answer")
        body = dict(model=self.model, messages=msgs, max_tokens=max_out, temperature=self.a.temperature,
                    top_p=self.a.top_p, top_k=self.a.top_k, seed=seed)
        r = _http(self.base + "/chat/completions", body, timeout=self.a.timeout)
        c = r["choices"][0]
        return c["message"]["content"] or "", c.get("finish_reason"), n_in, max_out


def run_job(backend, job, out_dir, idx, a):
    row, base = job["row"], job["row"]["nn_code"]
    rec = dict(B=idx, nn=row["nn"], fam=family(row["nn"]), sample=job["k"], seed=job["seed"],
               extra_info=bool(job["extra"]))
    extra = (job.get("extra_header", d2.EXTRA_HEADER) + job["extra"].strip() + "\n\n") if job["extra"] else ""
    f = d2.FMT["whole"]
    budget = max(40, int(0.3 * len(base.splitlines())))
    msgs = d2.msgs_for(d2.fill(PROMPT, ACC=f"{float(row['accuracy']):.4f}", CODE=base, EXTRA=extra, BUDGET=budget,
                               FMT_RULES=f["rules"], FMT_EXAMPLE=f["example"]))
    work = out_dir / "work" / f"B{idx}"
    current, code, attempts, transcript = base, None, [], []
    t0 = time.time()
    for rnd in range(a.max_repairs + 1):
        try:
            out, finish, n_in, max_out = backend.generate(msgs, job["seed"] + 1000 * rnd)
        except Exception as e:
            attempts.append({"round": rnd, "stage": "llm", "error": f"{type(e).__name__}: {e}"[:300]})
            break
        transcript.append({"round": rnd, "user": msgs[-1]["content"], "answer": out, "finish": finish,
                           "prompt_tokens": n_in, "max_tokens": max_out})
        msgs = msgs + [{"role": "assistant", "content": out}]  # repairs continue the conversation (Self-Debug)
        new, err = d2.apply_edit("whole", current, out, budget)
        if not new:
            cut = finish == "length" and d2.extract_whole(out) is None
            attempts.append({"round": rnd, "stage": "apply", "ok": False, "truncated": cut,
                             "error": "answer cut off at the output limit" if cut else err})
            repair = REPAIR_TRUNC if cut else d2.fill(d2.REPAIR_APPLY, ERR=err, CODE=current, HINT="", OUT=f["out"])
            msgs = msgs + [{"role": "user", "content": repair}]
            continue
        code = current = new
        res = d2.smoke(code, row, work / f"r{rnd}", a.smoke_timeout)
        attempts.append({"round": rnd, "stage": "smoke", **res})
        if res.get("ok") or res.get("infra"):
            break
        where = f"Location: {res['where']}\n" if res.get("where") else ""
        msgs = msgs + [{"role": "user", "content": d2.fill(REPAIR_RUN, CODE=current, ERR=res["error"], WHERE=where,
                                                           OUT=f["out"])}]
    smokes = [x for x in attempts if x["stage"] == "smoke"]
    parses = False
    if code:
        try:
            ast.parse(code)
            parses = True
        except SyntaxError:
            pass
    d = out_dir / ("synth_nn" if code else "failed") / f"B{idx}"
    d.mkdir(parents=True, exist_ok=True)
    (d / new_out_file).write_text("\n\n===== next round =====\n\n".join(t["answer"] for t in transcript))
    (d / "transcript.json").write_text(json.dumps(transcript, indent=1))
    (d / "attempts.json").write_text(json.dumps(attempts, indent=1))
    (d / f"original_{row['nn']}.py").write_text(base)
    row.to_pickle(d / "dataframe.df")
    if code:
        (d / new_nn_file).write_text(code)
    rec.update(code=bool(code), parses=parses, rounds=len(transcript),
               apply_failures=sum(x["stage"] == "apply" for x in attempts),
               truncated=sum(bool(x.get("truncated")) for x in attempts),
               llm_errors=sum(x["stage"] == "llm" for x in attempts),
               finish=[t["finish"] for t in transcript],
               smoke_first=smokes[0].get("ok") if smokes else None, smoke_last=smokes[-1].get("ok") if smokes else None,
               smoke_infra=any(x.get("infra") for x in smokes),
               removed_checks=max(0, n_checks(base) - n_checks(code)) if code else None,
               changed_lines=d2.changed_lines(base, code) if code else None, seconds=round(time.time() - t0, 1))
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", required=True)
    p.add_argument("--extra_info", default=None, help="JSON {nn or family: text}; omit for the baseline condition")
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=3000)
    p.add_argument("--max_repairs", type=int, default=2)
    p.add_argument("--smoke_timeout", type=float, default=300)
    p.add_argument("--vllm_url", required=True)
    p.add_argument("--vllm_model", default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=8192, help="upper limit; lowered per request to fit")
    p.add_argument("--min_new_tokens", type=int, default=1024, help="below this, the request is not sent")
    p.add_argument("--ctx", type=int, default=None, help="context length; default: read from the server")
    p.add_argument("--ctx_margin", type=int, default=64)
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--dataset", default="cifar-10")
    p.add_argument("--only", default=None, help="comma-separated network names (for tests)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=str(Path.home() / "web_alter_runs_exp3"))
    p.add_argument("--name", default=None)
    a = p.parse_args()

    extra = json.load(open(a.extra_info)) if a.extra_info else {}
    cond = "web" if a.extra_info else "base"
    out_dir = Path(a.out) / (a.name or f"{datetime.now():%Y%m%d-%H%M}_{cond}")
    if out_dir.exists():
        sys.exit(f"[edit_alter] {out_dir} exists -- choose another --name")
    rows = load_rows(a.set, "img-classification", a.dataset)
    jobs = build_jobs(rows, a.samples, a.seed, extra)
    if a.only:  # filter after pairing, so a test job is identical to the same job in the full run
        keep = set(a.only.split(","))
        jobs = [j for j in jobs if j["row"]["nn"] in keep]
        if not jobs:
            sys.exit(f"[edit_alter] none of {sorted(keep)} in {a.set}")
    jobs = jobs[:a.limit]
    backend = Backend(a.vllm_url, a.vllm_model, a)
    out_dir.mkdir(parents=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        commit = "unknown"
    prompts_md5 = hashlib.md5((d2.SYSTEM + PROMPT + d2.EXTRA_HEADER + REPAIR_RUN + d2.REPAIR_APPLY + REPAIR_TRUNC
                               + json.dumps(d2.FMT["whole"])).encode()).hexdigest()
    meta = dict(vars(a), experiment="exp3-base-v3", condition=cond, backend=backend.name, nn_gpt_commit=commit,
                nn_dataset=version("nn-dataset"), prompts_md5=prompts_md5, networks=len({j["row"]["nn"] for j in jobs}), jobs=len(jobs),
                with_extra_info=sum(bool(j["extra"]) for j in jobs), started=datetime.now().isoformat(timespec="seconds"))
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[edit_alter] {cond}: {len(jobs)} jobs, {meta['networks']} networks, {backend.name} -> {out_dir}", flush=True)

    n = {"code": 0, "smoke_ok": 0, "repaired": 0, "truncated": 0}
    with open(out_dir / "results.jsonl", "w") as log, ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_job, backend, j, out_dir, i, a) for i, j in enumerate(jobs)]
        for done, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            log.write(json.dumps(r) + "\n")
            log.flush()
            n["code"] += r["code"]
            n["smoke_ok"] += bool(r["smoke_last"])
            n["repaired"] += bool(r["smoke_last"]) and r["smoke_first"] is False
            n["truncated"] += r["truncated"] > 0
            print(f"[{done}/{len(jobs)}] B{r['B']} {r['nn'][:26]:26s} code={r['code']} rounds={r['rounds']} "
                  f"finish={r['finish']} smoke first={r['smoke_first']} last={r['smoke_last']} "
                  f"changed={r['changed_lines']} {r['seconds']}s", flush=True)
    if not backend.exact_count:
        print("[edit_alter] note: server has no /tokenize; prompt tokens were estimated", flush=True)
    print(f"[edit_alter] done: {len(jobs)} jobs | code {n['code']} | smoke OK {n['smoke_ok']} "
          f"(of which repaired {n['repaired']}) | jobs with a cut-off answer {n['truncated']} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
