"""Experiment 3b: two-stage pipeline (advisor, 13 Oct 2026: "divide the task into separate stages").

Stage 1 (decide, no code): the LLM analyses the model and decides ONE architecture change for better accuracy after
         1 epoch, as JSON (what, where, which modules, why). In the web condition the SAME prompt additionally
         contains the web suggestions, with the option to use one of them or none.
Stage 2 (implement): the v3 generator (exp3.edit_alter: whole-file edit, smoke check, repairs) implements exactly
         that planned change; the plan is the only extra text in its prompt. Repairs do not repeat the plan.

Both conditions run both stages; the ONLY difference is the suggestions block in stage 1. So a difference between
staged_base and staged_web measures the web information, not the effect of planning itself.
If stage 1 fails (no valid JSON after its retry), the job continues without a plan (recorded as stage1_ok=False);
this rule is the same in both conditions.

Usage: python -m ab.gpt.util.knowledge.exp3.staged_alter --set SET.csv --vllm_url URL [--extra_info JSON] --name NAME
"""
import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from ab.gpt.act.alter.web_alter import build_jobs, load_rows
from ab.gpt.util.knowledge.exp3 import edit_alter as ea
from ab.gpt.util.knowledge.exp3.comprehension import code_index, parse_json

SYSTEM1 = "You are an expert in neural network architecture. You analyse precisely and answer only in the requested format."

STAGE1 = """Below is a PyTorch model for CIFAR-10 image classification. It is trained for exactly ONE epoch and then \
evaluated (current accuracy after 1 epoch: @ACC@).

Decide the ONE architecture change that most likely increases its accuracy after that single epoch. Optimise for \
one epoch, not for long training: a model that learns quickly (for example fewer parameters, simpler or shallower \
parts) can beat a larger one. The change must only touch the architecture (layers, blocks, forward pass): no new \
hyperparameters and no change to supported_hyperparameters, train_setup or learn. Do NOT write any code.

=== MODEL ===
@CODE@

@SUGG@Answer ONLY with one JSON object in exactly this format:
{"used_suggestion": "<the web suggestion you use, named briefly, or \"none\" if the change is your own>",
 "change": "<what to change, concretely, in 1-3 sentences>",
 "location": "<ClassName> or <ClassName>.<method> in the code above where the change is made",
 "modules": ["nn.<Name>", ...],
 "reason": "<one sentence: why this helps after one epoch>"}
"""

SUGG_BLOCK = """=== WEB RESEARCH SUGGESTIONS for this architecture ===
@SUGG@

You may use one of these suggestions if it is the most promising change for this model after one epoch; \
otherwise choose your own change and set "used_suggestion" to "none".

"""

STAGE1_RETRY = "Your answer was not a valid JSON object in the requested format. Answer again with ONLY the JSON object."

PLAN_HEADER = "=== PLANNED CHANGE (decided in an analysis step; implement exactly this change) ===\n"


def stage1(backend, job, out_dir, idx, a):
    row = job["row"]
    sugg = SUGG_BLOCK.replace("@SUGG@", job["extra"].strip()) if job["extra"] else ""
    user = (STAGE1.replace("@ACC@", f"{float(row['accuracy']):.4f}").replace("@CODE@", row["nn_code"])
            .replace("@SUGG@", sugg))
    msgs = [{"role": "system", "content": SYSTEM1}, {"role": "user", "content": user}]
    log, plan = [], None
    for attempt in range(2):
        try:
            out, finish, n_in, _ = backend.generate(msgs, job["seed"] + 500 + 1000 * attempt)
        except Exception as e:
            log.append({"attempt": attempt, "error": f"{type(e).__name__}: {e}"[:300]})
            break
        log.append({"attempt": attempt, "user": msgs[-1]["content"], "answer": out, "finish": finish, "prompt_tokens": n_in})
        js = parse_json(out)
        if isinstance(js, dict) and str(js.get("change") or "").strip():
            plan = js
            break
        msgs = msgs + [{"role": "assistant", "content": out}, {"role": "user", "content": STAGE1_RETRY}]
    d = out_dir / "stage1"
    d.mkdir(exist_ok=True)
    (d / f"B{idx}.json").write_text(json.dumps({"nn": row["nn"], "web": bool(job["extra"]), "plan": plan, "log": log}, indent=1))
    info = dict(stage1_ok=plan is not None)
    if plan is None:
        return "", info
    classes, _ = code_index(row["nn_code"])
    used = str(plan.get("used_suggestion") or "none").strip()
    info.update(used_suggestion=None if used.lower() in ("none", "-1", "") else used[:120],
                plan_location_ok=str(plan.get("location") or "").split(".")[0].strip("` ") in classes,
                plan_modules=plan.get("modules"))
    mods = ", ".join(m for m in plan.get("modules") or [] if isinstance(m, str)) or "none"
    text = (f"Change: {str(plan.get('change')).strip()}\nWhere: {str(plan.get('location') or '').strip()}\n"
            f"Modules: {mods}\nWhy: {str(plan.get('reason') or '').strip()}")
    return text, info


def run_job(backend, job, out_dir, idx, a):
    plan, info = stage1(backend, job, out_dir, idx, a)
    job2 = dict(job, extra=plan, extra_header=PLAN_HEADER)
    rec = ea.run_job(backend, job2, out_dir, idx, a)
    rec.update(info, web=bool(job["extra"]), extra_info=bool(job["extra"]))
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
    p.add_argument("--max_new_tokens", type=int, default=8192)
    p.add_argument("--min_new_tokens", type=int, default=1024)
    p.add_argument("--ctx", type=int, default=None)
    p.add_argument("--ctx_margin", type=int, default=64)
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--dataset", default="cifar-10")
    p.add_argument("--only", default=None, help="comma-separated network names (for tests)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=str(Path.home() / "web_alter_runs_exp3"))
    p.add_argument("--name", default=None)
    a = p.parse_args()

    extra = json.load(open(a.extra_info)) if a.extra_info else {}
    cond = "staged_web" if a.extra_info else "staged_base"
    out_dir = Path(a.out) / (a.name or f"{datetime.now():%Y%m%d-%H%M}_{cond}")
    if out_dir.exists():
        sys.exit(f"[staged_alter] {out_dir} exists -- choose another --name")
    rows = load_rows(a.set, "img-classification", a.dataset)
    jobs = build_jobs(rows, a.samples, a.seed, extra)
    if a.only:
        keep = set(a.only.split(","))
        jobs = [j for j in jobs if j["row"]["nn"] in keep]
        if not jobs:
            sys.exit(f"[staged_alter] none of {sorted(keep)} in {a.set}")
    jobs = jobs[:a.limit]
    backend = ea.Backend(a.vllm_url, a.vllm_model, a)
    out_dir.mkdir(parents=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ea.REPO, text=True).strip()
    except Exception:
        commit = "unknown"
    prompts_md5 = hashlib.md5((SYSTEM1 + STAGE1 + SUGG_BLOCK + STAGE1_RETRY + PLAN_HEADER + ea.d2.SYSTEM + ea.PROMPT
                               + ea.REPAIR_RUN + ea.d2.REPAIR_APPLY + ea.REPAIR_TRUNC).encode()).hexdigest()
    meta = dict(vars(a), experiment="exp3b-staged", condition=cond, backend=backend.name, nn_gpt_commit=commit,
                nn_dataset=version("nn-dataset"), prompts_md5=prompts_md5,
                networks=len({j["row"]["nn"] for j in jobs}), jobs=len(jobs),
                with_extra_info=sum(bool(j["extra"]) for j in jobs), started=datetime.now().isoformat(timespec="seconds"))
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[staged_alter] {cond}: {len(jobs)} jobs, {meta['networks']} networks, {backend.name} -> {out_dir}", flush=True)

    n = {"code": 0, "smoke_ok": 0, "plan": 0, "used_sugg": 0}
    with open(out_dir / "results.jsonl", "w") as log, ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_job, backend, j, out_dir, i, a) for i, j in enumerate(jobs)]
        for done, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            log.write(json.dumps(r) + "\n")
            log.flush()
            n["code"] += r["code"]
            n["smoke_ok"] += bool(r["smoke_last"])
            n["plan"] += r["stage1_ok"]
            n["used_sugg"] += bool(r.get("used_suggestion"))
            print(f"[{done}/{len(jobs)}] B{r['B']} {r['nn'][:26]:26s} plan={r['stage1_ok']} used={str(r.get('used_suggestion'))[:30]!r} "
                  f"code={r['code']} rounds={r['rounds']} smoke first={r['smoke_first']} last={r['smoke_last']} "
                  f"changed={r['changed_lines']} {r['seconds']}s", flush=True)
    print(f"[staged_alter] done: {len(jobs)} jobs | plan {n['plan']} | plan uses a web suggestion {n['used_sugg']} | "
          f"code {n['code']} | smoke OK {n['smoke_ok']} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
