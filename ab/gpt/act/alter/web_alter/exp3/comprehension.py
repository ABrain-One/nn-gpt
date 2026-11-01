"""Comprehension test (advisor, 13 Oct 2026): does the LLM understand the web suggestions for a given network?

Separate prompt, no code generation. For each network of the set, the LLM gets the model code and the same web
suggestions the generator gets, and must analyse every suggestion as JSON (which torch.nn modules it adds, whether
the model already has them, where in the code it would go, whether it is feasible as an architecture-only change,
its expected effect after ONE epoch). Answers that can be checked against the code are checked automatically:
  location_ok      the named class (and method, if given) exists in the model code
  modules_real     every named module exists in torch.nn (no invented layers)
  present_agrees   the claim "already present" matches the code (all named nn.<X> occur in it)
Everything else (feasibility, expected effect, reason) is saved for reading and for later comparison with the
generation results. Uses only the vLLM server, not the GPU. This analysis step is also stage 1 of a multi-stage
pipeline (analyse -> decide -> implement), so it is reusable.

Usage: python -m ab.gpt.act.alter.web_alter.exp3.comprehension --set SET.csv --extra_info JSON --vllm_url URL --name NAME
"""
import argparse
import ast
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from ab.gpt.act.alter.web_alter import build_jobs, load_rows
from ab.gpt.act.alter.web_alter.exp3.edit_alter import Backend

SYSTEM = "You are an expert in neural network architecture. You analyse precisely and answer only in the requested format."

PROMPT = """Below are a PyTorch model and suggestions from web research for improving its architecture.
Do NOT write any code. Analyse EACH suggestion and answer ONLY with one JSON object in exactly this format:

{"suggestions": [{"suggestion": "<short restatement>", "modules": ["nn.<Name>", ...], "already_present": true or false,
  "location": "<ClassName> or <ClassName>.<method> where the change would be made", "feasible": true or false,
  "effect_after_1_epoch": "help" or "hurt" or "neutral", "reason": "<one sentence>"}],
 "best": <0-based index of the most promising suggestion, or -1 if none is useful>}

Definitions:
- modules: the torch.nn layer classes the change would add, e.g. ["nn.BatchNorm2d"]; [] if it adds none.
- already_present: true if the model already uses all of these modules.
- location: a class (and method) that exists in the model code below.
- feasible: true if it can be done by changing only the architecture (no new hyperparameters, no change to
  supported_hyperparameters, train_setup or learn).
- effect_after_1_epoch: expected effect on CIFAR-10 accuracy after training for exactly ONE epoch.

=== MODEL (accuracy after 1 epoch: @ACC@) ===
@CODE@

=== WEB RESEARCH SUGGESTIONS ===
@EXTRA@
"""


def code_index(code):
    """{class name: set of method names} and the set of nn.<X> names used in the code."""
    classes = {}
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.ClassDef):
            classes[node.name] = {f.name for f in node.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))}
    used = set(re.findall(r"\bnn\.([A-Za-z_][A-Za-z0-9_]*)", code))
    return classes, used


def parse_json(text):
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(text[i:j + 1])
    except json.JSONDecodeError:
        return None


def check(item, classes, used):
    import torch.nn as tnn
    names = [m.split(".")[-1] for m in item.get("modules") or [] if isinstance(m, str)]
    loc = str(item.get("location") or "").strip().strip("`")
    cls, _, meth = loc.partition(".")
    location_ok = cls in classes and (not meth or meth.split("(")[0] in classes[cls])
    modules_real = all(hasattr(tnn, n) for n in names)
    present_truth = bool(names) and all(n in used for n in names)
    claim = item.get("already_present")
    return dict(location_ok=location_ok, modules_real=modules_real, present_truth=present_truth,
                present_agrees=(claim is present_truth) if isinstance(claim, bool) and names else None)


def run_one(backend, job):
    row = job["row"]
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT.replace("@ACC@", f"{float(row['accuracy']):.4f}")
             .replace("@CODE@", row["nn_code"]).replace("@EXTRA@", job["extra"].strip())}]
    rec = dict(nn=row["nn"], extra=job["extra"])
    try:
        out, finish, n_in, _ = backend.generate(msgs, job["seed"])
    except Exception as e:
        return dict(rec, error=f"{type(e).__name__}: {e}"[:300])
    classes, used = code_index(row["nn_code"])
    js = parse_json(out)
    items = js.get("suggestions") if isinstance(js, dict) else None
    rec.update(answer=out, finish=finish, prompt_tokens=n_in, json_ok=isinstance(items, list),
               best=js.get("best") if isinstance(js, dict) else None,
               items=[dict(it, **check(it, classes, used)) for it in items if isinstance(it, dict)] if items else [])
    return rec


def report(recs):
    items = [it for r in recs for it in r.get("items", [])]
    n = len(recs)
    def rate(key):
        v = [it[key] for it in items if it.get(key) is not None]
        return f"{sum(v)}/{len(v)}" + (f" = {sum(v) / len(v):.0%}" if v else "")
    lines = [f"networks {n} | answered as valid JSON {sum(r.get('json_ok', False) for r in recs)}/{n} | "
             f"errors {sum('error' in r for r in recs)} | suggestions analysed {len(items)}",
             f"location exists in the code:           {rate('location_ok')}",
             f"named modules exist in torch.nn:       {rate('modules_real')}",
             f"'already present' claim matches code:  {rate('present_agrees')}",
             f"judged feasible (architecture only):   {sum(it.get('feasible') is True for it in items)}/{len(items)}",
             "expected effect after 1 epoch:         " + ", ".join(
                 f"{k} {sum(it.get('effect_after_1_epoch') == k for it in items)}" for k in ("help", "neutral", "hurt")),
             f"networks where no suggestion is useful (best = -1): {sum(r.get('best') == -1 for r in recs)}/{n}"]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", required=True)
    p.add_argument("--extra_info", required=True)
    p.add_argument("--vllm_url", required=True)
    p.add_argument("--vllm_model", default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=4000)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=-1)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_new_tokens", type=int, default=3000)
    p.add_argument("--min_new_tokens", type=int, default=512)
    p.add_argument("--ctx", type=int, default=None)
    p.add_argument("--ctx_margin", type=int, default=64)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--dataset", default="cifar-10")
    p.add_argument("--only", default=None)
    p.add_argument("--out", default=str(Path.home() / "web_alter_runs_exp3"))
    p.add_argument("--name", default=None)
    a = p.parse_args()

    out_dir = Path(a.out) / (a.name or f"{datetime.now():%Y%m%d-%H%M}_comprehension")
    if out_dir.exists():
        sys.exit(f"[comprehension] {out_dir} exists -- choose another --name")
    extra = json.load(open(a.extra_info))
    jobs = [j for j in build_jobs(load_rows(a.set, "img-classification", a.dataset), 1, a.seed, extra) if j["extra"]]
    if a.only:
        jobs = [j for j in jobs if j["row"]["nn"] in set(a.only.split(","))]
    backend = Backend(a.vllm_url, a.vllm_model, a)
    out_dir.mkdir(parents=True)
    (out_dir / "run_meta.json").write_text(json.dumps(dict(vars(a), backend=backend.name, networks=len(jobs),
                                                           started=datetime.now().isoformat(timespec="seconds")), indent=1))
    print(f"[comprehension] {len(jobs)} networks with suggestions, {backend.name} -> {out_dir}", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        recs = list(ex.map(lambda j: run_one(backend, j), jobs))
    with open(out_dir / "answers.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    rep = report(recs)
    (out_dir / "report.txt").write_text(rep + "\n")
    print(rep, flush=True)


if __name__ == "__main__":
    main()
