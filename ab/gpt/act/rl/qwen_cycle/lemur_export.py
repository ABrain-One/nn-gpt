"""Export the networks generated in the Qwen self-improvement experiments to the LEMUR / NN Dataset format.

Two steps:
  prepare  (CPU)  select the strict-valid networks of the frozen evaluations, best first and unique, convert each to the
                  NN Dataset interface and check that it builds, runs and trains one batch. Writes <out>/<id>.py and
                  <out>/index.csv (source model, seed, prompt, our 10-epoch accuracy, parameters).
  register (GPU)  train every prepared network with the standard NN Dataset pipeline (ab.nn.api.check_nn), which saves
                  the model file and its per-epoch statistics in nn-dataset (ab/nn/nn, ab/nn/stat) under PREFIX.

Conversion (the network itself is not changed):
  1. the code is extracted exactly as in the strict reward (TunePPO._strict_extract) - this is the evaluated program;
  2. 'import torch', 'import torch.nn as nn', 'import torch.nn.functional as F' are added where missing (the strict
     check executed the code in a namespace that already contained torch, nn and F);
  3. top-level statements that would run on import (calls, 'if __name__ == ...' blocks) are removed; definitions stay;
  4. 'self.device = device' is set in Net.__init__ if the network does not store the device (NN Dataset interface);
  5. supported_hyperparameters() and the train_setup / learn methods are added with the general NNGPT functionality
     (ab.gpt.util.nn.PostprocessNN: _fix_hyperparams, _fix_missing_train_setup, _fix_missing_learn; the stubs use SGD
     with prm['lr'] and prm['momentum'] and cross-entropy). The other PostprocessNN repairs are not applied, because
     the networks already build and run as written.

Usage:
  python lemur_export.py prepare  <release_dir> <out_dir> [--limit N]
  python lemur_export.py register <out_dir> [--limit N] [--epochs 5]      (in the NNGPT environment, GPU)
"""
import argparse, ast, csv, hashlib, json, os, re, sys, glob

PREFIX = "qwcyc"
PRM = {"lr": 0.01, "momentum": 0.9, "dropout": 0.2}        # the prm the networks were built with (TunePPO._PRM)
HEADER = ("import torch", "import torch.nn as nn", "import torch.nn.functional as F")


def strict_extract(text):                                    # identical to TunePPO._strict_extract
    code = text
    if "<nn>" in code:
        code = code.split("<nn>", 1)[1].split("</nn>", 1)[0]
    m = re.search(r"```(?:python)?\s*\n(.*?)```", code, re.S)
    if m:
        code = m.group(1)
    return code.replace("```python", "").replace("```", "")


def norm_sig(code):
    """Syntax tree without identifiers' names and docstrings (as the repeat penalty): duplicates share it."""
    try:
        t = ast.parse(code)
    except SyntaxError:
        return None
    for n in ast.walk(t):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and n.body and isinstance(n.body[0], ast.Expr) \
                and isinstance(getattr(n.body[0], "value", None), ast.Constant) and isinstance(n.body[0].value.value, str):
            n.body = n.body[1:] or [ast.Pass()]
    return hashlib.md5(ast.dump(t, annotate_fields=False).encode()).hexdigest()


def convert(code):
    tree = ast.parse(code)
    keep = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and not (isinstance(node.value, ast.Constant)):
            continue                                                    # top-level call
        if isinstance(node, ast.If) and "__name__" in ast.unparse(node.test):
            continue                                                    # script entry point
        keep.append(node)
    tree.body = keep
    src = ast.unparse(tree)
    head = [h for h in HEADER if h not in src]
    src = "\n".join(head + [src]) if head else src
    tree = ast.parse(src)
    net = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Net")
    init = next((f for f in net.body if isinstance(f, ast.FunctionDef) and f.name == "__init__"), None)
    if init is not None and "self.device" not in ast.unparse(init):
        argnames = [a.arg for a in init.args.args]
        dev = "device" if "device" in argnames else "torch.device('cuda' if torch.cuda.is_available() else 'cpu')"
        stmt = ast.parse(f"self.device = {dev}").body[0]
        idx = next((i + 1 for i, s in enumerate(init.body) if "super(" in ast.unparse(s)), 0)
        init.body.insert(idx, stmt)
        src = ast.unparse(tree)
    if "def supported_hyperparameters" not in src:
        src += "\n\n\ndef supported_hyperparameters():\n    return {'lr', 'momentum', 'dropout'}\n"
    os.environ["NNGPT_STUB_FIXERS"] = "1"
    from ab.gpt.util.nn import PostprocessNN as P
    changes = []
    # only the fixes the interface needs: our networks already build and run as written, and the other repairs can
    # change valid code (e.g. _fix_in_shape_unpack turns a correct '_, c, h, w = in_shape' into a 5-value unpack)
    for fix in (P._fix_hyperparams, P._fix_missing_train_setup, P._fix_missing_learn):
        src, msg = fix(src)
        if msg:
            changes.append(msg)
    return src, changes


def check(src):
    """Build, forward, train_setup and one learn step on CPU - the NN Dataset interface is complete."""
    import torch
    ns = {"__name__": "lemur_check"}
    exec(compile(src, "net.py", "exec"), ns)
    net = ns["Net"]((1, 3, 32, 32), (10,), dict(PRM), torch.device("cpu"))
    out = net(torch.randn(2, 3, 32, 32))
    assert tuple(out.shape) == (2, 10), out.shape
    assert ns["supported_hyperparameters"]() <= {"lr", "momentum", "dropout"}
    net.train_setup(dict(PRM))
    net.learn([(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))])
    return sum(p.numel() for p in net.parameters())


def prepare(release, out, limit):
    os.makedirs(out, exist_ok=True)
    rows = []
    sources = [("base", "-", f) for f in sorted(glob.glob(f"{release}/runs/base/qwen_v3_run?_acc.jsonl"))]
    for f in sorted(glob.glob(f"{release}/runs/final/seed*/eval_*/run?_acc.jsonl")):
        parts = f.split("/")
        sources.append((parts[-2].replace("eval_", ""), parts[-3], f))
    for model, seed, f in sources:
        for l in open(f):
            r = json.loads(l)
            if r.get("strict") == "OK" and isinstance(r.get("accuracy"), (int, float)):
                rows.append(dict(model=model, seed=seed, run=re.search(r"run(\d)_acc", f).group(1), idx=r["idx"],
                                 accuracy=float(r["accuracy"]), code=strict_extract(r["text"]),
                                 prompt_hash=hashlib.md5(r["prompt"].strip().encode()).hexdigest()[:12]))
    rows.sort(key=lambda r: -r["accuracy"])
    seen, done, failed = set(), [], []
    import signal
    for r in rows:
        s = norm_sig(r["code"])
        if s is None or s in seen:
            continue
        seen.add(s)
        try:
            signal.alarm(120)
            src, changes = convert(r["code"])
            n = check(src)
            signal.alarm(0)
        except BaseException as e:
            signal.alarm(0)
            failed.append((r["model"], r["seed"], r["idx"], f"{type(e).__name__}: {str(e)[:80]}"))
            continue
        name = f"{len(done) + 1:04d}_{s[:10]}"
        open(f"{out}/{name}.py", "w").write(src)
        done.append(dict(file=f"{name}.py", source_model=r["model"], seed=r["seed"], run=r["run"], idx=r["idx"],
                         prompt_hash=r["prompt_hash"], accuracy_10ep=round(r["accuracy"], 4), params=n,
                         changes="; ".join(changes), signature=s))
        if limit and len(done) >= limit:
            break
    with open(f"{out}/index.csv", "w", newline="") as g:
        w = csv.DictWriter(g, fieldnames=list(done[0])); w.writeheader(); w.writerows(done)
    with open(f"{out}/failed.csv", "w", newline="") as g:
        csv.writer(g).writerows([("model", "seed", "idx", "error")] + failed)
    print(f"strict-valid trained networks: {len(rows)} | unique: {len(seen)} | prepared: {len(done)} | failed conversion: {len(failed)}")


def register(out, limit, epochs):
    from ab.nn import api
    rows = list(csv.DictReader(open(f"{out}/index.csv")))
    log = f"{out}/registered.csv"
    done = {r["file"] for r in csv.DictReader(open(log))} if os.path.exists(log) else set()
    prm = dict(PRM, batch=128, epoch=epochs, transform="norm")
    for i, r in enumerate(rows):
        if limit and i >= limit:
            break
        if r["file"] in done:
            continue
        code = open(f"{out}/{r['file']}").read()
        try:
            name, acc, acc_t, *_ = api.check_nn(code, "img-classification", "cifar-10", "acc", prm, True, PREFIX)
            res = dict(file=r["file"], lemur_name=name, lemur_accuracy=acc, accuracy_to_time=acc_t, error="")
        except Exception as e:
            res = dict(file=r["file"], lemur_name="", lemur_accuracy="", accuracy_to_time="", error=f"{type(e).__name__}: {str(e)[:120]}")
        new = not os.path.exists(log)
        with open(log, "a", newline="") as g:
            w = csv.DictWriter(g, fieldnames=list(res))
            if new:
                w.writeheader()
            w.writerow(res)
        print(f"[{i + 1}/{len(rows)}] {r['file']}: {res['lemur_name'] or res['error']} acc {res['lemur_accuracy']}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "register"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    a = ap.parse_args()
    if a.mode == "prepare":
        prepare(a.paths[0], a.paths[1], a.limit)
    else:
        register(a.paths[0], a.limit, a.epochs)
