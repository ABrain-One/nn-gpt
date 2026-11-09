"""Pool-wide TFLite FP32 conversion sweep for the denoise corpus.

Converts every model in the corpus (code from a pre-dumped codes.jsonl, so pods never
open the shared SQLite DB) to TFLite via litert-torch/ai-edge-torch -- the direct
torch->TFLite path tflite_bench.py established; there is no ONNX hop. Conversion is
weights-agnostic: latency does not depend on weight values, so models export at random
init and the .tflite files are for RUNTIME measurement only, never image quality.

Every model converts in its OWN SUBPROCESS with a timeout: the TF converter can hang or
segfault on hostile graphs, and one bad model must not kill a shard. A record is appended
to the shard report either way, so reruns resume past everything already attempted.

  # dump the corpus once (login node, read-only):
  python tflite_pool_convert.py --dump-codes --db .../ab.nn.db --out .../tflite_pool

  # convert one shard (pod, needs the tflite venv):
  python tflite_pool_convert.py --shard 0 --num-shards 6 --out /a/mm/sweep_out/tflite_pool
"""
import argparse
import json
import os
import subprocess
import sys
import time

SIZE = 256                       # corpus training resolution; frontier re-exports use 720x480
PRM = {"lr": 1e-4, "momentum": 0.9, "dropout": 0.1, "epoch": 10, "batch": 16}


def dump_codes(db, out):
    import sqlite3
    os.makedirs(out, exist_ok=True)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT DISTINCT n.name, n.code FROM nn n JOIN stat s ON s.nn = n.name "
        "WHERE s.task='img-denoising' AND n.code IS NOT NULL ORDER BY n.name").fetchall()
    con.close()
    p = os.path.join(out, "codes.jsonl")
    with open(p, "w") as f:
        for name, code in rows:
            f.write(json.dumps({"name": name, "code": code}) + "\n")
    print(f"[dump] {len(rows)} models -> {p}")


def convert_one(code_path, name, size, out_dir):
    """Worker mode: exec + instantiate + forward-check + convert. Prints one JSON line."""
    rec = {"model": name, "ok": False, "forward_ok": False, "params": None,
           "size_kb": None, "convert_s": None, "error": None}
    try:
        import torch
        ns = {}
        exec(compile(open(code_path).read(), name, "exec"), ns)
        net = ns["Net"](in_shape=(1, 3, size, size), out_shape=(3, size, size),
                        prm=dict(PRM), device="cpu")
        net.eval()
        rec["params"] = sum(p.numel() for p in net.parameters())
        sample = (torch.randn(1, 3, size, size),)
        with torch.no_grad():
            y = net(*sample)
        if not (hasattr(y, "shape") and tuple(y.shape[-2:]) == (size, size)):
            raise RuntimeError(f"forward shape {getattr(y, 'shape', type(y))}")
        rec["forward_ok"] = True
        try:
            import litert_torch as converter
        except ImportError:
            import ai_edge_torch as converter
        t0 = time.time()
        edge = converter.convert(net, sample)
        path = os.path.join(out_dir, f"{name}_{size}x{size}_fp32.tflite")
        edge.export(path)
        rec["convert_s"] = round(time.time() - t0, 1)
        rec["size_kb"] = os.path.getsize(path) // 1024
        rec["ok"] = True
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:400]
    print("\n@@RESULT@@" + json.dumps(rec))


def run_shard(a):
    models = [json.loads(l) for l in open(os.path.join(a.out, "codes.jsonl"))]
    mine = [m for i, m in enumerate(models) if i % a.num_shards == a.shard]
    tdir = os.path.join(a.out, "tflite")
    os.makedirs(tdir, exist_ok=True)
    report = os.path.join(a.out, f"report_shard{a.shard}.jsonl")
    done = set()
    if os.path.exists(report):
        done = {json.loads(l)["model"] for l in open(report) if l.strip()}
    todo = [m for m in mine if m["name"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"[shard {a.shard}/{a.num_shards}] {len(mine)} assigned, "
          f"{len(done)} already attempted, {len(todo)} to go", flush=True)

    scratch = os.environ.get("TMPDIR", "/tmp")
    n_ok = n_fail = 0
    for k, m in enumerate(todo):
        cp = os.path.join(scratch, f"cand_{a.shard}.py")
        with open(cp, "w") as f:
            f.write(m["code"])
        rec = None
        try:
            r = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--one", cp, "--out", a.out,
                 "--name", m["name"], "--size", str(a.size), "--tflite-dir", tdir],
                capture_output=True, text=True, timeout=a.timeout)
            for line in r.stdout.splitlines():
                if line.startswith("@@RESULT@@"):
                    rec = json.loads(line[len("@@RESULT@@"):])
            if rec is None:
                tail = (r.stderr or r.stdout or "").strip()[-300:]
                rec = {"model": m["name"], "ok": False,
                       "error": f"worker died rc={r.returncode}: {tail}"}
        except subprocess.TimeoutExpired:
            rec = {"model": m["name"], "ok": False, "error": f"timeout {a.timeout}s"}
        with open(report, "a") as f:
            f.write(json.dumps(rec) + "\n")
        n_ok += rec.get("ok", False); n_fail += not rec.get("ok", False)
        if (k + 1) % 10 == 0 or k + 1 == len(todo):
            print(f"[shard {a.shard}] {k+1}/{len(todo)}  ok {n_ok}  fail {n_fail}", flush=True)
    print(f"[shard {a.shard}] DONE  ok {n_ok}  fail {n_fail}")


def main():
    # containers run as a uid with no passwd entry; torch.export's inductor cache-dir
    # setup calls getpass.getuser(), which raises KeyError from getpwuid unless one of
    # its env fallbacks is set. HOME alone is not consulted by getuser.
    os.environ.setdefault("HOME", os.environ.get("TMPDIR", "/tmp"))
    os.environ.setdefault("USER", "mm")
    os.environ.setdefault("LOGNAME", "mm")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-codes", action="store_true")
    ap.add_argument("--db")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test cap, 0 = all")
    # internal worker mode
    ap.add_argument("--one", help=argparse.SUPPRESS)
    ap.add_argument("--name", help=argparse.SUPPRESS)
    ap.add_argument("--tflite-dir", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.one:
        convert_one(a.one, a.name, a.size, a.tflite_dir)
    elif a.dump_codes:
        dump_codes(a.db, a.out)
    else:
        run_shard(a)


if __name__ == "__main__":
    main()
