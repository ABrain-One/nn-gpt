"""Stage 1 of the A/B deployability profiling: staged model code -> ONNX. NO TFLITE HERE.

WHY A NEW STAGE-1 AT ALL. The direct torch->TFLite path every earlier artifact was built with is
gone from this image: `ai_edge_torch` now resolves to a deprecation shim ("renamed to litert-torch")
whose module has no .convert, and `litert_torch` is not installed anywhere on the share. So
tflite_pool_convert.py and reexport_convert.py both fail at the convert call -- not because they are
wrong, but because the package they depend on was replaced underneath them. The two-pass ONNX route
documented in export_mobile_job.sh is unaffected, because onnx2tf needs neither torch nor litert.

SUBPROCESS PER MODEL WITH A TIMEOUT, inherited from tflite_pool_convert.py for the same reason it
has one: torch.onnx.export traces an arbitrary generated graph and can hang or segfault, and an
in-process loop would lose every result collected before that point. A record is appended either
way so a rerun resumes past anything already attempted.

HOME/USER/LOGNAME are load-bearing. The container runs as uid 1067 with no passwd entry and
torch's inductor cache setup calls getpass.getuser(), which raises KeyError from getpwuid. HOME
alone is not consulted by getuser().

  python ab_onnx_export.py --codes <dir>/codes.jsonl --out <dir>/onnx --shard 0 --num-shards 4
"""
import argparse
import json
import os
import subprocess
import sys

SIZE = 256


def export_one(code_path, name, size, out_dir):
    """Trace one model to ONNX. Runs as its own process; prints an @@RESULT@@ line."""
    os.environ.setdefault("HOME", os.environ.get("TMPDIR", "/tmp"))
    os.environ.setdefault("USER", "mm")
    os.environ.setdefault("LOGNAME", "mm")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    rec = {"model": name, "ok": False}
    try:
        import torch
        from mutator.lemur_io import load_lemur_model
        net = load_lemur_model(open(code_path).read())
        net.eval()
        # Weights are irrelevant: latency is a property of the graph, so random init is correct
        # here and the artifacts must never be used to judge image quality.
        x = torch.randn(1, 3, size, size)
        p = os.path.join(out_dir, "%s_%d.onnx" % (name, size))
        with torch.no_grad():
            torch.onnx.export(net, (x,), p, input_names=["input"],
                              output_names=["output"], opset_version=17, dynamic_axes=None)
        rec.update(ok=True, onnx_kb=round(os.path.getsize(p) / 1024, 1))
    except Exception as e:
        rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
    print("@@RESULT@@" + json.dumps(rec), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--one")
    ap.add_argument("--name")
    a = ap.parse_args()

    if a.one:
        export_one(a.one, a.name, a.size, a.out)
        return
    if not a.codes:
        ap.error("--codes is required unless --one is given")

    os.makedirs(a.out, exist_ok=True)
    scratch = os.environ.get("TMPDIR", "/tmp")
    models = [json.loads(l) for l in open(a.codes) if l.strip()]
    mine = models[a.shard::a.num_shards]
    report = os.path.join(a.out, "onnx_report_shard%d.jsonl" % a.shard)
    done = set()
    if os.path.exists(report):
        for l in open(report):
            try:
                done.add(json.loads(l)["model"])
            except Exception:
                pass
    todo = [m for m in mine if m["name"] not in done]
    print("[shard %d/%d] %d assigned, %d already attempted, %d to go"
          % (a.shard, a.num_shards, len(mine), len(mine) - len(todo), len(todo)), flush=True)

    for i, m in enumerate(todo, 1):
        cp = os.path.join(scratch, "cand_%s.py" % m["name"][:60])
        open(cp, "w").write(m["code"])
        rec = {"model": m["name"], "ok": False, "error": "unknown"}
        try:
            r = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--one", cp, "--name", m["name"],
                 "--out", a.out, "--size", str(a.size)],
                capture_output=True, text=True, timeout=a.timeout)
            for line in (r.stdout or "").splitlines():
                if line.startswith("@@RESULT@@"):
                    rec = json.loads(line[len("@@RESULT@@"):])
                    break
            else:
                rec["error"] = "no result line (rc=%s) %s" % (r.returncode, (r.stderr or "")[-160:])
        except subprocess.TimeoutExpired:
            rec["error"] = "timeout %ds" % a.timeout
        with open(report, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("  [%d/%d] %-52s %s" % (i, len(todo), m["name"][:52],
                                      "OK" if rec.get("ok") else rec.get("error", "")[:70]),
              flush=True)
        try:
            os.remove(cp)
        except OSError:
            pass
    print("SHARD %d DONE" % a.shard, flush=True)


if __name__ == "__main__":
    main()
