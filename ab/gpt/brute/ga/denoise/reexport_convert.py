"""Convert staged models to TFLite at a NON-SQUARE resolution.

WHY NOT REUSE THE EXISTING TWO CONVERTERS.
  * to_tflite.py imports `litert_torch` unconditionally, which is NOT in the ai-linux image --
    all 16 models failed with ModuleNotFoundError before a single graph was traced.
  * tflite_pool_convert.py DOES work in this image (it built the 2,511-model pool) because it falls
    back to `ai_edge_torch`, but its `size` is one integer used for both dimensions, so it cannot
    express 720x480 or 1088x1920. Its own line 26 anticipates this exercise -- "frontier re-exports
    use 720x480" -- without supporting it.
This file keeps that proven conversion path and adds separate height and width. tflite_pool_convert.py
is deliberately not edited: it is the script that produced every artifact the earlier tables rest on.

TWO ENVIRONMENT DETAILS INHERITED FROM THE WORKING SCRIPT, BOTH LOAD-BEARING.
  * HOME/USER/LOGNAME must be set. The container runs as uid 1067 with no passwd entry, and
    torch.export's inductor cache setup calls getpass.getuser(), which raises KeyError from
    getpwuid. HOME alone is not consulted by getuser().
  * One SUBPROCESS PER MODEL. Tracing can segfault or hang on an arbitrary generated graph, and an
    in-process loop loses every result collected so far when that happens. A timeout turns a hang
    into one recorded failure instead of a dead job.

  python reexport_convert.py --src DIR --out DIR --h 480 --w 720
"""
import argparse
import json
import os
import subprocess
import sys
import time

PRM = {"lr": 1e-4, "momentum": 0.9, "dropout": 0.0, "batch": 1, "transform": "denoise"}


def convert_one(code_path, name, h, w, out_dir):
    """Worker mode: exec, instantiate, forward-check, convert. Prints one JSON line."""
    rec = {"model": name, "h": h, "w": w, "ok": False, "forward_ok": False,
           "params": None, "size_kb": None, "convert_s": None, "error": None}
    try:
        import torch
        ns = {"__name__": "reexport_model"}
        exec(compile(open(code_path).read(), name, "exec"), ns)
        net = ns["Net"](in_shape=(1, 3, h, w), out_shape=(3, h, w), prm=dict(PRM), device="cpu")
        net.eval()
        rec["params"] = sum(p.numel() for p in net.parameters())
        sample = (torch.randn(1, 3, h, w),)
        with torch.no_grad():
            y = net(*sample)
        # A denoiser must return the input resolution. Converting a graph that silently changes
        # shape would produce a .tflite that profiles fine and computes the wrong thing.
        if not (hasattr(y, "shape") and tuple(y.shape[-2:]) == (h, w)):
            raise RuntimeError(f"forward shape {getattr(y, 'shape', type(y))} != ({h}, {w})")
        rec["forward_ok"] = True
        try:
            import litert_torch as converter
        except ImportError:
            import ai_edge_torch as converter
        rec["converter"] = converter.__name__
        t0 = time.time()
        edge = converter.convert(net, sample)
        path = os.path.join(out_dir, f"{name}_{h}x{w}_fp32.tflite")
        edge.export(path)
        rec["convert_s"] = round(time.time() - t0, 1)
        rec["size_kb"] = os.path.getsize(path) // 1024
        rec["path"] = path
        rec["ok"] = True
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:400]
    print("\n@@RESULT@@" + json.dumps(rec))


def main():
    os.environ.setdefault("HOME", os.environ.get("TMPDIR", "/tmp"))
    os.environ.setdefault("USER", "mm")
    os.environ.setdefault("LOGNAME", "mm")
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir of staged *.py model sources")
    ap.add_argument("--out", required=True, help="dir to write *.tflite into")
    ap.add_argument("--h", type=int, required=True)
    ap.add_argument("--w", type=int, required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--one")
    ap.add_argument("--name")
    a = ap.parse_args()

    if a.one:
        convert_one(a.one, a.name, a.h, a.w, a.out)
        return

    os.makedirs(a.out, exist_ok=True)
    srcs = sorted(f for f in os.listdir(a.src) if f.endswith(".py"))
    print(f"=== convert {len(srcs)} models @ {a.h}x{a.w} -> {a.out} ===", flush=True)

    rows = []
    for i, f in enumerate(srcs, 1):
        name = f[:-3]
        cmd = [sys.executable, os.path.abspath(__file__), "--one", os.path.join(a.src, f),
               "--name", name, "--out", a.out, "--h", str(a.h), "--w", str(a.w),
               "--src", a.src]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
            line = [l for l in p.stdout.splitlines() if l.startswith("@@RESULT@@")]
            if line:
                rec = json.loads(line[-1][len("@@RESULT@@"):])
            else:
                rec = {"model": name, "ok": False,
                       "error": f"no result line (rc={p.returncode}): "
                                f"{(p.stderr or p.stdout or '')[-200:]}"}
        except subprocess.TimeoutExpired:
            rec = {"model": name, "ok": False, "error": f"timeout after {a.timeout}s"}
        rows.append(rec)
        st = "OK  " if rec.get("ok") else "FAIL"
        extra = (f"{rec['size_kb']:>7} KB  {rec['convert_s']:>6.1f}s  {rec.get('converter','?')}"
                 if rec.get("ok") else str(rec.get("error"))[:110])
        print(f"  [{i}/{len(srcs)}] {st} {name[:42]:<44}{extra}", flush=True)

    ok = [r for r in rows if r.get("ok")]
    print(f"\n=== {len(ok)}/{len(rows)} converted @ {a.h}x{a.w} ===")
    if a.report:
        json.dump(rows, open(a.report, "w"), indent=1)
        print(f"wrote {a.report}")
    # Exit non-zero when nothing converted so the caller cannot mistake this for success.
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
