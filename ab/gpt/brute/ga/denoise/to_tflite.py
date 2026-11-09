"""Convert denoise models to TFLite (LiteRT) and check their edge inference.

The thesis targets "Resource-Constrained Edge Systems", so every generated
architecture must ultimately run as a .tflite on device. This harness makes
that a one-command check over a whole batch of models:

  PyTorch Net (source)  --litert_torch.convert-->  .tflite  --ai_edge_litert-->  run

For each model it reports, in one table:
  * convert?      -- did it lower to TFLite at all (deployability signal)
  * tflite_KB     -- on-device model size (~flash/RAM footprint)
  * MACs          -- REAL deployment-graph MACs (parsed from converter), an
                     authoritative cross-check on edge_cost.py's hook estimate
  * latency_ms    -- median CPU (XNNPACK) inference time -> the real edge-cost
                     signal that replaces the MAC proxy in the GA objective
  * parity_dB     -- output-agreement PSNR torch-vs-tflite (~99 = conversion
                     preserved behaviour; low = the export changed the model)
  * error         -- failure reason if it did NOT convert (itself useful data:
                     which generated ops are not deployable)

Env (same as the other tools): nn-dataset venv python, both repos importable.
LiteRT/TFLite runs on CPU here; latency is a desktop-CPU proxy but consistent
across models, so it RANKS them correctly even if absolute ms differ from a phone.
"""
import argparse
import contextlib
import glob
import io
import json
import os
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, "/a/mm/nn-gpt/ab/gpt/brute/ast")

NN_DIR = "/a/mm/nn-dataset/ab/nn/nn"


def log(m):
    print(m, flush=True)


def _disable_db_prm_lookup():
    """load_lemur_model does a full-DB scan per model for stored prm (~2 min
    each); forward/convert only needs default prm, so stub it once."""
    import pandas as pd
    import mutator.lemur_io as lio
    empty = pd.DataFrame(columns=["nn", "nn_code", "accuracy", "prm"])
    lio.nn_dataset.data = lambda *a, **k: empty


def load_torch(source):
    from mutator.lemur_io import load_lemur_model
    model = load_lemur_model(source)
    model.eval()
    info = getattr(model, "_temp_module_info", None)
    return model.cpu(), info      # convert on CPU (LiteRT traces CPU tensors)


def _cleanup(info):
    if info:
        try:
            from mutator.lemur_io import cleanup_temp_module
            cleanup_temp_module(*info)
        except Exception:
            pass


def load_weights(model, ckpt_dir, name):
    """Load out/ckpt/<name>/best_model.pth into the model if it exists.

    LATENCY DOES NOT DEPEND ON WEIGHT VALUES, so a random-init export measures speed
    correctly. It does NOT measure quality: a randomly initialised denoiser outputs noise.
    Loading real weights is therefore only needed if the .tflite is to be judged on output,
    and the returned flag is reported so nobody mistakes one for the other.
    """
    import torch
    p = os.path.join(ckpt_dir, name, "best_model.pth")
    if not os.path.exists(p):
        return False, "no checkpoint"
    try:
        sd = torch.load(p, map_location="cpu", weights_only=False)
        # checkpoints are saved either as a bare state_dict or wrapped in a dict
        if isinstance(sd, dict):
            for k in ("state_dict", "model_state_dict", "model"):
                if k in sd and isinstance(sd[k], dict):
                    sd = sd[k]
                    break
        if not isinstance(sd, dict):
            return False, f"unexpected checkpoint type {type(sd).__name__}"
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            return False, f"{len(missing)} missing keys (architecture mismatch)"
        return True, f"loaded ({len(sd)} tensors)"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"[:100]


def convert_one(source, out_path, h, w, warmup, iters, ckpt_dir=None, name=None):
    """Convert one model source to out_path (.tflite), run it, and measure.
    Returns a dict row; on failure ok=False + error string."""
    import torch
    import numpy as np
    row = {"tflite_kb": None, "macs": None, "latency_ms": None,
           "parity_db": None, "ok": False, "error": None}
    model, info = None, None
    try:
        model, info = load_torch(source)
        row["weights"] = "random-init"
        if ckpt_dir and name:
            ok_w, msg = load_weights(model, ckpt_dir, name)
            row["weights"] = ("trained: " + msg) if ok_w else ("random-init (" + msg + ")")
            model.eval()
        sample = (torch.randn(1, 3, h, w),)

        import litert_torch
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            edge = litert_torch.convert(model, sample)
            edge.export(out_path)
        conv_log = buf.getvalue()
        mm = re.search(r"([\d.]+)\s*([GMk]?)\s*MACs", conv_log)
        if mm:
            scale = {"G": 1e9, "M": 1e6, "k": 1e3, "": 1.0}[mm.group(2)]
            row["macs"] = int(float(mm.group(1)) * scale)
        row["tflite_kb"] = round(os.path.getsize(out_path) / 1024, 1)

        # torch reference output
        with torch.no_grad():
            y_torch = model(sample[0]).numpy()

        # LiteRT inference + latency
        from ai_edge_litert.interpreter import Interpreter
        it = Interpreter(model_path=out_path)
        it.allocate_tensors()
        inp, out = it.get_input_details(), it.get_output_details()
        x = sample[0].numpy().astype(np.float32)
        for _ in range(warmup):
            it.set_tensor(inp[0]["index"], x); it.invoke()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            it.set_tensor(inp[0]["index"], x); it.invoke()
            ts.append((time.perf_counter() - t0) * 1000.0)
        ts.sort()
        row["latency_ms"] = round(ts[len(ts) // 2], 2)   # median
        y_tfl = it.get_tensor(out[0]["index"])

        # parity: output-agreement PSNR torch vs tflite
        mse = float(np.mean((y_torch - y_tfl) ** 2))
        row["parity_db"] = 99.0 if mse <= 1e-12 else round(
            10.0 * np.log10(1.0 / mse), 1)
        row["ok"] = True
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {str(e)[:120]}"
    finally:
        _cleanup(info)
    return row


def resolve_specs(seeds, models_dir):
    specs = []
    for nm in seeds:
        p = os.path.join(NN_DIR, nm + ".py")
        if os.path.exists(p):
            specs.append((nm, open(p).read()))
        else:
            log(f"  (seed {nm} not found — skipped)")
    if models_dir and os.path.isdir(models_dir):
        for p in sorted(glob.glob(os.path.join(models_dir, "*.py"))):
            src = open(p).read()
            label = os.path.basename(p)[:-3]
            first = src.splitlines()[0] if src else ""
            if first.startswith("# name="):
                label = first.split("name=", 1)[1].split()[0] + "_" + label[:6]
            specs.append((label, src))
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*",
                    default=[f"DenoiseUnet{i}F" for i in range(1, 11)] + ["LiteDenoiseNet"])
    ap.add_argument("--models-dir", default=None,
                    help="also convert every *.py in this dir (GA mutant sources)")
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "tflite"))
    ap.add_argument("--ckpt-dir", default="/a/mm/nn-dataset/out/ckpt",
                    help="load out/ckpt/<name>/best_model.pth when present")
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--w", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--results", default=os.path.join(_HERE, "tflite_report.json"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    _disable_db_prm_lookup()
    specs = resolve_specs(args.seeds, args.models_dir)
    log(f"=== TFLite convert + inference | {len(specs)} models @ {args.h}x{args.w} ===")

    rows = []
    for label, src in specs:
        safe = re.sub(r"[^A-Za-z0-9._+-]", "_", label)
        out_path = os.path.join(args.out_dir, safe + ".tflite")
        r = convert_one(src, out_path, args.h, args.w, args.warmup, args.iters,
                        args.ckpt_dir, label)
        r["model"] = label
        rows.append(r)
        if r["ok"]:
            log(f"  OK   {label:24s} {r['tflite_kb']:8.1f} KB  "
                f"{(r['macs'] or 0)/1e6:8.1f} MMac  {r['latency_ms']:8.2f} ms  "
                f"parity {r['parity_db']} dB  [{r.get('weights','?')}]")
        else:
            log(f"  FAIL {label:24s} {r['error']}")

    ok = [r for r in rows if r["ok"]]
    log(f"\n=== summary: {len(ok)}/{len(rows)} converted ===")
    if ok:
        log(f"{'model':24s}{'KB':>9}{'MMac':>10}{'ms':>9}{'parity':>8}")
        for r in sorted(ok, key=lambda r: r["latency_ms"]):
            log(f"{r['model'][:24]:24s}{r['tflite_kb']:9.1f}"
                f"{(r['macs'] or 0)/1e6:10.1f}{r['latency_ms']:9.2f}{r['parity_db']:8.1f}")
        bad = [r for r in ok if (r["parity_db"] or 99) < 40]
        if bad:
            log(f"\n  ⚠ low torch-vs-tflite parity (<40 dB) — export changed behaviour: "
                f"{[r['model'] for r in bad]}")
    failed = [r for r in rows if not r["ok"]]
    if failed:
        log(f"\n  NOT deployable ({len(failed)}): "
            + ", ".join(f"{r['model']}({r['error'].split(':')[0]})" for r in failed))

    json.dump(rows, open(args.results, "w"), indent=1)
    log(f"\nsaved -> {args.results}  |  .tflite files -> {args.out_dir}/")


if __name__ == "__main__":
    main()
