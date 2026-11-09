"""Stage 2 of the mobile export: ONNX -> TFLite, via onnx2tf. NEVER IMPORTS TORCH.

WHY THIS ROUTE. The obvious path is ai-edge-torch (now litert-torch) straight from the nn.Module.
It cannot work in this image: it pins torch 2.12.1, the container ships 2.7.1, and having both
importable loads two torches into one process ("Artifact of type=precompile already
registered"). Removing the pinned torch afterwards leaves litert_torch unimportable because the
same cleanup takes its dependencies. Both failures come from ONE cause -- the converter and the
model exporter sharing a process. Converting from the ONNX file breaks that coupling: this
script needs tensorflow + onnx and nothing else, so it can live in its own site-packages with
no version to negotiate.

NCHW -> NHWC MATTERS FOR THE MEASUREMENT. TFLite and every mobile delegate (GPU, NNAPI, XNNPACK)
are NHWC. A naive converter emits a Transpose around every convolution, and those transposes
would show up as latency that the architecture does not actually have -- which would silently
corrupt the very question being asked ("is 9053 MMac too slow?"). onnx2tf rewrites the graph to
native NHWC instead of wrapping it, so what gets timed is the model.

THREE FILES PER MODEL, because "half precision" means two different things in TFLite:
  <name>_float32.tflite   reference. fp32 throughout. Runs anywhere.
  <name>_fp16w.tflite     fp16 WEIGHTS, fp32 in/out, Dequantize before each kernel. Half the
                          size, runs on CPU/XNNPACK, and the GPU delegate takes the fp16
                          weights directly. THIS IS THE ONE TO SHIP AND TO BENCHMARK.
  <name>_float16.tflite   onnx2tf's true-fp16 graph: kernels themselves take fp16 tensors.
                          The CPU reference kernels reject it outright, so it is verifiable
                          only with the GPU delegate on the device. Kept, not recommended.
Emitting all three is what lets a slow float32 number be told apart from a genuinely
expensive architecture.

  python onnx_to_tflite.py --in-dir /a/mm/sweep_out/mobile_export
"""
import argparse
import glob
import json
import os
import shutil
import sys


def _from_saved_model(sm, dst, fp16_weights):
    """Re-convert the SavedModel with the stock TFLiteConverter.

    Needed for two independent reasons:

    1. FALLBACK. onnx2tf's newer "flatbuffer_direct" lowering emitted a float32 file for
       DenoiseUnet9F (592 nodes) whose interpreter refused it with "Input tensor 1264 lacks
       data" -- a constant with no buffer attached. The SavedModel it wrote alongside is
       fine, so routing that graph through the classic converter produces a working file.
       Only the big model tripped this, which is why the failure has to be detected by
       RUNNING each flatbuffer rather than by checking that a file appeared.

    2. THE ONLY CPU-RUNNABLE HALF-PRECISION FORM. onnx2tf's *_float16.tflite is a genuine
       fp16 graph: its conv/depthwise/space_to_depth kernels take fp16 tensors, which the
       CPU reference kernels reject outright (kTfLiteFloat32 || UInt8 || Int8 assertion).
       It is a GPU-delegate artefact and cannot be benchmarked or sanity-checked on CPU.
       fp16 WEIGHT quantisation is the shippable form instead -- fp32 in/out with Dequantize
       nodes in front of the weights, so it is half the size, runs on CPU/XNNPACK, and the
       GPU delegate still consumes the fp16 weights directly.
    """
    import tensorflow as tf
    c = tf.lite.TFLiteConverter.from_saved_model(sm)
    if fp16_weights:
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.target_spec.supported_types = [tf.float16]
    blob = c.convert()
    with open(dst, "wb") as f:
        f.write(blob)
    return round(os.path.getsize(dst) / 1024, 1)


# onnx2tf 2.6.7 defaults to "flatbuffer_direct", a hand-rolled FlatBuffer builder. It is
# fast and it produced a working file for three of the four architectures -- but for
# DenoiseUnet9F (592 nodes) it emitted a float32 flatbuffer the interpreter rejects with
# "Input tensor 1264 lacks data", i.e. a constant with no buffer behind it, and it writes no
# SavedModel so there is nothing to fall back to. "tf_converter" is the documented
# compatibility path: it goes through the real TFLiteConverter and leaves a SavedModel
# behind, which is also what the fp16-weight build needs. Try it FIRST and keep the direct
# builder only as a backstop -- correctness before conversion speed.
BACKENDS = ("tf_converter", "flatbuffer_direct")


def convert_one(onnx_p, out_dir, work_dir, rec=None):
    """-> dict describing what was produced. Raises nothing; reports instead."""
    import onnx2tf
    base = os.path.basename(onnx_p)[:-5]
    rec = rec if rec is not None else {"model": base, "onnx": os.path.basename(onnx_p)}
    f32_dst = os.path.join(out_dir, f"{base}_float32.tflite")

    for backend in BACKENDS:
        sm = os.path.join(work_dir, base)
        shutil.rmtree(sm, ignore_errors=True)
        try:
            onnx2tf.convert(
                input_onnx_file_path=onnx_p,
                output_folder_path=sm,
                output_signaturedefs=True,
                # copy_onnx_input_output_names_to_tflite is deliberately OFF. It pins the
                # OUTPUT back to the ONNX NCHW layout while the input is still rewritten to
                # NHWC, so the graph ends in a transpose that exists only to satisfy a name.
                # A benchmark artefact should be NHWC end to end -- that is what the phone
                # feeds it and what the delegates expect. The verifier prints the resulting
                # in/out shapes so the layout is stated rather than assumed.
                tflite_backend=backend,
                # NOT SET: flatbuffer_direct_output_saved_model. It would give the direct
                # builder a SavedModel, and hence an fp16-WEIGHT build for DenoiseUnet9F --
                # the one model tf_converter cannot handle. But enabling it made that same
                # model emit no .tflite at all, turning a working float32 file into nothing.
                # A CPU-runnable half-precision variant is not worth losing the float32 one,
                # so 9F ships fp32 + the GPU-delegate fp16 and no fp16w.
                non_verbose=True,
            )
        except Exception as e:
            # Keep the WHOLE message. TF's tf.constant TypeError ends with the element
            # counts ("has N elements, but got shape ... with M elements") and truncating at
            # 200 chars threw away the only part that identifies which tensor is wrong.
            msg = f"{type(e).__name__}: {e}"
            rec.setdefault("backend_errors", {})[backend] = msg[:400] + " ... " + msg[-400:] \
                if len(msg) > 800 else msg
            continue

        for suffix in ("float32", "float16"):
            src = os.path.join(sm, f"{base}_{suffix}.tflite")
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(out_dir, f"{base}_{suffix}.tflite"))
                rec[f"{suffix}_kb"] = round(os.path.getsize(src) / 1024, 1)
        rec["backend"] = backend

        # The fp16-WEIGHT build is the shippable half-precision form and needs the
        # SavedModel, so it is attempted from whichever backend left one.
        if os.path.exists(os.path.join(sm, "saved_model.pb")):
            rec["saved_model"] = True
            try:
                rec["fp16w_kb"] = _from_saved_model(
                    sm, os.path.join(out_dir, f"{base}_fp16w.tflite"), True)
            except Exception as e:
                rec["fp16w_error"] = f"{type(e).__name__}: {e}"[:200]

        shutil.rmtree(sm, ignore_errors=True)
        # A file that exists is not a file that runs -- that distinction is the whole reason
        # the 9F breakage was caught here instead of on the phone.
        if verify(f32_dst).get("ok"):
            break
        rec.setdefault("backend_errors", {})[backend] = "float32 flatbuffer failed to run"

    if not any(k.endswith("_kb") for k in rec):
        rec["error"] = "no .tflite produced by any backend"
    return rec


def verify(tfl_p):
    """Run the flatbuffer once so a file that exists but cannot load is not reported as a
    success. A .tflite that the interpreter rejects is worse than no file: it sends the user
    to the phone to debug our export."""
    if not os.path.exists(tfl_p):
        return {"ok": False, "error": "not produced"}
    try:
        import numpy as np
        try:
            from ai_edge_litert.interpreter import Interpreter
        except Exception:
            from tensorflow.lite.python.interpreter import Interpreter
        it = Interpreter(model_path=tfl_p)
        it.allocate_tensors()
        i, o = it.get_input_details()[0], it.get_output_details()[0]
        x = np.random.rand(*i["shape"]).astype(i["dtype"])
        it.set_tensor(i["index"], x)
        it.invoke()
        y = it.get_tensor(o["index"])
        # int(), not the raw shape: TFLite hands back numpy int32, which json.dump refuses.
        # The first run died on exactly that AFTER all eight conversions had succeeded,
        # leaving a truncated report and no per-file error text -- so a working converter
        # looked like a failing one.
        return {"ok": True, "in": [int(v) for v in i["shape"]],
                "out": [int(v) for v in y.shape], "dtype": str(i["dtype"]),
                "out_finite": bool(np.isfinite(y).all())}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="/a/mm/sweep_out/mobile_export")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--work-dir", default="/scratch/o2tf")
    ap.add_argument("--out", default="")
    ap.add_argument("--only", nargs="*", default=[],
                    help="substrings; convert only matching ONNX files (debug loop)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    # WHAT COUNTS AS A PASS, made explicit because it is not the same question for every
    # caller. mobile_export wants a shippable artefact, so it needs the fp16-WEIGHT build and
    # "f32+fp16w" stays the default. The A/B residency pool only needs a flatbuffer AI Hub can
    # place on the NPU, and fp16w is unobtainable for it anyway: tf_converter fails on these
    # generated graphs with an NCHW/NHWC conv-depth mismatch, and the flatbuffer_direct
    # backstop that does succeed writes no SavedModel for the fp16w build to start from. With
    # the default criterion a fully successful shard would exit non-zero and be recorded as a
    # failed Job, which is the opposite of what happened.
    ap.add_argument("--require", choices=("f32", "f32+fp16w"), default="f32+fp16w")
    a = ap.parse_args()
    out_dir = a.out_dir or a.in_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(a.work_dir, exist_ok=True)

    try:
        import onnx2tf
        import tensorflow as tf
        print(f"[o2tf] onnx2tf {getattr(onnx2tf,'__version__','?')}  "
              f"tensorflow {tf.__version__}")
    except Exception as e:
        print(f"FATAL: converter stack unimportable ({type(e).__name__}: {e})")
        return 2
    # A torch import here would mean the isolation failed and we are back in the two-torch
    # situation that broke the first attempt. Say so rather than letting it pass silently.
    if "torch" in sys.modules:
        print("WARN: torch is loaded in this process -- isolation is not what it should be")

    onnxs = sorted(glob.glob(os.path.join(a.in_dir, "*.onnx")))
    if a.only:
        onnxs = [p for p in onnxs if any(s in os.path.basename(p) for s in a.only)]
    if not onnxs:
        print(f"FATAL: no .onnx under {a.in_dir}")
        return 2

    # SHARDING AND A PER-RECORD JSONL, added for the 192-model A/B pool. tf_converter goes
    # through the real TFLiteConverter twice per model (float32, then the fp16-weight build),
    # so one process over 192 generated graphs runs for hours -- and a single segfault in an
    # arbitrary graph would lose every result collected before it, which is the same reason
    # ab_onnx_export.py forks per model. The record is appended as each model finishes and read
    # back on start, so a rerun resumes past anything already attempted. num_shards=1 keeps the
    # original single-file behaviour for the 8-model mobile_export caller.
    if a.num_shards > 1:
        onnxs = onnxs[a.shard::a.num_shards]
    name = ("tflite_report_shard%d.jsonl" % a.shard) if a.num_shards > 1 \
        else "tflite_report.jsonl"
    rp = a.out or os.path.join(out_dir, name)
    done = set()
    if os.path.exists(rp):
        for line in open(rp):
            try:
                done.add(json.loads(line)["model"])
            except Exception:
                pass
    todo = [q for q in onnxs if os.path.basename(q)[:-5] not in done]
    print(f"[o2tf] shard {a.shard}/{a.num_shards}: {len(onnxs)} assigned, "
          f"{len(onnxs) - len(todo)} already attempted, {len(todo)} to go\n")

    rows, bad = [], 0
    for p in todo:
        base = os.path.basename(p)[:-5]
        try:
            rec = convert_one(p, out_dir, a.work_dir)
            # Every backend failed on the default graph. Retry on the alt/ export, which
            # expresses chunk() as Slice rather than Split and so does not trip the
            # depthwise-kernel misderivation. Same output filenames -- the fallback is an
            # implementation detail of producing a working file, and `onnx_used` records
            # which graph it came from.
            if not verify(os.path.join(out_dir, f"{base}_float32.tflite")).get("ok"):
                alt = os.path.join(a.in_dir, "alt", f"{base}.onnx")
                if os.path.exists(alt):
                    print(f"  {base}: retrying via alt/ (chunk rewritten as Slice)",
                          flush=True)
                    rec["retried_with_alt_onnx"] = True
                    rec = convert_one(alt, out_dir, a.work_dir, rec=rec)
                    if verify(os.path.join(out_dir, f"{base}_float32.tflite")).get("ok"):
                        rec["onnx_used"] = "alt/ (chunk exported as Slice)"
        except Exception as e:
            rec = {"model": base, "error": f"{type(e).__name__}: {e}"[:300]}
        for suffix in ("float32", "float16", "fp16w"):
            rec[f"{suffix}_check"] = verify(
                os.path.join(out_dir, f"{rec['model']}_{suffix}.tflite"))
        c32 = rec.get("float32_check", {})
        cw = rec.get("fp16w_check", {})
        ok32, okw = c32.get("ok"), cw.get("ok")
        # float16 is EXPECTED to fail on CPU (true-fp16 graph, GPU delegate only), so it is
        # deliberately not part of the pass criterion -- counting it would flag every model.
        if not (ok32 and (okw or a.require == "f32")):
            bad += 1
        rows.append(rec)
        # default=str so one unserialisable value degrades a field instead of losing the record.
        with open(rp, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        # Print the REASON, not just the boolean. "f16=False" with no message is what made
        # the first run unactionable.
        why = rec.get("error") or ""
        if not ok32 and c32.get("error"):
            why += f"  f32: {c32['error'][:150]}"
        if not okw and cw.get("error"):
            why += f"  fp16w: {cw['error'][:150]}"
        print(f"  {rec['model']:<24} [{rec.get('backend','-')}] "
              f"f32={rec.get('float32_kb','-')}KB "
              f"fp16w={rec.get('fp16w_kb','-')}KB f16gpu={rec.get('float16_kb','-')}KB  "
              f"runs(cpu): f32={ok32} fp16w={okw}{why}", flush=True)

    print(f"\nwrote {rp}")
    print(f"  converted+verified {len(rows)-bad}/{len(rows)} attempted this run")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
