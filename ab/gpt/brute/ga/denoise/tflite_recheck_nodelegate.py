"""Re-run the flatbuffers onnx_to_tflite.py marked bad, with XNNPACK TURNED OFF.

WHY THIS IS A SEPARATE QUESTION. verify() in onnx_to_tflite.py builds a plain Interpreter,
which silently applies XNNPACK, so its failures come in two kinds that must not be pooled:

  "XNNPack delegate failed to reshape runtimeNode N ... failed to prepare"
        the CPU delegate declined the graph. The flatbuffer itself may be perfectly valid.
  anything else (tensor lacks data, allocate_tensors raising, non-finite output)
        the file is broken and no backend will save it.

The distinction decides whether a model may enter the AI Hub run. XNNPACK is a CPU delegate
and has nothing to do with NPU placement, so excluding a model because XNNPACK declined it
would drop a measurable architecture for an irrelevant reason -- and, worse, the A/B compares
two arms: if those exclusions land unevenly across arms they confound the residency result
the whole pool exists to produce. So the exclusion has to be justified per model, not assumed.

BUILTIN_WITHOUT_DEFAULT_DELEGATES is the op-resolver that gives the reference kernels with no
delegate attached. It is the same interpreter AI Hub's own CPU reference path uses.

  python tflite_recheck_nodelegate.py --dir /a/mm/sweep_out/llmgate_ab/tflite
"""
import argparse
import glob
import json
import os


def check(p, no_delegate):
    if not os.path.exists(p):
        return {"ok": False, "error": "not produced"}
    try:
        import numpy as np
        try:
            from ai_edge_litert.interpreter import Interpreter, OpResolverType
        except Exception:
            from tensorflow.lite.python.interpreter import Interpreter, OpResolverType
        kw = {}
        if no_delegate:
            kw["experimental_op_resolver_type"] = \
                OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        it = Interpreter(model_path=p, **kw)
        it.allocate_tensors()
        i, o = it.get_input_details()[0], it.get_output_details()[0]
        it.set_tensor(i["index"], np.random.rand(*i["shape"]).astype(i["dtype"]))
        it.invoke()
        y = it.get_tensor(o["index"])
        return {"ok": True, "out": [int(v) for v in y.shape],
                "out_finite": bool(np.isfinite(y).all())}
    except Exception as e:
        return {"ok": False, "error": "%s: %s" % (type(e).__name__, e)[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    recs = []
    for p in sorted(glob.glob(os.path.join(a.dir, "tflite_report_shard*.jsonl"))):
        for line in open(p):
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    bad = [r for r in recs if not r.get("float32_check", {}).get("ok")]
    print("%d records, %d failed the delegated check -- rechecking without XNNPACK\n"
          % (len(recs), len(bad)))

    out = []
    for r in bad:
        m = r["model"]
        res = check(os.path.join(a.dir, "%s_float32.tflite" % m), no_delegate=True)
        # Keep the ORIGINAL error too: "valid flatbuffer, XNNPACK declined it" and "broken
        # flatbuffer" are different findings and the report has to be able to say which.
        row = {"model": m, "xnnpack_error": r.get("float32_check", {}).get("error", ""),
               "nodelegate": res}
        out.append(row)
        print("  %-56s nodelegate_ok=%s %s"
              % (m[:56], res.get("ok"), str(res.get("error", ""))[:90]))

    rescued = [r for r in out if r["nodelegate"].get("ok")]
    print("\n%d of %d run fine once XNNPACK is out of the way" % (len(rescued), len(out)))
    print("%d are genuinely broken flatbuffers" % (len(out) - len(rescued)))
    rp = a.out or os.path.join(a.dir, "nodelegate_recheck.json")
    json.dump(out, open(rp, "w"), indent=2, default=str)
    print("wrote %s" % rp)


if __name__ == "__main__":
    main()
