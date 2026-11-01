"""Build one generated transform and push an image through it. Subprocess.

    python -I _transform_child.py <tr_file> <args.json> <result.json>

Separate process for the same reason the model smoke test is: generated code
runs arbitrary imports, and a crash must cost one candidate rather than the
cycle. The verdict is in the result file; a non-zero exit means the harness
itself failed, which the parent treats as unknown.
"""

import json
import sys
import traceback


def main():
    tr_file, args_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    args = json.loads(open(args_path, encoding="utf-8").read())
    verdict = {"ok": False, "reason": "unknown", "stage": "harness"}
    try:
        import importlib.util

        import torch
        import torchvision.transforms as transforms
        from PIL import Image

        verdict["stage"] = "import"
        spec = importlib.util.spec_from_file_location("nngpt_tr_subject", tr_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        verdict["stage"] = "entry_point"
        fn = getattr(module, "transform", None)
        if not callable(fn):
            raise AttributeError("no callable transform(_) in the module")

        verdict["stage"] = "build"
        norm = (tuple(args["mean"]), tuple(args["std"]))
        pipeline = fn(norm)
        if pipeline is None:
            raise RuntimeError("transform(norm) returned None")
        if not callable(pipeline):
            raise TypeError(
                f"transform(norm) returned {type(pipeline).__name__}, "
                "which cannot be applied to an image")

        verdict["stage"] = "apply"
        size = int(args.get("size", 32))
        img = Image.new("RGB", (size, size))
        out = pipeline(img)
        if not torch.is_tensor(out):
            raise TypeError(
                f"the pipeline produced {type(out).__name__}; a transform must "
                "end in ToTensor() so the trainer receives a tensor")
        if out.dim() != 3:
            raise RuntimeError(
                f"the pipeline produced a {out.dim()}-dimensional tensor, "
                "expected 3 (channels, height, width)")
        verdict.update(ok=True, reason="transform_ok", stage="done")
    except Exception as exc:
        verdict.update(
            ok=False,
            reason=f"{verdict['stage']}: {type(exc).__name__}: {exc}",
            traceback=traceback.format_exc()[-4000:])
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(verdict, f)
    except Exception:
        print(json.dumps(verdict))
    return 0


if __name__ == "__main__":
    sys.exit(main())
