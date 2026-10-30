"""Build one generated model and push a batch through it. Run as a subprocess.

Separate file, not an inline ``-c`` string, so a traceback has real line
numbers and so the parent can run it with ``-I`` -- isolated mode, which keeps
the generated file's own directory off sys.path. A planted ``json.py`` next to
a generated model would otherwise be imported by this very script.

    python -I _smoke_child.py <nn_file> <args.json> <result.json>

Exit code is 0 whether the model builds or not; the verdict is in the result
file. A non-zero exit means the harness itself failed, which the parent treats
as "unknown" rather than as a broken model.
"""

import json
import sys
import traceback


def _load_module(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("nngpt_smoke_subject", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    nn_file, args_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    args = json.loads(open(args_path, encoding="utf-8").read())
    verdict = {"ok": False, "reason": "unknown", "stage": "harness"}
    try:
        # Stage "harness" means the failure is ours, not the model's: the
        # parent keeps a model whose verdict never got past this point.
        import torch
        verdict["stage"] = "import"
        module = _load_module(nn_file)

        verdict["stage"] = "supported_hyperparameters"
        names = module.supported_hyperparameters()
        if not isinstance(names, (set, frozenset, list, tuple)):
            raise TypeError(
                f"supported_hyperparameters() returned {type(names).__name__}, "
                "expected a set of names")
        # Built the way ab/nn/util/Train.py builds it, which is: the record's
        # own prm, passed through. Train.py does not synthesise a value per
        # declared name, so neither does this.
        #
        # Inventing one made the gate stricter than the trainer and wrong in
        # the same move. Every declared key absent from PRM_DEFAULTS got 0.1,
        # so `prm.get('use_amp', False)` returned 0.1 -- truthy -- and turned
        # on an autocast('cuda') path in a subprocess run with
        # CUDA_VISIBLE_DEVICES="", while `prm.get('patch_size', 16)` returned
        # 0.1 and reached Conv2d as a kernel size. Under the real trainer those
        # keys are simply missing and each model falls back to its own default.
        #
        # So: the caller's prm, with the standard keys filled in only where it
        # lacks them. A declared name the record does not carry stays absent,
        # exactly as in training -- a `.get` finds the model's default and a
        # direct `prm[name]` raises KeyError here instead of an hour later.
        defaults = dict(args.get("defaults") or {})
        prm = dict(args["prm"])
        for key in names:
            key = str(key)
            if key not in prm and key in defaults:
                prm[key] = defaults[key]

        device = torch.device("cpu")
        in_shape = tuple(args["in_shape"])
        out_shape = tuple(args["out_shape"])

        verdict["stage"] = "construct"
        # Exactly how ab/nn/util/Train.py calls it, including the 4-tuple
        # in_shape -- the single most common failure in this pipeline was a
        # generated helper that unpacked in_shape into five names.
        net = module.Net(in_shape, out_shape, prm, device)
        net.to(device)

        verdict["stage"] = "train_setup"
        net.train_setup(prm)

        verdict["stage"] = "forward"
        light = bool(args.get("light"))
        batch = max(1 if light else 2, int(args.get("batch", 2)))
        x = torch.randn((batch,) + in_shape[1:], device=device)
        net.train()
        if light:
            # No graph is kept, so the activations are freed as the forward
            # walks the model instead of being held for a backward that is not
            # going to happen. That is the difference between a few hundred
            # megabytes and several gigabytes at 256 square.
            with torch.no_grad():
                y = net(x)
        else:
            y = net(x)
        if y is None:
            raise RuntimeError("forward() returned None")
        if hasattr(y, "shape") and len(y.shape) >= 2 and out_shape:
            got, want = int(y.shape[-1]), int(out_shape[-1])
            if got != want:
                raise RuntimeError(
                    f"forward() produced {got} outputs for out_shape {out_shape}")

        verdict["stage"] = "learn"
        if args.get("check_learn", True):
            # One optimiser step. This is what catches an in-place operation on
            # a tensor the graph still needs, and a loss/target dtype mismatch
            # -- both of which otherwise surface only after the dataset has been
            # loaded and a GPU worker has been started.
            import torch.utils.data as tud
            targets = torch.randint(0, max(1, int(out_shape[-1])), (batch,))
            xs = torch.randn((batch,) + in_shape[1:])
            loader = tud.DataLoader(tud.TensorDataset(xs, targets), batch_size=batch)
            net.learn(loader)

        verdict.update(ok=True, reason="smoke_ok", stage="done")
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
