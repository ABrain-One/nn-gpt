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
        prm = dict(args["prm"])
        # The trainer passes exactly the keys the file asks for, so a file that
        # asks for a key nobody supplies fails at training time, not here.
        for key in names:
            prm.setdefault(str(key), 0.1)

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
        batch = max(2, int(args.get("batch", 2)))
        x = torch.randn((batch,) + in_shape[1:], device=device)
        net.train()
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
