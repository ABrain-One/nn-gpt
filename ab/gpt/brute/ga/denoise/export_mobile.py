"""Stage 1 of the mobile export: torch -> ONNX. TFLite is stage 2 (onnx_to_tflite.py).

WHY SPLIT. The direct torch->TFLite converters (ai-edge-torch, renamed litert-torch) pin
torch 2.12.1. Installing that beside the container's 2.7.1 loads two torches at once and
fails with an "Artifact already registered" assertion that took the ONNX export down with
it; deleting the pinned torch afterwards leaves litert_torch unimportable. Going through ONNX
sidesteps the conflict entirely -- stage 2 runs tensorflow only and never imports torch.

WHAT THE EXPORT IS FOR. Answering "is 9053 MMac a problem on mobile?" needs a SCALING CURVE,
not one model, so four are exported spanning ~7x in cost at each of two resolutions. Latency
does not depend on weight VALUES, so random-init models measure speed correctly -- but their
output is noise, so only the checkpointed models can be judged on image quality. Every row
states which it is, and reports MACs MEASURED from the exported net rather than quoted.
"""
import argparse, json, os, sys
from contextlib import contextmanager

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, "/a/mm/nn-gpt/ab/gpt/brute/ast"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

SPECS = [
    ("DenoiseThunder",  1274, "inside published mobile envelope (19.4 kMAC/px)"),
    ("DenoiseSubPixel", 4247, "our best by PSNR 35.246 dB; 2.7x over that envelope"),
    ("DenoiseUnet9F",   9053, "EXACTLY the cap under test; 5.8x over"),
    ("LiteDenoiseNet",     0, "highest-scoring CHECKPOINTED family; trained weights"),
]
NN_DIR = "/a/mm/nn-dataset/ab/nn/nn"


@contextmanager
def chunk_as_slice(torch):
    """Make Tensor.chunk() export as Slice instead of Split, for the duration of a block.

    Monkeypatching a torch method is a heavy hammer, so: it is scoped to one export call,
    restored in a finally, and it changes only WHICH ONNX NODE describes the operation.
    chunk() and basic slicing both return views over the same memory, so the exported graph
    computes exactly what the module computes. Uneven splits fall through to the original,
    since equal-size slicing cannot express them.
    """
    orig = torch.Tensor.chunk

    def patched(self, chunks, dim=0):
        n = self.shape[dim]
        if chunks <= 0 or n % chunks:
            return orig(self, chunks, dim)
        step = n // chunks
        out = []
        for i in range(chunks):
            idx = [slice(None)] * self.dim()
            idx[dim] = slice(i * step, (i + 1) * step)
            out.append(self[tuple(idx)])
        return tuple(out)

    torch.Tensor.chunk = patched
    try:
        yield
    finally:
        torch.Tensor.chunk = orig


def macs_mm(net, hw):
    """Measured MMac at 3xHWxHW via the GA's own counter, or None.

    Measured rather than taken from SPECS: the hardcoded figures came from a different
    script, and a latency claim should be paired with a cost number produced by the same
    process that produced the file being timed.
    """
    try:
        import torch
        from edge_cost import _mac_hooks
        handles, macs = _mac_hooks(net)
        try:
            with torch.no_grad():
                net(torch.rand(1, 3, hw, hw))
        finally:
            for h in handles:
                h.remove()
        return round(macs["v"] / 1e6, 1)
    except Exception:
        return None


def build(name, hw, ckpt_dir):
    import torch
    src = open(os.path.join(NN_DIR, name + ".py")).read()
    ns = {}
    exec(compile(src, name, "exec"), ns)
    net = ns["Net"]((1, 3, hw, hw), (3,),
                    {"lr": 1e-4, "batch": 16, "epoch_max": 50, "transform": "denoise"}, "cpu")
    net.eval()
    wstat = "no-checkpoint (random-init)"
    p = os.path.join(ckpt_dir, name, "best_model.pth")
    if os.path.exists(p):
        try:
            sd = torch.load(p, map_location="cpu", weights_only=False)
            if isinstance(sd, dict):
                for k in ("state_dict", "model_state_dict", "model"):
                    if k in sd and isinstance(sd[k], dict):
                        sd = sd[k]; break
            # A DataParallel/compile wrapper prefixes every key. Strip it before judging,
            # otherwise a fully-usable checkpoint reads as a total architecture mismatch.
            for pre in ("module.", "_orig_mod."):
                if sd and all(k.startswith(pre) for k in sd):
                    sd = {k[len(pre):]: v for k, v in sd.items()}
            missing, unexpected = net.load_state_dict(sd, strict=False)
            # NOT ALL MISSING KEYS MEAN MISSING WEIGHTS. `num_batches_tracked` is a BN
            # bookkeeping counter used only to compute the momentum of the running stats
            # DURING TRAINING; at eval the module reads running_mean/running_var and never
            # touches it. A checkpoint that omits it (torch<1.0 layout, or a save that
            # filtered non-float buffers) still carries every learned parameter. The previous
            # run reported "48/316 missing -> random-init" and exported an untrained net on
            # the strength of that counter alone.
            benign = [k for k in missing if k.endswith("num_batches_tracked")]
            real = [k for k in missing if not k.endswith("num_batches_tracked")]
            mk = list(net.state_dict().keys())
            if not real:
                wstat = "TRAINED"
                if benign:
                    wstat += f" ({len(benign)} num_batches_tracked defaulted; eval-irrelevant)"
            else:
                # Genuine mismatch: name the first few offenders on BOTH sides, because a
                # count alone cannot distinguish a renamed submodule from a resized one.
                ck = list(sd.keys())
                wstat = (f"PARTIAL/random-init ({len(real)} real + {len(benign)} benign "
                         f"of {len(mk)} missing; unexpected={len(unexpected)}; "
                         f"missing[:3]={real[:3]}; ckpt[0]={ck[0] if ck else '-'})")
        except Exception as e:
            wstat = f"random-init (load failed: {type(e).__name__}: {e})"[:200]
    return net, wstat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/a/mm/sweep_out/mobile_export")
    ap.add_argument("--ckpt-dir", default="/a/mm/nn-dataset/out/ckpt")
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 512])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import torch

    # TFLite IS NO LONGER ATTEMPTED HERE. ai-edge-torch / litert-torch pin their own torch
    # (2.12.1) and installing them alongside the container's 2.7.1 produced a double-loaded
    # torch that killed the ONNX pass too; stripping the pinned torch back out left
    # litert_torch unimportable. The torch->TFLite path therefore cannot run in this image.
    # onnx_to_tflite.py converts the ONNX written here instead, in a process that never
    # imports torch at all, which removes the conflict rather than fighting it.
    rows = []
    for hw in a.sizes:
        for name, mmac_256, note in SPECS:
            scaled = mmac_256 * (hw * hw) / (256 * 256)
            r = {"model": name, "size": hw, "note": note,
                 "mmac_expected": round(scaled, 1) if mmac_256 else None}
            try:
                net, wstat = build(name, hw, a.ckpt_dir)
                r["weights"] = wstat
                r["mmac_measured"] = macs_mm(net, hw)
                x = torch.randn(1, 3, hw, hw)

                onnx_p = os.path.join(a.out, f"{name}_{hw}.onnx")
                torch.onnx.export(net, (x,), onnx_p, input_names=["input"],
                                  output_names=["output"], opset_version=17,
                                  dynamic_axes=None)
                r["onnx_kb"] = round(os.path.getsize(onnx_p) / 1024, 1)

                # A SECOND ONNX WITH chunk() REWRITTEN AS SLICING, kept aside as a
                # conversion fallback. Tensor.chunk() exports as a Split node, and onnx2tf
                # then mis-derives the kernel of a depthwise conv fed by that split: it
                # reads the split COUNT (2) as the input-channel count and asks for a
                # (3,3,2,32) kernel where the weight has 32*1*3*3 = 288 elements. Every
                # block of DenoiseUnet9F does `x1, x2 = xp.chunk(2, 1); x1 = self.dw(x1)`,
                # so all 24 of them hit it and the model does not convert at all.
                # Basic slicing exports as Slice, which the converter tracks correctly, and
                # both forms return the same views -- the graph differs, the arithmetic does
                # not. Written into alt/ for every model so the fallback needs no
                # special-casing, and so it is never mistaken for the deliverable.
                alt_dir = os.path.join(a.out, "alt")
                os.makedirs(alt_dir, exist_ok=True)
                try:
                    with chunk_as_slice(torch):
                        torch.onnx.export(
                            net, (x,), os.path.join(alt_dir, f"{name}_{hw}.onnx"),
                            input_names=["input"], output_names=["output"],
                            opset_version=17, dynamic_axes=None)
                    r["alt_onnx"] = True
                except Exception as e:
                    r["alt_onnx_error"] = f"{type(e).__name__}: {e}"[:120]
            except Exception as e:
                r["error"] = f"{type(e).__name__}: {e}"[:160]
            rows.append(r)
            mm = r.get("mmac_measured")
            print(f"  {name:<18} {hw:>4}  {(f'{mm:9.0f}' if mm else '        ?')} MMac  "
                  f"onnx={r.get('onnx_kb','-')}KB  "
                  f"[{r.get('weights','?')}] {r.get('error') or ''}", flush=True)

    rp = os.path.join(a.out, "export_report.json")
    json.dump(rows, open(rp, "w"), indent=2)
    print(f"\nwrote {rp}")
    for f in sorted(os.listdir(a.out)):
        print(f"  {f}  {os.path.getsize(os.path.join(a.out,f))//1024} KB")


if __name__ == "__main__":
    raise SystemExit(main())
