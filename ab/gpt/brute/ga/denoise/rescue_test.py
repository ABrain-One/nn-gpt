"""NaN-guard rescue test (yield >90% proof).

Plain DenoiseUnet9F NaN-diverged to 0.0 in Gate 1. Here we train it @50ep with an
INSTRUMENTED guard (prints NANGUARD_FIRE each time it skips a non-finite batch) and
report the PSNR. Decisive outcomes:
  - PSNR>0 AND fires>0  -> the guard caught divergence(s) and rescued the model (PROOF).
  - PSNR>0 AND fires==0 -> 9F just didn't diverge this run (guard harmless, unproven).
  - PSNR==0             -> divergence is persistent, guard insufficient (must know before prod).

Run from nn-dataset root:
    cd /a/mm/nn-dataset
    PYTHONPATH=. python .../rescue_test.py
"""
import argparse, re, sys, time

sys.path.insert(0, "/a/mm/nn-dataset")
import ab.nn.api as api

NN = "/a/mm/nn-dataset/ab/nn/nn/DenoiseUnet9F.py"
TASK, DATASET, METRIC, PSNR_SCALE = "img-denoising", "denoise", "psnr", 48.0


def instrumented_guard(code):
    """Same guard as production but prints when it fires, so we can count rescues."""
    out = []
    for ln in code.splitlines(keepends=True):
        m = re.match(r'^(\s*)(\w+)\.backward\(\)\s*$', ln)
        if m and 'isfinite' not in (out[-1] if out else ''):
            ind, var = m.group(1), m.group(2)
            out.append(f"{ind}if not torch.isfinite({var}): print('NANGUARD_FIRE', flush=True); continue\n")
        out.append(ln)
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    code = instrumented_guard(open(NN).read())
    prm = {"lr": args.lr, "batch": args.batch, "transform": "denoise",
           "epoch": args.epoch, "num_workers": args.workers}
    print(f"=== NaN-guard rescue test: DenoiseUnet9F @{args.epoch}ep (was 0.0 in Gate 1) ===", flush=True)
    t0 = time.time(); psnr = 0.0; err = None
    try:
        _n, acc, _a2t, _q = api.check_nn(code, TASK, DATASET, METRIC, prm,
                                         save_to_db=False, prefix="RescueTest")
        psnr = acc * PSNR_SCALE
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:120]}"
        print("  FAILED:", err, flush=True)
    print("=== RESULT ===", flush=True)
    print(f"  9F guarded PSNR@{args.epoch} = {psnr:.3f} dB   ({round(time.time()-t0)}s)   err={err}", flush=True)
    print("  (grep NANGUARD_FIRE in the log for rescue count)", flush=True)


if __name__ == "__main__":
    main()
