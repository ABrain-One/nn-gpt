"""Report the REAL exception behind each LLM candidate's build failure.

edge_cost.measure_cost() ends with a bare `except Exception: return {"ok": False,
"params": 0, "macs": 0}` -- no message, by design ("never kill a model that otherwise
trained fine just because profiling hiccuped"). Correct for the GA, useless for triage:
the screener could only print "BUILD FAIL: None" for 8 of 9 candidates, which does not
distinguish "the LLM wrote broken code" from "our loader rejects valid code".

This calls load_lemur_model directly and lets the exception surface, so the failures can
be counted by CAUSE. That difference decides the next move: bad code means fix the prompt,
loader rejection means fix the harness.

Usage: python diag_candidates.py [run_dir]
"""
import glob
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, "/a/mm/nn-gpt/ab/gpt/brute/ast",
           os.path.abspath(os.path.join(HERE, "..", "..", "ast"))):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

run_dir = sys.argv[1] if len(sys.argv) > 1 else "/a/mm/sweep_out/llm_gen/run"

import torch                                          # noqa: E402
from mutator.lemur_io import load_lemur_model, get_dataset_params  # noqa: E402

h, w = get_dataset_params("denoise")["spatial_dims"]
print(f"input probe: 1x3x{h}x{w}\n")

buckets = {}
for cdir in sorted(glob.glob(os.path.join(run_dir, "cand", "*"))):
    cid = os.path.basename(cdir)
    src = os.path.join(cdir, "candidate.py")
    if not os.path.exists(src):
        print(f"{cid:<16} NO CODE"); buckets.setdefault("no code", []).append(cid); continue
    code = open(src).read()
    stage, err = "load", None
    try:
        model = load_lemur_model(code)
        stage = "forward"
        model.eval()
        with torch.no_grad():
            dev = next(model.parameters()).device
            out = model(torch.randn(1, 3, h, w, device=dev))
        stage = "shape"
        assert out.shape == (1, 3, h, w), f"output {tuple(out.shape)} != (1,3,{h},{w})"
        print(f"{cid:<16} OK   out={tuple(out.shape)}")
        buckets.setdefault("ok", []).append(cid)
        continue
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    print(f"{cid:<16} FAIL@{stage}  {err[:150]}")
    if os.environ.get("DIAG_TRACE"):
        traceback.print_exc()
    buckets.setdefault(f"{stage}: {type(Exception()).__name__ if not err else err.split(':')[0]}",
                       []).append(cid)

print("\n--- failures by cause ---")
for k, v in sorted(buckets.items(), key=lambda t: -len(t[1])):
    print(f"  {len(v):>2}  {k}   ({', '.join(v)})")
