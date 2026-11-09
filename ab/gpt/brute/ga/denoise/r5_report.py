"""One-shot Run-5 progress report. Isolates Run-5-NEW models by diffing the live R5
cache against the R4 cache it was warm-started from (both keyed by code checksum)."""
import json, os, sys

D = "/a/mm/nn-gpt/ab/gpt/brute/ga/denoise"


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return {}


def trained(c):
    return {k for k, e in c.items() if isinstance(e, dict) and (e.get("psnr") or 0) > 0}


def good(c, thr=34.0):
    return {k for k, e in c.items() if isinstance(e, dict) and (e.get("psnr") or 0) >= thr}


r5 = load(f"{D}/seed_ga_R5_cache.json")
r4 = load(f"{D}/seed_ga_R4_cache.json")   # warm-start baseline (Run 4 final, unchanged)

gen = "?"
try:
    import pickle
    d = pickle.load(open(f"{D}/seed_ga_R5_checkpoint.pkl", "rb"))
    gen = d.get("generation", "?")
except Exception:
    pass

new_trained = trained(r5) - trained(r4)
new_good = good(r5) - good(r4)
# best psnr among Run-5-new trained models
best = 0.0
best_key = None
for k in new_trained:
    p = r5[k].get("psnr") or 0
    if p > best:
        best, best_key = p, k

res = load(f"{D}/seed_ga_R5_results.json")
files = len([f for f in os.listdir(f"{D}/seed_ga_R5_models") if f.endswith(".py")]) \
    if os.path.isdir(f"{D}/seed_ga_R5_models") else 0

print("========== RUN 5 REPORT ==========")
print(f"generation:            {gen}/30")
print(f"R5-NEW distinct trained: {len(new_trained)}   (cache {len(r5)} vs warm-start {len(r4)})")
print(f"R5-NEW >=34 dB (good):   {len(new_good)}")
print(f"best R5-NEW proxy PSNR:   {best:.3f} dB" + (f"  [{best_key[:40]}]" if best_key else ""))
print(f"candidate files emitted:  {files}")
if res:
    print(f"results best_psnr_db:     {round(res.get('best_psnr_db', 0), 3)}  ({res.get('best_name', '?')[:44]})")
print("==================================")
