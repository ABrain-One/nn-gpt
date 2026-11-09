import sys, glob, os
sys.path.insert(0, "/a/mm/nn-gpt/ab/gpt/brute/ga/denoise")
sys.path.insert(0, "/a/mm/nn-gpt/ab/gpt/brute/ast")
import numpy as np, torch
from functional_similarity import get_noisy_batch, model_outputs, gram_centered, linear_cka

D = "/a/mm/nn-gpt/ab/gpt/brute/ga/denoise"
noisy, _ = get_noisy_batch(24)
files = glob.glob(D + "/seed_ga_P1_models/*.py")
grams, names = [], []
for i, fp in enumerate(files):
    out = model_outputs(open(fp).read(), noisy)
    if out is None:
        continue
    grams.append(gram_centered(out - noisy)); names.append(os.path.basename(fp)[:-3])
    if (i+1) % 25 == 0: print(f"  ...{i+1}/{len(files)} scored", flush=True)
n = len(grams)
print(f"functional fingerprints: {n} models", flush=True)
sims = []
for a in range(n):
    for b in range(a+1, n):
        sims.append(float(linear_cka(grams[a], grams[b])))
sims = np.array(sims)
print("=== FUNCTIONAL (residual-CKA) similarity ===")
print(f"  pairs:            {len(sims)}")
print(f"  mean pairwise:    {100*sims.mean():.1f}%")
print(f"  median:           {100*np.median(sims):.1f}%")
print(f"  min / max:        {100*sims.min():.1f}% / {100*sims.max():.1f}%")
print(f"  pairs >90% (functionally near-identical): {100*(sims>0.90).mean():.1f}%")
print(f"  pairs <50% (genuinely distinct behaviour): {100*(sims<0.50).mean():.1f}%")
