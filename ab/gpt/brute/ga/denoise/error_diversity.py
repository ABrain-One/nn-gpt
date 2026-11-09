"""Are our models FUNCTIONALLY different, or only structurally different?

WHAT WE COULD ALREADY COUNT, AND WHY IT IS NOT ENOUGH. Structural diversity is measured:
1052 distinct computation graphs (WL exact) collapsing to 228 neighbourhoods at
continuous-WL cosine >= 0.95. Neither number says anything about BEHAVIOUR, and the corpus
is suspicious on exactly that point -- 73% of models score between 33 and 34 dB, a 1.5 dB
band holding 90% of them. Different graphs can compute nearly the same function.

WHY THE PROJECT HAD NO ANSWER. diversity_gate.py records the functional metrics that were
tried and rejected:

    CKA on random-init models is weight-noise-dominated (same architecture reloaded scored
    0.69, distinct archs 0.92; could not separate 1F from 9F). Functional CKA and
    empirical-NTK were both tested seed-marginalized and do NOT transfer to denoising.

The diagnosis given is that reconstruction U-Nets share an encoder-decoder-skip skeleton, so
internal features and gradient kernels saturate. Note what BOTH failed metrics have in
common: they read INTERNAL REPRESENTATIONS of UNTRAINED networks. This script reads neither.
It reads the TASK OUTPUT of TRAINED weights -- which only became measurable at corpus scale
once save_pth_weights was enabled and 626 checkpoints accumulated.

THE RESIDUAL, NOT THE OUTPUT -- the single decision this file turns on. Correlating model
outputs directly is worthless here: every model approximates the same clean image, so every
pair correlates ~0.999 and a monoculture scores as maximally diverse. The discriminative
signal is the ERROR, (prediction - ground truth). Two denoisers that make THE SAME MISTAKES
ON THE SAME PIXELS have learned the same function however different their graphs; two that
fail differently have not. This is the standard ensemble-diversity measure, and it is the
number that decides whether this corpus is a useful ensemble or a redundant one.

CONTROLS. Every model sees the IDENTICAL fixed batch in the same order (get_noisy_batch is
seeded), so no correlation is an artefact of different data. Errors are mean-centred per
model before correlating, making this Pearson rather than raw cosine -- without that, any two
models sharing a constant brightness bias would correlate highly on the bias alone.

LIMITS, STATED RATHER THAN BURIED. High correlation proves functional similarity ON THIS
VALIDATION DISTRIBUTION and this noise type; it does not license claims about other noise.
Models whose weights were never saved are excluded and counted, never imputed.

  python error_diversity.py --batch 24 --corr-max 0.95
"""
import argparse
import collections
import json
import os
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, "/a/mm/nn-gpt/ab/gpt/brute/ast"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

TASK = "img-denoising"
IN_SHAPE = (1, 3, 256, 256)
OUT_SHAPE = (3,)
PRM = {"lr": 1e-4, "batch": 16, "epoch_max": 5, "transform": "denoise"}


def arm_of(name):
    if name.startswith("LLM_") or "_seeded_" in name:
        return "LLM"
    if name.startswith(("DenoiseGA-", "DnSeedGA-", "DenoiseSkip-", "DnSkip-",
                        "DnNewFam-", "DnSearch-")):
        return "GA"
    return "hand-written"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/a/mm/nn-dataset/db/ab.nn.db")
    ap.add_argument("--ckpt", default="/a/mm/sweep_out/ckpt")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--corr-max", type=float, default=0.95)
    # WITHOUT THIS THE HEADLINE NUMBER IS UNINTERPRETABLE. Every model denoises the SAME
    # images with the SAME noise realisation, so a large part of every residual is driven by
    # the shared input noise rather than by the model. That inflates every correlation by an
    # unknown constant. The first run reported a minimum pairwise correlation of 0.693 across
    # 315 models and I could not say whether that meant "these models are alike" or "this
    # metric reports 0.69 for anything", because there was no baseline. These two give one:
    #   identity  -- predict the noisy input unchanged (a model that does NO denoising)
    #   random-N  -- real architectures with their weights NEVER loaded
    # Trained-vs-random near 0.2 with trained-vs-trained bottoming at 0.693 means the
    # corpus really is functionally concentrated. Trained-vs-random ALSO near 0.69 means the
    # metric is measuring the noise and every number built on it is void.
    ap.add_argument("--controls", type=int, default=8,
                    help="random-init control models (0 disables); identity is always added")
    ap.add_argument("--out", default="/a/mm/sweep_out/manifest/error_diversity.json")
    a = ap.parse_args()

    import numpy as np
    import torch
    from functional_similarity import get_noisy_batch, _disable_db_prm_lookup
    _disable_db_prm_lookup()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== functional (error-correlation) diversity | device={dev} ===", flush=True)

    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    best = {n: (v or 0) * 48.0 for n, v in con.execute(
        "SELECT nn, MAX(accuracy) FROM stat WHERE task=? AND accuracy IS NOT NULL "
        "GROUP BY nn", (TASK,))}
    rows = con.execute("SELECT name, code FROM nn").fetchall()
    con.close()

    have = [(n, c) for n, c in rows
            if os.path.exists(os.path.join(a.ckpt, n, "best_model.pth"))]
    print(f"  {len(rows)} models in DB | {len(have)} have trained weights", flush=True)
    if a.limit:
        have = sorted(have, key=lambda r: -best.get(r[0], 0))[:a.limit]

    noisy, clean = get_noisy_batch(a.batch)
    noisy, clean = noisy.to(dev), clean.to(dev)
    print(f"  fixed batch {tuple(noisy.shape)} (seeded, identical for every model)",
          flush=True)

    errs, names, failed, t0 = [], [], [], time.time()
    for i, (name, code) in enumerate(have, 1):
        net = None
        try:
            ns = {}
            exec(compile(code, "<m>", "exec"), ns)
            net = ns["Net"](IN_SHAPE, OUT_SHAPE, dict(PRM), dev).to(dev)
            sd = torch.load(os.path.join(a.ckpt, name, "best_model.pth"),
                            map_location=dev, weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            # strict=False: some checkpoints carry num_batches_tracked / wrapper prefixes.
            # A silently EMPTY load would make the model a random net and correlate with
            # nothing, so require that something actually matched.
            res = net.load_state_dict(sd, strict=False)
            loaded = len(sd) - len(getattr(res, "unexpected_keys", []))
            if loaded <= 0:
                raise ValueError("no checkpoint tensor matched the model")
            net.eval()
            with torch.no_grad():
                p = net(noisy)
            if p.shape != clean.shape:
                raise ValueError(f"out {tuple(p.shape)} != target {tuple(clean.shape)}")
            e = (p - clean).flatten().float()
            if not torch.isfinite(e).all():
                raise ValueError("non-finite output")
            e = e - e.mean()
            nrm = e.norm()
            if nrm.item() == 0:
                raise ValueError("zero-variance error")
            errs.append((e / nrm).cpu().numpy())
            names.append(name)
        except Exception as ex:
            failed.append((name, f"{type(ex).__name__}: {ex}"[:110]))
        finally:
            del net
            if dev == "cuda":
                torch.cuda.empty_cache()
        if i % 25 == 0 or i == len(have):
            print(f"  [{i}/{len(have)}] ok={len(errs)} failed={len(failed)} "
                  f"{time.time()-t0:.0f}s", flush=True)

    if len(errs) < 2:
        print("FATAL: fewer than 2 usable error maps")
        for n, e in failed[:10]:
            print(f"   {n[:44]}: {e}")
        return 1

    n_trained = len(errs)

    def _push(vec, label):
        v = vec.flatten().float()
        v = v - v.mean()
        nr = v.norm()
        if nr.item() > 0 and torch.isfinite(v).all():
            errs.append((v / nr).cpu().numpy())
            names.append(label)
            return True
        return False

    ctrl_names = []
    if a.controls >= 0:
        # A denoiser that does nothing. Its "error" is exactly the input noise, so its
        # correlation with the real models is the clearest possible read on how much of the
        # measured similarity is the shared noise rather than shared behaviour.
        if _push(noisy - clean, "__CONTROL_identity__"):
            ctrl_names.append("__CONTROL_identity__")
        # Real architectures, weights never loaded.
        picked = 0
        for name, code in have:
            if picked >= a.controls:
                break
            try:
                ns = {}
                exec(compile(code, "<m>", "exec"), ns)
                net = ns["Net"](IN_SHAPE, OUT_SHAPE, dict(PRM), dev).to(dev).eval()
                with torch.no_grad():
                    p = net(noisy)
                if p.shape == clean.shape and _push(p - clean, f"__CONTROL_rand{picked}__"):
                    ctrl_names.append(f"__CONTROL_rand{picked}__")
                    picked += 1
                del net
                if dev == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                continue
        print(f"  controls added: {len(ctrl_names)} "
              f"(identity + {picked} random-init)", flush=True)

    M = np.stack(errs).astype(np.float32)
    Cfull = M @ M.T
    np.fill_diagonal(Cfull, 0.0)

    # CONTROLS ARE NEVER PART OF THE CORPUS STATISTICS. They are a yardstick held against it;
    # folding them into the clustering or the pairwise distribution would contaminate the
    # very number they exist to calibrate.
    C = Cfull[:n_trained, :n_trained]
    names_t = names[:n_trained]

    if ctrl_names:
        print(f"\n=== NEGATIVE CONTROLS -- is the metric measuring behaviour or noise? ===")
        tt = C[np.triu_indices(n_trained, k=1)]
        print(f"  trained x trained    mean {tt.mean():.3f}   min {tt.min():.3f}")
        for j, cn in enumerate(ctrl_names):
            col = Cfull[n_trained + j, :n_trained]
            print(f"  trained x {cn[10:-2]:<12} mean {col.mean():.3f}   "
                  f"min {col.min():.3f}   max {col.max():.3f}")
        rnd = [n_trained + j for j, c in enumerate(ctrl_names) if "rand" in c]
        if rnd:
            rc = Cfull[np.ix_(rnd, list(range(n_trained)))]
            gap = tt.mean() - rc.mean()
            print(f"\n  SEPARATION trained-vs-trained minus trained-vs-random: {gap:+.3f}")
            print("  " + ("=> the metric DISCRIMINATES; the corpus numbers below are real."
                          if gap > 0.15 else
                          "=> WARNING: little separation. The correlation is dominated by "
                          "the shared noise realisation, and the cluster counts below are "
                          "NOT trustworthy as a behaviour measure."))

    # Greedy clustering in descending PSNR -- the SAME procedure corpus_diversity.py uses for
    # the structural count, so "228 structural" and this number are comparable rather than
    # two different algorithms sharing a word.
    names = names_t
    order = sorted(range(len(names)), key=lambda i: -best.get(names[i], 0.0))
    reps, members = [], []
    for i in order:
        hit = next((k for k, r in enumerate(reps) if C[i, r] >= a.corr_max), None)
        if hit is None:
            reps.append(i)
            members.append([names[i]])
        else:
            members[hit].append(names[i])

    pw = C[np.triu_indices(len(names), k=1)]
    print(f"\n=== RESULT | {len(names)} models measured ===")
    print(f"  pairwise ERROR correlation")
    print(f"    min {pw.min():.3f}  median {np.median(pw):.3f}  "
          f"mean {pw.mean():.3f}  max {pw.max():.3f}")
    for t in (0.99, 0.95, 0.90, 0.80, 0.50, 0.0):
        print(f"    pairs >= {t:4.2f}  {(pw >= t).sum():>10d}  ({(pw >= t).mean():6.1%})")
    print(f"\n  FUNCTIONAL CLUSTERS at corr >= {a.corr_max}   {len(reps)}")
    print(f"  absorbed into an existing cluster          {len(names) - len(reps)}")
    sizes = sorted((len(m) for m in members), reverse=True)
    print(f"  largest cluster  {sizes[0]} models ({sizes[0]/len(names):.0%})")
    print(f"  singleton clusters {sum(1 for s in sizes if s == 1)}")

    by_arm = collections.Counter(arm_of(n) for n in names)
    rep_arm = collections.Counter(arm_of(names[i]) for i in reps)
    print(f"\n=== per arm (models -> functional clusters) ===")
    for arm in sorted(by_arm):
        print(f"  {arm:<14} {by_arm[arm]:5d} -> {rep_arm.get(arm, 0):5d}")

    idx = {arm: [i for i, n in enumerate(names) if arm_of(n) == arm] for arm in by_arm}
    print(f"\n=== mean error correlation WITHIN / ACROSS arms ===")
    arms = sorted(idx)
    for i, p in enumerate(arms):
        for q in arms[i:]:
            if not idx[p] or not idx[q]:
                continue
            sub = C[np.ix_(idx[p], idx[q])]
            v = sub[np.triu_indices(len(idx[p]), k=1)] if p == q else sub.ravel()
            if v.size:
                print(f"  {p:<14} x {q:<14} {v.mean():.3f}")

    print(f"\n=== 5 largest functional clusters ===")
    for k in sorted(range(len(members)), key=lambda k: -len(members[k]))[:5]:
        if len(members[k]) > 1:
            print(f"  {len(members[k]):4d} models behave like {members[k][0][:46]}")

    if failed:
        print(f"\n=== excluded ({len(failed)}) ===")
        for k, v in collections.Counter(e.split(":")[0] for _n, e in failed).most_common(8):
            print(f"  {v:4d}  {k}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"measured": len(names), "corr_max": a.corr_max,
                   "functional_clusters": len(reps),
                   "median_pairwise_corr": float(np.median(pw)),
                   "mean_pairwise_corr": float(pw.mean()),
                   "per_arm": {k: [by_arm[k], rep_arm.get(k, 0)] for k in by_arm},
                   "clusters": {members[k][0]: members[k] for k in range(len(members))},
                   "excluded": dict(failed)}, f, indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
