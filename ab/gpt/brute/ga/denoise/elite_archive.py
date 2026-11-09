"""Build the elite reference pool that gives the LLM run GENERATIONS.

WHAT THIS FIXES. Until now the LLM run was one-shot i.i.d. sampling: 550 independent
prompts, prompt 550 knowing nothing about prompt 1. No selection, no feedback, no
improvement across the run. The seeded arm referenced HAND-WRITTEN seeds only -- never the
LLM's own output -- so it was conditioning, not evolution. The GA meanwhile compounds:
generation N+1 is built from N's survivors.

This script closes that loop. It reads what the LLM actually produced and trained, ranks it,
and writes an elite pool that the next generation conditions on. That is the LLM's analogue
of GA selection pressure, and it is what makes the two approaches comparable rather than
merely co-run.

RANKING uses the mean of the last 3 of the 5 trained epochs, not the max. A max over epochs
rewards volatility: the first stage-2 model peaked at 34.145 dB on epoch 4 and settled at
33.527 on epoch 5, so ranking on peaks would advertise the noisiest models as the best. This
mirrors the last5 rule already used for the hand-written seed corpus.

DIVERSITY IS ENFORCED IN THE POOL, not just measured. An elite pool picked purely on dB
collapses: the top-K of a converging population are near-copies of each other, and
conditioning on K copies of one architecture is a fast route to mode collapse -- the
LLM-generation analogue of a GA losing population diversity. Elites are therefore admitted
greedily in dB order subject to the SAME gate the GA uses (exact WL graph hash, plus graded
cosine when --cos-max is given), so the pool spans architectures rather than ranking one.

CARRY-OVER. Hand-written seeds stay eligible via --floor-seeds: if the LLM's own output is
worse than the hand-written corpus, conditioning purely on LLM output would walk the
population downhill. The pool is the best of BOTH, which is elitism in the GA sense.

  python elite_archive.py --stage2-dirs /a/mm/sweep_out/llm_stage2 --k 8 \
      --out /a/mm/sweep_out/llm_evolve/elites_gen1.json
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, "/a/mm/nn-gpt/ab/gpt/brute/ast"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

PSNR_SCALE = 48.0


def curve_of(stat_dir):
    """[(epoch, dB)] from check_nn's per-epoch JSONs (a LIST wrapping one dict)."""
    out = []
    for p in glob.glob(os.path.join(stat_dir, "**", "*.json"), recursive=True):
        b = os.path.basename(p)[:-5]
        if not b.isdigit():
            continue
        try:
            d = json.load(open(p))
            if isinstance(d, list):
                d = d[0] if d else {}
            a = d.get("accuracy")
            if a is not None:
                out.append((int(b), a * PSNR_SCALE))
        except Exception as e:
            print(f"  [curve] skipped {p}: {type(e).__name__}: {e}")
    return sorted(out)


def macs_mm(src):
    """MMac at 3x256x256 via the GA's own counter, or None if unmeasurable."""
    try:
        import torch
        from edge_cost import _mac_hooks
        ns = {}
        exec(compile(src, "<elite>", "exec"), ns)
        net = ns["Net"]((1, 3, 256, 256), (3,),
                        {"lr": 1e-4, "batch": 16, "epoch_max": 5,
                         "transform": "denoise"}, "cpu")
        net.eval()
        handles, macs = _mac_hooks(net)
        try:
            with torch.no_grad():
                net(torch.rand(1, 3, 256, 256))
        finally:
            for h in handles:
                h.remove()
        return macs["v"] / 1e6
    except Exception as e:
        # Return the REASON, not just None. Generation 1 measured 0096 at 10787 MMac and
        # skipped it as over-cap; generation 2 called the same file "unmeasurable" -- along
        # with 0121 (1274 MMac) and 0066 (1463 MMac), our two CHEAPEST and most edge-relevant
        # LLM models. Identical code, identical candidates, different answer, and a bare
        # `except: return None` made it impossible to say why. The gate stays fail-closed
        # (unmeasurable is never treated as legal), but silently dropping the models that
        # best fit the budget is a real cost, so the cause has to reach the log.
        return None, f"{type(e).__name__}: {e}"[:160]


def _common_depth(a, b):
    """How many leading path components two paths share."""
    pa, pb = os.path.abspath(a).split(os.sep), os.path.abspath(b).split(os.sep)
    n = 0
    for x, y in zip(pa, pb):
        if x != y:
            break
        n += 1
    return n


def collect(stage2_dirs, run_dirs, min_epochs=5, mac_cap=9053.0):
    """[{id, arm, code, db, epoch, prm, curve}] for every LLM model that reached stage 2.

    CANDIDATE IDS ARE NOT UNIQUE ACROSS GENERATIONS. Every generation numbers its 150
    candidates 0000..0149 with the same _<arm>_<strategy> suffix, so gen 1 contains a
    completely different model also called `0026_seeded_direct_instruction`. The original
    `src[cid] = path` built one flat dict over all run dirs, where the last generation
    written silently won -- pairing one generation's TRAINING CURVE with another
    generation's SOURCE CODE.

    That is how gen 2 came to report 0026/0066/0096/0121 as "MACs unmeasurable": their
    curves came from gen 0's stage2, but their code resolved to gen 1's same-named
    candidates, 119 of which had failed preflight and do not build. It cost us 0121
    (1274 MMac) and 0066 (1463 MMac) -- the two cheapest, most edge-relevant models we
    have. It failed safe only by luck: had the colliding file merely been buildable, a
    wrong architecture would have entered the elite pool carrying someone else's dB, and
    the diversity gate would have hashed the wrong code too.

    So candidates are resolved PER STAGE2 DIR, preferring the run dir sharing the longest
    path prefix with it -- i.e. the one from the same generation.
    """
    by_id = {}
    for r in run_dirs:
        for p in glob.glob(os.path.join(r, "cand", "*", "candidate.py")):
            by_id.setdefault(os.path.basename(os.path.dirname(p)), []).append(p)
    got = []
    for sd in stage2_dirs:
        for d in sorted(glob.glob(os.path.join(sd, "full", "*"))):
            cid = os.path.basename(d)
            c = curve_of(d)
            if not c or cid not in by_id:
                continue
            # Same generation as this stage2 dir wins. Ties (gen 0, whose stage2 dir is a
            # sibling of both run dirs) fall back to run-dir order, which is safe there
            # because the arm is part of the id.
            cand_path = max(by_id[cid], key=lambda p: _common_depth(p, sd))
            # Ranking a partially-trained model against fully-trained ones is not a fair
            # comparison -- an interrupted 2-epoch run is undertrained, not worse. Drop it
            # rather than let it compete for an elite slot on a number that means something
            # different from everyone else's.
            if c[-1][0] < min_epochs:
                print(f"  [skip] {cid}: only {c[-1][0]} epochs (< {min_epochs})")
                continue
            # AN OVER-CAP ELITE POISONS EVERY LATER GENERATION. Measured on the first 9
            # trained LLM models: 2 were over (10787 and 16968 MMac), and the 10787 one had
            # already been selected as an elite. Nothing else in the LLM path checks this --
            # screen_and_train.py has no MAC check -- so the gate has to be here as well as
            # in preflight. Unmeasurable is treated as ineligible, never as legal.
            if mac_cap and mac_cap > 0:
                mm = macs_mm(open(cand_path).read())
                why = ""
                if isinstance(mm, tuple):
                    mm, why = mm
                if mm is None:
                    print(f"  [skip] {cid}: MACs unmeasurable ({why}) "
                          f"-> not eligible as an elite")
                    continue
                if mm > mac_cap:
                    print(f"  [skip] {cid}: {mm:.0f} MMac > {mac_cap:.0f} cap")
                    continue
            tail = [db for _e, db in c[-3:]]
            got.append({
                "id": cid,
                "arm": "cold" if "_cold_" in cid else ("seeded" if "_seeded_" in cid else "?"),
                "code": open(cand_path).read(),
                "db": sum(tail) / len(tail),      # last-3 mean, NOT the peak
                "epoch": c[-1][0],
                "prm": {"lr": 1e-4, "batch": 16, "epoch_max": c[-1][0],
                        "transform": "denoise"},
                "curve": c,
            })
    return got


def floor_seeds(n, seed_set="canonical"):
    """Best hand-written families, so the pool cannot walk downhill.

    WHICH POOL MATTERS MORE THAN IT LOOKS. "canonical" is the 18 swept families and its top
    four are DenoiseSubPixel/DilatedFlat/LPPool/ELUCtx -- 33.28-33.94 dB on the crop metric
    at 3,964-4,442 MMac. "named" adds the 23 hand-written seeds and its top four reach
    35.48-35.64 dB, one of them (DenoiseLowRank) at 865 MMac. Anchoring an LLM run on the
    canonical four hands it ancestors ~1.8 dB worse than the corpus already contains, which
    is the handicap under which llm_evolve_par5 ran 35 generations without ever displacing a
    seed. Default stays "canonical" so existing callers do not move underneath them.
    """
    try:
        import llm_denoise
        corpus = llm_denoise.seed_corpus(seed_set=seed_set)
    except Exception as e:
        print(f"  [floor] seed corpus unavailable ({type(e).__name__}: {e}); "
              f"proceeding with LLM output only")
        return []
    corpus = sorted(corpus, key=lambda s: -s["db"])[:n]
    for s in corpus:
        s["id"] = s["name"]
        s["arm"] = "seed"
    return corpus


def select(cands, k, cos_max):
    """Greedy by dB, admitting only architecturally novel entries."""
    from diversity_gate import DiversityGate
    g = DiversityGate(graded=cos_max is not None, cos_max=cos_max or 0.95,
                      log=lambda *a, **kw: None)
    chosen, rejected = [], []
    for c in sorted(cands, key=lambda x: -x["db"]):
        ok, fp = g.check(c["code"])
        if ok:
            g.add(c["id"], fp)
            chosen.append(c)
            if len(chosen) >= k:
                break
        else:
            rejected.append(c["id"])
    return chosen, rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2-dirs", nargs="+", required=True)
    ap.add_argument("--run-dirs", nargs="+",
                    default=["/a/mm/sweep_out/llm_gen2/cold/run",
                             "/a/mm/sweep_out/llm_gen2/seeded/run"])
    ap.add_argument("--k", type=int, default=8, help="elite pool size")
    ap.add_argument("--floor-seed-set", default="canonical",
                    help="pool the floor seeds are drawn from; see floor_seeds()")
    ap.add_argument("--floor-seeds", type=int, default=4,
                    help="best hand-written families kept eligible (0 to disable)")
    ap.add_argument("--cos-max", type=float, default=0.95,
                    help="graded near-duplicate threshold; <=0 for exact-hash only")
    ap.add_argument("--mac-cap", type=float, default=9053.0,
                    help="reject elites above this MMac (0 disables)")
    ap.add_argument("--min-epochs", type=int, default=5,
                    help="ignore models trained for fewer epochs than this; comparing a "
                         "part-trained model against fully-trained ones ranks it unfairly")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    llm = collect(a.stage2_dirs, a.run_dirs, a.min_epochs, a.mac_cap)
    print(f"=== elite archive ===")
    print(f"  LLM models with a trained curve : {len(llm)}")
    if llm:
        best = max(llm, key=lambda c: c["db"])
        print(f"  best LLM                        : {best['id']} {best['db']:.3f} dB "
              f"({best['arm']})")
    seeds = floor_seeds(a.floor_seeds, a.floor_seed_set) if a.floor_seeds else []
    print(f"  hand-written floor seeds        : {len(seeds)}")

    pool = llm + seeds
    if not pool:
        raise SystemExit("FATAL: nothing to build an elite pool from.")
    chosen, rejected = select(pool, a.k, a.cos_max if a.cos_max > 0 else None)

    print(f"\n  selected {len(chosen)} elites "
          f"({sum(1 for c in chosen if c['arm']=='seed')} hand-written, "
          f"{sum(1 for c in chosen if c['arm']!='seed')} LLM); "
          f"{len(rejected)} rejected as near-duplicates")
    for c in chosen:
        print(f"    {c['db']:7.3f} dB  {c['arm']:<7} {c['id']}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump([{k: v for k, v in c.items() if k != "curve"} for c in chosen],
                  f, indent=2)
    print(f"\nwrote {a.out}")
    # An all-hand-written pool means the LLM has not yet beaten the seed corpus. That is a
    # real result, not an error -- but the next generation is then NOT evolving its own
    # output, and any "the LLM improved across generations" claim would be false.
    if chosen and all(c["arm"] == "seed" for c in chosen):
        print("NOTE: every elite is hand-written -- the LLM has not yet produced anything "
              "that survives selection. The next generation is conditioning, not evolving.")


if __name__ == "__main__":
    raise SystemExit(main())
