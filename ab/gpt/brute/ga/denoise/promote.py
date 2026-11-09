"""Between-run promotion gate for the production seed-pool growth loop.

After a run's winners are full-trained (fulltrain_topk.py / a production loop), this
decides which of them are worth adding to the seed pool for the NEXT run. A model
becomes a seed ONLY if it clears all four gates (see PRODUCTION_RUN.md / the
denoise-model-similarity index):

  1. DISTINCT (primary)  not a graph-clone of any pool member OR already-promoted model.
                   THE metric is the exact WL graph hash from `arch_embed` (the unified
                   VOCABULARY-FREE algorithm: torch.fx trace; node identity = op-type +
                   ALL generically-introspected numeric hyperparams, no hand-authored op
                   list). Exact, free, name-agnostic, seed-independent, topology-aware.
                   Equal hash => same architecture => clone (greedy vs pool + promoted).
                   Untraceable models fall back to module-signature cosine < --sig-max.
                   This is the guarantee behind the anti-bias claim ("N provably distinct
                   architectures"). Recommended by Dmytro; validated (traces 9/9 seeds,
                   seed-independent, separates 1F/9F that CKA could not).
  2. FLOOR         psnr50 >= --bar (a real result, not a proxy fluke) — minimum quality.
  3. DEPLOYABLE    passes edge_deploy (undeployable != a result); within MAC cap.
  4. CROSSOVER-OK  exposes the Block(channels) contract so it can recombine as a parent.
                   SOFT by default (flagged, not fatal) — a mutation-only model is still
                   a useful seed, and _swappable_block detection is conservative (it even
                   fails on DenoiseUnet3F, a live production seed). Fatal with --require-xover.
  Optional: --pareto adds an efficiency prune (drop a model dominated on (psnr50 up,
  MACs down) by an ALREADY-PROMOTED model). Off by default — diversity-first: a distinct
  architecture is never vetoed for being slightly off the efficiency frontier.

  WHY WL-HASH, NOT CKA (measured, not assumed): promote.py evaluates models at RANDOM
  INIT (trained weights are never persisted). Init CKA is weight-noise-dominated (same
  arch scored 0.69, distinct 0.92; 1F vs 9F 0.99). Functional CKA AND empirical-NTK were
  both tested seed-marginalized and do NOT transfer to denoising (reconstruction U-Nets
  share a skeleton -> features and gradient kernels saturate). The WL graph hash is exact
  and weight-free. CKA/agreement remain available via --functional but INFORMATIONAL ONLY.
  See scratchpad/{wl_hash_test,seedavg_cka,proxy_cka,ntk_align,struct_validate}.py.

Emits: promote_report.json + (with --emit-dir) the promoted sources as .py files that
`run_seed_ga.py --seed-files <dir>/*.py` can load directly into the next run's pool
(promoted winners aren't in the DB while save_to_db is OFF). Nothing is trained here.

Usage:
  # combine both runs' full-trained models; WL-graph-hash distinctness + quality floor
  python promote.py \
      --results seed_ga_P1_fulltrain.json seed_ga_P2_fulltrain.json \
      --models-dirs seed_ga_P1_models seed_ga_P2_models \
      --bar 32.0 --emit-dir promoted_run2
"""
import argparse
import ast
import glob
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "predictor"))
sys.path.insert(0, "/a/mm/nn-gpt/ab/gpt/brute/ast")
sys.path.insert(0, "/a/mm/nn-gpt")   # for ab.gpt.util (MinHash textual sim)

NN_DIR = "/a/mm/nn-dataset/ab/nn/nn"
DEFAULT_POOL = ["DenoiseUnet1F", "DenoiseUnet6F", "DenoiseUnet7F", "DenoiseUnet9F",
                "DenoiseUnet10F", "DenoiseUnet3F", "DenoiseUnet4F", "DenoiseUnet8F",
                "LiteDenoiseNet"]     # the 9 production seeds (was a stale 5-seed list)


def _cos(a, b):
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def _struct_vec(source):
    import features as F
    return F.vector(source)


# --- module-signature descriptor: the VALIDATED architectural distinctness metric ---
# A weight-INDEPENDENT multiset of instantiated leaf-module reprs
# (e.g. Conv2d(16,32,kernel_size=(3,3),stride=(2,2))). Determined purely by the
# architecture: same arch -> identical bag (cosine 1.0, reproducible), distinct
# archs -> disjoint bags (cosine ~0). Diagnosed superior to residual-CKA, which on
# random-init models is weight-noise-dominated (same arch scored 0.69, distinct 0.92)
# and cannot separate e.g. 1F from 9F (0.99) that this cleanly separates (0.37). No
# GPU, no forward pass, no training-state dependence. See scratchpad/struct_validate.py.
def _module_sig(source):
    from collections import Counter
    from mutator.lemur_io import load_lemur_model, cleanup_temp_module
    model = None
    try:
        model = load_lemur_model(source)
        sig = Counter()
        for _n, mod in model.named_modules():
            if len(list(mod.children())) == 0:          # leaf module
                sig[str(mod)] += 1
        return sig
    except Exception:
        return None
    finally:
        if model is not None:
            info = getattr(model, "_temp_module_info", None)
            if info:
                try:
                    cleanup_temp_module(*info)
                except Exception:
                    pass


def _sig_cos(a, b):
    import math
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


# --- WL graph hash: THE distinctness metric (Dmytro's recommendation, validated) ---
# Trace each model to its computation graph (torch.fx), label nodes by OP TYPE (module
# class + kernel + stride + channels for call_module; function/method name otherwise), and
# Weisfeiler-Lehman hash the graph (channel-inclusive: a width-variant is a distinct model).
# Exact, free (no forward pass, no weights), name-agnostic, seed-independent, and topology-
# aware (captures the skip/cat/add wiring a plain layer-inventory misses). Equal hash =>
# same architecture => clone. Chosen over residual-CKA (weight-noise-dominated at init:
# same arch scored 0.69, distinct 0.92) and over the module-signature bag (fooled UP by
# width). Functional CKA and empirical-NTK were BOTH tested (seed-marginalized) and do NOT
# transfer to denoising: reconstruction U-Nets share an encoder-decoder-skip skeleton, so
# converged features AND gradient kernels saturate (NTK still had cross>self for 1F/9F).
# See scratchpad/{wl_hash_test,seedavg_cka,proxy_cka,ntk_align,struct_validate}.py.
def _wl_hash(source):
    """Exact WL graph hash via arch_embed (the unified vocabulary-free algorithm — single
    source of truth). None if the model can't be loaded/traced (caller falls back to the
    module-signature for those)."""
    import arch_embed
    return arch_embed.exact_source(source)


def _xover_ok(source):
    """True if the model exposes the Block(channels) contract block_crossover needs."""
    from block_crossover import _swappable_block
    try:
        return _swappable_block(ast.parse(source)) is not None
    except Exception:
        return False


def _load_candidates(results_paths, models_dirs, log):
    """From one or more fulltrain_*.json files: (name, checksum, psnr50, macs,
    deployable, source). A checksum full-trained in multiple runs is deduped
    (keep the higher psnr50), so combining Run 1 + Run 2 results is safe."""
    seen = {}
    for results_path in results_paths:
        if not os.path.exists(results_path):
            log(f"  (results file {results_path} not found — skipped)")
            continue
        data = json.load(open(results_path))
        runs = data.get("runs", data if isinstance(data, list) else [])
        for r in runs:
            if not r.get("psnr50") or r["psnr50"] <= 0:
                continue
            cks = r["checksum"]
            if cks in seen and seen[cks]["psnr50"] >= r["psnr50"]:
                continue                              # keep the better-trained copy
            src = None
            for md in models_dirs:
                p = os.path.join(md, cks + ".py")
                if os.path.exists(p):
                    src = open(p).read(); break
            if src is None:
                log(f"  (source not found for {r.get('name')} [{cks}] — skipped)")
                continue
            seen[cks] = dict(name=r.get("name"), checksum=cks,
                             psnr50=r["psnr50"], macs=r.get("macs"),
                             deployable=r.get("deployable"), source=src)
    out = list(seen.values())
    out.sort(key=lambda d: d["psnr50"], reverse=True)   # best (real) first
    return out


def _functional_index(pool_srcs, cand_srcs, n, log):
    """Return (residual-outputs dict keyed by id). Behavioural fingerprints on a
    FIXED noisy batch; residual = output - noisy (raw-output CKA saturates)."""
    from functional_similarity import get_noisy_batch, model_outputs, gram_centered
    noisy, _ = get_noisy_batch(n)
    grams, agree_out = {}, {}
    for key, src in list(pool_srcs.items()) + list(cand_srcs.items()):
        out = model_outputs(src, noisy)
        if out is None:
            continue
        res = out - noisy            # residual behaviour
        grams[key] = gram_centered(res)
        agree_out[key] = out
    return grams, agree_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="*",
                    default=[os.path.join(_HERE, "fulltrain_results.json")],
                    help="one or more fulltrain_*.json files (combined + deduped)")
    ap.add_argument("--models-dirs", nargs="*",
                    default=[os.path.join(_HERE, f"seed_ga_{a}_models")
                             for a in ("A0", "B0", "C0", "D0")])
    ap.add_argument("--pool", nargs="*", default=DEFAULT_POOL,
                    help="current seed pool (names resolved from the nn dir)")
    ap.add_argument("--pareto", action="store_true",
                    help="promote on efficiency-Pareto (psnr50 up / MACs down) instead of "
                         "a flat PSNR bar; --bar becomes a quality FLOOR. Keeps the grown "
                         "pool DIVERSE — distinct architectures are lower-PSNR but Pareto-"
                         "optimal at their cost, so a flat bar would re-monoculture the pool.")
    ap.add_argument("--bar", type=float, default=33.0,
                    help="min psnr50 to be competitive; with --pareto this is the quality floor")
    ap.add_argument("--mac-cap", type=float, default=9053.0,
                    help="MMac ceiling; over-cap is flagged, not auto-rejected (use --strict-mac)")
    ap.add_argument("--strict-mac", action="store_true",
                    help="hard-reject candidates over --mac-cap")
    ap.add_argument("--npu-verdict", default=None,
                    help="verdict.json from npu_gate.py: MEASURED on-device latency per candidate. "
                         "Without it, gate 3 is the MAC cap alone, which misses 26%% of undeployable "
                         "models (measured). Too-slow is a hard reject; unmeasured is a soft flag.")
    ap.add_argument("--npu-max-ms", type=float, default=10.0,
                    help="latency ceiling at 256x256 on the reference flagship. 10 ms sits inside the "
                         "largest gap in the measured distribution (6.60 -> 17.93 ms), so the verdict "
                         "is insensitive to moving it a few ms.")
    ap.add_argument("--require-xover", action="store_true",
                    help="make crossover-compat FATAL (default: soft flag). Use only if the "
                         "next run recombines by crossover exclusively; mutation-only seeds "
                         "are otherwise valid pool members.")
    ap.add_argument("--graded", action="store_true",
                    help="ALSO reject NEAR-duplicates (arch_embed continuous-WL cosine >= "
                         "--graded-max), not just exact graph-clones. Opt-in; for a mutation-only "
                         "generator this rejects same-family variants (cluster >=0.985).")
    ap.add_argument("--graded-max", type=float, default=0.95,
                    help="with --graded, reject if continuous-WL cosine to any pool/promoted "
                         "model >= this (0.95 calibrated; see metric_ablation/DECISION_LEDGER.md)")
    ap.add_argument("--sig-max", type=float, default=0.85,
                    help="FALLBACK only: for models that fail torch.fx tracing (no WL hash), "
                         "reject if module-signature cosine to any pool/promoted model exceeds "
                         "this. The primary distinctness gate is the WL graph hash (exact).")
    ap.add_argument("--struct-max", type=float, default=0.999,
                    help="INFORMATIONAL only now (AST-feature cosine; saturates, kept for "
                         "the report). The real gate is --sig-max (module-signature).")
    ap.add_argument("--functional", action="store_true",
                    help="INFORMATIONAL only: also compute residual-CKA + agreement dB (needs "
                         "GPU). Diagnosed unreliable for distinctness (weight-noise-dominated "
                         "on random-init models) — shown for the report, never gates.")
    ap.add_argument("--cka-max", type=float, default=0.90,
                    help="with --functional: reject if residual-CKA to a pool member exceeds this")
    ap.add_argument("--agree-max", type=float, default=30.0,
                    help="with --functional: reject if output-agreement dB to a pool member exceeds this")
    ap.add_argument("--n", type=int, default=24, help="noisy-batch size for functional")
    ap.add_argument("--emit-dir", default="",
                    help="write promoted sources here as .py (for run_seed_ga --seed-files)")
    ap.add_argument("--out", default=os.path.join(_HERE, "promote_report.json"))
    args = ap.parse_args()

    def log(m):
        print(m, flush=True)

    if not any(os.path.exists(p) for p in args.results):
        log(f"no results file among {args.results} — run fulltrain_topk.py first."); return

    cands = _load_candidates(args.results, args.models_dirs, log)
    # --pool accepts bare names (resolved from the nn dir) AND file paths
    # (e.g. new_seeds/*.py) — path members were previously silently dropped,
    # so the dedup gate never saw 14 of the 19 Run-3 seeds.
    pool_srcs = {}
    for nm in args.pool:
        if nm.endswith(".py") and os.path.exists(nm):
            pool_srcs[os.path.basename(nm)[:-3]] = open(nm).read()
        elif os.path.exists(os.path.join(NN_DIR, nm + ".py")):
            pool_srcs[nm] = open(os.path.join(NN_DIR, nm + ".py")).read()
    log(f"=== Promotion gate: {len(cands)} full-trained candidates vs pool "
        f"{list(pool_srcs)} ===")
    xg = "xover(req)" if args.require_xover else "xover(soft)"
    log(f"gates: DISTINCT(WL-graph-hash; module-sig<{args.sig_max} fallback)  "
        f"floor {args.bar}  deployable{'+npu<=' + format(args.npu_max_ms, '.0f') + 'ms' if args.npu_verdict else ''}  {xg}  pool-dedup"
        + (f"  +graded-near-dup(cos>={args.graded_max})" if args.graded else "")
        + ("  +pareto-prune" if args.pareto else "")
        + ("  +cka/agree(informational)" if args.functional else ""))

    from functional_similarity import _disable_db_prm_lookup
    _disable_db_prm_lookup()                    # skip ~2-min DB prm scan per instantiate
    log("computing WL graph hashes (topology) + module-signature fallback ...")
    # WL graph hash = PRIMARY distinctness (exact, topology-aware); module-sig = fallback
    pool_hashes = {k: _wl_hash(s) for k, s in pool_srcs.items()}
    cand_hashes = {c["checksum"]: _wl_hash(c["source"]) for c in cands}
    pool_sigs = {k: _module_sig(s) for k, s in pool_srcs.items()}
    cand_sigs = {c["checksum"]: _module_sig(c["source"]) for c in cands}
    n_untraced = sum(1 for v in cand_hashes.values() if v is None)
    if n_untraced:
        log(f"  ({n_untraced}/{len(cands)} candidates untraceable -> module-sig fallback)")

    # GRADED near-duplicate operating point (opt-in): continuous-WL embeddings
    pool_embeds = cand_embeds = {}
    if args.graded:
        import arch_embed
        log(f"computing continuous-WL embeddings (graded near-dup, cos>={args.graded_max}) ...")
        pool_embeds = {k: arch_embed.embed_source(s) for k, s in pool_srcs.items()}
        cand_embeds = {c["checksum"]: arch_embed.embed_source(c["source"]) for c in cands}

    # AST vectors kept for the informational column only
    pool_vecs = {k: _struct_vec(s) for k, s in pool_srcs.items()}
    cand_vecs = {c["checksum"]: _struct_vec(c["source"]) for c in cands}

    grams = agree = None
    if args.functional:
        log("computing behavioural fingerprints (residual-CKA) ...")
        grams, agree = _functional_index(
            pool_srcs, {c["checksum"]: c["source"] for c in cands}, args.n, log)
        from functional_similarity import linear_cka, out_psnr

    # greedy pass: a promoted candidate joins the reference set (pool + promoted-so-far)
    # so later candidates must differ from it too. The reference starts as the POOL ONLY
    # (the earlier bug seeded it with ALL candidates, so near-clone pairs mutually rejected
    # and nothing promoted).
    # MEASURED on-device latency, keyed by checksum AND by name so either resolves.
    # A file is used rather than a live call because profiling needs the AI Hub token and
    # network; promote.py must stay runnable inside a pod.
    npu_ms = None
    if args.npu_verdict:
        try:
            _v = json.load(open(args.npu_verdict))
            if _v.get("inconclusive"):
                log(f"  NPU verdict INCONCLUSIVE (coverage {_v.get('coverage', 0):.0%}) -- "
                    f"latency will be FLAGGED, never used to reject")
                npu_ms = {}
            else:
                npu_ms = {r["model"]: r.get("ms") for r in _v.get("rows", [])
                          if r.get("ms") is not None}
                log(f"  NPU verdict: {len(npu_ms)} measured, ceiling {args.npu_max_ms:.0f} ms "
                    f"on {_v.get('device', '?')} @ {_v.get('size', '?')}px")
        except Exception as e:
            log(f"  WARNING could not read --npu-verdict ({type(e).__name__}) -- "
                f"gate 3 falls back to the MAC cap alone")
            npu_ms = None

    ref_hashes = {h for h in pool_hashes.values() if h is not None}
    ref_embeds = [e for e in pool_embeds.values() if e is not None]
    ref_sigs = dict(pool_sigs)
    ref_vecs = dict(pool_vecs)
    ref_grams = dict(grams) if grams else {}
    ref_out = dict(agree) if agree else {}
    promoted, report = [], []

    for c in cands:
        cs = c["checksum"]; reasons = []
        # quality FLOOR (always) — don't seed sub-competitive models
        if c["psnr50"] < args.bar:
            reasons.append(f"below-floor {c['psnr50']:.2f}<{args.bar}")
        # OPTIONAL efficiency prune: dominated (worse AND costlier) by an ALREADY-PROMOTED
        # model — greedy, so a distinct cost-corner is never vetoed by a non-promotable clone
        if args.pareto and any(
                p["psnr50"] >= c["psnr50"] and (p["macs"] or 0) <= (c["macs"] or 0)
                and (p["psnr50"] > c["psnr50"] or (p["macs"] or 0) < (c["macs"] or 0))
                for p in promoted):
            reasons.append("pareto-dominated")
        if not c.get("deployable"):
            reasons.append("undeployable")
        macs_m = (c["macs"] or 0) / 1e6
        if args.mac_cap and macs_m > args.mac_cap:
            (reasons.append(f"macs {macs_m:.0f}>{args.mac_cap:.0f}")
             if args.strict_mac else reasons.append(f"OVER-CAP({macs_m:.0f})?"))

        # ---- MEASURED on-device latency: gate 3, using evidence instead of a proxy ----
        # The MAC cap above is the ONLY deployability test this gate has ever had, and it is blind:
        # of 128 models profiled on a Galaxy S26 Ultra, 33 (26%) exceed 10 ms at 256x256 against a
        # 1.72 ms median, and every one of them passes 9053 MMac. A multiply-accumulate count cannot
        # see whether an operator has an NPU kernel. Promoting such a model makes it a PARENT, so the
        # blindness compounds: its descendants inherit the operator that lands on the CPU.
        #
        # The verdict is READ FROM A FILE, not measured here. Profiling needs the AI Hub token and
        # network, which a pod does not have, and promote.py must stay runnable anywhere. Produce it
        # first with:  npu_gate_stage.sh <sources> <work>   then pass --npu-verdict <work>/verdict.json
        #
        # UNMEASURED IS SOFT, TOO-SLOW IS HARD. A candidate absent from the verdict has not been
        # tested, and "untested" must not read as "rejected" -- the device farm is a shared FIFO queue
        # that ran an hour deep tonight, and a congested queue silently vetoing a whole generation is
        # exactly the failure npu_gate.py's coverage check exists to prevent. Same reasoning here.
        if npu_ms is not None:
            ms = npu_ms.get(cs) if cs in npu_ms else npu_ms.get(c["name"])
            if ms is None:
                reasons.append("NPU-UNMEASURED?")
            elif ms > args.npu_max_ms:
                reasons.append(f"npu-slow {ms:.1f}ms>{args.npu_max_ms:.0f}ms")
        if c["name"] in args.pool:
            reasons.append("already-in-pool")           # architectural identity, not behaviour
        if not _xover_ok(c["source"]):
            reasons.append("no-xover-block")

        # DISTINCTNESS (the real gate): WL graph hash vs pool + promoted-so-far. Equal
        # hash = same computation graph = clone. Untraceable models fall back to the
        # module-signature cosine.
        ch = cand_hashes[cs]
        max_sig = max((_sig_cos(cand_sigs[cs], v) for v in ref_sigs.values()), default=0.0)
        if ch is not None:
            if ch in ref_hashes:
                reasons.append(f"graph-clone({ch[:8]})")
        elif cand_sigs[cs] is None:
            reasons.append("unloadable")
        elif max_sig > args.sig_max:
            reasons.append(f"struct-dup(sig {max_sig:.2f},untraced)")
        # GRADED near-duplicate (opt-in): continuous-WL cosine vs pool + promoted
        if args.graded and cand_embeds.get(cs) is not None and ref_embeds:
            import numpy as np
            max_cos = max(float(np.dot(cand_embeds[cs], e)) for e in ref_embeds)
            if max_cos >= args.graded_max:
                reasons.append(f"near-dup(cos {max_cos:.2f})")

        # AST cosine + optional CKA — INFORMATIONAL columns only (never gate)
        max_cos = max((_cos(cand_vecs[cs], v) for v in ref_vecs.values()), default=0.0)
        max_cka = max_agree = None
        if args.functional and grams and cs in ref_grams:
            from functional_similarity import linear_cka, out_psnr
            max_cka = max((linear_cka(ref_grams[cs], g) for k, g in ref_grams.items()
                           if k != cs), default=0.0)
            max_agree = max((out_psnr(ref_out[cs], o) for k, o in ref_out.items()
                             if k != cs and cs in ref_out), default=0.0)

        # decide PROMOTE on HARD reasons only. Soft (flag, not fail):
        #   OVER-CAP        — informational unless --strict-mac (handled above)
        #   no-xover-block  — soft unless --require-xover (mutation-only seeds are valid)
        def _is_soft(r):
            if r.startswith("OVER-CAP"):
                return True
            # Never measured != measured and bad. Trailing '?' marks flags throughout this file.
            if r.startswith("NPU-UNMEASURED"):
                return True
            if r.startswith("no-xover-block") and not args.require_xover:
                return True
            return False
        hard = [r for r in reasons if not _is_soft(r)]
        ok = not hard
        rec = dict(name=c["name"], checksum=cs, psnr50=c["psnr50"], macs_mmac=round(macs_m, 1),
                   deployable=c.get("deployable"), wl_hash=(ch[:12] if ch else None),
                   max_sig_cos=round(max_sig, 3), max_struct_cos=round(max_cos, 3),
                   max_residual_cka=(round(max_cka, 3) if max_cka is not None else None),
                   max_agree_db=(round(max_agree, 1) if max_agree is not None else None),
                   npu_ms=(npu_ms.get(cs) or npu_ms.get(c["name"])) if npu_ms else None,
                   verdict="PROMOTE" if ok else "reject", reasons=reasons)
        report.append(rec)
        if ok:
            promoted.append(c)
            if ch is not None:
                ref_hashes.add(ch)              # future candidates must differ from it too
            if args.graded and cand_embeds.get(cs) is not None:
                ref_embeds.append(cand_embeds[cs])
            ref_sigs[cs] = cand_sigs[cs]
            ref_vecs[cs] = cand_vecs[cs]
            if grams and cs in grams:
                ref_grams[cs] = grams[cs]; ref_out[cs] = agree[cs]

    # ---- report ---- (wl_hash = the distinctness identity; sigcos = fallback)
    log(f"\n{'verdict':9}{'name':32}{'psnr50':>8}{'MMac':>8}{'ms':>8}{'dep':>5}"
        f"{'wl_hash':>14}  reasons")
    for r in report:
        _ms = r.get("npu_ms")
        _mss = f"{_ms:.1f}" if _ms is not None else "-"
        log(f"{r['verdict']:9}{(r['name'] or '')[:31]:32}{r['psnr50']:8.2f}"
            f"{r['macs_mmac']:8.0f}{_mss:>8}"
            f"{('Y' if r['deployable'] else 'n'):>5}"
            f"{(r['wl_hash'] or 'untraced'):>14}"
            f"  {', '.join(r['reasons']) or '-'}")

    next_pool = list(pool_srcs) + [p["name"] for p in promoted]
    log(f"\nPROMOTED {len(promoted)}/{len(cands)}: {[p['name'] for p in promoted]}")
    log(f"NEXT-RUN pool ({len(next_pool)}): {len(pool_srcs)} seeds + {len(promoted)} promoted")

    if args.emit_dir and promoted:
        os.makedirs(args.emit_dir, exist_ok=True)
        for p in promoted:
            fp = os.path.join(args.emit_dir, p["checksum"] + ".py")
            with open(fp, "w") as f:
                f.write(f"# promoted name={p['name']} psnr50={p['psnr50']:.3f} "
                        f"macs={p['macs']}\n" + p["source"])
        log(f"emitted {len(promoted)} promoted sources -> {args.emit_dir}/ "
            f"(feed via: run_seed_ga.py --seed-files {args.emit_dir}/*.py)")

    json.dump({"bar": args.bar, "pareto": args.pareto, "sig_max": args.sig_max,
               "promoted": [p["name"] for p in promoted], "report": report},
              open(args.out, "w"), indent=2)
    log(f"report -> {args.out}")


if __name__ == "__main__":
    main()
