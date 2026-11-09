"""Profile every already-converted TFLite artifact on the device farm.

WHY THIS IS THE UNBLOCKER. 2,511 artifacts were converted at 256x256 for the earlier sweep and only
128 were ever profiled. The other 2,383 need no GPU, no conversion and no new code -- just farm
submissions -- and latency for the whole corpus is what three separate pieces of work are currently
blocked on:

  * GA fitness      the GA runs as ONE long-lived pod with an in-process fitness() closure, so it
                    cannot call the farm inline (no token, no network in-pod). A latency PREDICTOR
                    fit on the profiled corpus is the only way to put deployability inside its
                    selection loop -- and a predictor needs thousands of labels, not 128.
  * prompt ablation prompt style is already recorded in every candidate id (constraint_based,
                    direct_instruction, step_by_step, conversational) and in 34 Pg groups, so
                    "which prompt produces deployable models" is answerable RETROSPECTIVELY the
                    moment latency exists.
  * KTO reward      desirability is currently "valid AND novel". Making it the MAI Final Score
                    2^(2*PSNR)/(C*runtime) needs a runtime for each episode's model.

ORDERED LLM-FIRST ON PURPOSE. The LLM arm is the one with a known deployability problem (8 of 11
profiled LLM models run fully off the NPU) and it is the arm both the prompt ablation and the KTO
reward depend on, so its 470 models are submitted before the 2,000-odd GA models. If the farm rate
limits or the run is interrupted, the half that matters most is already done.

RESUMABLE AND THROTTLED. Job ids are written to an atomic state file after every submission, so a
crash costs nothing, and a rerun skips what is already known. Submission errors are counted and the
run aborts if they dominate -- hammering a shared farm through a quota error would be rude and
pointless.

  python bulk_profile.py --submit --limit 500
  python bulk_profile.py --collect
  python bulk_profile.py --report
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import naming  # noqa: E402

SWEEP = "/shared/ssd/home/b-b-ahmed/thesis/sweep_out"
POOL = os.path.join(SWEEP, "tflite_pool", "tflite")
STATE = os.path.join(SWEEP, "aihub", "bulk_jobs.json")
OUT = os.path.join(SWEEP, "aihub", "bulk_results.json")
# TWO DEVICES = TWO PARALLEL QUEUES. Measured drain on a single device was 0.72 jobs/min, which
# would put the full 2,383-artifact census at 13-55 h on one queue. Models are ROUND-ROBINED across
# devices rather than duplicated: each model is measured once, on one flagship, and the batch drains
# at roughly twice the rate.
#
# THE DEVICE IS RECORDED AND MUST BE TREATED AS A COVARIATE, NOT NORMALISED AWAY. Latency RANK
# agreement between these two generations is rho=0.9919 over 82 NPU-resident models, so either
# device orders architectures the same way -- but the ABSOLUTE offset is a median 1.63x with a
# per-model range of 1.23-2.41x. Rescaling S24 numbers by 1.63 would inject up to ~30% per-model
# error, so anything fit on this data uses `device` as a feature (or fits per-device) instead.
DEVICES = ["Samsung Galaxy S26 Ultra", "Samsung Galaxy S24 (Family)"]
DEVICE = DEVICES[0]                      # reporting default / back-compat
SUFFIX = "_256x256_fp32.tflite"
C_2021 = 2 ** 64


def load(p):
    return json.load(open(p)) if os.path.exists(p) else {}


def save(p, d):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    json.dump(d, open(tmp, "w"), indent=1)
    os.replace(tmp, p)


def already_profiled():
    done = set()
    for f in ("results_s26u.json", "results.json"):
        p = os.path.join(SWEEP, "aihub", f)
        if os.path.exists(p):
            done |= {m["model"] for m in json.load(open(p))["models"] if m.get("inference_us")}
    return done


def pending_models():
    arts = {re.sub(re.escape(SUFFIX) + "$", "", os.path.basename(p)): p
            for p in glob.glob(os.path.join(POOL, "*.tflite"))}
    skip = already_profiled() | set(load(STATE))
    todo = {k: v for k, v in arts.items() if k not in skip}
    # LLM first, then hand-written seeds, then the GA mass
    def rank(n):
        return (0 if n.startswith("LLM_") else
                1 if not (n.startswith("DenoiseGA-") or n.startswith("DnSeedGA-")) else 2)
    return sorted(todo.items(), key=lambda kv: (rank(kv[0]), kv[0])), len(arts), len(skip)


def stratify(todo, n):
    """Proportional-by-family random sample, deterministic under a name hash.

    WHY RANDOM WITHIN FAMILY AND NOT PSNR-STRATIFIED. The point of this sample is to train a LATENCY
    predictor, and PSNR is not the axis latency varies along -- stratifying on it would be a
    decorative choice that biases nothing useful. Proportional family sampling keeps the
    architectural mix of the corpus, and a name hash makes the draw reproducible without storing a
    seed. The census was never the goal: a predictor fit on a representative sample generalises as
    well as one fit on all 2,383 and finishes tonight instead of tomorrow afternoon.
    """
    import hashlib
    byfam = {}
    for name, path in todo:
        f = ("LLM_" if name.startswith("LLM_") else
             "DenoiseGA-" if name.startswith("DenoiseGA-") else
             "DnSeedGA-" if name.startswith("DnSeedGA-") else "hand")
        byfam.setdefault(f, []).append((name, path))
    tot = sum(len(v) for v in byfam.values()) or 1
    out = []
    for f, v in sorted(byfam.items()):
        v.sort(key=lambda kv: hashlib.sha1(kv[0].encode()).hexdigest())
        k = max(1, round(n * len(v) / tot))
        out.extend(v[:k])
        print(f"    {f:<12} {len(v):>5} available -> sampling {min(k, len(v)):>4}")
    return out[:n]


def cmd_submit(a):
    import qai_hub as hub
    todo, n_art, n_skip = pending_models()
    if a.family:
        before = len(todo)
        todo = [t for t in todo if t[0].startswith(a.family)]
        print(f"family filter {a.family!r}: {before} -> {len(todo)} pending")
        if not todo:
            sys.exit(f"nothing pending for family {a.family!r}")
    if a.stratify:
        print(f"stratified sample of {a.stratify} from {len(todo)} unsubmitted:")
        todo = stratify(todo, a.stratify)
    elif a.limit:
        todo = todo[:a.limit]
    print(f"{n_art} artifacts | {n_skip} already known | submitting {len(todo)} "
          f"round-robin across {len(DEVICES)} devices")
    st = load(STATE)
    devs = [hub.Device(d) for d in DEVICES]
    errs = 0
    for i, (name, path) in enumerate(todo, 1):
        dname = DEVICES[(i - 1) % len(DEVICES)]
        dev = devs[(i - 1) % len(devs)]
        try:
            job = hub.submit_profile_job(model=path, device=dev, name=f"bulk {name}")
            st[name] = {"job_id": job.job_id, "device_req": dname}
            errs = 0
            if i % 10 == 0 or i == len(todo):
                save(STATE, st)
                print(f"  {i}/{len(todo)} submitted", flush=True)
        except Exception as e:
            errs += 1
            print(f"  submit failed {name[:40]}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            # Consecutive failures almost always mean a quota or auth wall, not bad models.
            # Stop rather than grinding through two thousand identical errors.
            if errs >= 5:
                save(STATE, st)
                sys.exit(f"ABORT: {errs} consecutive submission failures -- likely a farm quota "
                         f"or token problem. {len(st)} jobs recorded; rerun to resume.")
            time.sleep(2)
    save(STATE, st)
    print(f"{len(st)} jobs in flight -> {STATE}")


def _fetch_one(hub, name, job_id):
    """Status-check and (if finished) download one job. Returns (name, record_or_None, state)."""
    try:
        job = hub.get_job(job_id)
        s = job.get_status()
        if not s.finished:
            return name, None, "pending"
        if not s.success:
            return name, {"status": "FAILED"}, "failed"
        prof = job.download_profile()
        ex = prof.get("execution_summary", {}) if isinstance(prof, dict) else {}
        us = ex.get("estimated_inference_time") or ex.get("inference_time")
        det = prof.get("execution_detail") or []
        unit = {}
        for lay in det:
            if isinstance(lay, dict):
                cu = lay.get("compute_unit") or "UNKNOWN"
                unit[cu] = unit.get(cu, 0) + (lay.get("execution_time") or 0)
        tot = sum(unit.values()) or 1
        return name, {"status": "OK" if us else "NO_LATENCY",
                      "ms": round(us / 1000.0, 3) if us else None,
                      "npu_time_frac": round(unit.get("NPU", 0) / tot, 4),
                      "n_layers": len(det), "device": job.device.name,
                      "fallback_units": sorted(u for u in unit if u != "NPU")}, "ok"
    except Exception as e:
        return name, None, f"error {type(e).__name__}"


def cmd_collect(a):
    """Parallel status polling.

    WHY THREADS. Each get_job+get_status is a ~469 ms HTTP round trip, so a serial pass over 789
    pending jobs took 6.2 MINUTES -- and the loop then slept another 120 s on top. The observed
    0.6 jobs/min was therefore a property of THIS COLLECTOR, not of the device farm: most wall
    time went on polling statuses rather than waiting for phones. These calls are pure I/O, so a
    modest thread pool cuts a pass to seconds. The pool is deliberately small (16): the farm is
    shared infrastructure and there is nothing to gain from hammering its API harder than needed.
    """
    import qai_hub as hub
    from concurrent.futures import ThreadPoolExecutor
    st, out = load(STATE), load(OUT)
    pend = fail = 0
    todo = [k for k in st if k not in out]
    print(f"{len(st)} jobs | {len(out)} collected | {len(todo)} to check")
    import time as _t
    t0 = _t.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex_:
        futs = [ex_.submit(_fetch_one, hub, n, st[n]["job_id"]) for n in todo]
        for i, f in enumerate(futs, 1):
            name, rec, state = f.result()
            if state == "pending":
                pend += 1
            elif rec is not None:
                out[name] = rec
                if state == "failed":
                    fail += 1
            else:
                fail += 1
                if fail <= 3:
                    print(f"  {name[:40]}: {state}")
            if i % 200 == 0:
                save(OUT, out)
    print(f"  pass took {_t.time()-t0:.0f}s with {a.workers} workers")
    save(OUT, out)
    ok = [v for v in out.values() if v.get("status") == "OK"]
    print(f"{len(ok)} OK | {pend} still running | {fail} failed -> {OUT}")
    sys.exit(2 if pend else 0)


def _row(f, vs):
    """One family line: p50/p90/over-10ms/off-NPU plus NPU-safe (<=10 ms AND fully on NPU),
    which is the count that actually answers 'how many of these would ship'."""
    ms = sorted(v["ms"] for v in vs if v.get("ms"))
    if not ms:
        return
    slow = sum(1 for m in ms if m > 10)
    off = sum(1 for v in vs if (v.get("npu_time_frac") or 0) <= 0.001)
    safe = sum(1 for v in vs
               if v.get("ms") and v["ms"] <= 10
               and (v.get("npu_time_frac") is None or v["npu_time_frac"] >= 0.99))
    print(f"{f:<14}{len(vs):>6}{ms[len(ms)//2]:>9.2f}{ms[int(.9*len(ms))]:>9.2f}"
          f"{slow/len(ms):>7.0%}{off/len(vs):>9.0%}{safe/len(vs):>10.0%}")


def cmd_report(a):
    import statistics as stat
    # ADDITIVE, NOT A REBUILD. latency_corpus.json is a union: bulk_results.json and the aihub
    # results files reach only 2,501 of its 3,841 rows, the rest having been folded in by
    # fold_profiles.py. The previous version dumped a freshly built dict over the file, which
    # silently deleted 1,340 rows -- every DnSeed2*/DnSeedGA2 measurement, 697 of them
    # DnSeed2Fast. Start from what is already on disk and only ever add.
    corpus_path = os.path.join(SWEEP, "aihub", "latency_corpus.json")
    merged = load(corpus_path)
    base_n = len(merged)
    out = load(OUT)
    for k, v in out.items():
        if v.get("status") == "OK":
            merged.setdefault(k, v)

    # EVERY results*.json, not one hardcoded name. results_llmgate_ab.json and
    # results_llm_lat.json were invisible to the old tuple, so 284 measurements sat unused.
    added_by_file = {}
    for p_ in sorted(glob.glob(os.path.join(SWEEP, "aihub", "results*.json"))):
        if ".bak" in p_:
            continue
        try:
            d = json.load(open(p_))
        except Exception as e:
            print(f"  WARNING skipping {os.path.basename(p_)}: {e}")
            continue
        if not (isinstance(d, dict) and isinstance(d.get("models"), list)):
            continue
        rows = [m for m in d["models"] if m.get("inference_us")]
        # THE DEVICE COMES FROM THE ROW. Tagging with the module constant is what produced the
        # device-pooling defect: it stamped S24 measurements as S26 Ultra. Per-row first, then
        # the file header, and only then the constant.
        devs = {m.get("device") for m in rows if m.get("device")}
        unknown = devs - set(DEVICES)
        if unknown and not a.allow_new_device:
            print(f"  HELD BACK {os.path.basename(p_)}: {len(rows)} rows on {sorted(unknown)}, "
                  f"a device not in the 2-device census. Pass --allow-new-device to admit it.")
            continue
        n0 = len(merged)
        for m in rows:
            merged.setdefault(m["model"], {
                "ms": m["inference_us"] / 1000.0,
                "npu_time_frac": m.get("npu_time_frac"),
                "device": m.get("device") or d.get("device") or DEVICE})
        if len(merged) > n0:
            added_by_file[os.path.basename(p_)] = len(merged) - n0
    if not merged:
        sys.exit("nothing collected yet")
    import collections
    bydev = collections.Counter(str(v.get("device") or "unrecorded") for v in merged.values())
    print(f"{len(merged)} models with measured latency, BY DEVICE:")
    for d, n in bydev.most_common():
        print(f"    {n:>5}  {d}")
    print("\n  Latencies are NOT comparable across devices and are never normalised: rank")
    print("  agreement is rho=0.9919 but the per-model offset is 1.23-2.41x. Cross-family claims")
    print("  must use a single-device table. The split is uneven by family, so pooling would")
    print("  penalise whichever arm drew more of the slower phone.\n")

    def table(rows, label):
        if not rows:
            return
        fam = collections.defaultdict(list)
        for n, v in rows:
            f = ("LLM_" if n.startswith("LLM_") else "DenoiseGA-" if n.startswith("DenoiseGA-")
                 else "DnSeedGA-" if n.startswith("DnSeedGA-")
                 else "DenoiseRS-" if n.startswith("DenoiseRS-")
                 else "DenoiseRW-" if n.startswith("DenoiseRW-") else "hand-written")
            fam[f].append(v)
        print(f"--- {label} ---")
        print(f"{'family':<14}{'n':>6}{'p50 ms':>9}{'p90 ms':>9}{'>10ms':>8}{'off-NPU':>9}"
              f"{'NPU-safe':>10}")
        for f, vs in sorted(fam.items(), key=lambda kv: -len(kv[1])):
            _row(f, vs)
        print()

    for dev in [d for d, _ in bydev.most_common()]:
        table([(n, v) for n, v in merged.items()
               if str(v.get("device") or "unrecorded") == dev], dev)

    fam = collections.defaultdict(list)
    for n, v in merged.items():
        # matches_family accepts 'LLM-' and the legacy 'LLM_' alike. 332 of the keys
        # already in latency_corpus.json carry the pre-rename spelling, and every
        # newly generated model carries the new one -- classifying on a single
        # spelling silently dumps one vintage into "hand-written". See naming.py.
        f = ("LLM" if naming.matches_family(n, "LLM-")
             else "DenoiseGA-" if n.startswith("DenoiseGA-")
             else "DnSeedGA-" if n.startswith("DnSeedGA-") else "hand-written")
        fam[f].append(v)
    print("--- ALL DEVICES POOLED (do not quote cross-family; kept only for coverage) ---")
    print(f"{'family':<14}{'n':>6}{'p50 ms':>9}{'p90 ms':>9}{'>10ms':>8}{'off-NPU':>9}")
    for f, vs in sorted(fam.items(), key=lambda kv: -len(kv[1])):
        ms = sorted(v["ms"] for v in vs if v.get("ms"))
        slow = sum(1 for m in ms if m > 10)
        off = sum(1 for v in vs if (v.get("npu_time_frac") or 0) <= 0.001)
        print(f"{f:<14}{len(vs):>6}{ms[len(ms)//2]:>9.2f}{ms[int(.9*len(ms))]:>9.2f}"
              f"{slow/len(ms):>7.0%}{off/len(vs):>9.0%}")
    print(f"\ncorpus {base_n} -> {len(merged)} rows (+{len(merged) - base_n})")
    for f_, n_ in sorted(added_by_file.items(), key=lambda kv: -kv[1]):
        print(f"    +{n_:5d}  {f_}")
    if a.dry_run:
        print("\nDRY RUN -- latency_corpus.json not written")
        return
    if len(merged) < base_n:
        sys.exit(f"REFUSING TO WRITE: {base_n - len(merged)} rows would be lost")
    o = corpus_path
    if os.path.exists(o):
        bak = o + ".bak-" + time.strftime("%Y%m%d-%H%M%S")
        shutil.copy(o, bak)
        print(f"    backup {bak}")
    tmp = o + ".tmp"
    json.dump(merged, open(tmp, "w"), indent=1)
    os.replace(tmp, o)
    print(f"\nwrote {o} -- the latency column the GA predictor, prompt ablation and KTO reward need")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="--report: show the merge and write nothing")
    ap.add_argument("--allow-new-device", action="store_true",
                    help="--report: admit results files measured on a device outside DEVICES. "
                         "Off by default: the corpus is a 2-device census and silently adding a "
                         "third turns every pooled statistic into a 3-device mixture.")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16,
                    help="parallel status pollers; a serial pass over 789 jobs took 6.2 min")
    ap.add_argument("--family", default=None,
                    help="only submit models whose name starts with this prefix. USE THIS FIRST: "
                         "of 2438 unprofiled models, 376 are DnSeedGA and only 9 of that family "
                         "have ever been profiled -- yet it is the arm with the best AWGN PSNR "
                         "(36.98), the best zero-shot SIDD transfer (32.59 dB vs 27.31 for LLM) "
                         "and the lowest p50 latency (1.66 ms). Its deployability is the most "
                         "valuable unmeasured quantity we have, and profiling in file order would "
                         "spend the farm on 1995 DenoiseGA models first.")
    ap.add_argument("--stratify", type=int, default=0,
                    help="submit a proportional-by-family random sample of this size instead of the "
                         "next --limit in order")
    a = ap.parse_args()
    if a.submit:
        cmd_submit(a)
    elif a.collect:
        cmd_collect(a)
    elif a.report:
        cmd_report(a)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
