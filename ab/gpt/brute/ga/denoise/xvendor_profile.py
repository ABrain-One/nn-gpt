"""Paired cross-VENDOR profiling: does the NPU-residency mechanism hold off Qualcomm?

WHY. The corpus is a 2-device census and both devices are Samsung/Snapdragon, so the
paper's central mechanism -- latency is bimodal in whether the compiler keeps the whole
graph on the accelerator -- is established on ONE vendor's stack. That is the single most
likely rejection sentence. AI Hub also carries Google Tensor (Edge TPU) and Samsung Exynos
silicon, and 2,511 TFLite artifacts at 256x256 already exist, so the test needs NO
conversion, NO GPU and NO retraining -- only farm submissions.

PAIRED ON PURPOSE. Every model submitted here already carries an S26 Ultra measurement, so
each new device yields a matched pair and the three questions are answerable directly:
  1. does the fixed-MAC latency dispersion reproduce off Qualcomm?
  2. does the residency binary reproduce, i.e. is latency still bimodal in npu_time_frac?
  3. does the source-code eviction gate TRANSFER -- train on Qualcomm, test on Tensor?
A Qualcomm device of an older generation is included as a control, so "vendor" is not
confounded with "generation".

WRITES TO ITS OWN STATE. bulk_profile.py's STATE/OUT hold the 2-device census and its
--report refuses a third device by default for a good reason: silently pooling a third
device turns every pooled statistic into a mixture. This script never touches those files.

  python xvendor_profile.py --plan              # stratified sample, writes nothing
  python xvendor_profile.py --submit --dry-run
  python xvendor_profile.py --submit
  python xvendor_profile.py --collect
  python xvendor_profile.py --report
"""
import argparse
import collections
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from bulk_profile import load, save, POOL, SUFFIX, _fetch_one   # noqa: E402

SWEEP = "/shared/ssd/home/b-b-ahmed/thesis/sweep_out"
OUTDIR = os.path.join(SWEEP, "xvendor")
STATE = os.path.join(OUTDIR, "jobs.json")
OUT = os.path.join(OUTDIR, "results.json")
REF = os.path.join(SWEEP, "aihub", "latency_corpus.json")
REF_DEVICE = "Samsung Galaxy S26 Ultra"

# (device name as AI Hub spells it, chipset, vendor, role)
TARGETS = [
    ("Google Pixel 9 Pro",      "google-tensor-g4",         "google",   "second vendor"),
    ("Samsung Galaxy A53 5G",   "samsung-exynos-1280",      "samsung",  "third vendor"),
    ("Samsung Galaxy S22 Ultra 5G", "qualcomm-snapdragon-8gen1", "qualcomm", "generation control"),
]
PER_DEVICE = 200


def family(n):
    for p in ("LLM", "DnSeedGA2", "DnSeedGA", "DnSeed2Fast", "DnSeed2Act", "DnSeed2Wide",
              "DnSeed3", "DenoiseGA", "DenoiseSkip", "DenoiseRW", "DnSearch", "DnSkip"):
        if n.startswith(p):
            return p
    return "hand"


def residency_class(frac):
    """The label the paper's gate predicts: fully resident vs partially/fully evicted."""
    if frac is None:
        return "unknown"
    if frac >= 0.999:
        return "resident"
    if frac <= 0.001:
        return "fully_evicted"
    return "partial"


def candidates():
    """Models with BOTH a 256x256 TFLite artifact and an S26 Ultra reference measurement."""
    arts = {re.sub(re.escape(SUFFIX) + "$", "", os.path.basename(p)): p
            for p in glob.glob(os.path.join(POOL, "*.tflite"))}
    ref = json.load(open(REF))
    out = {}
    for name, path in arts.items():
        r = ref.get(name)
        if not isinstance(r, dict) or r.get("status") != "OK" or not r.get("ms"):
            continue
        if r.get("device") != REF_DEVICE:
            continue
        out[name] = {"path": path, "ref_ms": r["ms"],
                     "ref_npu_frac": r.get("npu_time_frac"),
                     "ref_class": residency_class(r.get("npu_time_frac")),
                     "family": family(name)}
    return out


def plan(cands, n):
    """Stratify by (family x reference residency class), deterministic under a name hash.

    Stratifying on residency as well as family is deliberate: the gate-transfer test needs
    BOTH evicted and resident models on the new device, and eviction is rare in the genetic
    arms (~2%) and common in the LLM arm (~37%). A family-only sample would return almost
    no positives and could not estimate an AUC.
    """
    cells = collections.defaultdict(list)
    for name, c in cands.items():
        cells[(c["family"], c["ref_class"])].append(name)
    for k in cells:
        cells[k].sort(key=lambda s: hashlib.sha1(s.encode()).hexdigest())
    # proportional, but guarantee every non-empty cell at least one member
    tot = sum(len(v) for v in cells.values())
    quota = {k: max(1, round(n * len(v) / tot)) for k, v in cells.items()}
    pick = []
    for k, v in cells.items():
        pick += v[:min(quota[k], len(v))]
    pick.sort(key=lambda s: hashlib.sha1(s.encode()).hexdigest())
    return pick[:n], cells


def cmd_plan(a):
    cands = candidates()
    pick, cells = plan(cands, PER_DEVICE)
    print(f"candidates with TFLite + {REF_DEVICE} reference : {len(cands)}")
    print(f"sample per device                              : {len(pick)}")
    print(f"devices                                        : {len(TARGETS)}")
    print(f"total jobs                                     : {len(pick)*len(TARGETS)}\n")
    print("reference residency mix of the whole candidate pool:")
    for k, v in sorted(collections.Counter(c["ref_class"] for c in cands.values()).items()):
        print(f"   {k:16s} {v:5d}")
    print("\nsample, by family x reference residency class:")
    sm = collections.Counter((cands[n]["family"], cands[n]["ref_class"]) for n in pick)
    for (f, cl), v in sorted(sm.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"   {f:12s} {cl:16s} {v:4d}")
    print("\nsample residency mix (what the transfer test will have to work with):")
    for k, v in sorted(collections.Counter(cands[n]["ref_class"] for n in pick).items()):
        print(f"   {k:16s} {v:5d}")
    ms = sorted(cands[n]["ref_ms"] for n in pick)
    print(f"\nreference latency span of sample: {ms[0]:.3f} .. {ms[-1]:.3f} ms "
          f"(median {ms[len(ms)//2]:.3f})")
    for d, chip, vend, role in TARGETS:
        print(f"   -> {d:30s} {chip:26s} [{role}]")
    return pick, cands


def cmd_submit(a):
    import qai_hub as hub
    pick, cands = cmd_plan(a)
    if a.dry_run:
        print("\nDRY RUN: nothing submitted.")
        return
    avail = {d.name for d in hub.get_devices()}
    missing = [d for d, _, _, _ in TARGETS if d not in avail]
    if missing:
        return f"device(s) not on the farm: {missing}"
    st = load(STATE)
    nsub = nerr = 0
    for dname, chip, vend, role in TARGETS:
        dev = hub.Device(dname)
        for name in pick:
            key = f"{name}@@{dname}"
            if key in st:
                continue
            try:
                job = hub.submit_profile_job(model=cands[name]["path"], device=dev,
                                             name=f"xvendor {name} {chip}")
                st[key] = {"job_id": job.job_id, "model": name, "device_req": dname,
                           "chipset": chip, "vendor": vend, "role": role}
                nsub += 1
                if nsub % 10 == 0:
                    save(STATE, st)
                    print(f"  submitted {nsub} (last: {dname} / {name[:40]})", flush=True)
            except Exception as e:
                nerr += 1
                print(f"  ERROR {type(e).__name__} on {dname}/{name[:36]}: {e}", flush=True)
                save(STATE, st)
                if nerr > 15 and nerr > nsub:
                    save(STATE, st)
                    return "aborting: submission errors dominate (quota?)"
    save(STATE, st)
    print(f"\n{nsub} new jobs submitted, {nerr} errors; {len(st)} total in flight -> {STATE}")


def cmd_collect(a):
    import qai_hub as hub
    st, out = load(STATE), load(OUT)
    todo = [(k, v["job_id"]) for k, v in st.items() if k not in out]
    print(f"{len(st)} jobs, {len(out)} already collected, {len(todo)} to poll")
    ok = pend = fail = 0
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_fetch_one, hub, k, j): k for k, j in todo}
        for i, f in enumerate(cf.as_completed(futs), 1):
            k, rec, state = f.result()
            if rec is not None:
                meta = st[k]
                rec.update({"model": meta["model"], "device_req": meta["device_req"],
                            "chipset": meta["chipset"], "vendor": meta["vendor"],
                            "role": meta["role"]})
                out[k] = rec
                ok += state == "ok"
                fail += state == "failed"
            else:
                pend += 1
            if i % 50 == 0:
                save(OUT, out)
                print(f"  polled {i}/{len(todo)}  ok={ok} pending={pend} failed={fail}", flush=True)
    save(OUT, out)
    print(f"{ok} OK | {pend} still running | {fail} failed -> {OUT}")


def cmd_report(a):
    out = load(OUT)
    if not out:
        return "no results yet"
    rows = [r for r in out.values() if r.get("status") == "OK" and r.get("ms")]
    print(f"{len(out)} collected, {len(rows)} usable\n")
    bydev = collections.defaultdict(list)
    for r in rows:
        bydev[(r["vendor"], r["chipset"], r.get("device") or r["device_req"])].append(r)
    print(f"{'vendor':9s} {'chipset':26s} {'n':>5} {'resident':>9} {'partial':>8} "
          f"{'evicted':>8} {'med ms':>9}")
    for k in sorted(bydev):
        v = bydev[k]
        cl = collections.Counter(residency_class(r.get("npu_time_frac")) for r in v)
        ms = sorted(r["ms"] for r in v)
        print(f"{k[0]:9s} {k[1]:26s} {len(v):5d} {cl['resident']:9d} {cl['partial']:8d} "
              f"{cl['fully_evicted']:8d} {ms[len(ms)//2]:9.3f}")
    save(os.path.join(OUTDIR, "report.json"),
         {"n_collected": len(out), "n_usable": len(rows),
          "by_device": {f"{k[0]}|{k[1]}": len(v) for k, v in bydev.items()}})
    print(f"\nwrote {os.path.join(OUTDIR, 'report.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    if a.submit:
        return cmd_submit(a)
    if a.collect:
        return cmd_collect(a)
    if a.report:
        return cmd_report(a)
    cmd_plan(a)


if __name__ == "__main__":
    raise SystemExit(main())
