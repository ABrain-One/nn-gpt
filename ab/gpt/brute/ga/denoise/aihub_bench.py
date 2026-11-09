"""Profile the denoise TFLite artifacts on REAL Snapdragon hardware via Qualcomm AI Hub.

WHY AI HUB AND NOT "AI BENCHMARK". Ignatov's AI Benchmark is an Android APP that runs a FIXED
built-in model suite -- it cannot profile 124 arbitrary custom .tflite files, and it needs a
phone physically attached to the machine running it. Neither is available from this cluster.
AI Hub submits a model to a real device in Qualcomm's device farm and returns per-layer
placement plus wall-clock inference time, which is the citable number a paper needs.

HARDWARE CAVEAT THAT MUST TRAVEL WITH THE NUMBERS. AI Hub's farm is Snapdragon only. The
project's reference handset is a Dimensity 7300 (MediaTek), so these latencies are NOT that
phone's latencies. For a paper Snapdragon is arguably the better choice -- it is the platform
most efficient-denoising work reports on, including the MAI challenge lineage -- but the
device must be named explicitly in any table, never called "mobile latency" unqualified.

WHAT IS WORTH PROFILING. Only models that ALSO have a 50-epoch PSNR: latency alone is not a
result, the result is the PSNR-vs-latency Pareto front. Currently 124 models satisfy both
(2,387 further .tflite files have 5-epoch numbers only, and ep5 ranks 50-epoch performance at
just rho=0.684 -- pairing those with latency would draw a Pareto front out of noise).

SETUP (one time, requires a human -- an API token cannot be minted from here):
    1. sign in at https://app.aihub.qualcomm.com/  ->  Settings -> API token
    2. pip install qai-hub
    3. qai-hub configure --api_token <TOKEN>

    python aihub_bench.py --list-devices          # see what the farm offers
    python aihub_bench.py --submit                # submit profile jobs (resumable)
    python aihub_bench.py --collect               # poll + join with PSNR -> results

Submission and collection are SEPARATE because AI Hub jobs are asynchronous and queue behind
other tenants. Job ids are checkpointed to --state after every submission, so an interrupted
run resumes instead of resubmitting and burning quota twice.
"""
import argparse
import glob
import json
import os
import re
import sys

POOL = "/shared/ssd/home/b-b-ahmed/thesis/sweep_out/tflite_pool/tflite"
SWEEP = "/shared/ssd/home/b-b-ahmed/thesis/sweep_out"
SUFFIX = "_256x256_fp32.tflite"


def psnr_table():
    """model -> best 50-epoch dB, from ft50 and beatbar. 5-epoch numbers are NOT included."""
    out = {}
    for pat in ("fulltrain50/result_shard*.jsonl", "beatbar/result_shard*.jsonl"):
        for f in glob.glob(os.path.join(SWEEP, pat)):
            for line in open(f):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                name = r.get("key") or r.get("model")
                db = r.get("max_dB") or r.get("best_dB")
                if name and db and db > (out.get(name) or 0):
                    out[name] = db
    return out


def candidates(pool=POOL, suffix=SUFFIX, labels_path="", exclude_path=""):
    """-> [{model, path, psnr_dB, psnr_source, arm?, gen?}]

    LABELS MODE deliberately relaxes the 50-epoch requirement, and only it does. That
    requirement is right when the deliverable is a PSNR-vs-latency Pareto front, because ep5
    ranks ep50 at just rho=0.684 and a front drawn from ep5 is a front drawn from noise. The
    A/B residency pool asks a different question -- does the deployability gate change an
    arm's measured NPU placement -- and placement is a property of the graph, not of how long
    it was trained. The 5-epoch PSNR rides along as metadata and psnr_source records which
    kind it is, so the front below can refuse to draw itself from the wrong one.

    EXCLUDE exists because "a .tflite is on disk" is not "a .tflite runs". The A/B pool has 8
    flatbuffers that fail to invoke even with XNNPACK disabled; submitting them would spend
    farm quota to collect 8 guaranteed failures.
    """
    skip = set()
    if exclude_path:
        skip = {l.strip() for l in open(exclude_path) if l.strip()}
    if labels_path:
        lab = json.load(open(labels_path))
        rows = []
        for p in sorted(glob.glob(os.path.join(pool, "*" + suffix))):
            name = re.sub(re.escape(suffix) + "$", "", os.path.basename(p))
            rec = lab.get(name)
            if rec is None or name in skip:
                continue
            # TWO LABEL SHAPES. The A/B pool carries {arm, gen, psnr_ep5} per model because the
            # arm is the whole point there. A plain quality table -- crop_psnr_union.json and
            # anything else keyed model -> dB -- is a bare float, and demanding the dict form
            # made this crash after the TFLite stage had already run.
            if isinstance(rec, dict):
                rows.append({"model": name, "path": p, "psnr_dB": rec.get("psnr_ep5"),
                             "psnr_source": "ep5", "arm": rec.get("arm"),
                             "gen": rec.get("gen")})
            else:
                rows.append({"model": name, "path": p, "psnr_dB": float(rec),
                             "psnr_source": "ep5", "arm": None, "gen": None})
        return rows
    psnr = psnr_table()
    rows = []
    for p in sorted(glob.glob(os.path.join(pool, "*.tflite"))):
        name = re.sub(re.escape(suffix) + "$", "", os.path.basename(p))
        if name in psnr and name not in skip:
            rows.append({"model": name, "path": p, "psnr_dB": psnr[name],
                         "psnr_source": "ep50"})
    return rows


def load_state(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def save_state(path, st):
    tmp = path + ".tmp"
    json.dump(st, open(tmp, "w"), indent=1)
    os.replace(tmp, path)          # atomic: a crash mid-write must not lose every job id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="Samsung Galaxy S24 (Family)",
                    help="AI Hub device name; --list-devices to see valid values")
    ap.add_argument("--state", default=os.path.join(SWEEP, "aihub", "jobs.json"))
    ap.add_argument("--out", default=os.path.join(SWEEP, "aihub", "results.json"))
    ap.add_argument("--limit", type=int, default=0, help="0 = every eligible model")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be submitted; needs no API token")
    ap.add_argument("--pool", default=POOL)
    ap.add_argument("--suffix", default=SUFFIX)
    ap.add_argument("--labels", default="",
                    help="json {model: {arm, gen, psnr_ep5}}; selects LABELS MODE, which "
                         "profiles graphs for RESIDENCY and does not require a 50-epoch PSNR")
    ap.add_argument("--exclude", default="",
                    help="file of model names (one per line) never to submit")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.state), exist_ok=True)

    rows = candidates(a.pool, a.suffix, a.labels, a.exclude)
    if a.labels:
        arms = sorted({r.get("arm") or "?" for r in rows})
        print(f"{len(rows)} models eligible from {a.pool}")
        print(f"  LABELS MODE: residency measurement, PSNR source = ep5 (metadata only)")
        for arm in arms:
            print(f"    {arm:<16} {sum(1 for r in rows if r.get('arm') == arm)}")
    else:
        print(f"{len(rows)} models have BOTH a 50-epoch PSNR and a .tflite artifact")
    if a.limit:
        rows = sorted(rows, key=lambda r: -r["psnr_dB"])[:a.limit]
        print(f"  limited to the top {len(rows)} by PSNR")

    if a.dry_run:
        for r in rows[:10]:
            print(f"  {r['psnr_dB']:7.3f} dB  [{r.get('arm','-')}]  "
                  f"{os.path.basename(r['path'])}")
        print(f"  ... {len(rows)} total, {sum(os.path.getsize(r['path']) for r in rows)/1e6:.1f} MB")
        print(f"  device    {a.device}")
        print(f"  state     {a.state}")
        print(f"  out       {a.out}")
        return

    try:
        import qai_hub as hub
    except ImportError:
        sys.exit("qai-hub not installed:  pip install qai-hub")

    if a.list_devices:
        for d in hub.get_devices():
            print(f"  {d.name:<44} {d.os:<8} {' '.join(sorted(d.attributes))[:70]}")
        return

    st = load_state(a.state)

    if a.submit:
        dev = hub.Device(a.device)
        for i, r in enumerate(rows):
            if r["model"] in st:
                continue                       # resume: never resubmit, quota is finite
            try:
                job = hub.submit_profile_job(model=r["path"], device=dev,
                                             name=f"denoise-{r['model'][:40]}")
                st[r["model"]] = {"job_id": job.job_id, "psnr_dB": r["psnr_dB"],
                                  "psnr_source": r.get("psnr_source", "ep50"),
                                  "arm": r.get("arm"), "gen": r.get("gen")}
                save_state(a.state, st)
                print(f"  [{i+1}/{len(rows)}] submitted {r['model'][:44]} -> {job.job_id}", flush=True)
            except Exception as e:
                print(f"  [{i+1}/{len(rows)}] FAILED {r['model'][:44]}: {type(e).__name__}: {e}",
                      flush=True)
        print(f"submitted/known: {len(st)}")

    if a.collect:
        done, pending, failed = [], 0, 0
        for name, rec in st.items():
            try:
                job = hub.get_job(rec["job_id"])
                if not job.get_status().finished:
                    pending += 1
                    continue
                if not job.get_status().success:
                    failed += 1
                    continue
                prof = job.download_profile()
                ex = prof.get("execution_summary", prof) if isinstance(prof, dict) else {}
                # THE PROFILE SCHEMA IS NOT A STABLE CONTRACT. These key names were written
                # against the docs, not against a returned payload, and qai-hub has renamed
                # timing fields between releases. Probing a list of aliases and dumping the
                # raw payload once keeps a rename from silently producing None latencies --
                # which would look like "profiled successfully" and quietly poison the Pareto
                # front with zeros rather than failing.
                def pick(*aliases):
                    for k in aliases:
                        if isinstance(ex, dict) and ex.get(k) is not None:
                            return ex[k]
                    return None

                inf = pick("estimated_inference_time", "estimated_inference_time_us",
                           "inference_time", "execution_time")
                # scalar first: *_peak_range keys are [lo, hi] pairs, not a number
                mem = pick("estimated_inference_peak_memory", "inference_memory_peak_range",
                           "peak_memory_range")
                if isinstance(mem, (list, tuple)):
                    mem = mem[0] if mem else None

                # PER-LAYER COMPUTE UNIT -- the whole point of benchmarking a mid-range tier.
                # A model can sit inside the 9053 MMac budget and still be undeployable because
                # one operator has no NPU kernel: the layer silently falls back to CPU/GPU and
                # costs 10-50x. MMac cannot see that, and neither can total latency alone --
                # only the placement breakdown shows WHY a model is slow. Hexagon v73
                # (Snapdragon 7 Gen 4) is a full generation behind v75 (S24) and two behind v81
                # (S26 Ultra), so the same graph can place differently on each tier. The
                # fraction of RUNTIME off-NPU matters more than the fraction of layers: one
                # fallback convolution can dominate an otherwise fully-accelerated graph.
                det = prof.get("execution_detail") or []
                units, unit_us = {}, {}
                for lay in det:
                    if not isinstance(lay, dict):
                        continue
                    cu = lay.get("compute_unit") or "UNKNOWN"
                    units[cu] = units.get(cu, 0) + 1
                    unit_us[cu] = unit_us.get(cu, 0) + (lay.get("execution_time") or 0)
                tot_us = sum(unit_us.values()) or 1
                if inf is None and not os.path.exists(a.out + ".raw.json"):
                    json.dump(prof, open(a.out + ".raw.json", "w"), indent=1, default=str)
                    print(f"  !! no latency field matched; raw profile dumped to "
                          f"{a.out}.raw.json -- add the correct key to pick()")
                done.append({
                    "model": name,
                    "psnr_dB": rec["psnr_dB"],
                    "psnr_source": rec.get("psnr_source", "ep50"),
                    "arm": rec.get("arm"),
                    "gen": rec.get("gen"),
                    "inference_us": inf,
                    "peak_memory_bytes": mem,
                    "compile_us": pick("estimated_compile_time", "compile_time"),
                    # THE JOB'S REAL DEVICE, NOT a.device. Recording the ARGUMENT mislabels
                    # every row whenever --collect is run without repeating the --device that
                    # --submit used: three separate state files collected with the default
                    # argument all self-reported "Samsung Galaxy S24 (Family)" even though the
                    # farm had actually run them on S24 / S26 Ultra / Snapdragon 7 Gen 4 QRD.
                    # The latency numbers were fine; only the label was wrong -- which is worse,
                    # because a wrong device label is not detectable from the data afterwards.
                    "device": job.device.name,
                    "device_os": str(getattr(job.device, "os", "") or ""),
                    "n_layers": len(det),
                    "layers_by_unit": units,
                    "npu_layer_frac": round(units.get("NPU", 0) / max(1, len(det)), 4),
                    "npu_time_frac": round(unit_us.get("NPU", 0) / tot_us, 4),
                    "fallback_units": sorted(u for u in units if u not in ("NPU",)),
                })
            except Exception as e:
                print(f"  collect failed for {name[:40]}: {type(e).__name__}: {e}")
                failed += 1
        # A model whose latency never parsed must NOT enter the front: `or 1e18` would sort it
        # last and `pareto` would then never consider it, but it would still be counted as
        # "profiled". Separate them explicitly so the totals stay honest.
        unparsed = [r for r in done if r["inference_us"] is None]
        done = [r for r in done if r["inference_us"] is not None]
        if unparsed:
            print(f"  WARNING {len(unparsed)} jobs returned no parseable latency -- excluded")
        done.sort(key=lambda r: r["inference_us"])
        # File-level label comes from the rows, so it cannot disagree with them. Falls back to
        # the argument only when nothing was collected. A set is reported verbatim rather than
        # collapsed, so a state file that somehow mixes devices is visible instead of hidden.
        devs = sorted({r["device"] for r in done if r.get("device")})
        json.dump({"device": (devs[0] if len(devs) == 1 else devs or a.device),
                   "n": len(done), "models": done}, open(a.out, "w"), indent=1)
        print(f"\n{len(done)} profiled | {pending} pending | {failed} failed -> {a.out}")
        if done:
            print(f"\n{'ms':>8} {'dB':>7} {'NPU%':>6}  model")
            for r in done[:20]:
                ms = (r["inference_us"] or 0) / 1000.0
                print(f"{ms:8.2f} {r['psnr_dB']:7.3f} {100*r.get('npu_time_frac', 0):5.1f}%  "
                      f"{r['model'][:44]}")
            nfb = [r for r in done if r.get("npu_time_frac", 1) < 0.99]
            # a.device is the --submit default and is NOT what these rows were measured on;
            # printing it mislabelled every collect report. Use the same rows the header does.
            dlabel = devs[0] if len(devs) == 1 else (devs or a.device)
            print(f"\n{len(nfb)}/{len(done)} models spend runtime OFF the NPU on {dlabel}")
            for r in sorted(nfb, key=lambda r: r.get("npu_time_frac", 1))[:8]:
                print(f"  NPU {100*r.get('npu_time_frac',0):5.1f}% of runtime | "
                      f"fallback to {','.join(r.get('fallback_units') or []) or '-':<12} | "
                      f"{(r['inference_us'] or 0)/1000.0:7.2f} ms  {r['model'][:38]}")
            # ARM-WISE RESIDENCY, the deliverable in LABELS MODE. The A/B's own PSNR
            # comparison already came back null (ctrl 32.969 vs gate 32.964 over 115/79
            # survivors), so quality is not what is being tested here: the claim under test is
            # that the gate buys deployability, and until this table exists that claim is
            # unmeasured in either direction. Reported per arm with the off-NPU count, because
            # a mean npu_time_frac hides the shape -- a handful of fully-evicted models and a
            # broad partial fallback give the same mean and mean different things.
            if any(r.get("arm") for r in done):
                print(f"\nRESIDENCY BY ARM on {devs[0] if len(devs)==1 else devs}")
                for arm in sorted({r.get("arm") or "?" for r in done}):
                    g = [r for r in done if (r.get("arm") or "?") == arm]
                    fr = [r.get("npu_time_frac") for r in g
                          if r.get("npu_time_frac") is not None]
                    off = [x for x in fr if x < 0.99]
                    full = [x for x in fr if x < 0.01]
                    ms = sorted((r["inference_us"] or 0) / 1000.0 for r in g)
                    med = ms[len(ms) // 2] if ms else 0.0
                    print(f"  {arm:<16} n={len(g):3d}  "
                          f"mean npu_time_frac={sum(fr)/max(len(fr),1):.4f}  "
                          f"off-NPU={len(off):3d} ({100.0*len(off)/max(len(fr),1):5.1f}%)  "
                          f"fully-off={len(full):3d}  median={med:7.2f} ms")

            # The Pareto front is the actual deliverable WHEN the PSNR is a 50-epoch number.
            # In LABELS MODE it is not, and drawing one from ep5 is exactly the error the
            # candidates() docstring exists to prevent -- rho=0.684 against ep50. Refuse
            # rather than print something that will be screenshotted.
            srcs = {r.get("psnr_source", "ep50") for r in done}
            if srcs != {"ep50"}:
                print(f"\nPARETO FRONT SUPPRESSED: psnr_source={sorted(srcs)}. A front needs "
                      f"50-epoch PSNR (ep5 ranks ep50 at only rho=0.684).")
                return
            front, best = [], -1.0
            for r in done:                      # already sorted fastest-first
                if r["psnr_dB"] > best:
                    best = r["psnr_dB"]
                    front.append(r)
            print(f"\nPARETO FRONT ({len(front)} models, none dominated on both axes):")
            for r in front:
                print(f"  {(r['inference_us'] or 0)/1000.0:8.2f} ms  {r['psnr_dB']:7.3f} dB  {r['model'][:44]}")


if __name__ == "__main__":
    main()
