"""Profile the re-exported 720x480 and 1088x1920 artifacts on the AI Hub device farm.

Runs on the LOGIN NODE, not in a pod: the qai-hub API token lives in the user's config and the
cluster pods have no network credentials. Submission and collection are separate passes over an
atomic state file so a crash mid-run cannot lose job ids.

DEVICE IS READ BACK FROM EACH JOB, never taken from the argument. aihub_bench.py originally wrote
the --device flag into its results, and three collections run with the default flag all
self-reported "Galaxy S24" while the farm had actually run them on three different chips. The
latencies were fine; only the label was wrong, which is worse, because a wrong device label is
undetectable from the data afterwards.

  python reexport_submit.py --submit
  python reexport_submit.py --collect
"""
import argparse
import glob
import json
import os
import re

RX = "/shared/ssd/home/b-b-ahmed/thesis/sweep_out/reexport"
DEVICES = ["Samsung Galaxy S26 Ultra", "Samsung Galaxy S24 (Family)"]
RES = ["480x720", "1088x1920"]


def load(p):
    return json.load(open(p)) if os.path.exists(p) else {}


def save(p, st):
    tmp = p + ".tmp"
    json.dump(st, open(tmp, "w"), indent=1)
    os.replace(tmp, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--state", default=os.path.join(RX, "jobs_reexport.json"))
    ap.add_argument("--out", default=os.path.join(RX, "latency.json"))
    a = ap.parse_args()
    import qai_hub as hub

    st = load(a.state)

    if a.submit:
        n = 0
        for res in RES:
            d = os.path.join(RX, f"tflite_{res}")
            for p in sorted(glob.glob(os.path.join(d, "*.tflite"))):
                name = re.sub(rf"_{res}_fp32\.tflite$", "", os.path.basename(p))
                for dev in DEVICES:
                    key = f"{name}|{res}|{dev}"
                    if key in st:
                        continue
                    try:
                        job = hub.submit_profile_job(model=p, device=hub.Device(dev),
                                                     name=f"{name} {res} {dev}")
                        st[key] = {"job_id": job.job_id, "model": name, "res": res, "device": dev}
                        n += 1
                        print(f"  submitted {key} -> {job.job_id}", flush=True)
                        save(a.state, st)
                    except Exception as e:
                        print(f"  FAILED {key}: {type(e).__name__}: {e}", flush=True)
        print(f"\nsubmitted {n} new | {len(st)} total in {a.state}")

    if a.collect:
        out, pend, fail = {}, 0, 0
        for key, v in st.items():
            try:
                job = hub.get_job(v["job_id"])
                s = job.get_status()
                if not s.finished:
                    pend += 1
                    continue
                if not s.success:
                    fail += 1
                    continue
                prof = job.download_profile()
                ex = prof.get("execution_summary", {}) if isinstance(prof, dict) else {}
                us = ex.get("estimated_inference_time") or ex.get("inference_time")
                det = prof.get("execution_detail") or []
                unit_us = {}
                for lay in det:
                    if isinstance(lay, dict):
                        cu = lay.get("compute_unit") or "UNKNOWN"
                        unit_us[cu] = unit_us.get(cu, 0) + (lay.get("execution_time") or 0)
                tot = sum(unit_us.values()) or 1
                if us is None:
                    fail += 1
                    print(f"  {key}: finished but no latency field")
                    continue
                out[key] = {"model": v["model"], "res": v["res"],
                            "device": job.device.name,
                            "device_os": str(getattr(job.device, "os", "") or ""),
                            "ms": round(us / 1000.0, 3),
                            "npu_time_frac": round(unit_us.get("NPU", 0) / tot, 4),
                            "n_layers": len(det),
                            "fallback_units": sorted(u for u in unit_us if u != "NPU")}
            except Exception as e:
                fail += 1
                print(f"  {key}: {type(e).__name__}: {e}")
        save(a.out, out)
        print(f"\n{len(out)} profiled | {pend} pending | {fail} failed -> {a.out}")
        for res in RES:
            for dev in DEVICES:
                rows = [v for v in out.values() if v["res"] == res and dev.split()[2] in v["device"]]
                if not rows:
                    continue
                ms = sorted(r["ms"] for r in rows)
                full = sum(1 for r in rows if r["npu_time_frac"] >= 0.999)
                print(f"  {res:<10} {dev:<28} n={len(rows):<3} "
                      f"p50 {ms[len(ms)//2]:8.2f} ms  min {ms[0]:7.2f}  "
                      f"100%-NPU {full}/{len(rows)}")


if __name__ == "__main__":
    main()
