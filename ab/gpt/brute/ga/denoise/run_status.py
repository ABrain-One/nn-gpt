"""One status line per run for every generator we have in flight. Login-node safe.

WHY THIS EXISTS. The `struct` GA variant crashed 40 seconds after a restart and sat dead for
ELEVEN HOURS before anyone looked, because a completed K8s Job and a successful K8s Job are
indistinguishable in `kubectl get jobs` -- both read 1/1. The only evidence was a Python
traceback buried 2.6 MB into a log nobody was tailing. A generator that has stopped producing
looks exactly like one that is between models, so "is it alive" has to be answered from
LOG PROGRESS, not from pod state.

Everything here is stdlib and read-only: no torch, no kubectl required for the core numbers,
so it runs on the login node where the coordinator lives.

  python3 run_status.py              # human summary
  python3 run_status.py --json       # machine-readable, for the watcher loop
"""
import argparse
import json
import os
import re
import sqlite3
import subprocess
import time

HOST = "/shared/ssd/home/b-b-ahmed/thesis"
SWEEP = f"{HOST}/sweep_out"
DB = f"{HOST}/nn-dataset/db/ab.nn.db"
GA_VARIANTS = ("cheap", "normv3", "struct", "graded")

# A generator that has not written a line in this long is not "between models". The slowest
# single observed unit of work is a 5-epoch train at ~170 s; 25 minutes is an order of
# magnitude past that, so it means stalled or dead, not busy.
STALL_S = 1500


def _tail(path, n=4000):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - n * 200))
            return f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []


def ga_status(v):
    log = f"{SWEEP}/ga_{v}/run.log"
    st = {"name": f"GA:{v}", "alive": False, "gen": None, "gens": None,
          "models": 0, "idle_s": None, "note": ""}
    if not os.path.exists(log):
        st["note"] = "no log"
        return st
    st["idle_s"] = int(time.time() - os.path.getmtime(log))
    lines = _tail(log)
    for ln in reversed(lines):
        m = re.search(r"=====\s*Generation\s+(\d+)/(\d+)", ln)
        if m:
            st["gen"], st["gens"] = int(m.group(1)), int(m.group(2))
            break
    # Models THIS RUN trained, counted from its own _epoch_stats directory -- one subdir per
    # trained model. Counting rows in the private DB instead reports ~1030 for every variant,
    # because each private DB was seeded with an online backup of the whole master: that
    # number is the corpus, not the run's production, and it makes a dead run look prolific.
    # normv3 predates launch_ga_variant.sh and uses the older seed_ga_<NAME>_models layout,
    # so a single naming convention reports it as 0 models forever -- a live run looking
    # exactly like a dead one, which is the failure this whole script exists to prevent.
    # Matched case-INSENSITIVELY. The directory is seed_ga_NORMv3_models -- mixed case -- so
    # both f"seed_ga_{v}" and f"seed_ga_{v.upper()}" miss it, and the run reports 0 models
    # while producing normally.
    here = os.path.dirname(os.path.abspath(__file__))
    st["models"] = 0
    want = (f"ga_{v}_models".lower(), f"seed_ga_{v}_models".lower())
    for d in os.listdir(here):
        if d.lower() not in want:
            continue
        es = os.path.join(here, d, "_epoch_stats")
        try:
            n = len([x for x in os.listdir(es) if os.path.isdir(os.path.join(es, x))])
        except OSError:
            continue
        if n > st["models"]:
            st["models"] = n
    txt = "\n".join(lines[-400:])

    # LIVENESS COMES FROM THE POD AND THE LOG CLOCK, NOT FROM THE WORD "Traceback".
    #
    # The first version reported DEAD whenever a traceback appeared in the last 400 lines,
    # and promptly cried wolf: it declared `struct` dead at generation 17 while the pod had
    # been Running for 6h15m and the log was mid-epoch. A GA that generates candidate
    # architectures logs tracebacks CONSTANTLY -- every model that fails to build or train
    # prints one, and catching them is what the search is for. Treating a normal event as a
    # fatal one is worse than not watching at all, because a watcher that false-alarms gets
    # ignored, and then the real death goes unnoticed.
    #
    # Authoritative signals, in order:
    #   pod phase   Running / Succeeded / Failed  -- what the cluster actually believes
    #   log mtime   a live trainer writes constantly; silence past STALL_S means hung
    # A traceback is now a NOTE, never a verdict.
    pod = _pod_phase(f"denoise-ga-{v}")
    if pod == "Failed":
        st["note"] = "pod Failed"
    elif "COMPLETE" in txt and st["gen"] and st["gens"] and st["gen"] >= st["gens"]:
        st["note"] = "COMPLETE"
    elif pod == "Succeeded":
        st["note"] = "COMPLETE" if st["gen"] and st["gen"] >= (st["gens"] or 0) \
            else f"pod exited at gen {st['gen']}"
    elif st["idle_s"] > STALL_S:
        st["note"] = f"STALLED {st['idle_s']//60}m"
    elif pod in ("Running", None):
        st["alive"] = True
        if re.search(r"Traceback|TraceError", txt):
            st["note"] = "(tracebacks in tail - normal for a search)"
    return st


def _pod_phase(job):
    """Phase of the newest pod for a job, or None when kubectl is unavailable."""
    try:
        out = subprocess.run(
            ["kubectl", "get", "pods", "-n", "b-b-ahmed", "-l", f"job-name={job}",
             "--sort-by=.metadata.creationTimestamp",
             "-o", "jsonpath={.items[-1:].status.phase}"],
            capture_output=True, timeout=20).stdout.decode().strip()
        return out or None
    except Exception:
        return None


def llm_status(tag):
    log = f"{SWEEP}/{tag.replace('llm_evolve_par3','evolve_par3').replace('llm_evolve_par2','evolve_par')}.log"
    if not os.path.exists(log):
        log = f"{SWEEP}/evolve_par3.log"
    st = {"name": f"LLM:{tag}", "alive": False, "gen": None, "gens": None,
          "models": 0, "idle_s": None, "note": "", "per_gen": []}
    if not os.path.exists(log):
        st["note"] = "no log"
        return st
    st["idle_s"] = int(time.time() - os.path.getmtime(log))
    txt = "\n".join(_tail(log, 20000))
    for m in re.finditer(r"GENERATION (\d+)/(\d+)", txt):
        st["gen"], st["gens"] = int(m.group(1)), int(m.group(2))
    # "delta  : +N" on the COMMITTED merge is the models actually added that generation.
    blocks = txt.split("[6] merge")
    for b in re.finditer(r"delta\s*:\s*\+(\d+)\s+\+\d+\s+\+\d+[\s\S]{0,200}?COMMITTED", txt):
        st["per_gen"].append(int(b.group(1)))
    st["models"] = sum(st["per_gen"])
    st["alive"] = bool(subprocess.run(
        ["pgrep", "-f", "llm_evolve_par.sh"], capture_output=True).stdout.strip())
    if not st["alive"]:
        st["note"] = "COORDINATOR NOT RUNNING"
    elif st["idle_s"] > STALL_S:
        st["note"] = f"STALLED {st['idle_s']//60}m"
    return st


def corpus():
    try:
        c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        n = c.execute("SELECT COUNT(*) FROM nn").fetchone()[0]
        d = c.execute("SELECT COUNT(DISTINCT arch_uid) FROM nn "
                      "WHERE arch_uid IS NOT NULL").fetchone()[0]
        c.close()
        return n, d
    except sqlite3.Error:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--tag", default="llm_evolve_par3")
    a = ap.parse_args()

    rows = [ga_status(v) for v in GA_VARIANTS] + [llm_status(a.tag)]
    n, d = corpus()
    if a.json:
        print(json.dumps({"runs": rows, "nn_rows": n, "distinct_arch": d}))
        return 0

    print(f"=== generator status  {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"{'run':<22}{'state':<8}{'gen':<9}{'models':>7}  {'idle':>6}  note")
    for r in rows:
        state = "ALIVE" if r["alive"] else ("DONE" if r["note"] == "COMPLETE" else "DEAD")
        g = f"{r['gen']}/{r['gens']}" if r["gen"] else "-"
        idle = f"{r['idle_s']//60}m" if r["idle_s"] is not None else "-"
        print(f"{r['name']:<22}{state:<8}{g:<9}{r['models']:>7}  {idle:>6}  {r['note']}")
    if rows[-1]["per_gen"]:
        pg = rows[-1]["per_gen"]
        print(f"\nLLM yield/gen: {pg}")
        print(f"  total {sum(pg)} over {len(pg)} gens = {sum(pg)/len(pg):.1f}/gen")
    if n:
        print(f"\ncorpus: {n} rows | {d} distinct architectures (WL exact)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
