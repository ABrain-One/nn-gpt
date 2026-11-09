"""Fold npu_gate profile.json results into latency_corpus.json, the file everything else reads.

WHY A SEPARATE STEP. Two tools write latency and they do not share a path. bulk_profile.py owns
latency_corpus.json and derives its candidate list from tflite_pool/codes.jsonl -- a census of a
past run that does not contain the 1,341 models the dnseed2 arms just added. npu_gate.py profiles
whatever artifacts are in a work directory, which is exactly what we need for a fresh batch, but
writes its own profile.json. Rather than mutate codes.jsonl (it is the evidence base for
--ban-audit and should stay a record of what was actually converted then), this merges the new
measurements into the corpus file directly.

THE CORPUS ALREADY HOLDS TWO RECORD SHAPES AND THAT HAS BITTEN ONCE. 2,373 entries carry
"status": "OK"; 128 older ones have no status field at all and are perfectly valid -- they include
the DnSeedGA champions. A reader that insisted on status == "OK" silently dropped all 128, which
made the joint objective's target set look like it contained no DnSeedGA models when it did.
New rows written here therefore always carry an explicit status, so the ambiguity does not grow.

NEVER OVERWRITE A GOOD MEASUREMENT WITH A WORSE ONE. If a model is already in the corpus with a
latency, it is kept unless --force: re-profiling the same artifact should reproduce the number,
and if it does not, that discrepancy is worth seeing rather than silently resolving.

  python fold_profiles.py --profile W/profile.json --dry-run
  python fold_profiles.py --profile W/profile.json
"""
import argparse
import collections
import json
import os
import shutil
import sys

THESIS = "/a/mm" if os.path.isdir("/a/mm/nn-gpt") else "/shared/ssd/home/b-b-ahmed/thesis"
CORPUS = os.path.join(THESIS, "sweep_out", "aihub", "latency_corpus.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", action="append", required=True,
                    help="npu_gate profile.json (repeatable)")
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--ops-json", default=None,
                    help="optional {model: n_ops} to record alongside, since npu_gate's profile "
                         "rows carry no n_layers")
    ap.add_argument("--force", action="store_true", help="overwrite existing latency rows")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    corpus = json.load(open(a.corpus)) if os.path.exists(a.corpus) else {}
    ops = json.load(open(a.ops_json)) if (a.ops_json and os.path.exists(a.ops_json)) else {}
    before = len(corpus)

    incoming = {}
    for p in a.profile:
        if not os.path.exists(p):
            sys.exit(f"FATAL {p} does not exist")
        incoming.update(json.load(open(p)))

    added = skipped = kept = 0
    reasons = collections.Counter()
    for name, v in incoming.items():
        if not isinstance(v, dict) or not v.get("ms"):
            skipped += 1
            reasons[v.get("status", "no-ms") if isinstance(v, dict) else "malformed"] += 1
            continue
        if name in corpus and corpus[name].get("ms") and not a.force:
            kept += 1
            continue
        row = {"status": "OK", "ms": v["ms"], "npu_time_frac": v.get("npu_time_frac"),
               "device": v.get("device"), "fallback_units": v.get("fallback_units")}
        if ops.get(name):
            row["n_layers_tflite"] = ops[name]
        corpus[name] = row
        added += 1

    print(f"corpus {before} -> {before + added} entries")
    print(f"  added {added} | already present, left alone {kept} | unusable {skipped}")
    if reasons:
        print(f"  unusable breakdown: {dict(reasons)}")
    fam = lambda n: next((p for p in ("DnSeed2Act", "DnSeed2Fast", "DnSeed2Wide", "DnSeedGA2",
                                      "DnSeedGA", "DenoiseGA", "LLM") if n.startswith(p)), "other")
    newnames = [n for n, v in incoming.items() if isinstance(v, dict) and v.get("ms")]
    print(f"  by family: {dict(collections.Counter(fam(n) for n in newnames))}")

    if a.dry_run:
        print("\nDRY RUN -- nothing written")
        return 0
    if added:
        bak = a.corpus + ".bak"
        shutil.copy(a.corpus, bak) if os.path.exists(a.corpus) else None
        tmp = a.corpus + ".tmp"
        json.dump(corpus, open(tmp, "w"), indent=1)
        os.replace(tmp, a.corpus)
        print(f"\nwrote {a.corpus}  (previous copy at {bak})")
    else:
        print("\nnothing to add; corpus unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
