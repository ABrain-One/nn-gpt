"""Label every denoise model with full provenance, and report what git has NOT seen yet.

TWO JOBS, both read-only.

1. LABELLING. Every denoise model in the DB gets a provenance label derived from its name
   and the run that produced it, so "where did this model come from" is never a
   prefix-substring guess:

       seed          hand-written family (DenoiseUnet*, DenoiseSubPixel, ...)
       ga            DenoiseGA-*      -- source-level genetic algorithm
       ga-legacy     DnSeedGA-/DnSkip-/DnNewFam-/DnSearch-/DenoiseSkip-*  earlier GA runs
       llm-cold      LLM_Denoise-*  generated with NO reference architectures
       llm-seeded    LLM_Denoise-*  generated WITH reference architectures

   The cold/seeded split is not recoverable from the DB name alone -- both arms write the
   same LLM_Denoise-* prefix -- so it is joined in from the generation run's candidate ids
   via the eval results. Without that join the arm comparison is unreconstructible after
   the fact, which is why it is written to disk here rather than recomputed each time.

2. GIT DELTA. Lists exactly which denoise files are new or modified relative to HEAD, so a
   push adds only what has not been pushed before.

   THIS SCRIPT NEVER RUNS git add, git commit OR git push. It prints paths. Committing is
   the user's job, always. The output is a list to inspect, not a command to run.

   IT ALSO REFUSES TO LIST DELETIONS. The working tree is missing a very large number of
   files that HEAD contains (other tasks' training stats), so any blanket `git add -A` or
   `git commit -a` would record a mass deletion. Everything here is an explicit add-list of
   files that EXIST on disk; deletions are counted and reported, never staged.

  python dataset_manifest.py --repo /a/mm/nn-dataset --out /a/mm/sweep_out/manifest
"""
import argparse
import csv
import json
import glob
import os
import re
import sqlite3
import subprocess
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import naming  # noqa: E402

TASK = "img-denoising"
PSNR_SCALE = 48.0

SEED_RE = re.compile(r"^(DenoiseUnet\d+F?|DenoiseUNet\d*|LiteDenoiseNet|Denoise[A-Z][A-Za-z]*)$")
GA_RE = re.compile(r"^DenoiseGA-[0-9a-f]{16,}$")
GA_LEGACY_RE = re.compile(r"^(DnSeedGA-|DnSkip-|DnNewFam-|DnSearch-|DenoiseSkip-)")
# Two separate fixes here, both verified against the 470 LLM rows in the DB:
#
#   [-_]  -- models minted after the rename are LLM-Denoise-*, the ones that predate
#            it are LLM_Denoise_*. Both must classify as llm-*, or the manifest
#            reports the older half of the corpus as provenance "other".
#
#   no    -- this used to be `^LLM_Denoise`, which only ever matched the one-shot
#   "Denoise"  family. The evolve and parallel arms are named LLM_Eg<N>_* and
#   suffix     LLM_Pg<N>_*, so 372 of the 470 LLM models -- every model from the
#              parallel and evolve runs -- were being reported as provenance
#              "other", i.e. indistinguishable from hand-written seeds. Matching on
#              the LLM prefix alone fixes the count to 470.
LLM_RE = re.compile(r"^LLM[-_]")


def git(repo, *args):
    try:
        return subprocess.run(["git", "-C", repo] + list(args),
                              capture_output=True, text=True, timeout=180).stdout
    except Exception:
        return ""


def provenance(nn, llm_arm):
    if LLM_RE.match(nn):
        # Prefer the eval-results join, but fall back to the name itself: the
        # parallel/evolve arms bake the arm into the name (LLM-Pg7-seeded-<hash>),
        # so 'unknown' is only correct when neither source knows.
        arm = llm_arm.get(nn) or naming.arm_of(nn) or "unknown"
        return f"llm-{arm}"
    if GA_RE.match(nn):
        return "ga"
    if GA_LEGACY_RE.match(nn):
        return "ga-legacy"
    if SEED_RE.match(nn):
        return "seed"
    return "other"


def llm_arm_map(eval_dirs):
    """LLM-Denoise DB name -> 'cold'/'seeded', joined via the eval results."""
    out = {}
    for d in eval_dirs:
        for p in glob.glob(os.path.join(d, "results*.json")):
            try:
                for r in json.load(open(p)):
                    if r.get("db_name"):
                        # The candidate id keeps its underscores (it is run-local
                        # scratch, never a path); naming.arm_of reads either
                        # separator so this works for old and new runs alike.
                        arm = naming.arm_of(r.get("id", "")) or "unknown"
                        out[r["db_name"]] = arm
            except Exception:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/a/mm/nn-dataset")
    ap.add_argument("--out", default="/a/mm/sweep_out/manifest")
    ap.add_argument("--eval-dirs", nargs="*",
                    default=["/a/mm/sweep_out/llm_eval2", "/a/mm/sweep_out/llm_eval"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ---------- 1. labelled model inventory ----------
    db = os.path.join(a.repo, "db", "ab.nn.db")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    arm = llm_arm_map(a.eval_dirs)
    rows = con.execute(
        "SELECT nn, MAX(epoch), COUNT(DISTINCT epoch), MAX(accuracy) "
        "FROM stat WHERE task=? GROUP BY nn", (TASK,)).fetchall()

    recs = []
    for nn, ep_max, n_ep, acc in rows:
        recs.append({
            "nn": nn,
            "provenance": provenance(nn, arm),
            "epoch_max": ep_max,
            "db_rows": n_ep,
            "best_db": round(acc * PSNR_SCALE, 3) if acc is not None else None,
            "model_file": f"ab/nn/nn/{nn}.py",
            "stat_file": f"ab/nn/stat/nn/{nn}.json",
        })

    # ---------- 2. git delta, denoise only, ADDS ONLY ----------
    porcelain = git(a.repo, "status", "--porcelain", "--", "ab/nn")
    new, mod, deleted = [], [], 0
    for ln in porcelain.splitlines():
        if len(ln) < 4:
            continue
        code, path = ln[:2], ln[3:].strip().strip('"')
        is_denoise = ("denois" in path.lower() or "Denoise" in path
                      or "LLM_Denoise" in path or "LLM-Denoise" in path
                      or "LiteDenoise" in path)
        if not is_denoise:
            continue
        if code == "??":
            # untracked dir entries come back as a single path; expand to files
            full = os.path.join(a.repo, path)
            if os.path.isdir(full):
                for r, _d, fs in os.walk(full):
                    for f in fs:
                        new.append(os.path.relpath(os.path.join(r, f), a.repo))
            elif os.path.exists(full):
                new.append(path)
        elif "M" in code:
            mod.append(path)
        elif "D" in code:
            deleted += 1

    tracked = set(git(a.repo, "ls-files", "--", "ab/nn").splitlines())
    for r in recs:
        r["model_in_git"] = r["model_file"] in tracked
        r["stat_in_git"] = r["stat_file"] in tracked
        r["model_on_disk"] = os.path.exists(os.path.join(a.repo, r["model_file"]))

    # ---------- 3. write ----------
    mp = os.path.join(a.out, "models.csv")
    with open(mp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(sorted(recs, key=lambda r: (r["provenance"], r["nn"])))

    pl = os.path.join(a.out, "push_list.txt")
    with open(pl, "w") as f:
        f.write("# DENOISE files not yet in git (adds only -- no deletions, no renames).\n")
        f.write("# Review, then add them yourself. This tool never stages or commits.\n")
        f.write("#   git -C <repo> add --pathspec-from-file=push_list.txt\n\n")
        for p in sorted(set(new)):
            f.write(p + "\n")
    ml = os.path.join(a.out, "modified_list.txt")
    with open(ml, "w") as f:
        for p in sorted(set(mod)):
            f.write(p + "\n")

    # ---------- 4. report ----------
    c = Counter(r["provenance"] for r in recs)
    print("=== labelled denoise models in DB ===")
    for k, v in c.most_common():
        ing = sum(1 for r in recs if r["provenance"] == k and r["model_in_git"])
        print(f"  {k:<12} {v:5d} models   ({ing} already committed, {v-ing} not)")
    print(f"  {'TOTAL':<12} {len(recs):5d}")

    print("\n=== epoch_max distribution (the DB's 'epoch' is a CONFIG key) ===")
    for (p, e), n in sorted(Counter((r["provenance"], r["epoch_max"]) for r in recs).items()):
        print(f"  {p:<12} epoch_max={e:<4} {n:5d}")

    print(f"\n=== git delta (denoise paths under ab/nn only) ===")
    print(f"  NEW files to add     {len(set(new))}")
    print(f"  MODIFIED files       {len(set(mod))}")
    print(f"  deleted in worktree  {deleted}   <-- NOT listed for staging, on purpose")
    print(f"\nwrote {mp}")
    print(f"      {pl}   ({len(set(new))} paths)")
    print(f"      {ml}   ({len(set(mod))} paths)")
    print("\nNOTHING WAS STAGED OR COMMITTED. Pushing is yours to do.")


if __name__ == "__main__":
    raise SystemExit(main())
