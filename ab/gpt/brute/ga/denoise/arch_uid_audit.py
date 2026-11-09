"""How many of our models are ISOMORPHIC DUPLICATES? Measure it before changing anything.

CONTEXT. LEMUR keys the `nn` table on uuid4(code) = md5 of the source with whitespace
stripped (ab/nn/util/Util.py:111). That is a LEXICAL identity: rename a variable, reorder two
independent statements, restructure a helper class, and the same architecture gets a new
identifier and a new dataset row. The proposal is to key on architecture instead.

The machinery for that already exists here: arch_embed.exact() traces the module with torch.fx
and hashes the resulting computation GRAPH (Weisfeiler-Lehman, vocabulary-free -- it reads no
op NAME, only behavioural labels and topology). It is therefore invariant to exactly the
lexical variation uuid4 is sensitive to.

WHAT THIS SCRIPT DOES, and why it is a measurement rather than a migration. Re-keying the
`nn` table means changing a primary key that `stat.nn` references for ~1800 trial rows; that
is a migration to do deliberately, once, with evidence. This produces the evidence: for every
model in the DB it computes both identities and reports how many DISTINCT lexical ids collapse
to the same architectural one. That number is the answer to "is our corpus as diverse as its
row count claims", and it is also what tells us whether the four concurrent GA runs are
rediscovering each other's architectures.

TRACING CAN FAIL and that is not an error to hide. torch.fx cannot trace data-dependent
control flow, and a model that fails to trace has NO architectural id -- so any real migration
needs a documented fallback rather than silently reusing the lexical hash. Failures are
counted and reported separately here, never merged into the duplicate statistics.

  python arch_uid_audit.py --db /a/mm/nn-dataset/db/ab.nn.db --task img-denoising
"""
import argparse
import collections
import json
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, "/a/mm/nn-gpt/ab/gpt/brute/ast"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import naming  # noqa: E402  (needs HERE on sys.path first)


def arm_of(name):
    # Either separator: models minted before the '_'->'-' rename still carry
    # LLM_Pg*/LLM_Eg* in the DB (470 rows), and dropping them here would silently
    # reclassify the entire pre-rename corpus as "hand-written".
    if naming.matches_family(name, "LLM-Pg"):
        return "LLM-parallel"
    if naming.matches_family(name, "LLM-Eg"):
        return "LLM-evolve"
    if naming.matches_family(name, "LLM-"):
        return "LLM-oneshot"
    if name.startswith("DenoiseGA-"):
        return "GA"
    if name.startswith("DenoiseSkip-"):
        return "GA-skip"
    if "-" in name and len(name.rsplit("-", 1)[-1]) == 32:
        return "generated-other"
    return "hand-written"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/a/mm/nn-dataset/db/ab.nn.db")
    ap.add_argument("--task", default="img-denoising")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--out", default="/a/mm/sweep_out/manifest/arch_uids.json")
    a = ap.parse_args()

    c = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    rows = c.execute(
        "SELECT DISTINCT n.name, n.code FROM nn n JOIN stat s ON s.nn = n.name "
        "WHERE s.task = ? ORDER BY n.name", (a.task,)).fetchall()
    if a.limit:
        rows = rows[:a.limit]
    print(f"=== architectural-UID audit | {len(rows)} models with a {a.task} result ===\n")

    # THE SAME construction preflight uses. Previously this called exact_source, which
    # builds through mutator.lemur_io with different arguments -- so the ids written to
    # nn.arch_uid could never match the ids preflight computes, and the dedup check against
    # the dataset was a no-op that looked like a clean pass.
    from arch_uid import of_source

    by_arch = collections.defaultdict(list)
    failed = []
    for i, (name, code) in enumerate(rows, 1):
        h, err = of_source(code)
        if h is not None:
            by_arch[h].append(name)
        else:
            failed.append((name, err))
        if i % 50 == 0 or i == len(rows):
            print(f"  [{i}/{len(rows)}] traced={sum(len(v) for v in by_arch.values())} "
                  f"failed={len(failed)}", flush=True)

    traced = sum(len(v) for v in by_arch.values())
    dup_groups = {h: v for h, v in by_arch.items() if len(v) > 1}
    dup_models = sum(len(v) for v in dup_groups.values())

    print(f"\n=== RESULT ===")
    print(f"  models with a lexical id (rows in `nn`)   {len(rows)}")
    print(f"  traced successfully                       {traced}")
    print(f"  UNTRACEABLE (no architectural id)         {len(failed)}")
    print(f"  DISTINCT ARCHITECTURES among traced       {len(by_arch)}")
    if traced:
        print(f"  isomorphic-duplicate models               {dup_models - len(dup_groups)}"
              f"  ({(dup_models - len(dup_groups)) / traced:.1%} of traced)")
        print(f"  duplicate groups                          {len(dup_groups)}")
        print(f"\n  => the corpus claims {traced} architectures; it actually holds "
              f"{len(by_arch)}.")

    # WHICH ARMS COLLIDE. A group spanning two GA runs is the concrete answer to "will four
    # independent populations rediscover each other"; a group inside one arm is that
    # generator duplicating itself.
    cross, within = 0, 0
    arm_pairs = collections.Counter()
    for h, names in dup_groups.items():
        arms = sorted({arm_of(n) for n in names})
        if len(arms) > 1:
            cross += 1
            arm_pairs[" + ".join(arms)] += 1
        else:
            within += 1
            arm_pairs[arms[0] + " (self)"] += 1
    if dup_groups:
        print(f"\n=== where the duplicates come from ===")
        print(f"  groups spanning >1 arm  {cross}")
        print(f"  groups within one arm   {within}")
        for k, v in arm_pairs.most_common(12):
            print(f"    {v:4d}  {k}")

        print(f"\n=== 5 largest duplicate groups ===")
        for h, names in sorted(dup_groups.items(), key=lambda kv: -len(kv[1]))[:5]:
            print(f"  {len(names)} models share architecture {h[:16]}:")
            for n in names[:6]:
                print(f"      {n}")
            if len(names) > 6:
                print(f"      ... and {len(names)-6} more")

    if failed:
        print(f"\n=== untraceable ({len(failed)}) -- these would need a fallback id ===")
        cnt = collections.Counter(e.split(":")[0] for _n, e in failed)
        for k, v in cnt.most_common(6):
            print(f"  {v:4d}  {k}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"arch_to_names": {h: v for h, v in by_arch.items()},
               "untraceable": dict(failed)}, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
