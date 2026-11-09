"""One place that decides whether a name may enter the LEMUR repo.

WHY THIS EXISTS
---------------
LEMUR keys every training record by a directory whose name is four fields joined
by an underscore:

    ab/nn/stat/train/<task>_<dataset>_<metric>_<model>

and `ab/nn/util/Util.py:61` unpacks it with

    task, dataset, metric, nn = tuple(c.split(config_splitter))    # config_splitter = '_'

That is a positional 4-way split, so a single '_' *inside* any one of the four
fields shifts every later field by one and the import dies with
"too many values to unpack (expected 4)". The failure is not local to the bad
model either -- it aborts the whole import, so one malformed name can withhold
an entire batch.

We learned this the expensive way. Batch 1 shipped 372 models named
`LLM_Pg23_seeded-<hash>` and 111 stat dirs under dataset `denoise_awgn`; both
had to be renamed and withdrawn after review. The upstream repo had in fact been
telling us the convention all along -- every multi-word name it owns is
hyphenated (`cifar-10`, `celeba-gender`, `imagenette`, `psnrrgb`) -- we just
drifted from it in our own generators.

So: hyphens, never underscores, in all four fields. This module is the gate.
Call `check_*` at every boundary where a name is minted or where a name becomes
a path, and a non-conforming name can never reach a commit again.

Legacy names are a separate matter. 470 rows in the denoise DB still carry the
pre-rename `LLM_*` names, so readers must keep matching them; `canon()` maps an
old name to its new form and `matches_family()` accepts either spelling. Only
*minting* and *export* are strict -- reading history stays permissive, because
rejecting the past would just hide it.
"""

import re

# The separator LEMUR splits stat-dir names on. Mirrors ab/nn/util/Const.py:89.
# Not imported from there on purpose: this module has to be usable from the
# generator side, which does not always have nn-dataset importable.
CONFIG_SPLITTER = "_"

# The four fields of a stat-dir name, in order.
FIELDS = ("task", "dataset", "metric", "model")


class NameError_(ValueError):
    """A name that would corrupt the 4-way stat-dir split."""


def is_legal(name):
    """True when `name` is safe to use as any one of the four stat-dir fields."""
    return bool(name) and CONFIG_SPLITTER not in str(name)


def canon(name):
    """Rewrite a name into the convention: '_' -> '-'.

    Idempotent, so it is safe to call on names that are already clean. This is
    the *only* sanctioned transform -- do not strip or concatenate instead, or
    two distinct models can collide onto one name (`LLM_Pg2_cold` and
    `LLM_Pg2cold` both becoming `LLMPg2cold` would silently merge two records).
    """
    return str(name).replace(CONFIG_SPLITTER, "-")


def check(name, field="model"):
    """Return `name` if it is legal, else raise. Use at mint time.

    Deliberately raises instead of quietly canonicalising: if a generator is
    still producing underscored names we want the run to stop and the prefix to
    be fixed at source, not to accumulate a silent rewrite layer.
    """
    if not is_legal(name):
        raise NameError_(
            f"illegal {field} name {name!r}: contains {CONFIG_SPLITTER!r}, which "
            f"breaks the <task>_<dataset>_<metric>_<model> split in Util.py:61. "
            f"Use {canon(name)!r} instead."
        )
    return name


def check_stat_dir(dirname):
    """Verify a full stat-dir name splits into exactly the four fields.

    This is the assertion that actually matters -- it reproduces what LEMUR's
    importer does, rather than checking the fields in isolation, so it also
    catches a name assembled from individually-legal parts by buggy joining.
    """
    parts = str(dirname).split(CONFIG_SPLITTER)
    if len(parts) != 4:
        raise NameError_(
            f"stat dir {dirname!r} splits into {len(parts)} fields, expected 4 "
            f"({'_'.join('<%s>' % f for f in FIELDS)}). "
            f"Offending fields: {[p for p in parts]!r}"
        )
    return dict(zip(FIELDS, parts))


def stat_dir(task, dataset, metric, model):
    """Build a stat-dir name, refusing to build a broken one."""
    for value, field in ((task, "task"), (dataset, "dataset"),
                         (metric, "metric"), (model, "model")):
        check(value, field)
    return CONFIG_SPLITTER.join((task, dataset, metric, model))


# ---------------------------------------------------------------------------
# Reading legacy names
# ---------------------------------------------------------------------------

def variants(name):
    """Both spellings of a name, for looking it up in mixed-vintage sources.

    The on-disk tree was renamed; the DB was not (470 `LLM_*` rows). Anything
    that joins disk to DB has to try both.
    """
    n = str(name)
    out = [n]
    c = canon(n)
    if c != n:
        out.append(c)
    return out


def alias_both_ways(mapping):
    """Add canonical aliases to a dict keyed by names, without dropping originals.

    `setdefault` rather than assignment: when a key already exists under its new
    spelling that entry is authoritative and must not be clobbered by a legacy
    row.
    """
    for key in list(mapping):
        c = canon(key)
        if c != key:
            mapping.setdefault(c, mapping[key])
    return mapping


def family_prefixes(*prefixes):
    """Expand prefixes to match either spelling, for family/arm classification.

    `family_prefixes("LLM-")` -> ("LLM-", "LLM_") so a startswith() test keeps
    recognising the 470 legacy rows after the rename.
    """
    out = []
    for p in prefixes:
        for v in (p, canon(p), p.replace("-", CONFIG_SPLITTER)):
            if v not in out:
                out.append(v)
    return tuple(out)


def matches_family(name, *prefixes):
    """True when `name` belongs to one of `prefixes`, in either spelling."""
    return str(name).startswith(family_prefixes(*prefixes))


# Arm markers ('cold' / 'seeded') appear in two different namespaces that must
# not be conflated:
#   * candidate ids from the generation run  -- `p10_0000_seeded_constraint_based`
#     These are run-local scratch ids. They never become a model name and never
#     reach a path, so they keep their underscores. Do NOT canonicalise them.
#   * DB / on-disk model names -- `LLM-Denoise-seeded-<hash>` (was `..._seeded-...`)
# `arm_of` below reads either, which is why it checks both separators.
_ARM_RE = re.compile(r"[_-](cold|seeded)[_-]")


def arm_of(identifier):
    """'cold' / 'seeded' / '' from a candidate id or a model name, either spelling."""
    m = _ARM_RE.search(str(identifier))
    return m.group(1) if m else ""


def audit(names, field="model"):
    """Split `names` into (legal, illegal). For reporting, not enforcement."""
    legal, illegal = [], []
    for n in names:
        (legal if is_legal(n) else illegal).append(n)
    return legal, illegal


def _on_upstream(repo, relpath, _cache={}):
    """True when `relpath` already exists on origin/main, i.e. we do not own it.

    ab/nn/loader/imdb_wiki.py and ab/nn/metric/mai_psnr.py both violate the
    convention but are upstream's files for other tasks. They are real hazards for
    whoever uses them -- an `imdb_wiki` dataset cannot produce a parseable stat dir
    -- but renaming another team's modules is not ours to do, so the audit reports
    them separately instead of failing our gate forever on someone else's name.
    """
    import subprocess
    key = (repo, relpath)
    if key not in _cache:
        try:
            r = subprocess.run(["git", "-C", repo, "cat-file", "-e",
                                f"origin/main:{relpath}"],
                               capture_output=True, timeout=30)
            _cache[key] = r.returncode == 0
        except Exception:
            _cache[key] = False      # unknown provenance -> treat as ours, i.e. report
    return _cache[key]


def scan_repo(repo, only=("denois", "LiteDenoise", "LLM")):
    """Audit a checked-out nn-dataset for names that would break the import.

    Returns {'stat': [...], 'nn': [...], 'loader': [...], 'metric': [...],
             'upstream': [...]} where each item is (name, reason).

    Model files are scoped to denoise via `only`; loader/metric stems are always in
    scope because each one *is* a stat-dir field. Violations in files that already
    exist on origin/main are bucketed under 'upstream'.
    """
    import os

    def ours(s):
        return any(k.lower() in s.lower() for k in only)

    bad = {"stat": [], "nn": [], "loader": [], "metric": [], "upstream": []}

    stat = os.path.join(repo, "ab", "nn", "stat", "train")
    if os.path.isdir(stat):
        for d in os.listdir(stat):
            if not ours(d):
                continue
            try:
                check_stat_dir(d)
            except NameError_ as e:
                bad["stat"].append((d, str(e)))

    for kind, sub in (("nn", ("ab", "nn", "nn")),
                      ("loader", ("ab", "nn", "loader")),
                      ("metric", ("ab", "nn", "metric"))):
        p = os.path.join(repo, *sub)
        if not os.path.isdir(p):
            continue
        for f in os.listdir(p):
            if not f.endswith(".py") or f.startswith("__"):
                continue
            stem = f[:-3]
            # loader/metric stems are the dataset/metric FIELD, so they are always
            # in scope; model files only when they look like ours.
            if kind == "nn" and not ours(stem):
                continue
            if is_legal(stem):
                continue
            rel = "/".join(sub + (f,))
            why = f"contains {CONFIG_SPLITTER!r}; rename to {canon(stem)}.py"
            if _on_upstream(repo, rel):
                bad["upstream"].append((rel, why + "  [upstream file -- not ours]"))
            else:
                bad[kind].append((f, why))
    return bad


def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="audit a nn-dataset checkout for "
                                             "names that break the stat-dir split")
    ap.add_argument("--repo",
                    default="/shared/ssd/home/b-b-ahmed/thesis/nn-dataset")
    ap.add_argument("--all", action="store_true",
                    help="include names we do not own (upstream imdb_wiki, mai_psnr)")
    a = ap.parse_args()
    bad = scan_repo(a.repo, only=("",) if a.all else ("denois", "LiteDenoise", "LLM"))
    # Only our own violations set the exit code -- this is meant to be usable as a
    # pre-push check, and it would be useless if it always failed on upstream files.
    ours_total = sum(len(bad[k]) for k in ("stat", "nn", "loader", "metric"))
    for kind in ("stat", "nn", "loader", "metric"):
        items = bad[kind]
        print(f"{kind:<8} {len(items)} violation(s)")
        for name, why in items[:20]:
            print(f"    {name}\n      {why}")
        if len(items) > 20:
            print(f"    ... and {len(items) - 20} more")
    if bad["upstream"]:
        print(f"\nnot ours ({len(bad['upstream'])}) -- reported, does not fail this check:")
        for name, why in bad["upstream"]:
            print(f"    {name}\n      {why}")
    print(f"\nours: {ours_total} violation(s)"
          + ("  -> OK, safe to push" if not ours_total else "  -> FIX BEFORE PUSHING"))
    raise SystemExit(1 if ours_total else 0)


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1:
        _cli()

    # Self-check: the exact cases that burned us, kept as executable documentation.
    assert canon("LLM_Pg23_seeded-abc") == "LLM-Pg23-seeded-abc"
    assert canon(canon("LLM_Denoise_cold-x")) == "LLM-Denoise-cold-x"
    assert is_legal("LLM-Denoise-cold-x") and not is_legal("LLM_Denoise_cold-x")
    assert check_stat_dir("img-denoising_denoise_psnr_LLM-Denoise-cold-x")["model"] \
        == "LLM-Denoise-cold-x"
    for bad in ("img-denoising_denoise_awgn_psnr_M",          # dataset had '_'
                "img-denoising_denoise_psnr_LLM_Denoise-x"):  # model had '_'
        try:
            check_stat_dir(bad)
        except NameError_:
            pass
        else:
            raise AssertionError(f"should have rejected {bad}")
    assert matches_family("LLM_Pg1_seeded-x", "LLM-")   # legacy still classified
    assert matches_family("LLM-Pg1-seeded-x", "LLM-")
    assert arm_of("p10_0000_seeded_constraint_based") == "seeded"  # candidate id
    assert arm_of("LLM-Denoise-cold-abc") == "cold"               # new model name
    assert arm_of("LLM_Denoise_cold-abc") == "cold"               # legacy name
    assert alias_both_ways({"LLM_A": 1}) == {"LLM_A": 1, "LLM-A": 1}
    print("naming.py: all self-checks pass")
