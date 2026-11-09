"""LLM candidate generation for image denoising, in two arms, with full prompt capture.

WHY THIS EXISTS RATHER THAN ab/gpt/util/AlterNN.py:alter()
    alter() builds its prompts by iterating rows of nn_dataset.data(task=...). With no
    rows it emits no prompts, so it structurally cannot express the cold arm -- "generate
    a denoiser with NO reference to our seeds". It also never writes the prompt to disk:
    it saves the extracted code (new_nn_file), the raw response (new_out_file) and a
    dataframe, but the exact string sent to the model is discarded. Once sampling is
    stochastic and the seed corpus changes between runs, an unsaved prompt makes a result
    unreproducible. Both gaps are why this is a separate runner.

THE TWO ARMS
    cold    -- task spec + interface contract only. Measures the model's unconditioned
               prior over denoising architectures. Baseline for "did our 18 families bias
               the search space?"
    seeded  -- same text plus a primary reference and N supporting references drawn from
               the corpus, with their measured dB. Exploits what we already know.

    The shared blocks in the two JSON configs are byte-identical by construction; this
    runner asserts that at load time (--check-contract) so the ablation cannot silently
    decay into a comparison of two differently-worded prompts.

PROMPT PERSISTENCE
    Everything needed to replay a generation is written before the model is called:
      run_meta.json  llm name, sampling params, argv, config sha256s, git revision
      prompts.jsonl  append-only index, one JSON object per generation
      cand/<id>/     prompt_system.txt, prompt_user.txt (the EXACT strings sent),
                     response_raw.txt, candidate.py, meta.json
    prompt_sha256 is recorded so duplicate prompts across runs are detectable, and
    config_sha256 pins which version of the template produced a given candidate.

    --dry-run renders and saves every prompt WITHOUT loading the model, so the prompts can
    be reviewed before spending GPU time. Run it first.

Usage:
    python llm_denoise.py --mode cold   --n 20 --dry-run
    python llm_denoise.py --mode seeded --n 20 --supporting 3 --dry-run
    python llm_denoise.py --mode both   --n 20 --llm Qwen/Qwen2.5-Coder-7B-Instruct
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys

TASK, DATASET, METRIC = "img-denoising", "denoise", "psnr"
PSNR_SCALE = 48.0

HERE = os.path.dirname(os.path.abspath(__file__))
NNGPT_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
CONF_TEST_DIR = os.path.join(NNGPT_ROOT, "ab", "gpt", "conf", "prompt", "test")

COLD_CONF = "Denoise_gen_cold.json"          # legacy single-template (run 20260730-220434)
SEEDED_CONF = "Denoise_alter_seeded.json"    # legacy single-template
SHARED_CONF = "Denoise_shared_blocks.json"
COLD_STRAT_CONF = "Denoise_gen_cold_strategies.json"
SEEDED_STRAT_CONF = "Denoise_alter_seeded_strategies.json"

# Blocks that MUST match between the two arms. Anything outside these may differ.
SHARED_BLOCKS = ("TASK", "CONSTRAINTS", "INTERFACE CONTRACT", "OUTPUT")


def _resolve(*cands):
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


def nn_dir():
    return _resolve(
        "/a/mm/nn-dataset/ab/nn/nn",
        "/shared/ssd/home/b-b-ahmed/thesis/nn-dataset/ab/nn/nn",
        os.path.join(NNGPT_ROOT, "..", "nn-dataset", "ab", "nn", "nn"),
    )


def db_path():
    """The DB the reference corpus is read from.

    DENOISE_DB comes FIRST so a caller can point this at a pod-local copy. Reading the corpus
    straight off the shared hostPath is what stalled the reference-arm A/B: two generation pods
    sat in D-state at 0.33 MB/s while 13 training pods and the farm submitter competed for the
    same filesystem, and neither produced a candidate in 2.5 hours.

    Note this is NOT the same file as ab.nn.util.Const.db_file. Const derives its path from
    __project_root_path(), which walks up from the CWD -- and these jobs run `cd /a/mm/nn-gpt`,
    which has ab/ and README.md, so Const resolves the root to nn-gpt and opens
    /a/mm/nn-gpt/db/ab.nn.db: a 1.5 GB, 19,473-model all-tasks file with ZERO denoise models in it.
    That read is incidental (the reference pool correctly uses this function, which is why the
    production run reported 2,785 references), but it is most of the I/O. Running from a local root
    fixes that half; DENOISE_DB fixes this half.
    """
    env = os.environ.get("DENOISE_DB")
    if env and os.path.exists(env):
        return env
    return _resolve(
        "/a/mm/nn-dataset/db/ab.nn.db",
        "/shared/ssd/home/b-b-ahmed/thesis/nn-dataset/db/ab.nn.db",
        os.path.join(NNGPT_ROOT, "..", "nn-dataset", "db", "ab.nn.db"),
    )


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_rev(path):
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def load_conf(fname):
    p = os.path.join(CONF_TEST_DIR, fname)
    with open(p) as f:
        raw = f.read()
    d = json.loads(raw)
    key = next(k for k in d if not k.startswith("_"))
    return d[key], sha256(raw), p


def load_shared():
    """The four blocks every denoise prompt shares, plus the base system lines.

    Splicing these from one file is what makes the cold/seeded arms identical by
    CONSTRUCTION. The older two-config layout duplicated them and relied on
    --check-contract to notice drift after the fact; with 5 cold strategies x 4 framings
    that would have been nine copies of the same text waiting to diverge.
    """
    p = os.path.join(CONF_TEST_DIR, SHARED_CONF)
    with open(p) as f:
        raw = f.read()
    return json.loads(raw), sha256(raw)


def load_strategies(fname):
    """{key: {strategy, variants[], task, mode}} from a strategy config."""
    p = os.path.join(CONF_TEST_DIR, fname)
    with open(p) as f:
        raw = f.read()
    d = json.loads(raw)
    return {k: v for k, v in d.items() if not k.startswith("_")}, sha256(raw)


def compose(shared, variant, mode, primary=None, supporting=(), deploy=True):
    """Build (system, user) = strategy ask + shared blocks + references + output.

    Block ORDER is fixed here rather than in the configs so no strategy can reorder or
    omit a constraint: ask -> TASK -> CONSTRAINTS -> DEPLOYABILITY -> INTERFACE -> (refs) -> OUTPUT.

    `deploy` is the ONE block that can be switched off, and only because it is the treatment in the
    deployability prompt ablation. It sits immediately after CONSTRAINTS so that turning it off
    removes text without reordering anything else -- the control and treatment prompts are then
    identical except for that contiguous run of lines, which is what makes the contrast clean.
    Production generation must leave it on; ablation_prompt_deploy.py explains why the numbers
    matter. A config predating this key still works: the block is skipped if absent, rather than
    raising KeyError mid-run.
    """
    system = "\n".join(shared["system_base"])
    parts = [variant, ""]
    # deploy selects WHICH block goes in, or none:
    #   True / "v1"   the original convertibility block (rules 10-12)
    #   "v2"          adds the graph-size and attention rules -- see below
    #   "placebo"     length-matched neutral text for v1; "v2-placebo" likewise for v2
    #   False         nothing
    # The placebos exist because each real block adds 1.2-2.1k characters to a ~27k prompt, and
    # prompt length alone can change an LLM's output. Without them, an on-vs-off difference cannot
    # be attributed to the CONTENT of the guidance, only to something more having been said.
    #
    # V1 IS FROZEN, NOT SUPERSEDED. The deployability ablation launched 2026-08-12 generated its
    # arms against v1, so editing that block would destroy the record of what was tested. v2 is a
    # separate key for that reason, and both stay in the file.
    #
    # WHAT v2 ADDS AND WHY. v1 only addressed CONVERTIBILITY -- whether a TFLite graph can be built
    # at all, which affects 6.1% of the corpus. It does nothing about the dominant failure: 37% of
    # 321 profiled LLM models convert fine and then execute entirely on the CPU. Measured over 1350
    # profiled models, that is driven by GRAPH SIZE (off-NPU median 377 operations vs 70 on-NPU,
    # depth-alone AUC 0.965-0.973), which the 9053 MMac budget cannot see, plus attention blocks
    # (matmul+Softmax: 46/51 off-NPU vs 0/540 on). v2 states both. It deliberately does NOT list
    # the eight operators that topped the raw contrast (PixelUnshuffle, pad, gelu, normalize,
    # zeros, sqrt, ones, PixelShuffle): every one of them stopped separating once models of similar
    # depth were compared, so they were markers of deep graphs rather than causes of fallback.
    slot = {True: "deploy", "v1": "deploy", "v2": "deploy_v2",
            "placebo": "deploy_placebo", "v2-placebo": "deploy_v2_placebo"}.get(deploy)
    if deploy and slot is None:
        raise ValueError(f"unknown deploy guidance {deploy!r}; expected one of "
                         f"True/'v1'/'v2'/'placebo'/'v2-placebo'/False")
    keys = ("task", "constraints", slot, "interface") if slot \
        else ("task", "constraints", "interface")
    for key in keys:
        if key not in shared:
            continue
        parts += shared[key] + [""]
    if mode == "seeded":
        pb = ""
        if primary:
            pb = (f"PRIMARY REFERENCE - measured {primary['db']:.2f} dB PSNR at epoch "
                  f"{primary['epoch']}, hyperparameters {primary['prm']}:\n"
                  f"```python\n{primary['code']}\n```")
        sb = "\n".join(
            f"SUPPORTING REFERENCE {i} - measured {s['db']:.2f} dB PSNR:\n"
            f"```python\n{s['code']}\n```" for i, s in enumerate(supporting, 1))
        body = "\n".join(parts)
        body = body.replace("{primary_block}", pb).replace("{supporting_block}", sb)
        parts = [body]
    parts += shared["output"]
    return system, "\n".join(parts)


def _extract_block(lines, header):
    """Pull the contiguous run of lines starting at `header` up to the next ALL-CAPS
    header, so the two configs can be compared block-by-block rather than wholesale."""
    def is_header(ln):
        # A header is a line at column 0 whose FIRST WORD is all-caps. Testing the whole
        # line (s == s.upper()) fails here: real headers carry lowercase tails, e.g.
        # "CONSTRAINTS - a model violating any of these is discarded automatically:" and
        # "PRIMARY REFERENCE - measured {accuracy_db} dB PSNR ...". With the whole-line
        # test nothing terminated a block, so every block ran to the end of the prompt and
        # all four compared unequal -- a false drift report from the checker itself.
        if not ln or ln[:1].isspace():
            return False
        w = ln.split()[0].rstrip(":,")
        return len(w) >= 2 and w.isupper() and w.isalpha()

    out, on = [], False
    for ln in lines:
        if ln.startswith(header):
            on = True
            out.append(ln)
            continue
        if on:
            if is_header(ln):
                break
            out.append(ln)
    return "\n".join(out).rstrip()


def check_contract(cold, seeded):
    """Fail loudly if the arms have drifted apart on the shared text."""
    bad = []
    for blk in SHARED_BLOCKS:
        a = _extract_block(cold["prompt"], blk)
        b = _extract_block(seeded["prompt"], blk)
        if a != b:
            bad.append(blk)
    return bad


_PRM_CACHE = {}


def _prm(con, uid):
    """stat.prm is a 32-char content hash into the `prm` (uid, name, value) table.

    Rendered raw it reaches the model as 'hyperparameters be5ed7c965b01195...', which is
    noise occupying a slot the model will try to interpret. Resolve it to real key/values,
    and fall back to the empty dict rather than the hash if the join misses -- the hash is
    strictly worse than saying nothing.
    """
    if uid in _PRM_CACHE:
        return _PRM_CACHE[uid]
    try:
        rows = con.execute("SELECT name, value FROM prm WHERE uid=?", (uid,)).fetchall()
        d = {n: v for n, v in rows}
    except Exception:
        d = {}
    keep = {k: d[k] for k in ("lr", "batch", "momentum", "epoch_max", "transform")
            if k in d} or d
    _PRM_CACHE[uid] = keep
    return keep


def canonical_dir():
    return _resolve("/a/mm/sweep_out/canonical",
                    "/shared/ssd/home/b-b-ahmed/thesis/sweep_out/canonical")


GENERATED_SUFFIX = re.compile(r"-[0-9a-f]{16,}$")


# MEASURED over-cap models, from run_seed_ga.py's DEFAULT_SEEDS comment:
#   "DenoiseUnet2F/7F/10F ... are 1.5-3.3x over the edge cap (14005/30152/29054 MMac)"
# Keep these as recorded fact, because the cost backend cannot run everywhere: when
# measure_cost() is unavailable the legality check fails OPEN, so without this denylist an
# unmeasurable environment would admit precisely the three worst offenders. "Use the F
# ones" is very nearly the right rule -- the non-F variants are indeed the bulky ones --
# but it is not exactly right: these three F variants are the largest models in the tree.
KNOWN_OVER_CAP = {"DenoiseUnet2F": 14005, "DenoiseUnet4F": 12585,
                  "DenoiseUnet7F": 30152, "DenoiseUnet10F": 29054}
# DenoiseUnet4F (12585 MMac, 1.4x over) was NOT in the comment this list was transcribed
# from, so the first version of this denylist missed it and the "just use the F variants"
# rule let it into the seeded corpus. The live in-container gate caught it at runtime
# ("DROP over-cap seed DenoiseUnet4F: 12585 MMac > 9053 cap"), which is the only reason
# no over-cap reference reached the prompts. Measured 2026-07-30; 4 of the 11 F variants
# are over budget, not 3.


def is_seed_name(name):
    """True for the edge-legal hand-written seeds: DenoiseUnet<N>F plus LiteDenoiseNet.

    Only the F variants. The plain DenoiseUnet<N> models are the full-width originals and
    go over the 9053 MMac budget, so they are excluded by construction rather than left to
    a MAC check that may not be able to run.

    Also excludes every hash-suffixed generated model. Filtering only 'DenoiseGA-' and
    'DenoiseSkip-' missed the DnSeedGA-, DnSkip-, DnNewFam- and DnSearch- prefixes that
    are also in the stat table, which is why a "non-GA" query once returned 419 rows.

    KNOWN_OVER_CAP is applied on top, removing 2F/7F/10F. That leaves 1F, 3F, 4F, 5F, 6F,
    8F, 9F and LiteDenoiseNet -- a superset of the GA's own DEFAULT_SEEDS.
    """
    if GENERATED_SUFFIX.search(name) or name in KNOWN_OVER_CAP:
        return False
    return bool(re.match(r"(?i)^denoiseunet[0-9]+f$|^litedenoisenet$", name))


def is_family_name(name):
    """True for the Denoise<Word> families built by the family sweep (17 have stats).

    Everything that is a denoise model, is not hash-suffixed, and is not one of the 23
    hand-written seeds.
    """
    if GENERATED_SUFFIX.search(name) or is_seed_name(name) or name in KNOWN_OVER_CAP:
        return False
    if re.match(r"(?i)^denoiseunet[0-9]*f?$|^litedenoisenet$", name):
        return False          # a seed variant (incl. the excluded non-F originals)
    return bool(re.match(r"(?i)^denoise[A-Z]", name))


# A reference below this is a failed run, not an architecture worth imitating. Three of
# the 23 seeds sit at 4-6 dB (DenoiseUnet2, DenoiseUnet4, DenoiseUnet8) -- a denoiser that
# scores 4 dB is outputting near-noise. Handing one to the LLM as a labelled "reference
# implementation" actively teaches the wrong thing, and it would anchor the dB scale.
MIN_USABLE_DB = 25.0


def seed_corpus(names=None, limit=None, seed_set="named"):
    """[(name, code, dB, epoch, prm)] for denoise seeds, dB = mean of the last 5 epochs.

    NOT the max. A max over 50 epochs rewards volatility: DenoiseSubPixel peaks 0.34 dB
    above where it settles and DenoiseParallelMR moves from 4th by peak to last by last5,
    so feeding peaks would advertise the noisiest models as the best ones.

    seed_set='canonical' (default) reads the per-model DBs from the 50-epoch sweep, where
    every model ran the SAME recipe (lr 1e-4, batch 16, 50 epochs). seed_set='all' reads
    the master DB instead, and that is a trap worth spelling out:

      - it returns 168 entries, 132 of which are GA descendants of these same families,
        so the "references" are largely rediscoveries of each other;
      - the dB values are NOT comparable, because rows come from different batch/lr/epoch
        configurations (the LEMUR config key is (task, dataset, metric, nn, epoch_max) --
        batch and lr are not in it, so runs at different recipes land in the same table);
      - it ranks DenoiseUnet7 top at 36.86 dB, which is measured under a different recipe
        AND is one of the families outside the 9053 MMac edge budget. Presenting it as the
        primary reference teaches the model to imitate something the MAC gate then throws
        away.

    Over-cap seeds are dropped in both modes when the GA's edge check is importable.
    """
    nd = nn_dir()
    if not nd:
        raise SystemExit("cannot locate ab/nn/nn")

    stats = {}
    if seed_set in ("named", "seeds23"):
        want = is_seed_name if seed_set == "seeds23" else \
            (lambda n: is_seed_name(n) or is_family_name(n))
        dbp = db_path()
        if not dbp:
            raise SystemExit("master DB not found")
        con = sqlite3.connect(f"file:{dbp}?mode=ro", uri=True)
        for nn, ep, acc, prm in con.execute(
                "SELECT nn, epoch, accuracy, prm FROM stat "
                "WHERE task=? AND dataset=? AND metric=? AND accuracy IS NOT NULL",
                (TASK, DATASET, METRIC)):
            if want(nn):
                stats.setdefault(nn, []).append((ep, acc * PSNR_SCALE, _prm(con, prm)))
        con.close()
        # Prefer the canonical 50-epoch curve where one exists: same recipe for every
        # model, so the dB shown to the LLM is comparable across references. Master-DB
        # numbers stay for the seeds that were never in that sweep.
        cdir = canonical_dir()
        if cdir:
            for p in sorted(glob.glob(os.path.join(cdir, "db_*.db"))):
                nn = os.path.basename(p)[3:-3]
                if not want(nn):
                    continue
                c = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
                rows = [(e, a * PSNR_SCALE, _prm(c, pr)) for e, a, pr in c.execute(
                    "SELECT epoch, accuracy, prm FROM stat WHERE nn=? AND task=? "
                    "AND dataset=? AND metric=? AND accuracy IS NOT NULL",
                    (nn, TASK, DATASET, METRIC))]
                c.close()
                if rows:
                    stats[nn] = rows
    elif seed_set == "canonical":
        cdir = canonical_dir()
        if not cdir:
            raise SystemExit("canonical sweep dir not found; pass --seed-set all to "
                             "fall back to the master DB (see the docstring first)")
        for p in sorted(glob.glob(os.path.join(cdir, "db_*.db"))):
            nn = os.path.basename(p)[3:-3]
            con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
            for ep, acc, prm in con.execute(
                    "SELECT epoch, accuracy, prm FROM stat WHERE nn=? AND task=? "
                    "AND dataset=? AND metric=? AND accuracy IS NOT NULL",
                    (nn, TASK, DATASET, METRIC)):
                stats.setdefault(nn, []).append((ep, acc * PSNR_SCALE, _prm(con, prm)))
            con.close()
    else:
        dbp = db_path()
        if not dbp:
            raise SystemExit("master DB not found")
        con = sqlite3.connect(f"file:{dbp}?mode=ro", uri=True)
        for nn, ep, acc, prm in con.execute(
                "SELECT nn, epoch, accuracy, prm FROM stat "
                "WHERE task=? AND dataset=? AND metric=? AND accuracy IS NOT NULL",
                (TASK, DATASET, METRIC)):
            stats.setdefault(nn, []).append((ep, acc * PSNR_SCALE, _prm(con, prm)))
        con.close()

    edge_ok = _edge_checker()
    out, dropped, failed = [], [], []
    for name, rows in stats.items():
        if names and name not in names:
            continue
        fp = os.path.join(nd, f"{name}.py")
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            code = f.read()
        if edge_ok and not edge_ok(code, name):
            dropped.append(name)
            continue
        rows = sorted(rows)
        tail = rows[-5:]
        db = sum(r[1] for r in tail) / len(tail)
        if db < MIN_USABLE_DB:
            failed.append((name, db))
            continue
        out.append({"name": name, "code": code, "db": db,
                    "epoch": rows[-1][0], "prm": rows[-1][2]})
    if dropped:
        print(f"[seeds] dropped {len(dropped)} over-cap seed(s): {', '.join(sorted(dropped))}")
    if failed:
        print(f"[seeds] dropped {len(failed)} failed run(s) below {MIN_USABLE_DB:.0f} dB: "
              + ", ".join(f"{n} ({d:.1f})" for n, d in sorted(failed)))
    out.sort(key=lambda r: -r["db"])
    return out[:limit] if limit else out


def db_corpus(min_db=None, mac_cap=9053.0):
    """Every edge-legal, architecturally-distinct model in the master DB, code read from
    `nn.code` -- the wide reference pool.

    WHY THIS EXISTS AND WHY seed_set='all' IS NOT IT. `all` reads source from
    ab/nn/nn/<name>.py, so it only ever sees models that exist as FILES: ~168 of them, and
    not one GA or LLM descendant, because those live only in the database. The reference
    pool it can build is therefore a rounding error next to the corpus we have actually
    produced.

    That mattered more than it sounds. Sixteen generations ran with --seed-set elite, whose
    pool is --k 8, and the four hand-written anchors in it (35.1-35.2 dB) were never once
    displaced -- the best generated model reached 33.90 dB. Combined with the deterministic
    `corpus[i % len(corpus)]` reference pick, 2400 prompts were issued and only 140 of them
    were distinct; the most-repeated was sent 48 times. 34% of candidates were then thrown
    out by the diversity gate as near-duplicates and another 16% as already-in-dataset. The
    generator was not failing to be creative. It was answering the same 140 questions.

    dB IS NOT COMPARABLE ACROSS THIS POOL and must not be used to rank it directly. The
    LEMUR config key is (task, dataset, metric, nn, epoch_max), so a 50-epoch hand-written
    seed and a 5-epoch GA descendant sit in the same table with no column distinguishing
    the budget they were given. Ranking on raw dB would put every 50-epoch model above
    every 5-epoch one and hand tournament selection the old 8-model pool right back. So we
    rank each model by PERCENTILE WITHIN ITS OWN epoch_max COHORT and select on that. The
    dB and epoch shown in the prompt stay the true measured ones -- the normalisation
    affects which models get chosen, never what the model is told about them.
    """
    dbp = db_path()
    if not dbp:
        raise SystemExit("master DB not found")
    con = sqlite3.connect(f"file:{dbp}?mode=ro", uri=True)
    have_uid = any(r[1] == "arch_uid" for r in con.execute("PRAGMA table_info(nn)"))

    # Best row per (model, epoch_max cohort). accuracy*48 = real dB.
    rows = con.execute(
        "SELECT s.nn, s.epoch, MAX(s.accuracy), s.prm FROM stat s "
        "WHERE s.task=? AND s.dataset=? AND s.metric=? AND s.accuracy IS NOT NULL "
        "GROUP BY s.nn, s.epoch", (TASK, DATASET, METRIC)).fetchall()
    src = {n: (c, u) for n, c, u in con.execute(
        "SELECT name, code, %s FROM nn" % ("arch_uid" if have_uid else "NULL"))}

    cand = {}
    for nn, ep, acc, prm in rows:
        if nn not in src or not src[nn][0]:
            continue
        rec = {"name": nn, "code": src[nn][0], "db": acc * PSNR_SCALE,
               "epoch": ep, "prm": _prm(con, prm), "arch": src[nn][1]}
        # keep the model's best cohort
        if nn not in cand or rec["db"] > cand[nn]["db"]:
            cand[nn] = rec
    con.close()

    floor = MIN_USABLE_DB if min_db is None else min_db
    pool = [r for r in cand.values() if r["db"] >= floor]
    below = len(cand) - len(pool)

    # Percentile rank inside the epoch_max cohort -- see the docstring.
    cohorts = {}
    for r in pool:
        cohorts.setdefault(r["epoch"], []).append(r)
    # A cohort needs members before a percentile within it means anything. 1026 of the 1044
    # models ran epoch_max=5; the rest are scattered one-per-budget across ep1, ep31, ep43,
    # ep48, ep49. Ranking those by position in their own cohort gives every one of them
    # rank 1.0 -- "best in cohort" when the cohort is itself -- so they would win every
    # tournament they entered, and one of them is a single-epoch row measuring 37.4 dB.
    # Small cohorts get the median instead: they compete, they do not dominate.
    MIN_COHORT = 5
    for ep, group in cohorts.items():
        group.sort(key=lambda r: r["db"])
        n = len(group)
        if n < MIN_COHORT:
            for r in group:
                r["rank"] = 0.5
            continue
        for i, r in enumerate(group):
            r["rank"] = (i + 1) / n

    # ONE ENTRY PER ARCHITECTURE. Without this the pool inherits the corpus's isomorphic
    # duplicates and tournament selection over-samples whichever architecture happens to
    # have the most lexical copies -- reintroducing, in a subtler form, exactly the
    # concentration this function was written to remove.
    if have_uid:
        by_arch, no_uid = {}, []
        for r in pool:
            if not r["arch"]:
                no_uid.append(r)
                continue
            k = r["arch"]
            if k not in by_arch or r["rank"] > by_arch[k]["rank"]:
                by_arch[k] = r
        deduped = list(by_arch.values()) + no_uid
        collapsed = len(pool) - len(deduped)
        pool = deduped
    else:
        collapsed = 0
        print("[seeds] WARNING: nn.arch_uid absent -- pool NOT deduplicated by architecture")

    # MAC GATE, CACHED BY CODE HASH. Measuring costs a build plus a profiled forward per
    # model. Over a ~1100-model pool that is ~15 minutes, and db_corpus runs in all four
    # generation shards on every generation -- an hour of GPU time per generation spent
    # re-deriving a constant. The verdict depends only on the source, so it caches perfectly.
    # Shared file, atomic replace: concurrent shards may duplicate work the first time and
    # never again.
    # Deliberately NOT in the DB directory: that is inside the nn-dataset git worktree, and
    # a generated cache file does not belong in the dataset repo.
    cache_p = _resolve("/a/mm/sweep_out/manifest",
                       "/shared/ssd/home/b-b-ahmed/thesis/sweep_out/manifest") \
        or os.path.dirname(dbp)
    cache_p = os.path.join(cache_p, "mac_gate_cache.json")
    cache = {}
    try:
        with open(cache_p) as f:
            cache = json.load(f)
    except Exception:
        pass

    edge_ok = _edge_checker()
    if edge_ok:
        legal, computed = [], 0
        for r in pool:
            k = hashlib.sha256(r["code"].encode()).hexdigest()[:32]
            if k not in cache:
                cache[k] = bool(edge_ok(r["code"], r["name"]))
                computed += 1
            if cache[k]:
                legal.append(r)
        over = len(pool) - len(legal)
        pool = legal
        if computed:
            try:
                tmp = f"{cache_p}.{os.getpid()}"
                with open(tmp, "w") as f:
                    json.dump(cache, f)
                os.replace(tmp, cache_p)
            except OSError as e:
                print(f"[seeds] MAC cache not written ({e}); next run will re-measure")
        print(f"[seeds] MAC gate: {computed} measured, {len(pool) + over - computed} cached")
    else:
        over = 0

    pool.sort(key=lambda r: -r["db"])
    print(f"[seeds] db pool: {len(pool)} references "
          f"(dropped {below} below {floor:.0f} dB, {collapsed} isomorphic duplicates, "
          f"{over} over {mac_cap:.0f} MMac) across {len(cohorts)} epoch cohorts")
    return pool


def _by_arm(corpus):
    """Split the pool by producing arm, so sampling can be balanced across them.

    DnSeedGA IS ITS OWN ARM. It used to be bucketed with DenoiseGA under a single "GA" key, and
    that quietly starved the best producer we have. Arm-balanced sampling draws roughly one
    reference per arm, so a merged GA slot was shared between 2,049 DenoiseGA models and 376
    DnSeedGA ones -- DnSeedGA won that internal lottery about 15% of the time. Measured over the
    1,752-candidate production run: DnSeedGA supplied only 309 of 7,008 reference uses (4.4%),
    less than its 12.7% share of the corpus, while 62% of uses pointed at the two arms with the
    worst deployability.

    The arms are not interchangeable, which is the whole point of balancing:

        arm         p50 latency   NPU-safe   zero-shot SIDD
        DnSeedGA        1.52 ms        80%    31.5-32.6 dB
        hand            3.85 ms        54%    23.7-24.1 dB
        DenoiseGA      32.00 ms        39%    --
        LLM            42.94 ms        22%    25.7-27.3 dB

    So the generator was being shown slow, off-NPU designs most of the time and the one
    simultaneously-accurate-and-deployable arm almost never. The comment on pick_references
    records that arm balancing was introduced to stop "the LLM imitating the GA"; merging both GA
    arms meant it went on imitating the WRONG GA.

    Splitting them raises DnSeedGA's expected share of reference uses from ~4.4% to ~25%.
    """
    out = {}
    for r in corpus:
        n = r["name"]
        arm = ("LLM" if (n.startswith("LLM_") or "_seeded_" in n)
               # the seeded-GA arm first: DnSeedGA- would otherwise fall into the GA bucket below.
               # DENOISE_MERGE_GA_ARMS=1 restores the old merged behaviour, so the A/B control and
               # treatment are the same binary differing only by this env var.
               else "GAseed" if (n.startswith("DnSeedGA-")
                                 and os.environ.get("DENOISE_MERGE_GA_ARMS") != "1")
               else "GA" if n.startswith(("DenoiseGA-", "DnSeedGA-", "DenoiseSkip-",
                                          "DnSkip-", "DnNewFam-", "DnSearch-"))
               else "hand")
        out.setdefault(arm, []).append(r)
    return out


def pick_references(corpus, i, run_id, shard, k_supporting, tournament=3,
                    elite_share=0.25, elites=None, arm_balance=True):
    """(primary, supporting) for candidate i -- TOURNAMENT SELECTION, not modular cycling.

    The old line was `primary = corpus[i % len(corpus)]` with supporting picked by the same
    arithmetic. That makes the whole prompt a pure function of i mod a small number: with an
    8-model pool and 4 strategies the entire prompt space was 140 strings, re-issued 2400
    times over 16 generations. Temperature was the only thing varying, and asking one model
    the same question 48 times is not a search.

    Tournament-of-3 is the GA's own selection operator, and it is used here for the same
    reason the GA uses it: it keeps selection pressure toward good references without
    collapsing onto the single best one, which is what an argmax (or a frozen elite list)
    does. Ranking is by within-cohort percentile, so a 5-epoch descendant can win against a
    50-epoch seed on merit rather than on budget.

    elite_share keeps a fraction of prompts anchored on the measured best models, so the
    wide pool broadens the search without abandoning the quality signal that made the
    seeded arm beat the cold arm 24-to-2 on the PSNR gate.

    Seeded on (run_id, shard, i): stochastic across candidates, exactly reproducible from
    the saved metadata. A wall-clock seed would have made prompts unreplayable, which is
    the failure this whole script was written to avoid.
    """
    import random
    rng = random.Random(f"{run_id}|{shard}|{i}")

    # ARM-BALANCED SAMPLING. Uniform sampling from the corpus means 87% GA references,
    # because that is the corpus composition -- and the LLM duly produced GA-like models.
    # Measured on generation 1 of the 0.90 run: of 31 near-duplicate rejections that named
    # their match, 24 (77%) collided with a GA model. The wide pool fixed the LLM repeating
    # ITSELF and replaced it with the LLM imitating the GA.
    #
    # Drawing one reference per arm where possible exposes it to hand-written and LLM designs
    # at rates the raw corpus never would, without narrowing the pool back down.
    pools = _by_arm(corpus) if arm_balance else {"all": corpus}

    if elites and rng.random() < elite_share:
        primary = rng.choice(elites)
    else:
        src = rng.choice([v for v in pools.values() if v]) if arm_balance else corpus
        primary = max(rng.sample(src, min(tournament, len(src))),
                      key=lambda r: r.get("rank", 0.0))

    sup, used = [], {primary["name"]}
    if arm_balance:
        # round-robin the arms so a 3-reference prompt spans up to 3 different producers.
        # GAseed MUST be listed: _by_arm now separates DnSeedGA from DenoiseGA, and an arm missing
        # from this tuple is silently never drawn for a supporting reference no matter how good it
        # is. Order is shuffled below, so listing it first carries no bias.
        arms = [k for k in ("GAseed", "hand", "LLM", "GA") if pools.get(k)]
        rng.shuffle(arms)
        for k in arms:
            if len(sup) >= k_supporting:
                break
            cand = [r for r in pools[k] if r["name"] not in used]
            if cand:
                pick = rng.choice(cand)
                sup.append(pick)
                used.add(pick["name"])
    rest = [r for r in corpus if r["name"] not in used]
    rng.shuffle(rest)
    sup += rest[:max(0, k_supporting - len(sup))]
    return primary, sup


def _edge_checker():
    """run_seed_ga._seed_is_edge_legal, or None if the cost backend is genuinely absent.

    _seed_is_edge_legal FAILS OPEN by design -- if `import edge_cost` raises or the cost
    is unmeasurable it returns True, so a thop hiccup never empties the seed pool. That is
    reasonable inside the GA, where it logs the reason. It is a trap here: edge_cost.py
    lives in THIS directory and is only importable when that directory is on sys.path, so
    calling the checker from elsewhere silently returned True for every seed. The first
    run of this script duly reported "0 over-cap" out of 36 -- not because the seeds were
    legal, but because the cap was never applied. Probe the backend explicitly instead of
    trusting a True.
    """
    # HERE -> edge_cost; the ast tree -> the `mutator` package that measure_cost imports
    # lazily. Without the second path, `import edge_cost` succeeds and every measurement
    # then dies with "No module named 'mutator'". run_seed_ga.py adds both (lines 31-32).
    ast_dir = _resolve("/a/mm/nn-gpt/ab/gpt/brute/ast",
                       os.path.join(NNGPT_ROOT, "ab", "gpt", "brute", "ast"))
    for p in (HERE, NNGPT_ROOT, ast_dir):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    try:
        import edge_cost as EC
        from ab.gpt.brute.ga.denoise.run_seed_ga import _seed_is_edge_legal
    except Exception as e:
        print(f"[seeds] MAC CAP NOT ENFORCED: cost backend unimportable "
              f"({type(e).__name__}: {e}).")
        return None

    # Importing edge_cost is not enough: measure_cost() needs torch and thop at CALL time.
    # On the submit node neither is installed, so every seed took the fail-open path and
    # emitted a "keeping, fail-open" line -- 19 lines that read like a gate had run. Probe
    # once with a trivial module and, if the backend cannot actually measure, say so a
    # single time and disable the check honestly.
    try:
        probe = EC.measure_cost(
            "import torch.nn as nn\n"
            "class Net(nn.Module):\n"
            "    def __init__(self, in_shape=None, out_shape=None, prm={}, device='cpu'):\n"
            "        super().__init__()\n"
            "        self.c = nn.Conv2d(3, 3, 3, padding=1)\n"
            "    def forward(self, x):\n"
            "        return self.c(x)\n")
        working = bool(probe.get("ok"))
    except Exception as e:
        probe, working = {"err": f"{type(e).__name__}: {e}"}, False

    if not working:
        print(f"[seeds] MAC CAP NOT ENFORCED here: cost backend present but non-functional "
              f"({probe.get('err') or probe}). This node has no torch/thop; run generation "
              f"in-container to get the 9053 MMac gate. Seeds pass through unfiltered.")
        return None
    return lambda code, name: _seed_is_edge_legal(
        code, name, 9053.0, lambda m: print(f"[seeds]{m}"))


def render(conf, mode, primary=None, supporting=()):
    system = "\n".join(conf.get("system", []))
    body = "\n".join(conf["prompt"])
    if mode == "cold":
        return system, body

    sup = []
    for i, s in enumerate(supporting, 1):
        sup.append(f"SUPPORTING REFERENCE {i} - measured {s['db']:.2f} dB PSNR:\n"
                   f"```python\n{s['code']}\n```\n")
    body = body.replace("{supporting_models_prompt}",
                        "\n".join(sup) if sup else "(no supporting references this round)")
    body = (body.replace("{nn_code}", primary["code"])
                .replace("{accuracy_db}", f"{primary['db']:.2f}")
                .replace("{epoch}", str(primary["epoch"]))
                .replace("{prm}", str(primary["prm"])))
    return system, body


def extract_nn(text):
    m = re.search(r"<nn>(.*?)</nn>", text, re.S)
    code = m.group(1) if m else None
    if code is None:                       # tolerate a fenced block if tags were dropped
        m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
        code = m.group(1) if m else None
    if not code:
        return None
    code = code.strip()
    # The model sometimes nests markdown fences INSIDE the <nn> tags despite the prompt
    # saying "no markdown fences". Taking the tag contents verbatim then wrote a
    # candidate.py whose line 1 was "```python" -> SyntaxError at load. That was 2 of the
    # 8 build failures in run 20260730-220434, i.e. our bug, not the model's.
    if code.startswith("```"):
        code = re.sub(r"^```[a-zA-Z]*\n", "", code)
        code = re.sub(r"\n```\s*$", "", code)
        code = code.strip()
    return code if "class Net" in code else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cold", "seeded", "both"], default="both")
    ap.add_argument("--run-id", default=None,
                    help="pin the run id, which is ALSO the reference-selection seed. Pass the "
                         "same value to several runs to make candidate i see identical primary "
                         "and supporting references in each -- required for the deployability "
                         "prompt ablation to be paired rather than confounded by seed draw")
    ap.add_argument("--deploy-guidance",
                    choices=["on", "off", "placebo", "v2", "v2-placebo"], default="on",
                    help="include the DEPLOYABILITY block (rules 10-12: no AdaptiveMaxPool2d, no "
                         "self-attribute assignment in forward, avoid fft/amax). Default on. "
                         "'off' and 'placebo' are the control arms of the prompt ablation; "
                         "placebo substitutes length-matched neutral text so an on-vs-off "
                         "difference can be attributed to content rather than prompt length")
    ap.add_argument("--n", type=int, default=10, help="candidates per arm")
    ap.add_argument("--supporting", type=int, default=3)
    ap.add_argument("--llm", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--seeds", nargs="*", default=None, help="restrict the seed corpus")
    ap.add_argument("--seed-set",
                    choices=["named", "seeds23", "canonical", "all", "elite", "db"],
                    default="named",
                    help="named (default) = the 23 hand-written seeds UNION the 18 "
                         "families we built; seeds23 = seeds only; canonical = the "
                         "swept families under one recipe; all = master DB but source "
                         "read from disk files, so ~168 refs and no generated model; "
                         "elite = the k=8 pool from elite_archive.py; db = EVERY "
                         "edge-legal architecturally-distinct model in the master DB "
                         "(~1000), selected by tournament -- use this one")
    # The elite pool is the LLM's selection pressure. Without it every prompt is drawn
    # i.i.d. and the run cannot improve across its own samples, however many are drawn.
    #
    # With --seed-set db the elites are still loaded when --elite-file is given, but they
    # become an ANCHOR rather than the entire pool: --elite-share of prompts use one as the
    # primary reference and the rest come from the wide pool. Selection pressure without
    # the 140-distinct-prompt collapse that k=8 produced.
    ap.add_argument("--elite-file", default=None,
                    help="JSON from elite_archive.py; required for --seed-set elite, "
                         "optional anchor for --seed-set db")
    # THE GATE'S VERDICT, FED BACK. Without this the loop is open: ~32 candidates a
    # generation are refused as near-duplicates and the generator is never told, so it
    # proposes the same shape again. Produced by rejection_digest.py from the PREVIOUS
    # generation's preflight.json.
    ap.add_argument("--feedback-file", default=None,
                    help="prompt block describing last generation's rejections")
    ap.add_argument("--tournament", type=int, default=3,
                    help="tournament size for reference selection under --seed-set db")
    ap.add_argument("--elite-share", type=float, default=0.25,
                    help="fraction of prompts anchored on an elite as primary reference")
    ap.add_argument("--out", default=None)
    # SHARDING. Generation was 2 h 02 m of generation 1's 4 h 33 m -- 45% of the cycle --
    # and it is the one stage that parallelises for free: 150 independent samples, no shared
    # state, and NOT ONE database write. (The train stage is 41% but writes the LEMUR SQLite
    # file on NFSv3, where concurrent writers corrupt rather than block, so it cannot simply
    # be forked the same way.) Shards interleave by index rather than taking contiguous
    # blocks, so each still sees the full strategy round-robin and one shard's output is a
    # uniform sample of the whole generation, not a slice of it.
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    # Candidate ids are index-based, so every generation independently produces
    # `0026_seeded_direct_instruction`. That collision paired one generation's training curve
    # with another's source code in the elite archive and silently dropped our two cheapest
    # models. elite_archive.py now disambiguates by path, but that is a workaround; a tag per
    # run makes the ids unique at the source, which is where it should have been.
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter dir to apply on top of --llm (run-3 fine-tuned weights)")
    ap.add_argument("--id-prefix", default="",
                    help="prepended to every candidate id, e.g. 'g3' -> g3_0026_seeded_...")
    ap.add_argument("--dry-run", action="store_true",
                    help="render and SAVE prompts without loading the model")
    ap.add_argument("--check-contract", action="store_true",
                    help="verify the shared blocks are identical, then exit")
    a = ap.parse_args()

    cold_conf, cold_sha, cold_p = load_conf(COLD_CONF)
    seeded_conf, seeded_sha, seeded_p = load_conf(SEEDED_CONF)
    shared, shared_sha = load_shared()
    cold_strats, cold_strat_sha = load_strategies(COLD_STRAT_CONF)
    seeded_strats, seeded_strat_sha = load_strategies(SEEDED_STRAT_CONF)
    print(f"[strategies] cold: {len(cold_strats)} "
          f"({sum(len(v['variants']) for v in cold_strats.values())} variants)  "
          f"seeded: {len(seeded_strats)} "
          f"({sum(len(v['variants']) for v in seeded_strats.values())} variants)")

    drift = check_contract(cold_conf, seeded_conf)
    if drift:
        print(f"[contract] SHARED BLOCKS DIFFER between arms: {', '.join(drift)}")
        print("[contract] the ablation would measure wording, not conditioning. Fix first.")
        if a.check_contract:
            return 1
    else:
        print(f"[contract] OK - {len(SHARED_BLOCKS)} shared blocks identical across arms")
    if a.check_contract:
        return 0 if not drift else 1

    # PINNABLE, BECAUSE run_id IS THE REFERENCE-SELECTION SEED. pick_references seeds its RNG on
    # f"{run_id}|{shard}|{i}", so a wall-clock run_id means two runs launched a second apart draw
    # DIFFERENT primary and supporting models for candidate i. For the deployability prompt
    # ablation that is a confound, not just noise: the arms would differ in which reference
    # architectures the LLM was shown, on top of the one block of text they are supposed to differ
    # in -- and references are the most likely place a banned operator gets copied from.
    #
    # Passing the same --run-id to every arm makes candidate i see the identical references in all
    # three, which turns an unpaired between-arm comparison into a PAIRED one at no cost.
    run_id = a.run_id or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    root = a.out or os.path.join(HERE, "llm_out", run_id)
    # Per-shard sidecars. Shards share `root` (so cand/ is one directory) but must not share
    # prompts.jsonl or run_meta.json: concurrent appends to one file on NFS interleave
    # partial lines and the last writer wins the meta.
    _sfx = f"_shard{a.shard}" if a.num_shards > 1 else ""
    os.makedirs(root, exist_ok=True)

    def _load_elites(path):
        with open(path) as _f:
            pool = json.load(_f)
        # Fail loudly rather than silently generating an unconditioned run: an empty or
        # malformed elite file would otherwise degrade into a cold run wearing a seeded
        # label, and the generation-over-generation comparison would be meaningless.
        need = {"name", "code", "db", "epoch", "prm"}
        for _e in pool:
            _e.setdefault("name", _e.get("id", "elite"))
            missing = need - set(_e)
            if missing:
                raise SystemExit(f"elite entry {_e.get('name')} missing {sorted(missing)}")
        print(f"[elite] {len(pool)} references from {path} "
              f"(best {max(e['db'] for e in pool):.3f} dB)")
        return pool

    elites = None
    if a.seed_set == "elite":
        if not a.elite_file:
            raise SystemExit("--seed-set elite requires --elite-file "
                             "(produce it with elite_archive.py)")
        corpus = _load_elites(a.elite_file)
    elif a.seed_set == "db":
        corpus = db_corpus() if a.mode in ("seeded", "both") else []
        if a.elite_file and os.path.exists(a.elite_file):
            elites = _load_elites(a.elite_file)
            for _e in elites:
                _e.setdefault("rank", 1.0)
        else:
            print("[elite] no elite file -- all references drawn from the db pool")
    else:
        corpus = (seed_corpus(a.seeds, seed_set=a.seed_set)
                  if a.mode in ("seeded", "both") else [])
    if a.mode in ("seeded", "both") and not corpus:
        raise SystemExit("seeded arm requested but no denoise seeds with stats were found")

    feedback = ""
    if a.feedback_file and os.path.exists(a.feedback_file):
        with open(a.feedback_file) as _f:
            feedback = _f.read().strip()
        print(f"[feedback] {len(feedback)} chars from {a.feedback_file}")

    meta = {
        "run_id": run_id, "argv": sys.argv, "mode": a.mode, "n_per_arm": a.n,
        "llm": a.llm, "dry_run": a.dry_run, "seed_set": a.seed_set,
        "sampling": {"temperature": a.temperature, "top_k": a.top_k,
                     "top_p": a.top_p, "max_new_tokens": a.max_new_tokens},
        "configs": {SHARED_CONF: {"sha256": shared_sha},
                    COLD_STRAT_CONF: {"sha256": cold_strat_sha,
                                      "strategies": sorted(cold_strats)},
                    SEEDED_STRAT_CONF: {"sha256": seeded_strat_sha,
                                        "strategies": sorted(seeded_strats)},
                    COLD_CONF: {"path": cold_p, "sha256": cold_sha, "legacy": True},
                    SEEDED_CONF: {"path": seeded_p, "sha256": seeded_sha, "legacy": True}},
        "contract_drift": drift,
        "seed_corpus": [{"name": s["name"], "db": round(s["db"], 3)} for s in corpus],
        "git": {"nn-gpt": git_rev(NNGPT_ROOT),
                "nn-dataset": git_rev(os.path.join(NNGPT_ROOT, "..", "nn-dataset"))},
        "started": datetime.datetime.now().isoformat(),
    }
    with open(os.path.join(root, f"run_meta{_sfx}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[run] {root}  ({len(corpus)} seeds in corpus)")

    gen = None
    if not a.dry_run:
        sys.path.insert(0, NNGPT_ROOT)
        from ab.gpt.util.LLM import LLM
        import torch
        loader = LLM(a.llm, load_in_4bit=True)
        model, tok = loader.get_model(), loader.get_tokenizer()
        if a.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, a.adapter)
            print(f"[llm] adapter applied from {a.adapter}")
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

        def gen(system, user):
            msgs = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": user}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                          return_tensors="pt").to(model.device)
            with torch.no_grad():
                model.eval()
                out = model.generate(ids, max_new_tokens=a.max_new_tokens, do_sample=True,
                                     temperature=a.temperature, top_k=a.top_k,
                                     top_p=a.top_p, pad_token_id=tok.pad_token_id)
            return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    index = open(os.path.join(root, f"prompts{_sfx}.jsonl"), "a")
    arms = ["cold", "seeded"] if a.mode == "both" else [a.mode]
    made = {}

    for arm in arms:
        strats = cold_strats if arm == "cold" else seeded_strats
        keys = sorted(strats)
        ok = 0
        for i in range(a.n):
            # CONTIGUOUS BLOCKS, not i % num_shards. The obvious interleave aliases with
            # the strategy round-robin two lines below: strategies are picked by
            # i % len(keys), so with 4 shards and 4 seeded strategies EVERY candidate in
            # shard 2 was direct_instruction. Verified, not theorised. One evicted shard
            # would then delete an entire strategy from the generation and confound the
            # per-strategy comparison with which GPU it landed on. A contiguous block spans
            # the round-robin regardless of how the two counts divide.
            if a.num_shards > 1:
                lo = a.shard * a.n // a.num_shards
                hi = (a.shard + 1) * a.n // a.num_shards
                if not (lo <= i < hi):
                    continue
            # Round-robin the strategies so each gets ~n/len(keys) samples, then walk the
            # variants within a strategy. Deterministic, so (i, arm) alone replays the
            # exact prompt from the saved metadata.
            sk = keys[i % len(keys)]
            variants = strats[sk]["variants"]
            vi = (i // len(keys)) % len(variants)
            if arm == "cold":
                primary, sup = None, ()
            elif a.seed_set == "db":
                primary, sup = pick_references(
                    corpus, i, run_id, a.shard, a.supporting,
                    tournament=a.tournament, elite_share=a.elite_share, elites=elites)
            else:
                # Legacy modular cycling. Kept only so the older seed-sets replay exactly;
                # it is the reason 2400 prompts collapsed to 140 distinct ones, so anything
                # generating at volume should use --seed-set db.
                primary = corpus[i % len(corpus)]
                pool = [s for s in corpus if s["name"] != primary["name"]]
                sup = [pool[(i + j) % len(pool)] for j in range(min(a.supporting, len(pool)))]

            system, user = compose(shared, variants[vi], arm, primary, sup,
                                   deploy={"on": True, "off": False,
                                           "placebo": "placebo", "v2": "v2",
                                           "v2-placebo": "v2-placebo"}[a.deploy_guidance])
            # Prepended, not appended: the constraint has to be read before the references,
            # or the model anchors on the examples and treats the feedback as an afterthought.
            if feedback:
                user = feedback + "\n\n" + user
            cid = f"{a.id_prefix}_{i:04d}_{arm}_{sk}" if a.id_prefix \
                else f"{i:04d}_{arm}_{sk}"
            cdir = os.path.join(root, "cand", cid)
            os.makedirs(cdir, exist_ok=True)
            with open(os.path.join(cdir, "prompt_system.txt"), "w") as f:
                f.write(system)
            with open(os.path.join(cdir, "prompt_user.txt"), "w") as f:
                f.write(user)

            rec = {
                "id": cid, "arm": arm, "run_id": run_id,
                "config": COLD_STRAT_CONF if arm == "cold" else SEEDED_STRAT_CONF,
                "config_sha256": cold_strat_sha if arm == "cold" else seeded_strat_sha,
                "shared_sha256": shared_sha,
                "strategy": sk, "strategy_label": strats[sk].get("strategy"),
                "variant_idx": vi,
                # THE JOIN KEY FOR THE DEPLOYABILITY ABLATION. The prompt style is recoverable
                # from the candidate id, but this flag is not, and ablation_prompt.py already
                # documents what happens when a factor is only knowable from the run that
                # produced it: the analysis becomes impossible after the fact. Recorded per
                # candidate rather than once per run so a mixed run is still analysable.
                "deploy_guidance": a.deploy_guidance,
                "llm": a.llm, "adapter": a.adapter, "sampling": meta["sampling"],
                "prompt_sha256": sha256(system + "\n" + user),
                "prompt_chars": len(system) + len(user),
                "primary_seed": primary["name"] if primary else None,
                "supporting_seeds": [s["name"] for s in sup],
                "ts": datetime.datetime.now().isoformat(),
            }

            if gen is not None:
                resp = gen(system, user)
                with open(os.path.join(cdir, "response_raw.txt"), "w") as f:
                    f.write(resp)
                code = extract_nn(resp)
                rec["response_chars"] = len(resp)
                rec["extracted"] = bool(code)
                if code:
                    with open(os.path.join(cdir, "candidate.py"), "w") as f:
                        f.write(code)
                    rec["code_sha256"] = sha256(code)
                    ok += 1
            else:
                rec["extracted"] = None

            with open(os.path.join(cdir, "meta.json"), "w") as f:
                json.dump(rec, f, indent=2)
            index.write(json.dumps(rec) + "\n")
            index.flush()
            print(f"  [{cid}] prompt {rec['prompt_chars']:6d} ch  "
                  f"sha {rec['prompt_sha256'][:12]}  "
                  + ("saved (dry-run)" if gen is None
                     else ("extracted" if rec.get("extracted") else "NO CODE")))
        made[arm] = ok

    index.close()
    meta["finished"] = datetime.datetime.now().isoformat()
    meta["extracted_per_arm"] = made
    with open(os.path.join(root, f"run_meta{_sfx}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[done] {root}")
    if a.dry_run:
        print("       prompts saved; re-run without --dry-run to generate")
    else:
        for k, v in made.items():
            print(f"       {k}: {v}/{a.n} candidates yielded parseable code")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
