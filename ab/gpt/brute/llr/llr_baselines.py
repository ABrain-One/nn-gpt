"""
Shared read-only helpers for the layerwise-LR (LLR) fine-tuning pipeline.

Single source of truth for:
  - locating the LEMUR sqlite DB
  - loading the strategy-metadata CSV (hash -> architecture/strategy)
  - loading vanilla-architecture code and accuracy (uniform-control preferred,
    DB-vanilla fallback), transform-matched
  - the full strategy catalog (all 39 strategies, including never-evaluated
    ones) from the three brute-force generator modules

Used by:
  - ab.gpt.util.prompt.NNGenPrompt   (SFT training-pair construction)
  - ab.gpt.util.Tune                 (generation-time candidate sampling)
  - ab.gpt.brute.llr.analyze_llr_vs_vanilla (offline analysis)
"""

import csv
import re
import sqlite3
from pathlib import Path

# Some non-LLR generated/synthetic models are stored under bare 32-char hex
# hashes with no separating hyphen (unlike llr-<hash>), so they slip past a
# plain "no '-' in name" filter. Exclude them from the vanilla-architecture pool.
_HEX_HASH_RE = re.compile(r'^[0-9a-f]{32}$')

# Strategy-metadata columns carried alongside a strategy spec.
LLR_META_COLS = ('strategy', 'strategy_type', 'n_groups', 'multipliers',
                  'split_ratios', 'description', 'architecture')


def db_path() -> str | None:
    """Locate the LEMUR sqlite DB (local checkout or cluster mount)."""
    cand = Path(__file__).resolve().parents[3] / 'db' / 'ab.nn.db'
    if cand.exists():
        return str(cand)
    for p in ('/a/mm/db/ab.nn.db',):
        if Path(p).exists():
            return p
    return None


_METADATA_CSV_FIELDS = ('nn', 'architecture', 'strategy', 'strategy_type', 'n_groups',
                        'multipliers', 'split_ratios', 'dataset', 'task', 'metric',
                        'transform', 'description', 'meta_source')


def metadata_csv_path() -> Path:
    return Path(__file__).resolve().parents[3] / 'out' / 'nngpt' / 'llr_evaluated_with_meta.csv'


def load_llr_metadata() -> dict:
    """
    Load llr_evaluated_with_meta.csv into a dict keyed by nn name (the
    prefix-hash DB name, e.g. 'llr-<md5>'). Returns {} if the file is missing.
    """
    csv_path = metadata_csv_path()
    if not csv_path.exists():
        return {}
    meta = {}
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            meta[row['nn']] = row
    return meta


def record_llm_generated_provenance(nn_name: str, model_dir, dataset: str, task: str,
                                     metric: str, transform: str) -> bool:
    """
    Append a metadata row for an LLM-GENERATED (not brute-force) evaluated model,
    so it becomes eligible for the same vanilla-vs-LLR matching (training pairs,
    analyze_llr_vs_vanilla) that brute-force models already get via
    llr_evaluated_with_meta.csv.

    Brute-force provenance is recovered by re-generating and checksum-matching
    (build_metadata_csv.py) because the generator is deterministic. LLM output
    isn't reproducible that way, so instead we recover provenance from the
    generation-time `dataframe.df` still sitting in the model's synth_nn
    directory at evaluation-success time — it was written by
    llr_generation.py's samplers and carries the source architecture ('nn') plus
    the target strategy spec ('strategy_2', 'multipliers_2', ...).

    No-ops (returns False) if dataframe.df is missing or lacks 'strategy_2' —
    i.e. this model didn't come from our LLR generation path (brute-force rows,
    or any other NNEval consumer, are left untouched).
    """
    import pandas as pd

    df_path = Path(model_dir) / 'dataframe.df'
    if not df_path.exists():
        return False
    try:
        row = pd.read_pickle(df_path)
    except Exception:
        return False
    if 'strategy_2' not in row or not row.get('strategy_2'):
        return False

    def _s(v):
        return str(v) if v is not None else ''

    record = {
        'nn': nn_name,
        'architecture': _s(row.get('nn')),
        'strategy': _s(row.get('strategy_2')),
        'strategy_type': _s(row.get('strategy_type_2')),
        'n_groups': _s(row.get('n_groups_2')),
        'multipliers': _s(row.get('multipliers_2')),
        'split_ratios': _s(row.get('split_ratios_2')),
        'dataset': _s(dataset),
        'task': _s(task),
        'metric': _s(metric),
        'transform': _s(transform),
        'description': _s(row.get('description_2')),
        'meta_source': 'llm_generated',
    }

    csv_path = metadata_csv_path()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_METADATA_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(record)
    return True


def load_all_strategies() -> list[dict]:
    """
    Full strategy catalog (39 entries: 12 + 13 + 14, including llr_uniform),
    from the three generator modules' LAYERWISE_STRATEGIES definitions —
    independent of whether/where each strategy has actually been evaluated.
    Each entry also carries 'prefix' (llr/llr2/llr3) for reference.
    """
    from ab.gpt.brute.llr.layerwise_lr import LAYERWISE_STRATEGIES as s1
    from ab.gpt.brute.llr.layerwise_lr2 import LAYERWISE_STRATEGIES as s2
    from ab.gpt.brute.llr.layerwise_lr3 import LAYERWISE_STRATEGIES as s3
    out = []
    for prefix, strategies in (('llr', s1), ('llr2', s2), ('llr3', s3)):
        for s in strategies:
            out.append({**s, 'prefix': prefix})
    return out


def load_vanilla_baselines(arch_names, uniform_map=None) -> tuple[dict, dict, dict]:
    """
    Read baseline data straight from the DB (read-only).

    Two baseline accuracy sources, in priority order:
      1. The llr_uniform CONTROL (single group, 1.0x LR) — evaluated under the
         same conditions as the strategies. Identified via `uniform_map`
         {hash: arch} (built from the metadata CSV). Preferred, consistent.
      2. The pre-existing DB vanilla (plain arch name, e.g. 'ResNet') — fallback
         where a uniform control was not evaluated.

    The brute-force llr models are stored under opaque hash names (llr-<md5>), so
    a model's architecture is only recoverable via the metadata CSV. Vanilla
    *code* (for the LLM's baseline input) lives in nn.code under the plain
    architecture name.

    Returns (code_lut, vanilla_acc_lut, uniform_acc_lut):
        code_lut         : {arch: vanilla_source_code}
        vanilla_acc_lut  : {(arch, dataset, epoch, transform): best DB-vanilla acc}
        uniform_acc_lut  : {(arch, dataset, epoch, transform): best uniform-control acc}
    All accuracy keys include transform so the baseline uses the SAME
    preprocessing as the llr model — removing transform as a confounder.
    """
    dbp = db_path()
    arch_names = [a for a in arch_names if a]
    if not dbp or not arch_names:
        return {}, {}, {}
    qmarks = ",".join("?" * len(arch_names))
    con = sqlite3.connect(f"file:{dbp}?mode=ro", uri=True)
    try:
        cur = con.cursor()
        code_lut = {
            name: code
            for name, code in cur.execute(
                f"SELECT name, code FROM nn WHERE name IN ({qmarks})", arch_names)
        }
        vanilla_acc_lut = {}
        for nn, ds, ep, tf, acc in cur.execute(
            f"SELECT nn, dataset, epoch, transform, MAX(accuracy) FROM stat "
            f"WHERE nn IN ({qmarks}) GROUP BY nn, dataset, epoch, transform", arch_names):
            try:
                vanilla_acc_lut[(nn, ds, int(ep), tf)] = acc
            except (TypeError, ValueError):
                vanilla_acc_lut[(nn, ds, ep, tf)] = acc

        uniform_acc_lut = {}
        uniform_map = uniform_map or {}
        if uniform_map:
            hashes = list(uniform_map.keys())
            hm = ",".join("?" * len(hashes))
            for nn, ds, ep, tf, acc in cur.execute(
                f"SELECT nn, dataset, epoch, transform, MAX(accuracy) FROM stat "
                f"WHERE nn IN ({hm}) GROUP BY nn, dataset, epoch, transform", hashes):
                arch = uniform_map.get(nn)
                if not arch:
                    continue
                try:
                    key = (arch, ds, int(ep), tf)
                except (TypeError, ValueError):
                    key = (arch, ds, ep, tf)
                if key not in uniform_acc_lut or acc > uniform_acc_lut[key]:
                    uniform_acc_lut[key] = acc
    finally:
        con.close()
    print(f"[VANILLA] loaded baselines: {len(code_lut)} arch codes, "
          f"{len(vanilla_acc_lut)} db-vanilla + {len(uniform_acc_lut)} uniform-control "
          f"(arch,dataset,epoch,transform) accuracies")
    return code_lut, vanilla_acc_lut, uniform_acc_lut


def load_all_vanilla_archs(task: str | None = 'img-classification') -> dict:
    """
    {architecture_name: code} for plain-named architectures in the DB's nn
    table (not just the ~26 we've brute-forced LLR on). Used for generation-
    time exploration onto architectures the LLR generators have never touched —
    the actual generalization target of the fine-tune.

    A row is treated as a "plain architecture" if its name has no '-' (llr
    hash-prefixed names all contain one: 'llr-<hash>', 'llr2-<hash>', ...).

    LEMUR's vanilla-architecture pool spans many tasks (img-captioning, GANs,
    denoising, segmentation, detection, super-resolution, ...) whose train_setup
    doesn't follow the classification self.parameters()-optimizer pattern our
    injection assumes. By default this filters to task='img-classification' —
    pass task=None for the unfiltered set.
    """
    dbp = db_path()
    if not dbp:
        return {}
    con = sqlite3.connect(f"file:{dbp}?mode=ro", uri=True)
    try:
        cur = con.cursor()
        if task:
            query = ("SELECT DISTINCT nn.name, nn.code FROM nn "
                      "JOIN stat ON stat.nn = nn.name "
                      "WHERE nn.name NOT LIKE '%-%' AND stat.task = ?")
            rows = cur.execute(query, (task,))
        else:
            rows = cur.execute("SELECT name, code FROM nn WHERE name NOT LIKE '%-%'")
        return {name: code for name, code in rows if not _HEX_HASH_RE.match(name)}
    finally:
        con.close()


def load_arch_best_accuracy(arch_names) -> dict:
    """
    {(arch, dataset, epoch, transform): best_accuracy} for the given plain
    architecture names, across ALL their stat rows (any transform/epoch present
    in the DB) — used to quote a real "current accuracy" for exploratory
    generation targets that were never LLR-evaluated.
    """
    dbp = db_path()
    arch_names = [a for a in arch_names if a]
    if not dbp or not arch_names:
        return {}
    qmarks = ",".join("?" * len(arch_names))
    con = sqlite3.connect(f"file:{dbp}?mode=ro", uri=True)
    try:
        cur = con.cursor()
        out = {}
        for nn, ds, ep, tf, acc in cur.execute(
            f"SELECT nn, dataset, epoch, transform, MAX(accuracy) FROM stat "
            f"WHERE nn IN ({qmarks}) GROUP BY nn, dataset, epoch, transform", arch_names):
            try:
                out[(nn, ds, int(ep), tf)] = acc
            except (TypeError, ValueError):
                out[(nn, ds, ep, tf)] = acc
        return out
    finally:
        con.close()
