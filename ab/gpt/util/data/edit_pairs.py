"""Training corpus of code edits, for any architecture in the dataset.

The design-space corpus (``design_pairs``) reads a model as a point in a closed
13-factor space. That is only definable for one generator template, and it buys
a factor model whose estimate of a design is far better than the design's own
single training run -- which is what keeps measurement noise out of the pairs.

This module gives that up to work on anything in LEMUR. Pairs are found by
code similarity instead: two models that differ by a handful of lines, where
the better one measures better. What is lost is stated plainly in the stats --
without a factor model the only guard against noise is the size of the gain,
and at this budget noise alone clears 0.05 about one time in six.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ab.gpt.util.data.code_edit import (
    EDIT_CLOSE, EDIT_OPEN, TUNABLE_PRM, apply_edits, hp_text, model_identity,
    parse_edits, parse_full, render_change, render_edits,
    uses_pretrained_weights, validate_model, verify_round_trip)

CLASS_RE = re.compile(r"^class\s+(\w+)", re.M)
LENGTH_BUCKET = 40


class Record:
    __slots__ = ("nn", "code", "observed", "n_trials", "prm", "transform",
                 "transform_code")

    def __init__(self, nn, code, observed, n_trials, prm, transform, transform_code):
        self.nn, self.code, self.observed = nn, code, observed
        self.n_trials, self.prm = n_trials, prm
        self.transform, self.transform_code = transform, transform_code


def epoch_record_is_valid(rec: dict, file_epoch: int) -> bool:
    if not isinstance(rec, dict) or rec.get("accuracy") is None:
        return False
    best = rec.get("best_epoch")
    return best is None or best == file_epoch


def load_offline(root, task, dataset, metric, epoch, nn_prefixes=None) -> List[Record]:
    """Records from a checked-out ``nn-dataset/ab/nn`` tree, any architecture."""
    root_p = Path(root)
    nn_dir, stat_dir = root_p / "nn", root_p / "stat" / "train"
    tr_dir, pipeline = root_p / "transform", f"{task}_{dataset}_{metric}_"
    tr_cache: Dict[str, str] = {}

    def transform_code(name):
        if name not in tr_cache:
            p = tr_dir / f"{name}.py"
            tr_cache[name] = p.read_text(encoding="utf-8") if p.exists() else ""
        return tr_cache[name]

    out: List[Record] = []
    for d in sorted(stat_dir.glob(pipeline + "*")):
        nn = d.name[len(pipeline):]
        if nn_prefixes and not any(nn.startswith(p) for p in nn_prefixes):
            continue
        f, code_p = d / f"{epoch}.json", nn_dir / f"{nn}.py"
        if not f.exists() or not code_p.exists():
            continue
        code = code_p.read_text(encoding="utf-8", errors="replace")
        if validate_model(code):
            continue
        try:
            trials = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        good = [t for t in trials if epoch_record_is_valid(t, epoch)]
        if not good:
            continue
        accs = [t["accuracy"] for t in good]
        best = max(good, key=lambda t: t["accuracy"])
        prm = {k: best[k] for k in ("lr", "momentum", "batch", "dropout")
               if k in best and best[k] is not None}
        prm["transform"] = best.get("transform", "")
        out.append(Record(nn, code, sum(accs) / len(accs), len(accs), prm,
                          best.get("transform", ""),
                          transform_code(best.get("transform", ""))))
    return out


#: Architecture-similarity bands to draw training neighbourhoods from, as
#: measured on one corpus (improving pairs / of those, expressible as <=40-line
#: edits): very_high 7,290/7,290 from 218 models, high 13,857/13,857 from 289,
#: low 71,668/18,344 from 600. "high" is the default because every improving
#: pair in it is already a small edit, so no diff work is wasted. "medium"
#: returned 347 improving pairs from 600 models and "very_low_near" returned
#: none, which does not match the others; neither is used until that is
#: understood.
DEFAULT_BANDS = ("high",)


def sample_anchors(task, dataset, metric, epoch, n, seed=17,
                   nn_prefixes=None, pool_factor=4) -> List[str]:
    """Model names to centre similarity neighbourhoods on.

    A band with no anchor is one neighbourhood around one automatically chosen
    model, so the whole corpus -- training and test alike -- comes from a single
    architecture family. Measured: every test baseline was an rl-* model.
    Spreading anchors across the dataset is what makes the corpus, and the
    holdout drawn from it, span families.
    """
    import ab.nn.api as lemur
    df = lemur.data(task=task, dataset=dataset, metric=metric, epoch=epoch,
                    only_best_accuracy=True, max_rows=20000)
    names = []
    for _, row in df.iterrows():
        nn, code = row.get("nn"), row.get("nn_code")
        if not nn or not code:
            continue
        if nn_prefixes and not any(str(nn).startswith(p) for p in nn_prefixes):
            continue
        names.append(str(nn))
    names = sorted(set(names))
    if not names:
        return []
    # Not every model in the dataset has a row in nn_minhash, and an anchor
    # that is missing from it makes the band query raise rather than return
    # nothing. Hand back a larger shuffled pool than asked for so the caller
    # can skip the unusable ones and still reach n neighbourhoods.
    rng = random.Random(seed)
    pool = rng.sample(names, min(n * pool_factor, len(names)))
    return pool


def load_band(task, dataset, metric, epoch, band, size, anchor=None,
              nn_prefixes=None) -> List[Record]:
    """Models within one architecture-similarity band, as Records.

    The band is LEMUR's own MinHash Jaccard over the architecture graph, which
    is a real similarity measure; the previous grouping here was a heuristic
    over class names and file length that produced 13 groups from 3,909 models.
    """
    import ab.nn.api as lemur
    from ab.nn.util.db.Query import JoinConf
    conf = JoinConf(num_joint_nns=max(2, int(size)),
                    similarity_mode="anchor_band_db_minhash",
                    similarity_band=band, anchor_nn=anchor)
    df = lemur.data(sql=conf, task=task, dataset=dataset, metric=metric,
                    epoch=epoch, max_rows=5000)
    out: List[Record] = []
    raw = dropped_prefix = 0
    for _, row in df.iterrows():
        nn, code = row.get("nn"), row.get("nn_code")
        acc = row.get("accuracy")
        if not nn or not code or acc is None:
            continue
        raw += 1
        # A prefix filter here fights the band. The band already decides which
        # models are comparable, by architecture similarity, and it draws from
        # the whole dataset -- so asking for "GenFractalNet-*" as well removed
        # every row it returned. Left available for the rare case where the
        # corpus really must be restricted, but it is not the selector.
        if nn_prefixes and not any(str(nn).startswith(p) for p in nn_prefixes):
            dropped_prefix += 1
            continue
        if validate_model(code):
            continue
        out.append(Record(nn, code, float(acc), 1, dict(row.get("prm") or {}),
                          row.get("transform") or "", row.get("transform_code") or ""))
    if raw and not out and dropped_prefix:
        print(f"[edit-pairs] band {band}: all {raw} models were removed by "
              f"nn_prefixes={nn_prefixes}; the band selects by architecture "
              f"similarity across the whole dataset, so set NN_PREFIXES=None",
              flush=True)
    return out


def load_from_lemur(task, dataset, metric, epoch, nn_prefixes=None) -> List[Record]:
    import ab.nn.api as lemur
    kwargs = dict(only_best_accuracy=False, task=task, dataset=dataset,
                  metric=metric, epoch=epoch)
    if nn_prefixes:
        kwargs["nn_prefixes"] = tuple(nn_prefixes)
    df = lemur.data(**kwargs)
    by_nn: Dict[str, List] = collections.defaultdict(list)
    for _, row in df.iterrows():
        by_nn[row["nn"]].append(row)
    out: List[Record] = []
    for nn, rows in by_nn.items():
        code = rows[0].get("nn_code") or ""
        if not code or validate_model(code):
            continue
        accs = [r["accuracy"] for r in rows if r.get("accuracy") is not None]
        if not accs:
            continue
        best = max(rows, key=lambda r: r.get("accuracy") or 0.0)
        out.append(Record(nn, code, sum(accs) / len(accs), len(accs),
                          dict(best.get("prm") or {}), best.get("transform") or "",
                          best.get("transform_code") or ""))
    return out


def bucket_key(code: str):
    """Cheap similarity key, so pairing is not quadratic over the whole corpus."""
    return (tuple(sorted(set(CLASS_RE.findall(code)))),
            len(code.splitlines()) // LENGTH_BUCKET)


def changed_lines(a: str, b: str) -> int:
    sm = difflib.SequenceMatcher(None, a.splitlines(), b.splitlines(), autojunk=False)
    return sum(max(i2 - i1, j2 - j1)
               for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")


SYSTEM_PROMPT = """\
You are an expert PyTorch neural network architect. You are given one model that was trained for \
{epoch} epoch on '{dataset}' and measured by '{metric}'; return a better one.

Do not rewrite the file. Return the smallest set of edits that makes it better, as <edit> blocks:

<edit>
<find>
the exact lines to replace, copied character for character from the file
</find>
<with>
the lines to put there instead
</with>
</edit>

Each <find> block must appear exactly once in the file, so include a line or two of surrounding \
context when the lines alone would be ambiguous.

You may offer up to three alternative designs. Write each as its own run of <edit> blocks, every \
<find> quoted from the original file as given -- not from the file as your earlier edits would \
leave it. The best is trained first, so put it first.

If what you want to do would touch most of the file, do not force it into edits: return the whole \
new file in a single <nn> block instead. An edit carries both the old lines and the new ones, so \
past about half the file it is more to write, and more to get wrong, than the file itself. Use \
whichever is shorter.

The learning rate and the momentum are read from the training configuration, not from the file, \
so no edit can change them. To change them, add one <hp> block after your edits:

<hp>{{"lr": 0.003, "momentum": 0.95}}</hp>

Give only the ones you are changing, and omit the block entirely if you are changing neither. \
Nothing else about the training protocol is yours to set.

Keep class Net, supported_hyperparameters, train_setup and learn: the file is evaluated through \
them. Use only PyTorch, torchvision and torchvision.transforms. Pretrained backbones are allowed: which backbone, how many, where each is cut, how their features are fused, and the stem and head built around them for this input size are all yours to choose. The accuracy to beat below was measured under the same rules, so simply loading a pretrained backbone is not by itself an improvement over it.
One epoch is a single pass over the data, so favour fast early convergence over regularisation \
that only pays off over many epochs.

Then give one <reason> block: one short sentence per change. Begin your answer with <edit> or \
<nn> immediately and write no prose before or between the blocks."""


USER_PROMPT = """\
Baseline model, measured on '{dataset}' for task '{task}' after exactly {epoch} epoch of \
training. The accuracy below was produced by the same protocol that will be used to score you, so \
it is directly comparable to your own result.

Measured '{metric}': {accuracy:.4f}
Hyperparameters: {hyper}

<nn>{nn_code}</nn>
{examples}
Return edits that score above {target:.4f} under that same {epoch}-epoch protocol. That is the \
best model recorded in this corpus, so matching the baseline is not enough.
Change as much as you can justify and no more: every line you edit is a line that can break, and \
a file that does not run scores nothing."""




#: How many measured neighbours to show alongside the baseline.
CONTEXT_EXAMPLES = 4


def render_context(baseline: Record, neighbours: Sequence[Record],
                   limit: int = CONTEXT_EXAMPLES) -> str:
    """Nearby architectures, written as edits to the baseline, with their scores.

    Showing them as diffs rather than whole files is what makes several of them
    affordable: each costs a few dozen lines instead of several hundred. Some
    score better than the baseline and some worse, and that is the point -- the
    corpus of improving pairs alone never shows the model what *hurts*, and the
    measured mean improvement of a generation was negative.
    """
    # Both callers hand over a band ranked best-first, and taking the head of
    # that gives a low baseline four examples that all score better -- the one
    # shape the docstring above says is useless. Interleave the strongest of
    # the better ones with the weakest of the worse ones so the contrast
    # survives wherever the baseline sits in its band.
    better = [n for n in neighbours if n.observed > baseline.observed]
    worse = [n for n in neighbours if n.observed <= baseline.observed]
    better.sort(key=lambda r: -r.observed)
    worse.sort(key=lambda r: r.observed)
    ordered = []
    for i in range(max(len(better), len(worse))):
        if i < len(better):
            ordered.append(better[i])
        if i < len(worse):
            ordered.append(worse[i])

    blocks = []
    for n in ordered:
        if len(blocks) >= limit or n.nn == baseline.nn:
            continue
        body = render_edits(baseline.code, n.code)
        if body is None:
            continue
        inner = body[len(EDIT_OPEN):-len(EDIT_CLOSE)].strip()
        verdict = "better" if n.observed > baseline.observed else "worse"
        blocks.append(f'<example measured="{n.observed:.4f}" than_baseline="{verdict}" '
                      f'hyperparameters="{hp_text(n.prm)}">\n'
                      f'{inner}\n</example>')
    if not blocks:
        return ""
    return ("\nRelated architectures, written as edits to the baseline above, with what each one "
            "actually measured under the same protocol and the hyperparameters it was "
            "measured at. An example that scores better at different hyperparameters may "
            "owe the gain to those and not to its architecture:\n\n" + "\n".join(blocks) + "\n")


def describe_edits(before: str, after: str) -> str:
    """What changed, in the words of the code itself."""
    sm = difflib.SequenceMatcher(None, before.splitlines(), after.splitlines(),
                                 autojunk=False)
    bits = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        frm = " ".join(before.splitlines()[i1:i2]).strip()
        to = " ".join(after.splitlines()[j1:j2]).strip()
        if tag == "replace" and frm and to:
            bits.append(f"replaced `{frm[:70]}` with `{to[:70]}`")
        elif tag == "insert" and to:
            bits.append(f"added `{to[:70]}`")
        elif tag == "delete" and frm:
            bits.append(f"removed `{frm[:70]}`")
    return "; ".join(bits[:6])


def render_example(a: Record, b: Record, cfg, ceiling, neighbours=()) -> Optional[dict]:
    body, kind = render_change(a.code, b.code)
    if body is None:
        return None
    if kind == "edit":
        built, _why = apply_edits(a.code, parse_edits(body))
    else:
        built = parse_full(body)
    if built is None or built.strip() != b.code.strip():
        return None
    system = SYSTEM_PROMPT.format(dataset=cfg["dataset"], metric=cfg["metric"],
                                  epoch=cfg["epoch"])
    # Never show the answer among the examples.
    context = render_context(a, [n for n in neighbours if n.nn not in (a.nn, b.nn)])
    user = USER_PROMPT.format(
        dataset=cfg["dataset"], task=cfg["task"], metric=cfg["metric"],
        epoch=cfg["epoch"], accuracy=a.observed, nn_code=a.code.strip(),
        hyper=hp_text(a.prm), examples=context, target=max(ceiling, a.observed))
    reason = (f"{describe_edits(a.code, b.code)}. Measured {b.observed:.3f} "
              f"against the baseline's {a.observed:.3f}, "
              f"{b.observed - a.observed:+.3f}.")
    hp = {k: b.prm[k] for k in TUNABLE_PRM
          if k in b.prm and b.prm.get(k) != a.prm.get(k)}
    hp_block = f"\n<hp>{json.dumps(hp, sort_keys=True)}</hp>" if hp else ""
    assistant = f"{body}{hp_block}\n<reason>{reason}</reason>"
    return {
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user},
                     {"role": "assistant", "content": assistant}],
        "meta": {"baseline_nn": a.nn, "target_nn": b.nn,
                 "baseline_observed": round(a.observed, 4),
                 "target_observed": round(b.observed, 4),
                 "observed_gain": round(b.observed - a.observed, 4),
                 "changed_lines": changed_lines(a.code, b.code),
                 "context_examples": context.count("<example "),
                 "hp": hp,
                 "format": kind,
                 "edit_blocks": len(parse_edits(body)) if kind == "edit" else 0},
    }


#: Pairing inside a bucket is quadratic in its size, but only pairs that
#: already improve by min_gain reach the diff, which is what keeps it
#: affordable: measured at 1.05M candidate pairs in 112s. At 80 only 1,040 of
#: 3,389 baselines were ever considered and the corpus came out at 417 rows
#: against a max_prompts of 768 -- the trainer saw the same rows every cycle
#: and most of the corpus never at all. Members are kept by score, because
#: those are the ones worth learning to reach, plus a random tail so the pairs
#: are not all near the top.
MAX_BUCKET = 400


#: How many pairs one target may anchor. Without a limit every baseline picks
#: the same globally best networks and the corpus degenerates into a handful of
#: memorisation targets.
MAX_TARGET_USES = 8


def build_pairs(clusters, min_gain, max_changed, per_baseline, rng,
                max_bucket=MAX_BUCKET, max_target_uses=MAX_TARGET_USES):
    pairs = []
    target_uses = collections.Counter()
    for members in clusters:
        if len(members) < 2:
            continue
        if len(members) > max_bucket:
            top = sorted(members, key=lambda r: -r.observed)[:max_bucket // 2]
            rest = rng.sample(members, max_bucket // 2)
            members = list({r.nn: r for r in top + rest}.values())
        for a in members:
            cands = []
            for b in members:
                if b.nn == a.nn or b.observed - a.observed < min_gain:
                    continue
                # The accuracy test above is the cheap filter and it removes
                # most pairs. A line-multiset prefilter was tried here and
                # pruned 0% on a single-template corpus, where every model is a
                # near-copy, while costing 14% -- so the diff is run directly.
                ch = changed_lines(a.code, b.code)
                if ch < 1 or ch > max_changed:
                    continue
                cands.append((-b.observed, ch, b))
            if not cands:
                continue
            cands.sort(key=lambda t: t[:2])
            taken = 0
            for _score, _ch, b in cands:
                if taken >= per_baseline:
                    break
                # Ranking every baseline by the target's absolute accuracy
                # pointed them all at the same few best networks: measured,
                # 5,810 pairs shared SIX distinct targets. That is the corpus
                # teaching "emit this memorised network" rather than "improve
                # the one you were given" -- the failure this whole corpus
                # exists to avoid. A target may anchor only so many pairs; past
                # that the baseline takes its next-best reachable one.
                if target_uses[b.nn] >= max_target_uses:
                    continue
                target_uses[b.nn] += 1
                pairs.append((a, b))
                taken += 1
    rng.shuffle(pairs)
    return pairs


def emit_seed_context(seed_file, out_path, task="img-classification",
                      dataset="cifar-10", metric="acc", epoch=1,
                      band=DEFAULT_BANDS[0], size=60, nn_prefixes=None,
                      records=None) -> str:
    """Neighbour examples for each pinned test baseline, written once.

    Generation must show the same thing training did, or the model is tested on
    a prompt shape it never saw. The neighbours are fetched here rather than
    inside the generation loop so the test prompts are fixed for the whole run
    and cost no database work per cycle.
    """
    names = json.loads(Path(seed_file).read_text(encoding="utf-8")).get("nn") or []
    # build() writes context for every held-out baseline from the clusters it
    # already has. Anything it covered is better than a fresh band query, so
    # only the gaps are fetched here.
    context = {}
    if Path(out_path).is_file():
        try:
            context = {k: v for k, v in json.loads(
                Path(out_path).read_text(encoding="utf-8")).items() if v}
        except Exception:
            context = {}
    for name in names:
        if context.get(name):
            continue
        try:
            if records is not None:
                by_key = collections.defaultdict(list)
                for r in records:
                    by_key[bucket_key(r.code)].append(r)
                me = next((r for r in records if r.nn == name), None)
                members = by_key[bucket_key(me.code)] if me else []
            else:
                members = load_band(task, dataset, metric, epoch, band, size,
                                    anchor=name, nn_prefixes=nn_prefixes)
                me = next((r for r in members if r.nn == name), None)
            if me is None or len(members) < 2:
                context[name] = ""
                continue
            ranked = sorted((r for r in members if r.nn != name),
                            key=lambda r: -r.observed)
            context[name] = render_context(me, ranked)
        except Exception as exc:
            print(f"[edit-pairs] no context for {name}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            context[name] = ""
    context = {n: context.get(n, "") for n in names}
    have = sum(1 for v in context.values() if v)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(context, indent=1), encoding="utf-8")
    print(f"[edit-pairs] neighbour context for {have}/{len(names)} test baselines")
    return out_path


def holdout_members(names, holdout_frac, seed, pinned_seed_file=None) -> set:
    import hashlib
    pinned = set()
    if pinned_seed_file and Path(pinned_seed_file).is_file():
        try:
            pinned = set(json.loads(Path(pinned_seed_file).read_text(
                encoding="utf-8")).get("nn") or [])
        except Exception:
            pinned = set()
    out = set(pinned)
    for n in names:
        h = hashlib.sha1(f"{seed}:{n}".encode()).digest()
        if int.from_bytes(h[:4], "big") / 2**32 < holdout_frac:
            out.add(n)
    return out


def build(out_dir, task="img-classification", dataset="cifar-10", metric="acc",
          epoch=1, nn_prefixes=None, min_observed_gain=0.05, max_changed_lines=40,
          per_baseline=12, val_frac=0.08, holdout_frac=0.12, seed=17,
          offline_root=None, pinned_seed_file=None,
          max_bucket=MAX_BUCKET, bands=DEFAULT_BANDS, band_size=600,
          n_anchors=24, min_holdout=30) -> dict:
    rng = random.Random(seed)
    cfg = dict(task=task, dataset=dataset, metric=metric, epoch=epoch)

    # A cluster is a set of architecturally similar models: one similarity band
    # from LEMUR, or -- offline, where there is no database -- the class-name
    # grouping this module used before. Pairs are formed inside a cluster, so
    # the two models are comparable and the diff between them is small.
    clusters: List[List[Record]] = []
    if offline_root:
        records = load_offline(offline_root, task, dataset, metric, epoch, nn_prefixes)
        grouped = collections.defaultdict(list)
        for r in records:
            grouped[bucket_key(r.code)].append(r)
        clusters = [v for v in grouped.values() if len(v) >= 2]
    else:
        seen_names = set()
        records = []
        anchors = sample_anchors(task, dataset, metric, epoch, n_anchors,
                                 seed=seed, nn_prefixes=nn_prefixes) or [None]
        print(f"[edit-pairs] {len(anchors)} anchor(s) across the dataset")
        for band in bands:
            used = missing = empty = 0
            for anchor in anchors:
                if used >= n_anchors:
                    break
                try:
                    members = load_band(task, dataset, metric, epoch, band,
                                        band_size, anchor=anchor,
                                        nn_prefixes=nn_prefixes)
                except ValueError as exc:
                    # A model with no nn_minhash row cannot anchor a band. That
                    # is a property of the anchor, not a broken run, so take
                    # the next candidate from the pool instead of dying.
                    if "nn_minhash" not in str(exc):
                        raise
                    missing += 1
                    continue
                if len(members) < 2:
                    empty += 1
                    continue
                used += 1
                clusters.append(members)
                for r in members:
                    if r.nn not in seen_names:
                        seen_names.add(r.nn)
                        records.append(r)
            print(f"[edit-pairs] band {band}: {used} neighbourhood(s) from "
                  f"{len(anchors)} candidate anchor(s) "
                  f"({missing} not in nn_minhash, {empty} too small), "
                  f"{len(records)} distinct models")
            if not used:
                raise RuntimeError(
                    f"band {band!r}: no anchor produced a neighbourhood -- "
                    f"{missing} of {len(anchors)} candidates were missing from "
                    f"nn_minhash and {empty} returned fewer than 2 models")
    if len(records) < 50:
        raise RuntimeError(f"only {len(records)} usable records -- check the filters")
    # Pretrained and from-scratch models sit at different accuracies after one
    # epoch, so one shared ceiling makes every scratch prompt ask for something
    # unreachable and every pretrained prompt ask for something already met.
    # Each baseline is given the ceiling of its own group.
    ceilings = {}
    for flag in (False, True):
        group = [r.observed for r in records if uses_pretrained_weights(r.code) is flag]
        if group:
            ceilings[flag] = max(group)
    ceiling = max(ceilings.values())
    ceiling_of = lambda r: ceilings.get(uses_pretrained_weights(r.code), ceiling)

    # The hashed fraction has to leave enough baselines for the test set. With
    # 289 models at 0.12 it left 37 against a TEST_NN of 30, which is no margin
    # at all: a slightly smaller corpus and the run fails at the seed assert
    # rather than here, where the cause is visible.
    holdout = holdout_members([r.nn for r in records], holdout_frac, seed,
                              pinned_seed_file=pinned_seed_file)
    if len(holdout) < min_holdout:
        for frac in (0.2, 0.3, 0.4):
            holdout = holdout_members([r.nn for r in records], frac, seed,
                                      pinned_seed_file=pinned_seed_file)
            if len(holdout) >= min_holdout:
                print(f"[edit-pairs] holdout raised to {frac:.0%} to reach "
                      f"{len(holdout)} test baselines", flush=True)
                break
    if len(holdout) < min_holdout:
        raise RuntimeError(
            f"only {len(holdout)} baselines can be held out of {len(records)} "
            f"models, and {min_holdout} are needed for the test set; widen the "
            f"corpus (more anchors or bands) or lower TEST_NN")
    train_pool = [r for r in records if r.nn not in holdout]
    hold_pool = [r for r in records if r.nn in holdout]

    holdout_names = set(holdout)
    train_clusters = [[r for r in c if r.nn not in holdout_names] for c in clusters]
    pairs = build_pairs(train_clusters, min_observed_gain, max_changed_lines,
                        per_baseline, rng, max_bucket=max_bucket)
    if not pairs:
        raise RuntimeError("no pairs at the configured thresholds")

    report = verify_round_trip([(a.code, b.code) for a, b in pairs])
    if report["mismatch"]:
        raise RuntimeError(f"edit round trip failed: {report}")
    print(f"[edit-pairs] edit round trip verified: {report}")

    # Each baseline's neighbours are the rest of its own similarity band, best
    # first, so the examples shown are genuinely comparable architectures.
    neighbours_of = collections.defaultdict(list)
    for members in train_clusters:
        ranked = sorted(members, key=lambda r: -r.observed)
        for r in members:
            neighbours_of[r.nn] = ranked
    examples = [e for e in
                (render_example(a, b, cfg, ceiling_of(a), neighbours_of.get(a.nn, ()))
                 for a, b in pairs)
                if e is not None]
    if not examples:
        raise RuntimeError("no renderable pairs")
    rng.shuffle(examples)
    n_val = max(1, int(len(examples) * val_frac))
    val, train = examples[:n_val], examples[n_val:]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Neighbour context for the held-out baselines, written here because this
    # is the only place that knows which cluster each one came from. Deriving
    # it from neighbours_of would always yield nothing -- that map is built
    # from train_clusters, which has the holdout removed -- and re-querying the
    # band per seed asks the database for a neighbourhood centred on a model
    # that may not come back in its own result. Test baselines are shown only
    # training-pool neighbours, so no test accuracy appears in a test prompt.
    cluster_of = {}
    for c in clusters:
        for r in c:
            cluster_of.setdefault(r.nn, c)
    seed_ctx = {}
    for r in hold_pool:
        nb = [x for x in cluster_of.get(r.nn, ()) if x.nn not in holdout_names]
        seed_ctx[r.nn] = render_context(r, nb) if nb else ""
    (out / "seed_context.json").write_text(
        json.dumps(seed_ctx, indent=1), encoding="utf-8")
    with_ctx = sum(1 for v in seed_ctx.values() if v)
    print(f"[edit-pairs] neighbour context for {with_ctx}/{len(seed_ctx)} "
          f"held-out baselines", flush=True)
    if not with_ctx:
        raise RuntimeError(
            "no held-out baseline has neighbour context; test prompts would "
            "carry no examples while training prompts carry "
            f"{CONTEXT_EXAMPLES}, which is a train/test mismatch")

    def dump(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pinned = set()
    if pinned_seed_file and Path(pinned_seed_file).is_file():
        try:
            pinned = set(json.loads(Path(pinned_seed_file).read_text(
                encoding="utf-8")).get("nn") or [])
        except Exception:
            pinned = set()
    if pinned:
        used = {e["meta"]["baseline_nn"] for e in examples}
        used |= {e["meta"]["target_nn"] for e in examples}
        leaked = sorted(pinned & used)
        if leaked:
            raise RuntimeError(
                f"{len(leaked)} pinned test baseline(s) appear in the training "
                f"pairs, e.g. {leaked[:5]}")
        print(f"[edit-pairs] {len(pinned)} pinned test baselines, none used in training")

    # Every architecture LEMUR already holds, by its own graph identity.
    # Generation reads this so a candidate that merely re-spells a network in
    # the corpus is dropped before training rather than after, when the insert
    # is refused and the GPU time is already spent.
    known = sorted({model_identity(r.code, r.prm) for r in records})
    (out / "known_uids.json").write_text(json.dumps(known), encoding="utf-8")
    print(f"[edit-pairs] {len(known)} known architectures recorded")

    dump(out / "train.jsonl", train)
    dump(out / "val.jsonl", val)
    hold_pool.sort(key=lambda r: r.observed)
    dump(out / "holdout_baselines.jsonl",
         [{"nn": r.nn, "observed": round(r.observed, 4)} for r in hold_pool])

    changed = sorted(e["meta"]["changed_lines"] for e in examples)
    stats = {
        "records": len(records),
        "architectures": len({bucket_key(r.code) for r in records}),
        "train_baselines": len(train_pool), "holdout_baselines": len(hold_pool),
        "pairs": len(pairs), "train_examples": len(train), "val_examples": len(val),
        "min_observed_gain": min_observed_gain, "max_changed_lines": max_changed_lines,
        "changed_lines_p50": changed[len(changed) // 2],
        "changed_lines_p90": changed[int(0.9 * (len(changed) - 1))],
        # Target concentration is the thing to watch: if a few networks anchor
        # most pairs, the corpus is teaching the model to emit them rather than
        # to improve what it was given.
        "distinct_targets": len({e["meta"]["target_nn"] for e in examples}),
        "distinct_baselines": len({e["meta"]["baseline_nn"] for e in examples}),
        "edit_blocks_mean": round(
            sum(e["meta"]["edit_blocks"] for e in examples) / len(examples), 2),
        # How often a change was too large to be worth patching. The model is
        # trained on both forms so it can pick the shorter one itself.
        "format_edit": sum(1 for e in examples if e["meta"]["format"] == "edit"),
        "format_full": sum(1 for e in examples if e["meta"]["format"] == "full"),
        "seed_context_covered": with_ctx,
        "seed_context_total": len(seed_ctx),
        "observed_max": round(ceiling, 4),
        "observed_max_scratch": round(ceilings.get(False, ceiling), 4),
        "observed_max_pretrained": round(ceilings.get(True, ceiling), 4),
        "known_architectures": len(known),
        "round_trip": report,
        # Said plainly: there is no factor model here, so a pair rests on two
        # single training runs and the only guard is the size of the gain.
        "noise_guard": "observed gain only (no factor model)",
    }
    with open(out / "corpus_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats
