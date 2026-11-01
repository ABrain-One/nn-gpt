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
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ab.gpt.util.data.code_edit import (
    EDIT_CLOSE, EDIT_OPEN, TUNABLE_PRM, apply_edits, hp_text, model_identity,
    parse_edits, parse_full, render_change, render_edits,
    uses_pretrained_weights, validate_model, verify_round_trip)
from ab.gpt.util.data import minhash_clusters
from ab.gpt.util.data.family import (
    cap_per_family, describe as describe_families, family_histogram, family_of,
    stratified_sample)

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


#: Optional prefix-based floor on the anchor pool, off by default.
#:
#: It was 10, and that was wrong: it excluded a model from the anchor pool
#: because its *name* was rare, even when that model sat in a large genuine
#: MinHash neighbourhood. Cross-family similarity is the thing this pipeline
#: exists to exploit, and a name cannot see it -- one generator gives every
#: batch its own prefix, another puts 1,900 near-identical models under one.
#:
#: Nothing needs it now. The pool is ordered by a *content* grouping (shared
#: class names and file length, the same key the offline path clusters by) and
#: the loop below skips an anchor it has already seen inside another band, so
#: isolated models sit at the end of the pool and are simply never reached
#: rather than being removed from it. Left as an escape hatch at 1 = disabled.
MIN_ANCHOR_FAMILY = int(os.environ.get("NNGPT_MIN_ANCHOR_FAMILY", "1") or 1)


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
    # The query already returns the source, so a content grouping is free here
    # -- and it is what replaces the name prefix. bucket_key is the same
    # shared-class-names-and-length key the offline path clusters by: crude
    # next to MinHash, but it reads the code rather than the file name, so a
    # model named like nothing else in the dataset still lands beside the
    # models it resembles.
    shape_of = {}
    for _, row in df.iterrows():
        nn, code = row.get("nn"), row.get("nn_code")
        if not nn or not code:
            continue
        if nn_prefixes and not any(str(nn).startswith(p) for p in nn_prefixes):
            continue
        names.append(str(nn))
        shape_of[str(nn)] = bucket_key(str(code))
    names = sorted(set(names))
    if not names:
        return []
    # Not every model in the dataset has a row in nn_minhash, and an anchor
    # that is missing from it makes the band query raise rather than return
    # nothing. Hand back a larger shuffled pool than asked for so the caller
    # can skip the unusable ones and still reach n neighbourhoods.
    rng = random.Random(seed)
    want = min(n * pool_factor, len(names))
    # Uniform over names is uniform over the *dataset*, not over architectures.
    # One search family contributed 1,903 of 2,343 names, so a uniform pool of
    # 96 handed 24 neighbourhoods to that family and the corpus, the holdout and
    # the pinned test set all inherited it: 29 of 30 test baselines were
    # rl-bb-init-*. Round-robin over families spends the anchor budget on
    # distinct architectures instead, which is the whole reason this pipeline
    # is architecture-agnostic.
    # Round-robin over content groups, biggest group first. Biggest first
    # because build() stops once it has n_anchors neighbourhoods, so the head
    # of the pool should be the models most likely to return one; round-robin
    # because taking one per group per pass means the head is n *different*
    # kinds of architecture. Nothing is excluded -- a model alone in its group
    # is at the back of the pool, not missing from it.
    # The dataset ships a 128-permutation MinHash of every model's source in
    # ab/nn/stat/nn, so the clusters are known before a single query is made
    # and the anchors can be one per cluster, largest first. Measured on 2,500
    # models: 575 clusters in 0.2s, of which 38 span more than one name prefix
    # -- and one prefix ("divpo") is spread over 119 of them. The prefix was
    # wrong in both directions. Code shape is the fallback for a model with no
    # signature, so nothing is excluded for want of one.
    mh = minhash_clusters.cluster(names)
    if mh:
        group_key = lambda n: mh.get(n) or f"shape:{shape_of.get(n)}"
    else:
        group_key = lambda n: shape_of.get(n)
    pool = stratified_sample(names, want, rng, key=group_key,
                             min_members=MIN_ANCHOR_FAMILY, biggest_first=True,
                             raw_key=True)
    if len(pool) < want:                      # groups exhausted; top up
        extra = [x for x in names if x not in set(pool)]
        rng.shuffle(extra)
        pool.extend(extra[:want - len(pool)])
    groups = len({group_key(n) for n in pool})
    print(f"[edit-pairs] anchor pool: {len(pool)} candidate(s) over {groups} "
          f"similarity group(s) from {len(names)} model(s)"
          + ("" if mh else " (code-shape grouping: no MinHash signatures)"),
          flush=True)
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
    t_parse = time.time()
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
    # Every returned row is AST-parsed twice (validate_model, and later
    # uses_pretrained_weights). At 600 rows of ~700 lines that is not free, and
    # it is invisible in a profile taken from the outside -- so it is counted.
    if os.environ.get("NNGPT_ANCHOR_TRACE", "1") not in ("0", "false", "no"):
        print(f"[edit-pairs]   anchor {anchor or '(auto)'}: {raw} row(s) -> "
              f"{len(out)} usable in {time.time() - t_parse:.1f}s", flush=True)
    if raw and not out and dropped_prefix:
        print(f"[edit-pairs] band {band}: all {raw} models were removed by "
              f"nn_prefixes={nn_prefixes}; the band selects by architecture "
              f"similarity across the whole dataset, so set NN_PREFIXES=None",
              flush=True)
    return out


def load_all(task, dataset, metric, epoch, nn_prefixes=None,
             max_rows=20000) -> List[Record]:
    """Every model for this pipeline, in one query.

    The band loop asks the database which models resemble an anchor, one anchor
    at a time, and that is how the corpus was discovered. It does not have to
    be: ab/nn/stat/nn carries a MinHash of every model, so the groups can be
    computed here and the only thing left to fetch is the code and the
    accuracy -- which is a single query, the same one sample_anchors already
    makes.

    Measured reason to prefer it: with anchors chosen one per MinHash cluster,
    the band query returned 2, 2, 4 and 6 models for the four largest clusters
    (2475, 2106, 1577, 1320 members). LEMUR's "high" band and the signatures
    disagree, so probing for a neighbourhood that the signatures already
    describe spends a query per anchor to get a worse answer.
    """
    import ab.nn.api as lemur
    df = lemur.data(task=task, dataset=dataset, metric=metric, epoch=epoch,
                    only_best_accuracy=True, max_rows=max_rows)
    out: List[Record] = []
    for _, row in df.iterrows():
        nn, code, acc = row.get("nn"), row.get("nn_code"), row.get("accuracy")
        if not nn or not code or acc is None:
            continue
        if nn_prefixes and not any(str(nn).startswith(p) for p in nn_prefixes):
            continue
        if validate_model(code):
            continue
        out.append(Record(str(nn), code, float(acc), 1,
                          dict(row.get("prm") or {}),
                          row.get("transform") or "",
                          row.get("transform_code") or ""))
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

The learning rate, the momentum and the dropout are read from the training configuration, not \
from the file, so no edit can change them. To change them, add one <hp> block after your edits:

<hp>{{"lr": 0.003, "momentum": 0.95, "dropout": 0.1}}</hp>

Give only the ones you are changing, and omit the block entirely if you are changing none. The \
baseline's values are stated below; they are the ones it was measured at, not ones chosen for your \
architecture, and a design that wants a different learning rate has to ask for it here.

The <hp> block is not limited to those three. If your design needs a hyperparameter of its own -- \
a normalisation epsilon, a patch size, a mixup weight, a flag -- name it in \
supported_hyperparameters() and give it a value in <hp>. Those two have to agree exactly: \
supported_hyperparameters() returns the names the file really takes out of prm, every one of them \
gets its value from <hp> or from the standard set, and a name in neither place is a prm key that \
will not exist when the file reads it.

The input pipeline is also not in the file, and it is scored with your architecture: the baseline's \
accuracy below was measured with the module shown under it, and yours will be measured with the one \
you return. So every answer ends with exactly one <tr> block, holding a complete module with the \
same entry point, `def transform(_)` returning a `torchvision.transforms.Compose`:

<tr>
import torchvision.transforms as transforms
def transform(_):
    return transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
</tr>

This block is required, not optional. If the baseline's pipeline is already the right one for your \
architecture, return it unchanged -- but say so deliberately rather than by leaving the block out. \
One epoch is a single pass, so augmentation that speeds early convergence is worth more here than \
augmentation that only pays off over many epochs. Nothing else about the training protocol is \
yours to set.

Keep class Net, supported_hyperparameters, train_setup and learn: the file is evaluated through \
them. Everything you read out of prm must be named in supported_hyperparameters, and everything \
named there must be read: the framework supplies exactly the declared keys, so index prm directly \
as prm['lr'] and never defend against a missing one -- a key that is absent is an error in the \
declaration, not something to work around.

Build every layer in __init__ and move it to the device you were given; nothing may be constructed \
inside forward, which leaves the new weights on the CPU while the data is on the GPU.

torchvision.transforms.Normalize takes two sequences, a mean and a standard deviation, not six \
numbers: write transforms.Normalize(*norm) and let the dataset supply them. Use only PyTorch, torchvision and torchvision.transforms. Pretrained backbones are allowed: which backbone, how many, where each is cut, how their features are fused, and the stem and head built around them for this input size are all yours to choose. The accuracy to beat below was measured under the same rules, so simply loading a pretrained backbone is not by itself an improvement over it.
One epoch is a single pass over the data, so favour fast early convergence over regularisation \
that only pays off over many epochs.

Answer in this order and nothing else: the <edit> blocks (or one <nn> block), then <hp> if you are \
changing a hyperparameter, then the required <tr> block, then one <reason> block of one short \
sentence per change. Begin with <edit> or <nn> immediately and write no prose before or between \
the blocks."""


USER_PROMPT = """\
Baseline model, measured on '{dataset}' for task '{task}' after exactly {epoch} epoch of \
training. The accuracy below was produced by the same protocol that will be used to score you, so \
it is directly comparable to your own result.

Measured '{metric}': {accuracy:.4f}
Hyperparameters: {hyper}

<nn>{nn_code}</nn>
{transform}{examples}
Return edits that score above {target:.4f} under that same {epoch}-epoch protocol. That is the \
best model recorded in this corpus, so matching the baseline is not enough.
Change as much as you can justify and no more: every line you edit is a line that can break, and \
a file that does not run scores nothing."""




#: How many measured neighbours to show alongside the baseline.
#:
#: This is the prompt-length dial. The window is 16k; a ~700-line baseline is
#: ~9k tokens, and four examples take most of what is left -- which is why the
#: run lowered the minimum answer budget to 256 tokens to make its prompts fit
#: and then lost 71 generations to truncation. Fewer examples is the fix that
#: does not cost answer room. Both the corpus and the test prompts read this,
#: so they cannot drift apart.
CONTEXT_EXAMPLES = int(os.environ.get("NNGPT_CONTEXT_EXAMPLES", "4") or 4)


def transform_prompt_block(record: Record) -> str:
    """The baseline's input pipeline, so an edit can reason about changing it.

    Without it the <tr> block the system prompt offers is unusable: the model
    cannot keep an entry point it has never seen, and it cannot tell whether
    the baseline already crops and flips. Transform modules are a few lines, so
    this costs far less than one neighbour example.
    """
    code = (record.transform_code or "").strip()
    if not code:
        return ""
    name = record.transform or "the configured transform"
    return (f'\n<tr name="{name}">{code}</tr>\n')


def transform_answer_block(a: Record, b: Record) -> str:
    """The <tr> block for the target's pipeline. Always, when one is known.

    This used to fire only when the target's transform differed from the
    baseline's, which on a single-dataset corpus is almost never: measured,
    zero generations across a whole cycle contained a <tr> block, at epoch 0,
    before any fine-tuning. The prompt asked for it as an optional extra and
    the model declined; a corpus whose targets never carry one would then have
    taught the omission from cycle 1 regardless.

    So the block is now unconditional and the prompt requires it. A target that
    keeps the baseline's pipeline says so by returning it, which is also what
    the prompt asks a generation to do -- train and test demand the same thing.
    """
    code = (b.transform_code or a.transform_code or "").strip()
    if not code:
        return ""
    return f"\n<tr>{code}</tr>"


def render_context(baseline: Record, neighbours: Sequence[Record],
                   limit: Optional[int] = None) -> str:
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
    # Read at call time, not bound as a default: a caller that sets
    # CONTEXT_EXAMPLES on this module (or NNGPT_CONTEXT_EXAMPLES, after this
    # module was imported) must be able to change it, and a default argument is
    # fixed at import.
    limit = CONTEXT_EXAMPLES if limit is None else limit
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
        hyper=hp_text(a.prm), transform=transform_prompt_block(a),
        examples=context, target=max(ceiling, a.observed))
    reason = (f"{describe_edits(a.code, b.code)}. Measured {b.observed:.3f} "
              f"against the baseline's {a.observed:.3f}, "
              f"{b.observed - a.observed:+.3f}.")
    hp = {k: b.prm[k] for k in TUNABLE_PRM
          if k in b.prm and b.prm.get(k) != a.prm.get(k)}
    hp_block = f"\n<hp>{json.dumps(hp, sort_keys=True)}</hp>" if hp else ""
    tr_block = transform_answer_block(a, b)
    assistant = f"{body}{hp_block}{tr_block}\n<reason>{reason}</reason>"
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
                 "transform_changed": bool(tr_block),
                 # Which family the pair came from, so the sampler that feeds
                 # the trainer can balance over families instead of over rows.
                 "baseline_family": family_of(a.nn),
                 "target_family": family_of(b.nn),
                 "task": "improve",
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


#: A component may not swallow more than this share of the corpus.
#:
#: Similarity is not transitive: A resembles B and B resembles C does not make
#: A resemble C, so merging overlapping bands transitively can collapse the
#: whole corpus into one group. Past this share the merge is abandoned and each
#: model keeps the first band that returned it, which is a coarser grouping but
#: never a degenerate one.
MAX_COMPONENT_SHARE = float(os.environ.get("NNGPT_MAX_COMPONENT_SHARE", "0.6") or 0.6)


def similarity_components(clusters, cluster_keys) -> Dict[str, str]:
    """Group every fetched model by similarity alone, with no reference to names.

    Each band is a set of models LEMUR's MinHash says are comparable. Two bands
    that share a member are talking about the same region of architecture
    space, so the bands are merged by union-find and the resulting components
    are a partition of everything fetched -- a clustering over the models
    themselves rather than over anchors, and the closest thing to a global
    clustering that costs no extra queries.

    Guarded, because transitive merging can collapse: see MAX_COMPONENT_SHARE.
    """
    parent: Dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for members in clusters:
        names = [r.nn for r in members]
        for n in names:
            find(n)
        for n in names[1:]:
            union(names[0], n)

    total = len(parent)
    groups: Dict[str, List[str]] = collections.defaultdict(list)
    for n in parent:
        groups[find(n)].append(n)
    biggest = max((len(v) for v in groups.values()), default=0)

    if total and biggest / total > MAX_COMPONENT_SHARE:
        first_band: Dict[str, str] = {}
        for key, members in zip(cluster_keys, clusters):
            for r in members:
                first_band.setdefault(r.nn, key)
        print(f"[edit-pairs] similarity merge abandoned: one component held "
              f"{100.0 * biggest / total:.0f}% of {total} models (limit "
              f"{100.0 * MAX_COMPONENT_SHARE:.0f}%); falling back to "
              f"{len(set(first_band.values()))} band-of-origin group(s)",
              flush=True)
        return first_band

    # Name each component after one of its own members, so the id is stable
    # for a given corpus and means something in a log.
    label = {root: f"sim:{sorted(members)[0]}" for root, members in groups.items()}
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    print(f"[edit-pairs] {len(groups)} similarity component(s) over {total} "
          f"model(s); largest {sizes[:5]}", flush=True)
    return {n: label[find(n)] for n in parent}


def holdout_members(names, holdout_frac, seed, pinned_seed_file=None,
                    min_per_family=None, group_of=None) -> set:
    """Which baselines are withheld from training to be the test set.

    The hash is uniform over names, so the holdout mirrors the corpus's family
    mix -- and a corpus that is 90% one family hands back a test set that is
    90% that family, whatever the fraction. ``min_per_family`` adds a floor:
    every group present contributes at least that many baselines to the
    holdout, chosen by the same stable hash so the set is still reproducible
    and still independent of accuracy.

    ``group_of`` says what a group is. The caller passes LEMUR's own MinHash
    neighbourhood, a real similarity measure over the architecture graph; the
    name-prefix fallback is only a heuristic, and an unreliable one -- one
    generator names every batch differently ("MoE4Own-batch-00067-..."), so a
    single family looks like hundreds, while another lumps 1,900
    near-identical models under one prefix.
    """
    import hashlib

    def h_of(n):
        return int.from_bytes(
            hashlib.sha1(f"{seed}:{n}".encode()).digest()[:4], "big") / 2**32

    pinned = set()
    if pinned_seed_file and Path(pinned_seed_file).is_file():
        try:
            pinned = set(json.loads(Path(pinned_seed_file).read_text(
                encoding="utf-8")).get("nn") or [])
        except Exception:
            pinned = set()
    out = set(pinned)
    for n in names:
        if h_of(n) < holdout_frac:
            out.add(n)
    if min_per_family:
        key = group_of or family_of
        by_family = collections.defaultdict(list)
        for n in names:
            by_family[key(n)].append(n)
        for fam, members in by_family.items():
            have = sum(1 for n in members if n in out)
            if have >= min_per_family:
                continue
            for n in sorted(members, key=h_of):
                if n in out:
                    continue
                out.add(n)
                have += 1
                if have >= min_per_family:
                    break
    return out


#: How many measured standard deviations a pair's gain must clear.
NOISE_SIGMAS = float(os.environ.get("NNGPT_NOISE_SIGMAS", "3") or 3)


def _noise_floor(out_dir, configured) -> float:
    """The larger of the configured gain and what repeat measurements say.

    Reads the repeat spread stage2.py recorded for this run. Absent (cycle 0,
    or the ladder turned off) the configured value stands, so this can only
    ever tighten the threshold, never loosen it.
    """
    try:
        from ab.gpt.util.Const import nngpt_dir
        paths = sorted(Path(nngpt_dir).glob("llm/epoch/*/synth_nn/stage2_results.json"))
    except Exception:
        return configured
    stds = []
    for p in paths:
        try:
            payload = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            continue
        v = payload.get("repeat_std")
        if isinstance(v, (int, float)) and v > 0:
            stds.append(float(v))
    if not stds:
        return configured
    measured = NOISE_SIGMAS * (sum(stds) / len(stds))
    if measured <= configured:
        print(f"[edit-pairs] noise floor: {NOISE_SIGMAS:.0f} sigma over "
              f"{len(stds)} stage-2 cycle(s) is {measured:.4f}, below the "
              f"configured {configured:.4f}; keeping the configured value",
              flush=True)
        return configured
    print(f"[edit-pairs] noise floor raised {configured:.4f} -> "
          f"{measured:.4f} ({NOISE_SIGMAS:.0f} sigma of the measured repeat "
          f"spread over {len(stds)} cycle(s))", flush=True)
    return measured


#: Smallest cluster a test baseline may be taken from.
#:
#: A seed needs neighbours: training prompts carry example edits from the
#: baseline's own cluster, and a seed whose cluster has none is tested on a
#: prompt shape it never saw. Holding one member out of a cluster of 3 leaves 2
#: behind, which is enough for context.
SEED_CLUSTER_MIN = int(os.environ.get("NNGPT_SEED_CLUSTER_MIN", "3") or 3)

#: Name prefixes a test baseline may never be drawn from, comma separated.
#:
#: Set to this run's own nn_name_prefix. A model this pipeline produced is good
#: training data -- it is measured, and a (parent -> child) pair carrying a
#: real gain is exactly the signal the corpus is built on -- but it is poor
#: test data: a benchmark made of your own outputs measures editing your own
#: outputs, and it drifts, because every cycle changes what the test set is
#: made of. That is the opposite of the frozen benchmark the pinned seeds are
#: meant to be.
#:
#: Without this the fraction is left to cluster-size accidents: generated
#: models cluster together, their clusters grow past SEED_CLUSTER_MAX, and they
#: drop out of the holdout eventually -- but a new one unlike the others still
#: lands in a small cluster and can be drawn. Measured: 10 of 60 held-out
#: baselines were llm-*, and 7 of 30 pinned seeds in an earlier run.
#:
#: A prefix is the right instrument here and nowhere else: "llm-" is a fact
#: about where the model came from, not a claim about what it looks like.
#: Read at call time, not bound at import: the run script sets the variable
#: inside prepare(), which runs after this module is imported, so a constant
#: evaluated here would always be empty.
def holdout_exclude_prefixes() -> tuple:
    return tuple(p.strip() for p in os.environ.get(
        "NNGPT_HOLDOUT_EXCLUDE_PREFIXES", "").split(",") if p.strip())


#: How many candidate baselines to withhold, as a multiple of the test set.
#:
#: Taking one from every eligible cluster withheld 173 models so that 30 could
#: be used -- about 143 removed from the training pairs for no benefit, since
#: nothing is ever measured on them. A reserve is still wanted, because
#: emit_pinned_seeds picks by accuracy and a set of exactly 30 would leave it no
#: choice at all, but twice the test set is enough for that.
HOLDOUT_RESERVE = float(os.environ.get("NNGPT_HOLDOUT_RESERVE", "2") or 2)


#: Largest cluster a test baseline may be taken from.
#:
#: The pair-rich neighbourhoods are the large ones, and the hashed holdout used
#: to take members out of them -- punching holes in exactly the clusters the
#: corpus is built from. Above this size a cluster is never touched and
#: contributes every member to training.
SEED_CLUSTER_MAX = int(os.environ.get("NNGPT_SEED_CLUSTER_MAX", "10") or 10)


def _excluded_from_holdout(name, prefixes=None) -> bool:
    """Is this model barred from the test set by where it came from?"""
    prefixes = holdout_exclude_prefixes() if prefixes is None else prefixes
    if not prefixes:
        return False
    text = str(name or "")
    return any(text.startswith(p if p.endswith("-") else p + "-")
               for p in prefixes)


def holdout_from_small_clusters(cluster_id_of, min_size=None, max_size=None,
                                seed=17, pinned_seed_file=None,
                                min_holdout=30, per_cluster=1) -> set:
    """Test baselines taken one per small cluster, leaving large ones whole.

    Two things the hashed holdout could not do. It drew uniformly over models,
    so it removed members from the biggest clusters -- the ones that supply
    almost all the training pairs -- and it could leave a seed with no
    neighbours at all, which is a train/test prompt mismatch.

    Here a cluster in [min_size, max_size] contributes exactly one baseline and
    keeps the rest for training, so every seed is guaranteed real neighbour
    context; a cluster above max_size contributes none and stays intact. One
    seed per cluster also means 30 seeds are 30 distinct architectures, which
    makes "this architecture already exists in the database" unlikely rather
    than routine.

    Selection inside a cluster is by a stable hash, so it is reproducible and
    independent of accuracy. The band widens upward only if it cannot fill
    min_holdout.
    """
    import hashlib

    def h_of(n):
        return int.from_bytes(
            hashlib.sha1(f"{seed}:{n}".encode()).digest()[:4], "big") / 2**32

    min_size = SEED_CLUSTER_MIN if min_size is None else min_size
    max_size = SEED_CLUSTER_MAX if max_size is None else max_size

    pinned = set()
    if pinned_seed_file and Path(pinned_seed_file).is_file():
        try:
            pinned = set(json.loads(Path(pinned_seed_file).read_text(
                encoding="utf-8")).get("nn") or [])
        except Exception:
            pinned = set()

    members_of = collections.defaultdict(list)
    for name, cid in cluster_id_of.items():
        members_of[cid].append(name)

    target = max(min_holdout, int(min_holdout * max(1.0, HOLDOUT_RESERVE)))

    def draw(hi):
        out = set(pinned)
        eligible = []
        for cid, members in members_of.items():
            if min_size <= len(members) <= hi:
                eligible.append((cid, members))
        # Clusters that already hold a pinned seed come first -- the pinned set
        # is the frozen benchmark and cannot be dropped -- then the rest in a
        # stable hashed order, so the reserve is reproducible and is not the
        # database's own ordering.
        eligible.sort(key=lambda kv: (
            0 if any(m in pinned for m in kv[1]) else 1,
            hashlib.sha1(f"{seed}:{kv[0]}".encode()).hexdigest()))
        for cid, members in eligible:
            if len(out) >= target:
                break
            # Never take so many that fewer than 2 are left for context.
            room = min(per_cluster, max(0, len(members) - 2))
            if room < 1:
                continue
            if any(m in out for m in members):
                continue
            takeable = [m for m in sorted(members, key=h_of)
                        if not _excluded_from_holdout(m)]
            if not takeable:
                continue
            for m in takeable[:room]:
                out.add(m)
        return out, len(eligible)

    hi = max_size
    held, eligible = draw(hi)
    while len(held) < min_holdout and hi < 10 ** 6:
        hi *= 2
        held, eligible = draw(hi)
        print(f"[edit-pairs] seed cluster band widened to "
              f"[{min_size}, {hi}] to reach {len(held)} test baseline(s)",
              flush=True)
    _prefixes = holdout_exclude_prefixes()
    if _prefixes:
        barred = sum(1 for n in cluster_id_of if _excluded_from_holdout(n, _prefixes))
        print(f"[edit-pairs] holdout: {barred} model(s) with prefix "
              f"{'/'.join(_prefixes)} are training-only "
              f"(this run's own output is not its own benchmark)", flush=True)
    print(f"[edit-pairs] holdout: {len(held)} baseline(s), one per cluster, "
          f"from {eligible} cluster(s) sized [{min_size}, {hi}] "
          f"(reserve {HOLDOUT_RESERVE:g}x of {min_holdout}); the other "
          f"{max(0, eligible - len(held))} eligible and "
          f"{len([1 for m in members_of.values() if len(m) > hi])} larger "
          f"cluster(s) stay whole for training", flush=True)
    return held


def build(out_dir, task="img-classification", dataset="cifar-10", metric="acc",
          epoch=1, nn_prefixes=None, min_observed_gain=0.05, max_changed_lines=40,
          per_baseline=12, val_frac=0.08, holdout_frac=0.12, seed=17,
          offline_root=None, pinned_seed_file=None,
          max_bucket=MAX_BUCKET, bands=DEFAULT_BANDS, band_size=600,
          n_anchors=24, min_holdout=30, holdout_min_per_family=2) -> dict:
    rng = random.Random(seed)
    cfg = dict(task=task, dataset=dataset, metric=metric, epoch=epoch)

    # A pair rests on two single training runs, so the gain it claims has to
    # clear the noise of one. stage2.py re-trains the best candidates several
    # times and writes the spread it measures; once that number exists it is a
    # better floor than a constant guessed in advance. Measured on one run: the
    # same model scored 0.9562 and 0.9621, so a 0.05 threshold admits pairs
    # whose entire gain could be noise.
    min_observed_gain = _noise_floor(out_dir, min_observed_gain)

    # A cluster is a set of architecturally similar models: one similarity band
    # from LEMUR, or -- offline, where there is no database -- the class-name
    # grouping this module used before. Pairs are formed inside a cluster, so
    # the two models are comparable and the diff between them is small.
    clusters: List[List[Record]] = []
    # A label per cluster, so every later stage can group models by the
    # similarity neighbourhood they came from instead of by the shape of their
    # name. Named after the anchor the band was centred on, which is
    # deterministic for a given seed and readable in a log.
    cluster_keys: List[str] = []
    if offline_root:
        records = load_offline(offline_root, task, dataset, metric, epoch, nn_prefixes)
        grouped = collections.defaultdict(list)
        for r in records:
            grouped[bucket_key(r.code)].append(r)
        clusters = [v for v in grouped.values() if len(v) >= 2]
        cluster_keys = [f"offline:{i}" for i in range(len(clusters))]
    elif minhash_clusters.enabled() and minhash_clusters.stat_nn_dir() is not None:
        # One query, then group by signature. No probing and no anchors: the
        # clusters are a property of the models, already on disk, so there is
        # nothing to discover and nothing to guess from a name.
        t0 = time.time()
        records = load_all(task, dataset, metric, epoch, nn_prefixes)
        print(f"[edit-pairs] {len(records)} usable model(s) loaded in "
              f"{time.time() - t0:.0f}s", flush=True)
        if len(records) < 2:
            raise RuntimeError(f"only {len(records)} usable models in one query")
        ids = minhash_clusters.cluster([r.nn for r in records])
        grouped: Dict[str, List[Record]] = collections.defaultdict(list)
        for r in records:
            grouped[ids.get(r.nn) or f"shape:{bucket_key(r.code)}"].append(r)
        ranked = sorted(((k, v) for k, v in grouped.items() if len(v) >= 2),
                        key=lambda kv: -len(kv[1]))
        if not ranked:
            raise RuntimeError(
                "no similarity cluster holds two models; the signatures in "
                "ab/nn/stat/nn may not cover this pipeline's models")
        # Every cluster with two members, not the largest few. Keeping only
        # the top n would hand the corpus back to the big generators, which is
        # the concentration all of this exists to undo. Cost is bounded where
        # it actually arises: build_pairs trims any cluster to MAX_BUCKET
        # before pairing, so total work scales with the number of clusters
        # rather than with the square of the biggest one.
        try:
            limit = int(os.environ.get("NNGPT_MAX_CLUSTERS", "0") or 0)
        except ValueError:
            limit = 0
        kept = ranked[:limit] if limit > 0 else ranked
        clusters = [v for _k, v in kept]
        cluster_keys = [k for k, _v in kept]
        covered = {r.nn for v in clusters for r in v}
        records = [r for r in records if r.nn in covered]
        print(f"[edit-pairs] corpus from {len(clusters)} multi-member "
              f"cluster(s) of {len(grouped)}: sizes "
              f"{[len(v) for v in clusters[:8]]}..., {len(records)} model(s), "
              f"no band queries and no name prefixes", flush=True)
    else:
        seen_names = set()
        records = []
        anchors = sample_anchors(task, dataset, metric, epoch, n_anchors,
                                 seed=seed, nn_prefixes=nn_prefixes) or [None]
        print(f"[edit-pairs] {len(anchors)} anchor(s) across the dataset")
        # A silent loop over up to 4*n_anchors database joins, each of which
        # can return 600 rows of source code to parse. One measured run spent
        # over half an hour here with nothing on stdout between "N anchor(s)"
        # and the band summary, which is a long time to not know whether a run
        # is working. Each attempt is timed, and the loop gives up once the
        # budget is spent as long as it has something to build a corpus from.
        try:
            anchor_budget = float(os.environ.get("NNGPT_ANCHOR_BUDGET_SECONDS", "600") or 600)
        except ValueError:
            anchor_budget = 600.0
        # Every model a band returns is already in the corpus, and a band
        # centred on one of them would largely repeat it. Skipping those is
        # free -- no query is made -- and it is the only grouping that uses
        # LEMUR's own similarity measure rather than a guess about names: the
        # band itself says which models belong together. The effect is that
        # each query buys a genuinely new neighbourhood, and that no model is
        # ever excluded from the pool for what it is called.
        covered: set = set()
        for band in bands:
            used = missing = empty = skipped = 0
            t_band = time.time()
            for n_try, anchor in enumerate(anchors, 1):
                if used >= n_anchors:
                    break
                if anchor is not None and anchor in covered:
                    skipped += 1
                    continue
                spent = time.time() - t_band
                if anchor_budget and spent > anchor_budget and used:
                    print(f"[edit-pairs] band {band}: stopping the anchor "
                          f"search after {spent / 60:.1f} min with {used} "
                          f"neighbourhood(s) from {n_try - 1} attempt(s) "
                          f"(NNGPT_ANCHOR_BUDGET_SECONDS={anchor_budget:.0f})",
                          flush=True)
                    break
                t0 = time.time()
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
                covered.update(r.nn for r in members)
                cluster_keys.append(f"{band}:{anchor or 'auto'}")
                print(f"[edit-pairs] band {band}: {used}/{n_anchors} from "
                      f"attempt {n_try}/{len(anchors)}: "
                      f"{len(members)} model(s) in {time.time() - t0:.1f}s, "
                      f"{time.time() - t_band:.0f}s elapsed", flush=True)
                clusters.append(members)
                for r in members:
                    if r.nn not in seen_names:
                        seen_names.add(r.nn)
                        records.append(r)
            print(f"[edit-pairs] band {band}: {used} neighbourhood(s) from "
                  f"{len(anchors)} candidate anchor(s) "
                  f"({missing} not in nn_minhash, {empty} too small, "
                  f"{skipped} already inside another band), "
                  f"{len(records)} distinct models")
            if not used:
                raise RuntimeError(
                    f"band {band!r}: no anchor produced a neighbourhood -- "
                    f"{missing} of {len(anchors)} candidates were missing from "
                    f"nn_minhash and {empty} returned fewer than 2 models")
    if len(records) < 50:
        raise RuntimeError(f"only {len(records)} usable records -- check the filters")

    # Same clustering for every downstream decision -- the holdout floor, the
    # pinned test set and the per-cycle SFT sample -- so "one group" means the
    # same thing everywhere. A model whose signature is missing keeps the
    # component of the bands it appeared in.
    mh_records = minhash_clusters.cluster([r.nn for r in records])
    band_components = similarity_components(clusters, cluster_keys)
    cluster_id_of = {}
    for r in records:
        cid = mh_records.get(r.nn) or band_components.get(r.nn)
        if cid:
            cluster_id_of[r.nn] = cid
    if mh_records:
        covered = sum(1 for r in records if r.nn in mh_records)
        print(f"[edit-pairs] clustering: MinHash for {covered} of "
              f"{len(records)} record(s), band components for the rest; "
              f"{len(set(cluster_id_of.values()))} group(s) in total",
              flush=True)
    # Every record came out of a band, so the fallback is for safety only. It
    # is a code-shape key rather than a name prefix: if it ever fires, the
    # answer should still be about the model and not about its name.
    shape_fallback = {r.nn: f"shape:{bucket_key(r.code)}" for r in records}
    group_of = lambda n: (cluster_id_of.get(n) or shape_fallback.get(n)
                          or family_of(n))
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
    known = {r.nn: group_of(r.nn) for r in records}
    holdout = set()
    if SEED_CLUSTER_MIN > 0:
        try:
            holdout = holdout_from_small_clusters(
                known, seed=seed, pinned_seed_file=pinned_seed_file,
                min_holdout=min_holdout)
        except Exception as exc:
            print(f"[edit-pairs] cluster-banded holdout failed "
                  f"({type(exc).__name__}: {exc}); falling back to the hashed "
                  f"holdout", flush=True)
            holdout = set()
    if len(holdout) < min_holdout:
        holdout = holdout_members([r.nn for r in records], holdout_frac, seed,
                                  pinned_seed_file=pinned_seed_file,
                                  min_per_family=holdout_min_per_family,
                                  group_of=group_of)
        for frac in (0.2, 0.3, 0.4):
            if len(holdout) >= min_holdout:
                break
            holdout = holdout_members([r.nn for r in records], frac, seed,
                                      pinned_seed_file=pinned_seed_file,
                                      min_per_family=holdout_min_per_family,
                                      group_of=group_of)
            if len(holdout) >= min_holdout:
                print(f"[edit-pairs] holdout raised to {frac:.0%} to reach "
                      f"{len(holdout)} test baselines", flush=True)
    if len(holdout) < min_holdout:
        raise RuntimeError(
            f"only {len(holdout)} baselines can be held out of {len(records)} "
            f"models, and {min_holdout} are needed for the test set; widen the "
            f"corpus (more anchors or bands) or lower TEST_NN")
    train_pool = [r for r in records if r.nn not in holdout]
    hold_pool = [r for r in records if r.nn in holdout]
    # The two numbers that say whether this corpus can teach anything general.
    print(f"[edit-pairs] records:  {describe_families(r.nn for r in records)}",
          flush=True)
    print(f"[edit-pairs] holdout:  {describe_families(r.nn for r in hold_pool)}",
          flush=True)

    holdout_names = set(holdout)
    train_clusters = [[r for r in c if r.nn not in holdout_names] for c in clusters]
    pairs = build_pairs(train_clusters, min_observed_gain, max_changed_lines,
                        per_baseline, rng, max_bucket=max_bucket)
    if not pairs:
        raise RuntimeError("no pairs at the configured thresholds")

    # A sample check that render_edits -> apply_edits reproduces the target.
    # It used to abort the run on a single mismatch, which was defensible while
    # every pair came from one narrow band: a failure then meant the edit
    # machinery was broken. Over 391 diverse clusters it does not. The usual
    # mismatch is "edit N matches 2 places" -- a <find> block that is not
    # unique in the baseline, which is a property of that one pair and not a
    # bug: 4 of 500 on one corpus.
    #
    # Those pairs are dropped either way. render_example re-applies the edits
    # for every pair it renders and returns None when the result does not come
    # back byte for byte, so an ambiguous <find> never reaches the corpus. The
    # check stays as a tripwire with a threshold: a few percent is the corpus
    # being broad, a large fraction is the machinery being wrong.
    report = verify_round_trip([(a.code, b.code) for a, b in pairs])
    rendered = report.get("rendered") or 0
    rate = (report["mismatch"] / rendered) if rendered else 0.0
    try:
        limit = float(os.environ.get("NNGPT_ROUND_TRIP_LIMIT", "0.1") or 0.1)
    except ValueError:
        limit = 0.1
    if rate > limit:
        raise RuntimeError(
            f"edit round trip failed for {100 * rate:.1f}% of the sample "
            f"(limit {100 * limit:.0f}%): {report}")
    if report["mismatch"]:
        print(f"[edit-pairs] edit round trip: {report['mismatch']} of "
              f"{rendered} sampled pair(s) do not round-trip "
              f"({100 * rate:.1f}%, first: {report['first_failure']}); those "
              f"pairs are dropped when the corpus is rendered", flush=True)
    else:
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
    # So the per-cycle sampler can balance over similarity clusters rather than
    # over name prefixes. render_example cannot know this -- only build() has
    # the band membership -- so it is attached here.
    for e in examples:
        meta = e["meta"]
        meta["baseline_cluster"] = cluster_id_of.get(meta["baseline_nn"], "")
        meta["target_cluster"] = cluster_id_of.get(meta["target_nn"], "")
    rng.shuffle(examples)
    n_val = max(1, int(len(examples) * val_frac))
    val, train = examples[:n_val], examples[n_val:]

    # Auxiliary tasks over the same training pool. Added after the split, and
    # after the statistics below are computed over `examples`, so the
    # improvement corpus is reported exactly as before and the only thing that
    # changes is what the trainer is fed. Nothing here sees a held-out
    # baseline: the run is still validated on the improvement task alone.
    aux = []
    try:
        from ab.gpt.util.Const import nngpt_dir as _nngpt_dir
        from ab.gpt.util.data import aux_tasks
        aux = aux_tasks.build_all(
            train_pool, pairs, train_clusters, cfg, seed=seed,
            nngpt_dir=_nngpt_dir, n_improve=len(train))
        for e in aux:
            meta = e["meta"]
            base = meta.get("baseline_nn")
            if base:
                meta["baseline_cluster"] = cluster_id_of.get(base, "")
        train = train + aux
        rng.shuffle(train)
    except Exception as exc:
        print(f"[aux] auxiliary tasks skipped: {type(exc).__name__}: {exc}",
              flush=True)

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
    # Written once. Every cycle rebuilds the corpus, and rewriting this file
    # each time silently changed the 30 test prompts underneath a ceiling
    # (NNGPT_DESIGN_CEILING) that stays frozen at cycle 0 -- so the benchmark
    # drifted while the target it was measured against did not. The pinned
    # seeds are fixed for the run; their context has to be too.
    ctx_path = out / "seed_context.json"
    freeze = os.environ.get("NNGPT_FREEZE_SEED_CONTEXT", "1").strip().lower() \
        not in ("0", "false", "no", "off")
    if ctx_path.is_file() and freeze:
        try:
            kept = json.loads(ctx_path.read_text(encoding="utf-8"))
        except Exception:
            kept = {}
        # Merged, not compared. Two writers share this file -- build() writes
        # every held-out baseline, prepare() writes only the 30 pinned ones --
        # so an all-or-nothing comparison never matched and the file was
        # rewritten every cycle, which is the drift the freeze exists to stop.
        # A baseline that already has context keeps it byte for byte; new ones
        # are added. The test prompts therefore cannot change under a frozen
        # ceiling, while the holdout is still free to grow.
        frozen = sum(1 for n, v in seed_ctx.items() if kept.get(n))
        for name, old_ctx in kept.items():
            if old_ctx and name in seed_ctx:
                seed_ctx[name] = old_ctx
        print(f"[edit-pairs] seed context: {frozen} baseline(s) kept frozen, "
              f"{len(seed_ctx) - frozen} new", flush=True)
        ctx_path.write_text(json.dumps(seed_ctx, indent=1), encoding="utf-8")
    else:
        ctx_path.write_text(json.dumps(seed_ctx, indent=1), encoding="utf-8")
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
         [{"nn": r.nn, "observed": round(r.observed, 4),
           # The test-set picker caps how many baselines one group may supply;
           # without this it had only the name to group by.
           "cluster": group_of(r.nn)} for r in hold_pool])

    changed = sorted(e["meta"]["changed_lines"] for e in examples)
    stats = {
        "records": len(records),
        "architectures": len({bucket_key(r.code) for r in records}),
        "train_baselines": len(train_pool), "holdout_baselines": len(hold_pool),
        "pairs": len(pairs), "train_examples": len(train), "val_examples": len(val),
        "min_observed_gain": min_observed_gain, "max_changed_lines": max_changed_lines,
        "noise_sigmas": NOISE_SIGMAS,
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
        # Family concentration, the quantity that decides whether the test set
        # measures generalisation or one search family's local neighbourhood.
        "record_families": family_histogram(r.nn for r in records),
        "holdout_families": family_histogram(r.nn for r in hold_pool),
        "record_clusters": len(set(cluster_id_of.values())),
        "holdout_clusters": len({group_of(r.nn) for r in hold_pool}),
        "seed_cluster_band": [SEED_CLUSTER_MIN, SEED_CLUSTER_MAX],
        "holdout_reserve": HOLDOUT_RESERVE,
        "holdout_exclude_prefixes": list(holdout_exclude_prefixes()),
        "holdout_own_output": sum(
            1 for r in hold_pool if _excluded_from_holdout(r.nn)),
        # One baseline per cluster means this should equal holdout_baselines.
        "holdout_per_cluster_max": max(
            (collections.Counter(group_of(r.nn) for r in hold_pool).values()),
            default=0),
        "pair_baseline_families": family_histogram(
            e["meta"]["baseline_nn"] for e in examples),
        "transform_changed_pairs": sum(
            1 for e in examples if e["meta"].get("transform_changed")),
        # How many targets carry the <tr> block the prompt requires. If this is
        # 0 the model is being asked for a format the corpus never shows, which
        # is exactly how the transform axis stayed empty for a whole run.
        "pairs_with_transform_block": sum(
            1 for e in examples
            if "<tr>" in e["messages"][2]["content"]),
        # Training rows that are not the improvement task. The run is still
        # scored on improvement only; these exist so the loss cannot be
        # lowered by learning the shape of one answer.
        "aux_rows": len(aux),
        "aux_by_task": dict(collections.Counter(
            e["meta"].get("task") for e in aux)),
        "train_rows_total": len(train),
        "round_trip": report,
        # Said plainly: there is no factor model here, so a pair rests on two
        # single training runs and the only guard is the size of the gain.
        "noise_guard": "observed gain only (no factor model)",
    }
    with open(out / "corpus_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats
