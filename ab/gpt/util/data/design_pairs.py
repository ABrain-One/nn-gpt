"""Build the SFT corpus for design-space improvement fine-tuning.

Why the production pairing cannot work
--------------------------------------
``NN_gen_train_nn_improve.json`` sets ``no_repeat: ["nn"]``, ``improve: 0.05``,
``num_joint_nns: 2``. In ``nn-dataset/ab/nn/util/db/Query.py`` the partner is
then resolved by ``ORDER BY RANDOM() ... LIMIT 1`` among any *different* model
scoring at least 0.05 higher. The prompt asks the model to edit the baseline;
the label is an architecturally unrelated network. No function maps that prompt
to that target, so SFT can only learn "ignore the input, emit a memorised
network".

Two measurement problems compound it, both quantified on LEMUR cifar-10 at
epoch 1:

* Seed noise is large at a one-epoch budget. Over 1,218 repeat groups -- same
  model, every hyperparameter identical, misfiled records excluded -- the
  median sd is 0.025 and 41.8% of groups span more than 0.08 accuracy.
* Taken over 9,664 pairs whose true improvement is zero, the chance that noise
  alone clears a 0.05 threshold is **16.4%**. Roughly one production training
  pair in six is pure noise. A 0.10 threshold still leaves 8.9%, and the tail
  is heavy (p99 null gap 0.39), so no threshold fixes this on its own.

The GenFractalNet design space
------------------------------
``ab/nn/nn/GenFractalNet-*.py`` is a designed experiment: 3,846 models trained
on cifar-10 for one epoch, each a distinct point in a **13-factor** space. Nine
factors are declared in the file header; four more live in the body and are
easy to miss:

    header:  Optimizer, Norm Type, LR, Momentum, FC Dropout, Pooling,
             Kernel, Activation, Conv Type
    body:    Drop Path (FractalDropPath drop_prob), Fractal Columns,
             Start Channels, Total Blocks

With all thirteen extracted, the 3,846 models occupy 3,846 *distinct* design
points with zero collisions -- the design point determines the architecture
exactly. With only the nine header factors, 602 points collide while their code
still differs, which would make any claim of the form "only these two decisions
changed" false.

Why pairing uses a fitted model, not the raw numbers
----------------------------------------------------
Each design point was trained once, so its observed accuracy carries a full
measurement error. The *factor effects*, by contrast, are estimated across
hundreds to thousands of models each.

A ridge regression on the 13 factors plus all two-way interactions (354
parameters) achieves **5-fold CV RMSE 0.046, R2 0.86** against an observed
spread of 0.126, and its fitted gains correlate 0.885 with observed gains. The
fitted value is therefore a much better estimate of a design point's quality
than its own single run, and pairing on it removes the noise problem at its
source instead of trying to threshold it away.

A pair is kept only when **both** signals agree: the fitted gain clears
``min_fitted_gain`` and the observed gain is at least ``min_observed_gain``.
The fitted value guards against seed noise; the observed value guards against
model misspecification. Neither alone is trusted.

Adjusted main effects (regression, interactions averaged over the corpus).
These differ substantially from raw marginal means, which are confounded
because the space is sampled at only 0.46% density:

    Optimizer        AdamW 0.508  ->  SGD 0.340           +0.169
    Norm Type    BatchNorm 0.499  ->  InstanceNorm 0.376  +0.123
    Drop Path          0.0 0.497  ->  0.3 0.395           +0.102
    Start Channels      64 0.481  ->  16 0.417            +0.064
    FC Dropout         0.0 0.472  ->  0.5 0.425           +0.047
    Pooling            Max 0.472  ->  Avg 0.432           +0.041
    Total Blocks         3 0.474  ->  2 0.436             +0.038
    Conv Type     Standard 0.471  ->  Depthwise 0.436     +0.034
    Fractal Columns      2 0.473  ->  3 0.441             +0.031
    Kernel               5 0.467  ->  3 0.445             +0.022
    Activation        GELU 0.468  ->  ReLU 0.446          +0.022
    LR               0.003 0.466  ->  0.01 0.449          +0.017
    Momentum          0.92 0.461  ->  0.75 0.453          +0.008

Median observed accuracy is 0.472 and the best observed model reaches 0.730,
so there is roughly 0.26 of real headroom for a model that learns this space.

Data hygiene
------------
LEMUR's ``<n>.json`` holds a run measured at epoch n: ``accuracy`` is the value
at that epoch and ``train_stat.best_epoch`` is the best epoch so far, so
``best_epoch <= n`` must hold. 619 records across the dataset violate that --
whole-run summaries misfiled into ``1.json`` and ``5.json``, 479 of them
verbatim duplicates of a later file's record. One reports 0.9000 for a model
whose true epoch-1 accuracy is near 0.38. GenFractalNet is clean, but this
module filters defensively so the corpus can never ingest one.

Output
------
``train.jsonl`` / ``val.jsonl`` in the ``{"messages": [...]}`` chat format that
``ab/gpt/util/prompt/NNGenPrompt._raw_dataset_from_disk`` already consumes, so
passing ``data_dir`` to ``Tune.main`` bypasses the LEMUR join with no change to
the SFT path.

Usage
-----
    python -m ab.gpt.util.data.design_pairs --out ab/gpt/data/design_improve
    python -m ab.gpt.util.data.design_pairs --out /tmp/c --offline-root <nn-dataset>/ab/nn
"""

from __future__ import annotations

import argparse
import itertools
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# The 13-factor design space
# --------------------------------------------------------------------------

HEADER_KEYS: Tuple[str, ...] = (
    "Optimizer", "Norm Type", "LR", "Momentum", "FC Dropout",
    "Pooling", "Kernel", "Activation", "Conv Type",
)

#: Factors that live in the body. Missing these is not a detail: without them
#: 602 design points collide while their code differs, and Drop Path alone is
#: the third-largest effect in the whole space.
BODY_PATTERNS: Dict[str, str] = {
    "Drop Path":       r"def __init__\(self, drop_prob: float = ([0-9.]+)\)",
    "Fractal Columns": r"FractalBlock\(int\((\d+)\)",
    "Start Channels":  r"start_chan\s*=\s*int\((\d+)\)",
    "Total Blocks":    r"total_blocks\s*=\s*int\((\d+)\)",
}

DESIGN_KEYS: Tuple[str, ...] = HEADER_KEYS + tuple(BODY_PATTERNS)

KEY_PHRASE: Dict[str, str] = {
    "Optimizer": "the optimiser",
    "Norm Type": "the normalisation layer",
    "Drop Path": "the drop-path probability",
    "Start Channels": "the stem width",
    "FC Dropout": "the classifier dropout",
    "Pooling": "the pooling operator",
    "Total Blocks": "the number of fractal blocks",
    "Conv Type": "the convolution type",
    "Fractal Columns": "the fractal column count",
    "Kernel": "the convolution kernel size",
    "Activation": "the activation function",
    "LR": "the learning rate",
    "Momentum": "the momentum / beta1",
}

#: Why each factor matters *at a one-epoch budget*. The magnitude is filled in
#: per pair from the fitted model, so every clause is checkable.
KEY_MECHANISM: Dict[str, str] = {
    "Optimizer": (
        "the optimiser dominates at a one-epoch budget, because an adaptive method "
        "reaches a usable solution within a single pass while plain momentum SGD is "
        "still in its warmup phase when training stops"),
    "Norm Type": (
        "batch statistics calibrate the activations within the first few hundred "
        "steps, whereas per-sample normalisation throws away the cross-batch signal "
        "that makes early training stable"),
    "Drop Path": (
        "stochastic depth is a long-horizon regulariser; inside a single epoch it "
        "only removes gradient signal from paths that have not yet learned anything "
        "worth regularising"),
    "Start Channels": (
        "the stem width sets how much capacity the first layers have to form features "
        "in a single pass, and that width propagates through every later stage"),
    "FC Dropout": (
        "regularisation pays off over many epochs; inside one epoch it mostly removes "
        "capacity the model has not yet had time to overfit"),
    "Pooling": (
        "max pooling preserves the strongest activations and gives a sharper early "
        "signal than averaging, which dilutes it"),
    "Total Blocks": (
        "more blocks add representational depth, but every extra block is one more "
        "stage that has to converge within the same single pass"),
    "Conv Type": (
        "a dense convolution spends more parameters per layer but converges in fewer "
        "steps than a factorised one"),
    "Fractal Columns": (
        "the column count trades representational richness against how far any one "
        "path can get, since every column competes for the same gradient budget "
        "within a single epoch"),
    "Kernel": (
        "the kernel size sets how far spatial information travels per layer, and that "
        "reach compounds quickly when there are few steps to compound over"),
    "Activation": (
        "a smooth activation keeps gradients alive in units that a hard rectifier "
        "switches off outright, which matters most while the network is still far "
        "from convergence"),
    "LR": (
        "the step size sets how far the model can travel in a single pass over the "
        "data, and both too-small and too-large values waste that one pass"),
    "Momentum": (
        "beta1 sets how much gradient history is carried, which matters less when "
        "there is only one epoch of gradients to accumulate"),
}

HEADER_WINDOW = 2000

_PRETRAINED = re.compile(
    r"pretrained\s*=\s*True|weights\s*=\s*[\"']DEFAULT"
    r"|weights\s*=\s*[A-Za-z_]*Weights\.|torchvision\.models"
    r"|torch\.hub\.load|timm\.create_model")


def uses_pretrained_weights(code: str) -> bool:
    """True if the file imports trained weights.

    Models wrapping an ImageNet-pretrained torchvision backbone reach far higher
    epoch-1 accuracy than anything trained from scratch. Leaving them in teaches
    one degenerate lesson -- wrap a pretrained ResNet -- and collapses
    architectural diversity, so they are excluded from both sides of every pair.
    """
    return bool(_PRETRAINED.search(code))


def parse_design(code: str) -> Optional[Dict[str, str]]:
    """Extract the full 13-factor design point, or None if any factor is absent."""
    head = code[:HEADER_WINDOW]
    out: Dict[str, str] = {}
    for key in HEADER_KEYS:
        m = re.search(rf"^#\s*{re.escape(key)}:\s*(.+?)\s*$", head, re.M)
        if not m:
            return None
        out[key] = m.group(1).strip()
    for key, pat in BODY_PATTERNS.items():
        m = re.search(pat, code)
        if not m:
            return None
        out[key] = m.group(1).strip()
    return out


# --------------------------------------------------------------------------
# Record loading, with the misfiled-record filter
# --------------------------------------------------------------------------

def epoch_record_is_valid(rec: dict, file_epoch: int) -> bool:
    """Reject records that cannot belong to the epoch file they sit in.

    LEMUR stores a run measured at epoch n in ``<n>.json``: ``accuracy`` is the
    value at that epoch, ``train_stat.best_epoch`` the best epoch so far, so
    ``best_epoch <= n``. 619 records across the dataset break that -- whole-run
    summaries copied into ``1.json`` and ``5.json`` -- and one reports 0.9000
    for a model whose real epoch-1 accuracy is about 0.38. Ingesting one would
    put a 50-epoch result on the target side of a training pair.
    """
    ts = rec.get("train_stat") or {}
    be = ts.get("best_epoch")
    if isinstance(be, int) and be > file_epoch:
        return False
    ep = rec.get("epoch")
    if ep is not None and ep != file_epoch:
        return False
    return rec.get("accuracy") is not None


class Record:
    __slots__ = ("nn", "code", "design", "observed", "fitted", "n_trials",
                 "prm", "transform", "transform_code")

    def __init__(self, nn, code, design, observed, n_trials, prm, transform,
                 transform_code):
        self.nn, self.code, self.design = nn, code, design
        self.observed, self.n_trials = observed, n_trials
        self.prm, self.transform, self.transform_code = prm, transform, transform_code
        self.fitted = None


def load_from_lemur(task, dataset, metric, epoch, nn_prefixes) -> List[Record]:
    import ab.nn.api as lemur
    df = lemur.data(only_best_accuracy=False, task=task, dataset=dataset,
                    metric=metric, epoch=epoch, nn_prefixes=tuple(nn_prefixes))
    if df.empty:
        raise RuntimeError(
            f"LEMUR returned no rows for task={task} dataset={dataset} "
            f"metric={metric} epoch={epoch} nn_prefixes={list(nn_prefixes)}.")
    by_nn = defaultdict(list)
    for row in df.to_dict("records"):
        by_nn[row["nn"]].append(row)
    out = []
    for nn, rows in by_nn.items():
        code = rows[0].get("nn_code") or ""
        if not code or uses_pretrained_weights(code):
            continue
        design = parse_design(code)
        if design is None:
            continue
        accs = [r["accuracy"] for r in rows if r.get("accuracy") is not None]
        if not accs:
            continue
        best = max(rows, key=lambda r: r.get("accuracy") or -1.0)
        out.append(Record(nn, code, design, sum(accs) / len(accs), len(accs),
                          best.get("prm") or {}, best.get("transform") or "",
                          best.get("transform_code") or ""))
    return out


def load_offline(root, task, dataset, metric, epoch, nn_prefixes) -> List[Record]:
    """Load the same records from a checked-out ``nn-dataset/ab/nn`` tree."""
    root_p = Path(root)
    nn_dir = root_p / "nn"
    stat_dir = root_p / "stat" / "train"
    tr_dir = root_p / "transform"
    pipeline = f"{task}_{dataset}_{metric}_"
    tr_cache: Dict[str, str] = {}

    def transform_code(name):
        if name not in tr_cache:
            p = tr_dir / f"{name}.py"
            tr_cache[name] = p.read_text(encoding="utf-8") if p.exists() else ""
        return tr_cache[name]

    out: List[Record] = []
    skipped = 0
    for d in sorted(stat_dir.glob(pipeline + "*")):
        nn = d.name[len(pipeline):]
        if nn_prefixes and not any(nn.startswith(p) for p in nn_prefixes):
            continue
        f = d / f"{epoch}.json"
        code_p = nn_dir / f"{nn}.py"
        if not f.exists() or not code_p.exists():
            continue
        code = code_p.read_text(encoding="utf-8")
        if uses_pretrained_weights(code):
            continue
        design = parse_design(code)
        if design is None:
            continue
        try:
            trials = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        good = [t for t in trials if epoch_record_is_valid(t, epoch)]
        skipped += len(trials) - len(good)
        if not good:
            continue
        accs = [t["accuracy"] for t in good]
        best = max(good, key=lambda t: t["accuracy"])
        prm = {k: best[k] for k in ("lr", "momentum", "batch", "dropout")
               if k in best and best[k] is not None}
        prm["transform"] = best.get("transform", "")
        out.append(Record(nn, code, design, sum(accs) / len(accs), len(accs),
                          prm, best.get("transform", ""),
                          transform_code(best.get("transform", ""))))
    if skipped:
        print(f"[design-pairs] rejected {skipped} records that cannot belong to "
              f"epoch {epoch} (best_epoch > {epoch}, or epoch mismatch)")
    return out


# --------------------------------------------------------------------------
# The factor model
# --------------------------------------------------------------------------

class FactorModel:
    """Ridge regression on the 13 factors plus all two-way interactions.

    Each design point was trained once, so its observed accuracy carries a full
    measurement error (sd about 0.025-0.035 at this budget). The factor effects
    are estimated across hundreds to thousands of models each, so the fitted
    value is a much better estimate of a point's quality than its own run.

    5-fold CV on the current corpus: RMSE 0.046 against an observed spread of
    0.126 (R2 0.86), and fitted gains correlate 0.885 with observed gains.
    """

    def __init__(self, records: Sequence[Record], ridge: float = 1.0,
                 interactions: bool = True):
        self.keys = DESIGN_KEYS
        self.levels = {k: sorted({r.design[k] for r in records}) for k in self.keys}
        self.interactions = interactions
        self.ridge = ridge
        self.w = self._solve([self.vector(r.design) for r in records],
                             [r.observed for r in records], ridge)

    def vector(self, design: Dict[str, str]) -> List[float]:
        v = [1.0]
        for k in self.keys:
            for lev in self.levels[k][1:]:
                v.append(1.0 if design[k] == lev else 0.0)
        if self.interactions:
            for a, b in itertools.combinations(self.keys, 2):
                for la in self.levels[a][1:]:
                    for lb in self.levels[b][1:]:
                        v.append(1.0 if (design[a] == la and design[b] == lb) else 0.0)
        return v

    @staticmethod
    def _solve(X, y, ridge):
        import numpy as np
        A = np.asarray(X, dtype=float)
        b = np.asarray(y, dtype=float)
        return np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ b).tolist()

    def predict(self, design: Dict[str, str]) -> float:
        return float(sum(a * b for a, b in zip(self.vector(design), self.w)))

    def cross_validate(self, records, folds: int = 5, seed: int = 0) -> Dict[str, float]:
        import numpy as np
        X = np.asarray([self.vector(r.design) for r in records], dtype=float)
        y = np.asarray([r.observed for r in records], dtype=float)
        idx = np.arange(len(y))
        np.random.RandomState(seed).shuffle(idx)
        parts = np.array_split(idx, folds)
        pred = np.zeros_like(y)
        for i in range(folds):
            te = parts[i]
            tr = np.concatenate([parts[j] for j in range(folds) if j != i])
            w = np.linalg.solve(X[tr].T @ X[tr] + self.ridge * np.eye(X.shape[1]),
                                X[tr].T @ y[tr])
            pred[te] = X[te] @ w
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        base = float(np.sqrt(np.mean((y - y.mean()) ** 2)))
        return {"cv_rmse": round(rmse, 4), "spread": round(base, 4),
                "r2": round(1 - rmse ** 2 / base ** 2, 4),
                "corr": round(float(np.corrcoef(pred, y)[0, 1]), 4)}

    def adjusted_effects(self, records) -> Dict[str, Dict[str, float]]:
        """Effect of each level with the other factors averaged over the corpus.

        This is what the rationale quotes. Raw marginal means would be
        confounded -- the space is sampled at 0.46% density, so levels are not
        balanced against each other, and the marginal figures for LR (+0.069)
        and Momentum (+0.058) shrink to +0.017 and +0.008 once adjusted.
        """
        import numpy as np
        base = np.asarray([self.vector(r.design) for r in records], dtype=float)
        w = np.asarray(self.w, dtype=float)
        out: Dict[str, Dict[str, float]] = {}
        offset = 1
        spans = {}
        for k in self.keys:
            spans[k] = (offset, offset + len(self.levels[k]) - 1)
            offset += len(self.levels[k]) - 1
        for k in self.keys:
            out[k] = {}
            lo, hi = spans[k]
            for i, lev in enumerate(self.levels[k]):
                X = base.copy()
                X[:, lo:hi] = 0.0
                if i > 0:
                    X[:, lo + i - 1] = 1.0
                if self.interactions:
                    # Interaction columns involving k must be rebuilt; doing it
                    # exactly is simplest via the public path for a sample.
                    vals = []
                    for r in records:
                        d = dict(r.design); d[k] = lev
                        vals.append(self.predict(d))
                    out[k][lev] = float(sum(vals) / len(vals))
                    continue
                out[k][lev] = float((X @ w).mean())
        return out


def transition_effect(effects, key: str, frm: str, to: str) -> float:
    e = effects.get(key) or {}
    if frm not in e or to not in e:
        return 0.0
    return e[to] - e[frm]


# --------------------------------------------------------------------------
# Prompt text -- one definition, used for training and for generation
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert PyTorch neural network architect. You are given one image-classification model \
that was trained for {epoch} epoch on '{dataset}' and measured by '{metric}'; return a better one.
Follow LEMUR conventions exactly as the baseline below does: one top-level class Net(nn.Module), a \
module-level supported_hyperparameters(), train_setup(self, prm) and learn(self, train_data), with \
shapes read from in_shape and out_shape. Pure PyTorch and torchvision.transforms only, and never \
pretrained weights.
One epoch is a single pass over the data, so favour fast early convergence over regularisation \
that only pays off over many epochs.
Emit <hp> (a JSON object with "batch", "transform" and every key from supported_hyperparameters()), \
then <tr> (the complete transform code), then <nn> (the complete model file, no ellipses).
Begin your answer with <hp> immediately. Write no prose before or between the tags."""


USER_PROMPT = """\
Baseline model, measured on '{dataset}' for task '{task}' after exactly {epoch} epoch of \
training. The accuracy below was produced by the same protocol that will be used to score you, so \
it is directly comparable to your own result.

Measured '{metric}': {accuracy:.4f}

<hp>{prm}</hp>
<tr>{transform_code}</tr>
<nn>{nn_code}</nn>

Return a model that scores higher than {accuracy:.4f} under that same {epoch}-epoch protocol.

Change only the design decisions you can justify, and keep everything else identical to the \
baseline, so that whatever gain you achieve is attributable to the decisions you named. Do not \
rewrite the architecture wholesale, do not rename the class, and do not change the parts of the \
file that implement the LEMUR interface."""


_LEAD_IN = (
    "Changed {phrase} from {frm} to {to}",
    "Moved {phrase} from {frm} to {to}",
    "Replaced {frm} with {to} for {phrase}",
    "Set {phrase} to {to} instead of {frm}",
)


def changed_factors(a: Record, b: Record, effects) -> List[Tuple[str, float]]:
    """Changed factors with their adjusted effect, strongest first."""
    out = [(k, transition_effect(effects, k, a.design[k], b.design[k]))
           for k in DESIGN_KEYS if a.design[k] != b.design[k]]
    out.sort(key=lambda t: -t[1])
    return out


def build_reason(a: Record, b: Record, effects, rng: random.Random) -> str:
    """A rationale generated from the factor diff and the fitted effects.

    Every clause is checkable against the model, so the LLM is supervised on
    true statements about which decision is worth what -- not on plausible prose,
    and not merely on 150 lines of largely unchanged Python.
    """
    changed = changed_factors(a, b, effects)
    parts = []
    for k, eff in changed:
        frm, to = a.design[k], b.design[k]
        lead = rng.choice(_LEAD_IN).format(phrase=KEY_PHRASE[k], frm=frm, to=to)
        if eff > 0.03:
            worth = f"; across this design space that single move is worth about {eff:+.3f}"
        elif eff > 0.01:
            worth = f"; it is worth about {eff:+.3f} on its own, a small but consistent gain"
        else:
            worth = ("; on its own it is close to neutral, and it is here because it combines "
                     "with the changes above rather than for its own sake")
        parts.append(f"{lead}{worth} - {KEY_MECHANISM[k]}.")
    parts.append(
        f"Everything else is left exactly as the baseline had it. The redesign is predicted to "
        f"score {b.fitted:.3f} against the baseline's {a.fitted:.3f}, and measured "
        f"{b.observed:.3f} against {a.observed:.3f}, so the model and the measurement agree on "
        f"the {len(changed)} change{'s' if len(changed) != 1 else ''} above.")
    return " ".join(parts)


def render_example(a: Record, b: Record, effects, cfg, rng) -> dict:
    user = USER_PROMPT.format(
        dataset=cfg["dataset"], task=cfg["task"], metric=cfg["metric"],
        epoch=cfg["epoch"], accuracy=a.observed,
        prm=json.dumps(a.prm, sort_keys=True),
        transform_code=a.transform_code.strip(), nn_code=a.code.strip())
    system = SYSTEM_PROMPT.format(dataset=cfg["dataset"], metric=cfg["metric"],
                                  epoch=cfg["epoch"])
    # Order matters more than it looks. With <reason> first, an un-fine-tuned
    # model writes a long rationale, exhausts max_new_tokens and never reaches
    # the model file: measured at cycle 0, 19 of 41 generations produced no
    # <nn> at all and 28 produced no <hp>. Putting the functional tags first
    # means a truncated generation loses only the explanation, which nothing
    # downstream parses, instead of the one block the pipeline needs.
    assistant = (f"<hp>{json.dumps(b.prm, sort_keys=True)}</hp>\n"
                 f"<tr>{b.transform_code.strip()}</tr>\n"
                 f"<nn>{b.code.strip()}</nn>\n"
                 f"<reason>{build_reason(a, b, effects, rng)}</reason>")
    return {
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user},
                     {"role": "assistant", "content": assistant}],
        "meta": {"baseline_nn": a.nn, "target_nn": b.nn,
                 "baseline_observed": round(a.observed, 4),
                 "target_observed": round(b.observed, 4),
                 "baseline_fitted": round(a.fitted, 4),
                 "target_fitted": round(b.fitted, 4),
                 "fitted_gain": round(b.fitted - a.fitted, 4),
                 "observed_gain": round(b.observed - a.observed, 4),
                 "hamming": sum(1 for k in DESIGN_KEYS if a.design[k] != b.design[k]),
                 "target_design": b.design},
    }


# --------------------------------------------------------------------------
# Pairing
# --------------------------------------------------------------------------

def hamming(a, b) -> int:
    return sum(1 for k in DESIGN_KEYS if a[k] != b[k])


def build_pairs(records, min_fitted_gain, min_observed_gain, max_hamming,
                per_baseline, rng) -> List[Tuple[Record, Record]]:
    """Improving pairs at bounded edit distance, on two agreeing signals.

    ``max_hamming`` is what makes the target reachable from the prompt: the
    better model is the baseline with at most that many *named* decisions
    changed, so a learnable function from one to the other exists.

    The two gain conditions guard different failure modes. ``min_fitted_gain``
    uses the factor model, whose estimate of a design point is far more precise
    than the point's own single training run, and so is what keeps seed noise
    out of the corpus. ``min_observed_gain`` requires the measurement to agree,
    which keeps the corpus honest if the model is misspecified. Thresholding
    observations alone cannot work here: among pairs whose true improvement is
    zero, noise alone clears 0.05 sixteen percent of the time.
    """
    pairs = []
    for a in records:
        cands = []
        for b in records:
            if b.nn == a.nn:
                continue
            if b.fitted - a.fitted < min_fitted_gain:
                continue
            if b.observed - a.observed < min_observed_gain:
                continue
            h = hamming(a.design, b.design)
            if h < 1 or h > max_hamming:
                continue
            cands.append((h, -(b.fitted - a.fitted), b))
        if not cands:
            continue
        cands.sort(key=lambda t: t[:2])
        pairs.extend((a, b) for _, _, b in cands[:per_baseline])
    rng.shuffle(pairs)
    return pairs


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

def holdout_members(names, holdout_frac: float, seed: int,
                    pinned_seed_file=None) -> set:
    """Which models are held out, decided per name rather than per list.

    The obvious implementation -- shuffle the name list and take the first
    12% -- is unstable the moment the corpus changes size. Adding even one
    model reshuffles everything, so a model held out in one build lands in the
    training pool in the next. That matters because the generation seeds are
    pinned for the whole run: a baseline can stay fixed in the test set while
    quietly becoming a training baseline, and the improvement measured on it
    would then be partly recall rather than design.

    It bites hardest if generated models are ever written back under the corpus
    prefix to grow the dataset. The direct failure people expect -- training on
    the exact (X, X') pair the model just produced -- cannot happen while X is
    held out, because pairs are only built within the training pool. The real
    hazard is subtler: corpus growth silently moves X out of the holdout, and
    only then does that pair become constructible.

    Hashing the name makes membership a property of the model, so it never
    changes no matter how the corpus grows. Any model listed in the pinned seed
    file is held out unconditionally, which keeps the test set excluded by
    construction rather than by luck of the hash.
    """
    pinned = set()
    if pinned_seed_file:
        path = Path(pinned_seed_file)
        if path.is_file():
            try:
                pinned = set(json.loads(path.read_text(encoding="utf-8")).get("nn") or [])
            except Exception as exc:
                print(f"[design-pairs] could not read pinned seeds at {path}: {exc}")
    cut = int(0xFFFFFFFF * float(holdout_frac))
    out = set(pinned)
    for nn in names:
        digest = hashlib.sha1(f"{seed}:{nn}".encode("utf-8")).hexdigest()
        if int(digest[:8], 16) <= cut:
            out.add(nn)
    return {n for n in names if n in out}


def build(out_dir, task="img-classification", dataset="cifar-10", metric="acc",
          epoch=1, nn_prefixes=("GenFractalNet-",), min_fitted_gain=0.12,
          min_observed_gain=0.05, max_hamming=3, per_baseline=6,
          val_frac=0.08, holdout_frac=0.12, seed=17, offline_root=None,
          pinned_seed_file=None) -> dict:
    rng = random.Random(seed)
    cfg = dict(task=task, dataset=dataset, metric=metric, epoch=epoch)

    records = (load_offline(offline_root, task, dataset, metric, epoch, nn_prefixes)
               if offline_root else
               load_from_lemur(task, dataset, metric, epoch, nn_prefixes))
    if len(records) < 50:
        raise RuntimeError(f"only {len(records)} usable records -- check the filters")

    points = {tuple(r.design[k] for k in DESIGN_KEYS) for r in records}
    model = FactorModel(records)
    cv = model.cross_validate(records)
    for r in records:
        r.fitted = model.predict(r.design)
    effects = model.adjusted_effects(records)

    # Hold baselines out *before* pairing, so no held-out baseline ever appears
    # on either side of a training pair.
    holdout = holdout_members(
        [r.nn for r in records], holdout_frac, seed,
        pinned_seed_file=pinned_seed_file)
    train_pool = [r for r in records if r.nn not in holdout]
    hold_pool = [r for r in records if r.nn in holdout]

    pairs = build_pairs(train_pool, min_fitted_gain, min_observed_gain,
                        max_hamming, per_baseline, rng)
    if not pairs:
        raise RuntimeError("no pairs at the configured thresholds")

    examples = [render_example(a, b, effects, cfg, rng) for a, b in pairs]
    rng.shuffle(examples)
    n_val = max(1, int(len(examples) * val_frac))
    val, train = examples[:n_val], examples[n_val:]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def dump(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump(out / "train.jsonl", train)
    dump(out / "val.jsonl", val)
    hold_pool.sort(key=lambda r: r.observed)
    dump(out / "holdout_baselines.jsonl",
         [{"nn": r.nn, "observed": round(r.observed, 4),
           "fitted": round(r.fitted, 4), "design": r.design} for r in hold_pool])
    # The design points seen in training, so the evaluator can tell a designed
    # answer from a memorised one.
    dump(out / "train_design_points.jsonl",
         [{"design": e["meta"]["target_design"]} for e in train])
    with open(out / "factor_effects.json", "w", encoding="utf-8") as f:
        json.dump(effects, f, indent=2)

    best = {k: max(v, key=v.get) for k, v in effects.items()}
    obs = sorted(r.observed for r in records)
    stats = {
        "records": len(records),
        "unique_design_points": len(points),
        "factors": len(DESIGN_KEYS),
        "design_space_size": int(math.prod(len(model.levels[k]) for k in DESIGN_KEYS)),
        "factor_model": cv,
        "train_baselines": len(train_pool), "holdout_baselines": len(hold_pool),
        "pairs": len(pairs), "train_examples": len(train), "val_examples": len(val),
        "min_fitted_gain": min_fitted_gain, "min_observed_gain": min_observed_gain,
        "max_hamming": max_hamming,
        "mean_fitted_gain": round(
            sum(b.fitted - a.fitted for a, b in pairs) / len(pairs), 4),
        "observed_p50": round(obs[len(obs) // 2], 4),
        "observed_max": round(obs[-1], 4),
        "adjusted_effect_spread": {
            k: round(max(v.values()) - min(v.values()), 4) for k, v in effects.items()},
        "model_optimum": best,
        "model_optimum_fitted": round(model.predict(best), 4),
    }
    with open(out / "corpus_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats


# --------------------------------------------------------------------------
# Generation-time config and pinned seeds, from the same strings
# --------------------------------------------------------------------------

def emit_test_config(path, key="design_improve_test", task="img-classification",
                     dataset="cifar-10", metric="acc", epoch=1,
                     nn_prefixes=("GenFractalNet-",), min_accuracy=0.0) -> str:
    """Write the generation prompt config from SYSTEM_PROMPT / USER_PROMPT.

    The most damaging defect in the previous setup was that training and testing
    asked for different things: the train config said "raise this baseline from
    {accuracy} to {addon_accuracy}", while the test config asked, unpaired, for
    "the model that achieves a maximum value for the metric after the first
    epoch". Whatever the fine-tune taught was evaluated on a task it was never
    trained on. Generating this file from the same constants the corpus is
    rendered with makes that drift impossible.
    """
    system = SYSTEM_PROMPT.format(dataset=dataset, metric=metric, epoch=epoch)
    cfg = {key: {
        "comment": ["Generated by ab.gpt.util.data.design_pairs.emit_test_config.",
                    "Do not hand-edit: it must stay byte-identical to the prompt the",
                    "SFT corpus was rendered with, or train and test diverge again."],
        "system": system.split("\n"),
        "task": task, "dataset": dataset, "metric": metric, "epoch": epoch,
        "nn_prefixes": list(nn_prefixes), "min_accuracy": min_accuracy,
        "no_repeat": [], "keep_same": [],
        "input_list": [{"para": p, "value": p} for p in (
            "nn_code", "accuracy", "epoch", "dataset", "task", "metric",
            "metric_code", "transform_code", "prm")],
        "prompt": USER_PROMPT.split("\n"),
    }}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return path


def emit_pinned_seeds(corpus_dir, seed_file, n=40, seed=17) -> str:
    """Pin generation seeds to held-out baselines, spread across the range.

    ``ab/gpt/util/eval/Seed.py`` already reads ``{"nn": [...]}`` from
    ``NNGPT_SEED_FILE`` and holds that set fixed for the whole run, which is what
    makes cycle-to-cycle deltas comparable. Writing the held-out baselines into
    it is all that is needed to evaluate on models the fine-tune has never seen
    -- no pipeline change.

    Baselines are taken at even quantiles rather than at random, so the test set
    is not dominated by the easy low-accuracy cases where almost any change is an
    improvement.
    """
    rows = [json.loads(l) for l in open(Path(corpus_dir) / "holdout_baselines.jsonl",
                                        encoding="utf-8")]
    rows.sort(key=lambda r: r["observed"])
    if n >= len(rows):
        chosen = rows
    else:
        step = len(rows) / n
        chosen = [rows[min(len(rows) - 1, int(i * step + step / 2))] for i in range(n)]
    Path(seed_file).parent.mkdir(parents=True, exist_ok=True)
    Path(seed_file).write_text(
        json.dumps({"nn": [r["nn"] for r in chosen], "pinned_at_epoch": 0}, indent=2),
        encoding="utf-8")
    return seed_file


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--task", default="img-classification")
    p.add_argument("--dataset", default="cifar-10")
    p.add_argument("--metric", default="acc")
    p.add_argument("--epoch", type=int, default=1)
    p.add_argument("--nn-prefix", action="append", default=None)
    p.add_argument("--min-fitted-gain", type=float, default=0.12)
    p.add_argument("--min-observed-gain", type=float, default=0.05)
    p.add_argument("--max-hamming", type=int, default=3)
    p.add_argument("--per-baseline", type=int, default=6)
    p.add_argument("--holdout-frac", type=float, default=0.12)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--offline-root", default=None)
    p.add_argument("--pinned-seed-file", default=None,
                   help="models listed here are held out unconditionally")
    a = p.parse_args(argv)
    print(json.dumps(build(
        out_dir=a.out, task=a.task, dataset=a.dataset, metric=a.metric,
        epoch=a.epoch, nn_prefixes=tuple(a.nn_prefix or ("GenFractalNet-",)),
        min_fitted_gain=a.min_fitted_gain, min_observed_gain=a.min_observed_gain,
        max_hamming=a.max_hamming, per_baseline=a.per_baseline,
        holdout_frac=a.holdout_frac, seed=a.seed, offline_root=a.offline_root,
        pinned_seed_file=a.pinned_seed_file,
    ), indent=2))


if __name__ == "__main__":
    main()
