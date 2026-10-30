"""Auxiliary training tasks over the same corpus, to stop the model memorising.

The improvement task has one shape of answer: a set of <edit> blocks. Train on
nothing else and the cheapest way to lower the loss is to learn the shape --
measured, 60 of 60 valid generations of one run shared a class skeleton and
differed by a backbone string, while the training loss sat flat at 0.050 and
the best accuracy never beat what the base model managed before any
fine-tuning. That is a model reproducing its corpus, not one reading an
architecture.

Every task here answers a question about the *same* models with a *different*
kind of answer -- a number, a label, a choice, a hyperparameter, a transform.
There is no single template that satisfies them all, so the only way to lower
the loss across the mix is to actually read the code. Three of them (accuracy,
delta, cost) have numeric targets the model cannot reach by pattern-matching at
all.

Every target is ground truth already on disk. Nothing here needs a new
measurement, a judgement call or a label, and no task is included whose answer
could only be guessed:

    rank            which of two variants measured better          pairs
    predict_acc     what this model scored                         records
    predict_delta   what this edit was worth                       pairs
    best_of_k       which of k edits is the best one               cluster
    degrade         an edit that makes a model worse               pairs
    propose_hp      the hyperparameters it was measured at         prm
    propose_tr      the input pipeline it was measured with        transform
    predict_cost    its parameters, FLOPs and depth                stat/nn
    odd_one_out     which two of three are the same architecture   clusters
    will_it_run     whether generated code runs, and what breaks   own cycles

The last one needs this run's own rejected generations, so it produces nothing
in cycle 0 and grows from cycle 1. It is worth the wait: the valid-generation
rate fell to 5% over one run, and nothing in the corpus ever told the model
which of its own outputs would fail to build. It requires
NNGPT_VALIDITY_KEEP_REJECTED=1, because the gate otherwise deletes the code and
keeps only the reason -- a reason with no code is not a prompt.

Deliberately absent: "explain why this helped" and "repair this traceback".
Both need a target nobody measured. For repair in particular there is no fixed
version of a broken candidate anywhere -- nobody repaired those files -- so the
only available pairing would be a broken candidate against a sibling that
happened to run, which is a different architecture and not a repair. A task
whose answer is invented teaches the invention.

Validation is unaffected: these are training rows only. The test prompts stay
the improvement task, so the run is still judged on accuracy and on nothing
else.
"""

from __future__ import annotations

import collections
import json
import os
import random
from typing import Dict, List, Optional, Sequence

from ab.gpt.util.data.code_edit import (
    EDIT_CLOSE, EDIT_OPEN, hp_text, render_edits)

#: Share of the finished corpus these tasks may occupy. The improvement task
#: is still what the run is scored on, so it keeps the majority.
AUX_SHARE = float(os.environ.get("NNGPT_AUX_SHARE", "0.4") or 0.4)

#: Which to build. "all", "off", or a comma-separated list of the names below.
AUX_TASKS = os.environ.get("NNGPT_AUX_TASKS", "all").strip()

ALL_TASKS = ("rank", "predict_acc", "predict_delta", "best_of_k", "degrade",
             "propose_hp", "propose_tr", "predict_cost", "odd_one_out",
             "will_it_run")

#: Fields worth asking for out of ab/nn/stat/nn/<name>.json. Chosen because
#: each is a consequence of the architecture that a reader of the code could
#: work out, and none of them is in the file as a literal.
COST_FIELDS = ("total_params", "flops", "total_layers", "nn_depth",
               "is_resnet_like", "is_transformer_like")


def enabled_tasks() -> List[str]:
    raw = AUX_TASKS.lower()
    if raw in ("off", "0", "none", "false"):
        return []
    if raw in ("all", "", "1", "true"):
        return list(ALL_TASKS)
    want = [t.strip() for t in raw.split(",") if t.strip()]
    return [t for t in ALL_TASKS if t in want]


def _edits_only(before: str, after: str) -> Optional[str]:
    """The <edit> blocks between two files, without the wrapper tags."""
    body = render_edits(before, after)
    if body is None:
        return None
    inner = body[len(EDIT_OPEN):-len(EDIT_CLOSE)].strip()
    return inner or None


#: Longest prompt+answer an auxiliary row may be, in characters.
#:
#: NNGenPrompt drops any row longer than max_len (16384 tokens) without saying
#: which, so an oversized row is not an error -- it is a row that silently
#: never trains. Three full models in one odd_one_out prompt is ~27k tokens on
#: real files, so the task would have produced nothing but dropped rows. ~4
#: chars per token, kept well under the limit because the chat template and the
#: system prompt add to it.
MAX_ROW_CHARS = int(os.environ.get("NNGPT_AUX_MAX_CHARS", "40000") or 40000)


def _fits(*parts) -> bool:
    return sum(len(p or "") for p in parts) <= MAX_ROW_CHARS


def _keep(rows: List, row) -> None:
    """Append a row unless the length budget refused it."""
    if row is not None:
        rows.append(row)


def _msg(system, user, assistant, task, meta=None) -> dict:
    if not _fits(system, user, assistant):
        return None
    out = {"messages": [{"role": "system", "content": system},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": assistant}],
           "meta": {"task": task, "observed_gain": 0.0}}
    out["meta"].update(meta or {})
    return out


# --------------------------------------------------------------------------
# One builder per task. Each takes what it needs and returns rows or nothing.
# --------------------------------------------------------------------------

_RANK_SYS = ("You judge neural network architectures. You are given a baseline "
             "model and two candidate edits of it, each of which was trained "
             "for {epoch} epoch on '{dataset}' and measured by '{metric}'. Say "
             "which candidate measured higher. Answer with a single "
             "<answer>A</answer> or <answer>B</answer> and nothing else.")


def build_rank(pairs, cfg, rng, limit) -> List[dict]:
    """Which of two edits of one baseline scored higher.

    A comparison, so the answer carries no code at all: there is nothing to
    copy, and the only way to be right more often than chance is to read both
    edits against the baseline.
    """
    by_baseline = collections.defaultdict(list)
    for a, b in pairs:
        by_baseline[a.nn].append((a, b))
    rows = []
    keys = list(by_baseline)
    rng.shuffle(keys)
    for key in keys:
        group = by_baseline[key]
        if len(group) < 2:
            continue
        for _ in range(min(2, len(group) // 2)):
            (a, x), (_a2, y) = rng.sample(group, 2)
            if abs(x.observed - y.observed) < 0.01 or x.nn == y.nn:
                continue
            ex, ey = _edits_only(a.code, x.code), _edits_only(a.code, y.code)
            if not ex or not ey:
                continue
            first_is_x = rng.random() < 0.5
            one, two = (x, y) if first_is_x else (y, x)
            e1, e2 = (ex, ey) if first_is_x else (ey, ex)
            better = "A" if one.observed > two.observed else "B"
            user = (f"Baseline model, measured '{cfg['metric']}' "
                    f"{a.observed:.4f} at {hp_text(a.prm)}.\n\n"
                    f"<nn>{a.code.strip()}</nn>\n\n"
                    f"<candidate id=\"A\">\n{e1}\n</candidate>\n\n"
                    f"<candidate id=\"B\">\n{e2}\n</candidate>\n\n"
                    "Which candidate measured higher under the same protocol?")
            _keep(rows, _msg(
                _RANK_SYS.format(**cfg), user, f"<answer>{better}</answer>",
                "rank", {"baseline_nn": a.nn,
                         "margin": round(abs(x.observed - y.observed), 4)}))
            if len(rows) >= limit:
                return rows
    return rows


_ACC_SYS = ("You estimate how well a neural network will train. You are given "
            "one model and the hyperparameters it is trained with, for {epoch} "
            "epoch on '{dataset}', and you state the '{metric}' it reaches. "
            "Answer with a single <accuracy>0.0000</accuracy> and nothing else.")


def build_predict_acc(records, cfg, rng, limit) -> List[dict]:
    """What this model scored. A number: nothing about it can be memorised."""
    pool = list(records)
    rng.shuffle(pool)
    rows = []
    for r in pool[:limit]:
        user = (f"Model trained for {cfg['epoch']} epoch on '{cfg['dataset']}' "
                f"at {hp_text(r.prm)}.\n\n<nn>{r.code.strip()}</nn>\n\n"
                f"What '{cfg['metric']}' did it reach?")
        _keep(rows, _msg(_ACC_SYS.format(**cfg), user,
                         f"<accuracy>{r.observed:.4f}</accuracy>",
                         "predict_acc", {"baseline_nn": r.nn,
                                         "target_observed": round(r.observed, 4)}))
    return rows


_DELTA_SYS = ("You estimate what a change to a neural network is worth. You "
              "are given a baseline, its measured '{metric}' after {epoch} "
              "epoch on '{dataset}', and an edit of it. State the change in "
              "'{metric}' the edit produced. Answer with a single signed "
              "<delta>+0.0000</delta> and nothing else.")


def build_predict_delta(pairs, cfg, rng, limit) -> List[dict]:
    """What this edit was worth. The quantity the whole pipeline optimises."""
    pool = list(pairs)
    rng.shuffle(pool)
    rows = []
    for a, b in pool:
        inner = _edits_only(a.code, b.code)
        if not inner:
            continue
        user = (f"Baseline, measured '{cfg['metric']}' {a.observed:.4f} at "
                f"{hp_text(a.prm)}.\n\n<nn>{a.code.strip()}</nn>\n\n"
                f"<edit>\n{inner}\n</edit>\n\n"
                "What did this edit do to the measured score?")
        _keep(rows, _msg(_DELTA_SYS.format(**cfg), user,
                         f"<delta>{b.observed - a.observed:+.4f}</delta>",
                         "predict_delta",
                         {"baseline_nn": a.nn, "target_nn": b.nn,
                          "observed_gain": round(b.observed - a.observed, 4)}))
        if len(rows) >= limit:
            break
    return rows


_BEST_SYS = ("You choose between changes to a neural network. You are given a "
             "baseline and several edits of it, all measured the same way "
             "({epoch} epoch on '{dataset}', '{metric}'). Name the one that "
             "measured highest. Answer with a single <choice>N</choice> and "
             "nothing else.")


def build_best_of_k(pairs, cfg, rng, limit, k=3) -> List[dict]:
    """Which of k edits is the best one -- the pipeline's own question."""
    by_baseline = collections.defaultdict(list)
    for a, b in pairs:
        by_baseline[a.nn].append((a, b))
    rows = []
    keys = [x for x in by_baseline if len(by_baseline[x]) >= k]
    rng.shuffle(keys)
    for key in keys:
        picked = rng.sample(by_baseline[key], k)
        a = picked[0][0]
        blocks, scores = [], []
        for i, (_a, b) in enumerate(picked, 1):
            inner = _edits_only(a.code, b.code)
            if not inner:
                break
            blocks.append(f"<candidate id=\"{i}\">\n{inner}\n</candidate>")
            scores.append(b.observed)
        if len(blocks) != k or max(scores) - min(scores) < 0.01:
            continue
        best = scores.index(max(scores)) + 1
        user = (f"Baseline, measured '{cfg['metric']}' {a.observed:.4f} at "
                f"{hp_text(a.prm)}.\n\n<nn>{a.code.strip()}</nn>\n\n"
                + "\n\n".join(blocks)
                + "\n\nWhich candidate measured highest?")
        _keep(rows, _msg(_BEST_SYS.format(**cfg), user,
                         f"<choice>{best}</choice>", "best_of_k",
                         {"baseline_nn": a.nn, "k": k}))
        if len(rows) >= limit:
            break
    return rows


_DEGRADE_SYS = ("You know what makes a neural network worse as well as what "
                "makes it better. You are given a model measured after {epoch} "
                "epoch on '{dataset}'. Return the smallest set of <edit> blocks "
                "that would LOWER its '{metric}', in the format "
                "<edit><find>...</find><with>...</with></edit>. Answer with the "
                "edits and nothing else.")


def build_degrade(pairs, cfg, rng, limit) -> List[dict]:
    """An edit that makes a model worse, from the same pair read backwards.

    The improvement corpus only ever shows what helps, so a model trained on it
    has never been told what hurts -- and the measured mean improvement of a
    generation was negative. The reversed pair is ground truth for the opposite
    direction, at no cost, and it is the one task here that asks for code while
    still being impossible to answer with the memorised template.
    """
    pool = list(pairs)
    rng.shuffle(pool)
    rows = []
    for a, b in pool:
        if b.observed - a.observed < 0.05:
            continue
        inner = _edits_only(b.code, a.code)          # better -> worse
        if not inner:
            continue
        user = (f"Model measured '{cfg['metric']}' {b.observed:.4f} after "
                f"{cfg['epoch']} epoch on '{cfg['dataset']}' at "
                f"{hp_text(b.prm)}.\n\n<nn>{b.code.strip()}</nn>\n\n"
                "Return edits that would make this model measurably worse.")
        _keep(rows, _msg(
            _DEGRADE_SYS.format(**cfg), user,
            f"{EDIT_OPEN}\n{inner}\n{EDIT_CLOSE}\n"
            f"<reason>Measured {a.observed:.4f} against {b.observed:.4f}, "
            f"{a.observed - b.observed:+.4f}.</reason>",
            "degrade", {"baseline_nn": b.nn, "target_nn": a.nn}))
        if len(rows) >= limit:
            break
    return rows


_HP_SYS = ("You set training hyperparameters for a neural network. You are "
           "given a model to train for {epoch} epoch on '{dataset}'. State the "
           "learning rate and momentum it reaches its best '{metric}' with. "
           "Answer with a single <hp>{{\"lr\": 0.0, \"momentum\": 0.0}}</hp> "
           "block and nothing else.")


def build_propose_hp(records, cfg, rng, limit) -> List[dict]:
    """The hyperparameters a model was actually measured at.

    All 60 evaluated generations of one run trained at lr=0.01, momentum=0.9 --
    the values of the family that supplied nearly every baseline -- and two of
    them diverged to 0.13 there. A model that has never been asked what a given
    architecture wants cannot be expected to ask for something else.
    """
    pool = [r for r in records if "lr" in (r.prm or {})]
    rng.shuffle(pool)
    rows = []
    for r in pool[:limit]:
        hp = {k: r.prm[k] for k in ("lr", "momentum") if k in r.prm}
        if "lr" not in hp:
            continue
        user = (f"Model to train for {cfg['epoch']} epoch on "
                f"'{cfg['dataset']}'.\n\n<nn>{r.code.strip()}</nn>\n\n"
                "What hyperparameters does it want?")
        _keep(rows, _msg(_HP_SYS.format(**cfg), user,
                         f"<hp>{json.dumps(hp, sort_keys=True)}</hp>",
                         "propose_hp", {"baseline_nn": r.nn}))
    return rows


_TR_SYS = ("You design the input pipeline for a neural network. You are given "
           "a model to train for {epoch} epoch on '{dataset}'. Return the "
           "transform module it was measured with: a complete module defining "
           "`def transform(_)` that returns a "
           "torchvision.transforms.Compose, in one <tr> block and nothing else.")


def build_propose_tr(records, cfg, rng, limit) -> List[dict]:
    """The input pipeline a model was measured with.

    The transform slot was read 0 times in 210 generations of one run, so the
    cheapest accuracy on a one-epoch budget was never searched. This is the
    only task that teaches what belongs in that slot.
    """
    pool = [r for r in records if (r.transform_code or "").strip()]
    rng.shuffle(pool)
    rows, seen = [], set()
    for r in pool:
        code = r.transform_code.strip()
        # One example per distinct pipeline per model, so a transform shared by
        # a thousand models does not become a thousand identical rows.
        key = (r.nn, hash(code))
        if key in seen:
            continue
        seen.add(key)
        user = (f"Model to train for {cfg['epoch']} epoch on "
                f"'{cfg['dataset']}'.\n\n<nn>{r.code.strip()}</nn>\n\n"
                "What input pipeline was it measured with?")
        _keep(rows, _msg(_TR_SYS.format(**cfg), user, f"<tr>{code}</tr>",
                         "propose_tr", {"baseline_nn": r.nn}))
        if len(rows) >= limit:
            break
    return rows


_COST_SYS = ("You read a neural network and work out what it costs. Given the "
             "source, state its parameter count, FLOPs, layer count and depth, "
             "and whether it is residual or transformer-like. Answer with a "
             "single <cost>{{...}}</cost> JSON block and nothing else.")


def load_cost_facts(names) -> Dict[str, dict]:
    """Structural facts per model, from the dataset's own statistics files."""
    from ab.gpt.util.data.minhash_clusters import stat_nn_dir
    d = stat_nn_dir()
    if d is None:
        return {}
    out = {}
    for name in names:
        f = d / f"{name}.json"
        if not f.is_file():
            continue
        try:
            payload = json.loads(f.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        facts = {k: payload[k] for k in COST_FIELDS if payload.get(k) is not None}
        if len(facts) >= 4:
            out[name] = facts
    return out


def build_predict_cost(records, cfg, rng, limit) -> List[dict]:
    """Parameters, FLOPs, depth and family, from the source alone.

    30,000 of these are already computed in ab/nn/stat/nn. None of the numbers
    appears in the file as a literal -- every one has to be worked out by
    following the layers -- so this is the task that most directly asks the
    model to read an architecture rather than recognise one.
    """
    pool = list(records)
    rng.shuffle(pool)
    facts = load_cost_facts([r.nn for r in pool[:limit * 3]])
    if not facts:
        return []
    rows = []
    for r in pool:
        f = facts.get(r.nn)
        if not f:
            continue
        user = f"<nn>{r.code.strip()}</nn>\n\nWhat does this network cost?"
        _keep(rows, _msg(_COST_SYS.format(**cfg), user,
                         f"<cost>{json.dumps(f, sort_keys=True)}</cost>",
                         "predict_cost", {"baseline_nn": r.nn}))
        if len(rows) >= limit:
            break
    return rows


_ODD_SYS = ("You recognise when two neural networks are the same architecture "
            "written differently. You are given three models. Two are the same "
            "architecture; one is not. Name the one that does not belong. "
            "Answer with a single <answer>N</answer> and nothing else.")


def build_odd_one_out(cluster_members, cfg, rng, limit) -> List[dict]:
    """Which two of three are the same architecture.

    The label is LEMUR's own MinHash clustering, so it costs nothing, and the
    question cannot be answered from surface text -- two models in a cluster
    may be spelled very differently, and two in different clusters may look
    alike. Teaches the similarity notion the corpus is built on.
    """
    big = [m for m in cluster_members if len(m) >= 2]
    if len(big) < 2:
        return []
    rows = []
    for _ in range(limit * 4):
        if len(rows) >= limit:
            break
        same, other = rng.sample(range(len(big)), 2)
        if len(big[same]) < 2 or not big[other]:
            continue
        x, y = rng.sample(big[same], 2)
        z = rng.choice(big[other])
        if len({x.nn, y.nn, z.nn}) < 3:
            continue
        trio = [x, y, z]
        rng.shuffle(trio)
        odd = trio.index(z) + 1
        blocks = "\n\n".join(
            f"<model id=\"{i}\">{m.code.strip()}</model>"
            for i, m in enumerate(trio, 1))
        _keep(rows, _msg(_ODD_SYS.format(**cfg),
                         blocks + "\n\nWhich model is not the same "
                                  "architecture as the other two?",
                         f"<answer>{odd}</answer>", "odd_one_out", {}))
    return rows


_RUN_SYS = ("You tell whether generated PyTorch model code will actually run. "
            "The file is evaluated by constructing Net(in_shape, out_shape, "
            "prm, device) with a 4-tuple in_shape, calling train_setup and "
            "learn, and training it. Say whether it survives that. Answer with "
            "a single <verdict>runs</verdict>, or "
            "<verdict>fails</verdict><error>ExceptionType</error>, and nothing "
            "else.")

#: Where this run's own generations live, relative to nngpt_dir.
EPOCH_GLOB = "llm/epoch/*/synth_nn"


#: Failures that are not a property of the code. A file that ran out of memory
#: on a card already holding a 9B trainer, or that hit the epoch time limit,
#: would run elsewhere -- labelling it "fails" teaches the model to predict the
#: machine's state, which it cannot see and should not try to.
INFRA_ERRORS = ("outofmemoryerror", "cuda out of memory", "timeout",
                "timeouterror", "keyboardinterrupt", "systemexit",
                "oserror", "ioerror", "connectionerror", "brokenpipeerror")


def _error_class(text: str) -> Optional[str]:
    """The exception type a stored traceback ended on, if the code owns it."""
    import re
    low = (text or "").lower()
    if any(tok in low for tok in INFRA_ERRORS):
        return None
    hits = re.findall(r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception))\b",
                      text or "", re.M)
    return hits[-1] if hits else None


def collect_run_outcomes(nngpt_dir, limit_per_class=400) -> List[dict]:
    """(code, verdict) for generations this run has already evaluated.

    Reads the epoch directories rather than the database, because what matters
    here is the model's own output -- including the files that never reached
    the database at all.
    """
    from pathlib import Path
    root = Path(nngpt_dir)
    good, bad = [], []
    for synth in sorted(root.glob(EPOCH_GLOB)):
        for d in sorted(list(synth.iterdir()) + list((synth / "_rejected").iterdir()
                                                     if (synth / "_rejected").is_dir() else [])):
            if not d.is_dir():
                continue
            nn = d / "new_nn.py"
            if not nn.is_file():
                continue
            try:
                code = nn.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if not code.strip():
                continue
            err_file = d / "error.txt"
            if (d / "eval_info.json").is_file() and not err_file.is_file():
                if len(good) < limit_per_class:
                    good.append({"code": code, "verdict": "runs", "error": None})
            elif err_file.is_file():
                try:
                    err = err_file.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                cls = _error_class(err)
                if cls and len(bad) < limit_per_class:
                    bad.append({"code": code, "verdict": "fails", "error": cls})
    return good, bad


def build_will_it_run(nngpt_dir, cfg, rng, limit) -> List[dict]:
    """Whether a generated file runs, and which exception stops it.

    The one task whose data is the model's own mistakes. In one run 218 of the
    evaluation failures were a single generated helper unpacking a 4-tuple
    in_shape into five names, repeated across cycles because nothing ever told
    the model that it does not work. Classes are balanced, so the answer
    "fails" is never right more often than "runs".
    """
    good, bad = collect_run_outcomes(nngpt_dir)
    if not good or not bad:
        return []
    n = min(len(good), len(bad), max(1, limit // 2))
    rng.shuffle(good)
    rng.shuffle(bad)
    rows = []
    for item in good[:n] + bad[:n]:
        answer = ("<verdict>runs</verdict>" if item["verdict"] == "runs"
                  else f"<verdict>fails</verdict><error>{item['error']}</error>")
        user = (f"<nn>{item['code'].strip()}</nn>\n\n"
                "Does this file build and train?")
        _keep(rows, _msg(_RUN_SYS, user, answer, "will_it_run",
                         {"verdict": item["verdict"], "error": item["error"]}))
    rng.shuffle(rows)
    return rows


def build_all(records, pairs, clusters, cfg, seed=17, nngpt_dir=None,
              n_improve=0, share=None, tasks=None, verbose=True) -> List[dict]:
    """Auxiliary rows to sit alongside ``n_improve`` improvement examples.

    Callers pass the TRAINING pool only. Nothing here may see a held-out
    baseline: these are training rows, and the run is validated on the
    improvement task over the holdout alone.
    """
    names = tasks if tasks is not None else enabled_tasks()
    if not names or not records:
        return []
    share = AUX_SHARE if share is None else share
    if share <= 0:
        return []
    # share is of the finished corpus, so the budget is relative to the
    # improvement rows the caller already has.
    total = int(n_improve * share / max(1e-6, 1.0 - share)) if n_improve else len(records)
    per = max(1, total // len(names))
    rng = random.Random(seed)

    out: List[dict] = []
    built = {}
    for name in names:
        if name == "rank":
            rows = build_rank(pairs, cfg, rng, per)
        elif name == "predict_acc":
            rows = build_predict_acc(records, cfg, rng, per)
        elif name == "predict_delta":
            rows = build_predict_delta(pairs, cfg, rng, per)
        elif name == "best_of_k":
            rows = build_best_of_k(pairs, cfg, rng, per)
        elif name == "degrade":
            rows = build_degrade(pairs, cfg, rng, per)
        elif name == "propose_hp":
            rows = build_propose_hp(records, cfg, rng, per)
        elif name == "propose_tr":
            rows = build_propose_tr(records, cfg, rng, per)
        elif name == "predict_cost":
            rows = build_predict_cost(records, cfg, rng, per)
        elif name == "odd_one_out":
            rows = build_odd_one_out(clusters, cfg, rng, per)
        elif name == "will_it_run":
            rows = build_will_it_run(nngpt_dir, cfg, rng, per) if nngpt_dir else []
        else:
            rows = []
        built[name] = len(rows)
        out.extend(rows)

    if verbose:
        missing = [k for k, v in built.items() if not v]
        print(f"[aux] {len(out)} auxiliary row(s) over {len(built)} task(s) "
              f"(target {per} each): "
              + ", ".join(f"{k}={v}" for k, v in built.items())
              + (f"; produced nothing: {', '.join(missing)}" if missing else ""),
              flush=True)
    rng.shuffle(out)
    return out
