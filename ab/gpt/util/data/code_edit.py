"""Search/replace edits: a code patch format an LLM can actually produce.

A unified diff carries line numbers and @@ hunk headers that must be exactly
right, and an LLM has no reliable way to count lines. A search/replace block
carries neither: the applier finds the SEARCH text in the file and substitutes
REPLACE. Getting it wrong is detected (no match, or several) rather than
silently applied at the wrong offset.

The format is deliberately not architecture-aware. It works on any model file
in the dataset, which is what a closed factor space cannot do.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import random
import re
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

#: Tag markers, not git-conflict markers. The dataset contains comment banners
#: made of "=" characters, and a bare "=======" divider inside a SEARCH block
#: terminated it early -- the edit then applied with the divider left in the
#: file. A marker has to be something a Python source file cannot contain.
EDIT_OPEN, EDIT_CLOSE = "<edit>", "</edit>"
FIND_OPEN, FIND_CLOSE = "<find>", "</find>"
WITH_OPEN, WITH_CLOSE = "<with>", "</with>"

EDIT_RE = re.compile(r"<edit[^>]*>(.*?)</edit>", re.S)
BLOCK_RE = re.compile(
    r"^[ \t]*<find>[ \t]*\n(.*?)\n?^[ \t]*</find>[ \t]*\n"
    r"[ \t]*<with>[ \t]*\n(.*?)\n?^[ \t]*</with>[ \t]*$",
    re.S | re.M)

#: Files that import trained weights. A model wrapping an ImageNet-pretrained
#: torchvision backbone reaches far higher one-epoch accuracy than anything
#: trained from scratch, so leaving them in teaches one degenerate lesson --
#: wrap a pretrained ResNet -- and sets a corpus ceiling no from-scratch edit
#: can reach. design_pairs excluded them for exactly this reason; dropping that
#: filter here let a whole pretrained family into the corpus while the prompt
#: told the model never to use pretrained weights.
PRETRAINED = re.compile(
    r"pretrained\s*=\s*True|weights\s*=\s*[\"']DEFAULT"
    r"|weights\s*=\s*[A-Za-z_]*Weights\.|torchvision\.models"
    r"|torch\.hub\.load|timm\.create_model")


def uses_pretrained_weights(code: str) -> bool:
    return bool(PRETRAINED.search(code or ""))


#: Without these the file cannot be evaluated at all, whatever else it does.
REQUIRED_SYMBOLS = ("supported_hyperparameters", "train_setup", "learn")

MAX_CONTEXT = 12


FULL_RE = re.compile(r"<nn[^>]*>(.*?)</nn>", re.S)

#: Every tag that may follow one of the answer blocks. An unclosed block ends
#: where the next one begins.
SIBLING_OPEN = re.compile(r"<\s*/?\s*(?:nn|tr|hp|edit|find|with|reason)\b[^>]*>",
                          re.I)


def block_bodies(text: str, tag: str) -> List[str]:
    """Bodies of ``<tag>...</tag>``, tolerating a missing closing tag.

    The model sometimes writes a complete block and simply omits the close --
    goes straight from the end of its file to ``<tr>``. Requiring ``</nn>``
    threw those replies away whole: measured on one cycle, three of thirty
    seeds produced a finished file, a closed <tr> and a closed <reason>, and
    were discarded for one absent tag and then reported as truncated.

    So a block that is never closed ends at the next answer tag, or at the end
    of the reply. That is the only reading available, and it is the right one
    whenever the content is complete -- and when it is not, the file fails to
    parse a moment later in validate_model, which is where a genuinely cut-off
    block should be caught.
    """
    out: List[str] = []
    open_re = re.compile(rf"<\s*{tag}\b[^>]*>", re.I)
    close_re = re.compile(rf"<\s*/\s*{tag}\s*>", re.I)
    for m in open_re.finditer(text or ""):
        start = m.end()
        close = close_re.search(text, start)
        end = close.start() if close else len(text)
        # A sibling tag before the close means this block was never closed.
        for sib in SIBLING_OPEN.finditer(text, start):
            if sib.start() >= end:
                break
            if close_re.match(text, sib.start()):
                continue
            end = sib.start()
            break
        body = text[start:end]
        if body.strip():
            out.append(body)
    return out


def parse_full(text: str) -> Optional[str]:
    """A whole replacement file, when the reply gave one instead of edits."""
    for body in reversed(block_bodies(text, "nn")):
        body = body.strip()
        if body:
            return body
    return None


def parse_edits(text: str) -> List[Tuple[str, str]]:
    """(search, replace) pairs from a reply, in the order written."""
    out: List[Tuple[str, str]] = []
    bodies = EDIT_RE.findall(text or "")
    if not bodies:
        bodies = [text or ""]
    for body in bodies:
        for search, replace in BLOCK_RE.findall(body):
            if search.strip():
                out.append((search, replace))
    return out


def render_change(before: str, after: str) -> Tuple[Optional[str], str]:
    """The shorter of an edit set and the whole file, with which was chosen.

    An edit carries both the old lines and the new ones, so it is roughly twice
    the changed text. Past about half the file that is more to write than the
    file itself, and the shorter form is also the one with less to get wrong.
    Choosing per pair rather than per corpus means the model is trained on both
    and learns which fits, instead of being made to patch a rewrite.
    """
    edits = render_edits(before, after)
    full = "<nn>" + after.strip() + "</nn>"
    if edits is None:
        return full, "full"
    return (edits, "edit") if len(edits) <= len(full) else (full, "full")


def render_edits(before: str, after: str) -> Optional[str]:
    """Search/replace blocks turning ``before`` into ``after``.

    Each SEARCH block is grown with surrounding context until it occurs exactly
    once in ``before``, so applying it cannot be ambiguous. Returns None when a
    hunk cannot be made unique within MAX_CONTEXT lines, which is the honest
    outcome: that pair is not expressible as an unambiguous edit and does not
    belong in the corpus.
    """
    la, lb = before.splitlines(), after.splitlines()
    sm = difflib.SequenceMatcher(None, la, lb, autojunk=False)
    hunks = [(i1, i2, j1, j2) for tag, i1, i2, j1, j2 in sm.get_opcodes()
             if tag != "equal"]
    if not hunks:
        return None

    spans = []
    for i1, i2, j1, j2 in hunks:
        ctx = 1
        while True:
            # Identical context on both sides, so the REPLACE block re-emits
            # the surrounding lines verbatim and the edit stays a local one.
            lead = min(ctx, i1, j1)
            tail = min(ctx, len(la) - i2, len(lb) - j2)
            s1, s2 = i1 - lead, i2 + tail
            t1, t2 = j1 - lead, j2 + tail
            search = "\n".join(la[s1:s2])
            if search.strip() and before.count(search) == 1:
                spans.append((s1, s2, t1, t2))
                break
            ctx += 1
            if ctx > MAX_CONTEXT:
                return None

    # merge spans that now overlap, or the second edit would not find its text
    merged = []
    for span in sorted(spans):
        if merged and span[0] <= merged[-1][1]:
            p = merged[-1]
            merged[-1] = (p[0], max(p[1], span[1]), p[2], max(p[3], span[3]))
        else:
            merged.append(span)

    blocks = []
    for s1, s2, t1, t2 in merged:
        blocks.append(f"{FIND_OPEN}\n" + "\n".join(la[s1:s2])
                      + f"\n{FIND_CLOSE}\n{WITH_OPEN}\n" + "\n".join(lb[t1:t2])
                      + f"\n{WITH_CLOSE}")
    return EDIT_OPEN + "\n" + "\n".join(blocks) + "\n" + EDIT_CLOSE


def _normalise(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines())


def apply_edits(code: str, edits: Sequence[Tuple[str, str]]) -> Tuple[Optional[str], str]:
    """Apply every edit in order. Returns (new_code, reason_if_failed)."""
    if not edits:
        return None, "no edit blocks"
    out = code
    for n, (search, replace) in enumerate(edits):
        hits = out.count(search)
        if hits == 1:
            out = out.replace(search, replace, 1)
            continue
        if hits > 1:
            return None, f"edit {n} matches {hits} places"
        # tolerate trailing-whitespace drift, which the model cannot see
        norm_out, norm_search = _normalise(out), _normalise(search)
        if norm_out.count(norm_search) == 1:
            out = norm_out.replace(norm_search, _normalise(replace), 1)
            continue
        return None, f"edit {n} matches nothing"
    if out == code:
        return None, "edits change nothing"
    return out, ""


def architecture_uid(code: str) -> Optional[str]:
    """LEMUR's own identity for a network: a hash of its source graph.

    Comments, whitespace and local variable names do not change it, so two
    files that spell the same architecture differently collapse to one id. That
    is the check LEMUR applies when a trained model is written back -- but only
    *after* it has been trained. Applying it before training costs nothing and
    turns a wasted hour into a skipped candidate.

    Returns None when nn-dataset is unavailable or the code has no layers, and
    the caller falls back to hashing the text.
    """
    try:
        from ab.nn.util.ArchUID import is_model_code, source_graph, wl_hash
    except Exception:
        return None
    try:
        if not is_model_code(code):
            return None
        g = source_graph(code)
        return wl_hash(g) if len(g.keys) else None
    except Exception:
        return None


def drop_noop_edits(edits: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Edits whose replacement is identical to what they match.

    A reply often quotes a region and returns it unchanged, to say "this part
    is fine". Applied, they do nothing; counted, they make a no-op answer look
    like a seven-edit improvement.
    """
    return [(s, r) for s, r in edits if _normalise(s) != _normalise(r)]


def apply_edit_sets(baseline: str, edits: Sequence[Tuple[str, str]],
                    ) -> List[Tuple[str, int, int]]:
    """Apply edits, splitting the reply into alternative candidates.

    A reply often carries several *alternative* edit sets rather than one
    cumulative one: the model rewrites the same region twice, offering two
    designs. Applied in sequence the second fails -- its <find> text no longer
    exists, because the first edit replaced it -- and treating that as one
    broken answer threw away both candidates. Measured on one cycle, 26 of 30
    generations were discarded this way while every individual edit matched the
    baseline exactly once.

    So an edit that does not match the working copy but does match the
    untouched baseline is read as the start of a new alternative, and the
    candidate so far is kept. Returns (code, applied, skipped) per candidate.
    """
    results: List[Tuple[str, int, int]] = []
    current, applied, skipped = baseline, 0, 0

    def match(code, search):
        if code.count(search) == 1:
            return search
        norm = _normalise(search)
        if _normalise(code).count(norm) == 1:
            return norm
        return None

    for search, replace in edits:
        hit = match(current, search)
        if hit is not None:
            current = (current if hit == search else _normalise(current))
            current = current.replace(hit, replace if hit == search
                                      else _normalise(replace), 1)
            applied += 1
            continue
        if applied and match(baseline, search) is not None:
            results.append((current, applied, skipped))
            current, applied, skipped = baseline, 0, 0
            hit = match(current, search)
            current = (current if hit == search else _normalise(current))
            current = current.replace(hit, replace if hit == search
                                      else _normalise(replace), 1)
            applied = 1
            continue
        skipped += 1

    if applied:
        results.append((current, applied, skipped))
    return results


def validate_model(code: str) -> str:
    """Empty string when the file could be evaluated, else why not."""
    try:
        # Corpus files are parsed by the thousand and some contain sloppy string
        # escapes. Their SyntaxWarnings are about the model file, not about this
        # run, and printing them buries the lines that are.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code)
    except SyntaxError as exc:
        return f"syntax error line {exc.lineno}: {exc.msg}"
    names = {n.name for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    if "Net" not in names:
        return "no class Net"
    missing = [s for s in REQUIRED_SYMBOLS if s not in names]
    if missing:
        return "missing " + ", ".join(missing)
    return ""


HEADER_PRM = {"LR": "lr", "Momentum": "momentum"}

#: The only hyperparameters a reply may set. They are read out of prm at
#: training time rather than written in the file, so no edit can reach them,
#: and everything else (batch size, epoch count, transform) is the fixed
#: protocol the measurement depends on.
#: Which training-configuration keys an <hp> block may set.
#:
#: Was ("lr", "momentum"). Every one of 60 evaluated generations of one run
#: trained at lr=0.01, momentum=0.9, dropout=0.2 -- the values of the seed
#: whose family supplied almost every baseline -- and two of them diverged to
#: 0.13 accuracy at that learning rate. Dropout is added because it is the one
#: regularisation dial the file itself reads from prm; batch is deliberately
#: left out, as a model that raises it changes the memory envelope the eval
#: worker was sized for.
TUNABLE_PRM = ("lr", "momentum", "dropout")

#: Columns a LEMUR record carries that are results of a training run, not
#: inputs to one. They arrive in the record's prm and have no business in the
#: hyperparameters written beside a generated model.
MEASURED_PRM = frozenset({
    "uid", "accuracy", "metric_acc", "duration", "duration_seconds",
    "epoch_max", "nn", "task", "dataset", "metric", "train_stat",
    "layer_stat", "prm",
})

#: What an <hp> block may not set: the results of a past run, the record's
#: identity, and the epoch budget, which belongs to the cycle. Everything else
#: is the model's to propose.
RESERVED_PRM = MEASURED_PRM | frozenset({"epoch"})


def _hp_value(value):
    """Keep a proposed value in the type the model will read it as."""
    if isinstance(value, bool) or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return value
    raise TypeError(type(value).__name__)

def hp_text(prm) -> str:
    """The tunable hyperparameters of a record, as the prompt shows them.

    Without these, two records of the same architecture at different learning
    rates read as an architectural win. The model is scored against a baseline
    it can see the hyperparameters of, so it can tell an architectural gain
    from a tuning gain -- and it cannot aim an <hp> block without knowing what
    it is changing from.
    """
    prm = prm or {}
    vals = [(k, prm.get(k)) for k in TUNABLE_PRM]
    vals = [(k, v) for k, v in vals if isinstance(v, (int, float))]
    if not vals:
        return "not recorded"
    return ", ".join(f"{k}={float(v):g}" for k, v in vals)


HP_RE = re.compile(r"<hp[^>]*>(.*?)</hp>", re.S)
TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)


def parse_tr(text: str) -> Optional[str]:
    """The transform module in a <tr> block, or None.

    Guards the one failure this slot has: the model putting a whole network in
    it. A transform module defines ``transform`` and nothing that looks like a
    ``Net``; anything else is refused rather than written out as tr.py, where
    it would silently replace the data pipeline with a model.
    """
    for body in reversed(TR_RE.findall(text or "")):
        code = body.strip()
        if not code or "def transform" not in code:
            continue
        if re.search(r"^class\s+Net\b", code, re.M):
            continue
        if "supported_hyperparameters" in code:
            continue
        try:
            ast.parse(code)
        except SyntaxError:
            continue
        return code
    return None


def parse_hp(text: str) -> Dict:
    """Every hyperparameter an <hp> block proposes.

    Not a fixed list. supported_hyperparameters() names what a file really
    takes from prm, and a generated file may take whatever it likes -- a
    normalisation epsilon, a patch size, a mixup weight -- provided it proposes
    a value for it here, alongside the standard ones.

    This used to keep only lr, momentum and dropout. Every other name a model
    declared then reached the trainer with nothing behind it, so the file's own
    prm[name] raised KeyError: the model asked for a hyperparameter, the
    pipeline dropped the answer, and the candidate was discarded for it.
    """
    out: Dict = {}
    for body in HP_RE.findall(text or ""):
        try:
            payload = json.loads(body.strip())
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            key = str(key)
            if key in RESERVED_PRM:
                continue
            try:
                out[key] = _hp_value(value)
            except TypeError:
                pass
    return out


def derive_prm(code: str, base_prm, hp=None) -> Dict:
    """Training hyperparameters for the edited file.

    The file reads lr and momentum out of prm, so an edit to the code cannot
    change them. One corpus wrote them into a header comment and that was read
    back here -- but it is a convention of a single architecture family, and on
    anything else the model had no way to change them at all. An explicit <hp>
    block works for every architecture; the header comment is still honoured
    where it exists, and an <hp> block wins over it.
    """
    prm = dict(base_prm or {})
    head = code[:2000]
    for key, name in HEADER_PRM.items():
        m = re.search(rf"^#\s*{key}:\s*([0-9.eE+-]+)\s*$", head, re.M)
        if m:
            try:
                prm[name] = float(m.group(1))
            except ValueError:
                pass
    # Every proposed name, not a fixed three: the trainer has to carry a value
    # for each name the file declares, or the file cannot read it.
    for name, value in (hp or {}).items():
        if name not in RESERVED_PRM:
            prm[name] = value
    for key in MEASURED_PRM:
        prm.pop(key, None)
    return prm


def model_identity(code: str, prm=None) -> str:
    """How LEMUR tells two trained models apart: architecture plus prm.

    The graph id alone is not it. Learning rate and momentum are read out of
    prm and appear in the file only as comments, so a candidate that changes
    nothing but the learning rate has the *same* graph id as its baseline --
    measured on one corpus, 1,024 models collapsed to 256 architectures. Keying
    on the graph alone would reject those as duplicates, discarding a factor
    worth real accuracy. The hyperparameters that vary independently of the
    code are folded in here.
    """
    uid = architecture_uid(code)
    if uid is None:
        uid = hashlib.sha1(_normalise(code).encode()).hexdigest()
    tail = derive_prm(code, prm or {})
    bits = [f"{k}={tail[k]}" for k in sorted(HEADER_PRM.values()) if k in tail]
    return uid + ("|" + ",".join(bits) if bits else "")


def verify_round_trip(pairs: Sequence[Tuple[str, str]], sample: int = 500,
                      seed: int = 17) -> Dict[str, object]:
    """render_edits then apply_edits must reproduce the target byte for byte."""
    rng = random.Random(seed)
    pairs = list(pairs)
    if len(pairs) > sample:
        pairs = rng.sample(pairs, sample)
    checked = unrenderable = mismatch = invalid = 0
    first = None
    for before, after in pairs:
        rendered = render_edits(before, after)
        if rendered is None:
            unrenderable += 1
            continue
        checked += 1
        built, why = apply_edits(before, parse_edits(rendered))
        if built is None or _normalise(built) != _normalise(after):
            mismatch += 1
            if first is None:
                first = why or "result differs from the target"
        elif validate_model(built):
            invalid += 1
    return {"pairs": len(pairs), "rendered": checked,
            "unrenderable": unrenderable, "mismatch": mismatch,
            "invalid_after_apply": invalid, "first_failure": first}
