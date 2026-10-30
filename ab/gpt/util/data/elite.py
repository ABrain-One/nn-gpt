"""The best models this run has produced, kept and shown back to it.

The objective is the maximum accuracy a cycle reaches, and nothing in the loop
was aimed at it. A generation entered the corpus only if the corpus rebuild
happened to return it, the prompt named a ceiling frozen at cycle 0, and the
best model the run had ever produced was never mentioned to the model again.
Measured consequence: the highest accuracy of the whole run, 0.9708, was
reached in cycle 0 -- before any fine-tuning -- and sixteen cycles of training
never beat it.

This module keeps an explicit archive. After each evaluation the best results
are recorded with the edits that produced them, and the best of them is shown
in the next cycle's prompts as something to beat. That is the difference
between "improve this model" and "improve this model, and here is the best
anything has managed so far".

The archive is this run's own output, written under nngpt_dir, and is reset
with the run directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

ARCHIVE_NAME = "elite.json"

#: How many champions to keep. More than one so a single lucky measurement
#: cannot define the target for the rest of the run.
KEEP = int(os.environ.get("NNGPT_ELITE_KEEP", "5") or 5)

#: How many to put in a prompt. One: the point is a target, not a library, and
#: every example costs room the answer needs.
SHOW = int(os.environ.get("NNGPT_ELITE_SHOW", "1") or 1)


def archive_path(nngpt_dir) -> Path:
    return Path(nngpt_dir) / ARCHIVE_NAME


def load(nngpt_dir) -> List[dict]:
    p = archive_path(nngpt_dir)
    if not p.is_file():
        return []
    try:
        rows = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    return rows if isinstance(rows, list) else []


def _model_accuracy(model_dir: Path) -> Optional[float]:
    info = model_dir / "eval_info.json"
    if not info.is_file():
        return None
    try:
        payload = json.loads(info.read_text(encoding="utf-8"))
    except Exception:
        return None
    acc = payload.get("accuracy")
    if acc is None:
        res = payload.get("eval_results")
        if isinstance(res, dict):
            acc = res.get("accuracy", res.get("acc"))
    try:
        return float(acc)
    except (TypeError, ValueError):
        return None


def _baseline_code(model_dir: Path) -> Optional[str]:
    """The file this candidate was edited from, saved beside it."""
    for f in sorted(model_dir.glob("original_*.py")):
        try:
            return f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
    return None


def update(models_dir, nngpt_dir, epoch=None) -> List[dict]:
    """Fold this cycle's results into the archive. Returns it, best first.

    Each entry stores the edits rather than the file: an edit is a few dozen
    lines against several hundred, which is what makes showing it in a prompt
    affordable.
    """
    from ab.gpt.util.data.code_edit import render_edits

    rows = load(nngpt_dir)
    known = {r.get("code_hash") for r in rows}
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        return rows

    import hashlib
    for d in sorted(models_dir.glob("B*")):
        acc = _model_accuracy(d)
        if acc is None:
            continue
        nn_file = d / "new_nn.py"
        if not nn_file.is_file():
            continue
        try:
            code = nn_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        code_hash = hashlib.sha1(code.encode("utf-8", "replace")).hexdigest()
        if code_hash in known:
            continue
        base = _baseline_code(d)
        edits = render_edits(base, code) if base else None
        prm = {}
        hp = d / "hp.txt"
        if hp.is_file():
            try:
                loaded = json.loads(hp.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    prm = {k: loaded[k] for k in ("lr", "momentum", "dropout")
                           if k in loaded}
            except Exception:
                prm = {}
        rows.append({"accuracy": round(acc, 4), "epoch": epoch,
                     "model": d.name, "code_hash": code_hash,
                     "edits": edits, "prm": prm})
        known.add(code_hash)

    rows.sort(key=lambda r: -(r.get("accuracy") or 0.0))
    rows = rows[:KEEP]
    try:
        archive_path(nngpt_dir).parent.mkdir(parents=True, exist_ok=True)
        archive_path(nngpt_dir).write_text(json.dumps(rows, indent=1),
                                           encoding="utf-8")
    except Exception as exc:
        print(f"[elite] could not write the archive: {exc}", flush=True)
    if rows:
        print(f"[elite] best ever {rows[0]['accuracy']:.4f} "
              f"({rows[0].get('model')}, epoch {rows[0].get('epoch')}); "
              f"archive holds {len(rows)}", flush=True)
    return rows


def best_accuracy(nngpt_dir) -> Optional[float]:
    rows = load(nngpt_dir)
    return rows[0]["accuracy"] if rows else None


def prompt_block(nngpt_dir, show: Optional[int] = None) -> str:
    """The champions, as a block to append to a generation prompt.

    Shown as edits to *their own* baseline, which is not the baseline being
    edited now -- so it is context, not something to copy verbatim. Says what
    the number to beat actually is, instead of a corpus ceiling that was
    computed once at cycle 0 and never moved.
    """
    rows = [r for r in load(nngpt_dir) if r.get("edits")]
    if not rows:
        return ""
    rows = rows[:max(1, show if show is not None else SHOW)]
    blocks = []
    for r in rows:
        inner = str(r["edits"]).strip()
        blocks.append(f'<best_so_far measured="{r["accuracy"]:.4f}">\n'
                      f'{inner}\n</best_so_far>')
    return ("\nThe best result anything in this run has reached so far, and the "
            "edits that produced it, against its own baseline. Beat it:\n\n"
            + "\n".join(blocks) + "\n")
