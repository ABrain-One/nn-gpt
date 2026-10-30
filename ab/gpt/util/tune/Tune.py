# ab/gpt/util/Tune.py
"""
ab/gpt/util/Tune.py — Central tuning pipeline for NNGPT.
- tune() is the ONLY entry point.
- All logic lives here: nn_gen, trans_gen, generate_step, finetune_step.
- Agents only call generate_step() and finetune_step() from this file.
"""


import os
import random
import re
import shutil
import json
import subprocess
import sys
from os import makedirs
from os.path import isfile
import glob
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import torch
import ab.nn.api as lemur
import deepspeed
from ab.nn.util.Util import release_memory, create_file
from ab.gpt.util.HostMemory import release_host_memory
from peft import PeftModel
from tqdm import tqdm

import ab.gpt.act.eval.Eval as NNEval
from ab.gpt.util.llm.Chatbot import ChatBot
from ab.gpt.util.Const import *
from ab.gpt.util.Const import nngpt_dir

from ab.gpt.util.llm.LLMUtil import quantization_config_4bit
from ab.gpt.util.llm.LoRA import LoRA
from ab.gpt.util.Util import (
    exists,
    extract_delta,
    extract_code,
    extract_hyperparam,
    extract_transform,
    is_nn_model_code,
)
from ab.gpt.util.prompt.NNGenPrompt import NNGenPrompt
from ab.gpt.util.nn.DeltaUtil import apply_delta, validate_delta, repair_code
from ab.gpt.util.Const import nngpt_upload, DEFAULT_DATASET, DEFAULT_NN_PREFIXES
import ab.gpt.util.llm.SFTUtil as SFTUtil
from ab.gpt.brute.trans.TransformEval import run_eval
from ab.gpt.util.prompt.TransformGenPrompt import TransformGenPrompt, load_data_from_folders
from ab.gpt.act.agents.state import AgentState
import ab.gpt.util.training_runtime as TrainingRuntime

ds_conf = conf_dir / 'DeepSpeed.json'
TRANSFORM_OUT_DIR = trans_dir / 'dataset_epoch1'
TRANSFORM_RES_DIR = trans_dir / 'result_epoch1'


def _env_on(name, default: bool = False) -> bool:
    """Is this switch on? ``default`` applies when the variable is unset.

    Needed for switches that are on unless turned off; the original form could
    only express opt-in, so an unset variable always meant "off".
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _carry_probe(model, tokenizer, max_len, out_dir):
    """Mean token loss of the model as built for this cycle on fixed chat rows
    (NNGPT_CARRY_PROBE=comma-separated jsonl paths; NNGPT_CARRY_PROBE_N rows each).
    Measurement only: prints one line per file and writes carry_probe.json."""
    import hashlib
    paths = [p for p in os.environ.get("NNGPT_CARRY_PROBE", "").split(",") if p.strip()]
    n_rows = int(os.environ.get("NNGPT_CARRY_PROBE_N", "64"))
    results = {}
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    for path in paths:
        rows = []
        with open(path.strip(), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
                if len(rows) >= n_rows:
                    break
        per_row = []
        for row in rows:
            messages = row.get("messages", [])
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            # Same tokenization as SFT (Prompt.preprocess_batch).
            ids = tokenizer(text, truncation=True, max_length=max_len)["input_ids"]
            prompt = tokenizer.apply_chat_template(
                [m for m in messages if m["role"] != "assistant"], tokenize=False, add_generation_prompt=True)
            n_prompt = len(tokenizer(prompt, truncation=True, max_length=max_len)["input_ids"])
            x = torch.tensor([ids], device=device)
            with torch.no_grad():
                logits = model(input_ids=x).logits[0, :-1].float()
            tok_loss = torch.nn.functional.cross_entropy(logits, x[0, 1:], reduction="none")
            code = tok_loss[max(n_prompt - 1, 0):]
            per_row.append({
                "ids_sha1": hashlib.sha1(json.dumps(ids).encode()).hexdigest(),
                "n_tokens": len(ids) - 1,
                "loss_sum_all": float(tok_loss.sum()),
                "n_code": int(code.numel()),
                "loss_sum_code": float(code.sum()),
            })
        n_all = sum(r["n_tokens"] for r in per_row) or 1
        n_code = sum(r["n_code"] for r in per_row) or 1
        summary = {
            "n": len(per_row),
            "loss_all": sum(r["loss_sum_all"] for r in per_row) / n_all,
            "loss_code": sum(r["loss_sum_code"] for r in per_row) / n_code,
            "rows": per_row,
        }
        results[path.strip()] = summary
        print(f"[carry-probe] file={path.strip()} n={summary['n']} "
              f"loss_all={summary['loss_all']:.6f} loss_code={summary['loss_code']:.6f}", flush=True)
    if was_training:
        model.train()
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "carry_probe.json").write_text(json.dumps(results, indent=1))
    return results


# Delta mode constants
_MAX_DELTA_RETRIES = 2


def apply_sliding_window(example, max_length, stride, tokenizer):
    input_ids = example['input_ids']
    attention_mask = example['attention_mask']

    chunks = []
    for i in range(0, len(input_ids), stride):
        end = i + max_length
        if end <= len(input_ids):
            chunk_input_ids = input_ids[i:end]
            chunk_attention_mask = attention_mask[i:end]

            pad_len = max_length - len(chunk_input_ids)
            if pad_len > 0:
                chunk_input_ids += [tokenizer.pad_token_id] * pad_len
                chunk_attention_mask += [0] * pad_len

            chunks.append({"input_ids": chunk_input_ids,
                          "attention_mask": chunk_attention_mask})
    return {"chunks": chunks}


def flatten_chunks(data):
    all_chunks = sum(data["chunks"], [])  # flatten batched list
    return {
        "input_ids": [chunk["input_ids"] for chunk in all_chunks],
        "attention_mask": [chunk["attention_mask"] for chunk in all_chunks],
    }


# ============================================================
# SINGLE SOURCE OF TRUTH: GENERATION (nn_gen / trans_gen)
# ============================================================

# How many seed rows to draw per requested model. A prompt that does not leave
# MIN room for the answer is skipped, and a skipped prompt used to cost the epoch
# one model: test_nn rows in, fewer than test_nn models out, and an empty
# full_output.txt where the generation should have been. Drawing a surplus turns
# that into a refill -- oversized prompts are dropped and the next candidate
# takes the slot, until test_nn prompts have been accepted.
SEED_POOL_FACTOR = float(os.environ.get("NNGPT_SEED_POOL_FACTOR", "3") or 3)
SEED_POOL_MIN_SURPLUS = 8


# Smallest answer worth asking for. A complete reply is three blocks -- <hp>,
# <tr> and a full <nn> class -- which does not fit in a few hundred tokens. When
# the window left after the prompt is below this, the model is being asked for
# something it cannot deliver and the output comes back cut off mid-class, which
# is what filled the last cycles with unparseable generations. Such a prompt is
# dropped and the seed pool refills the slot instead.
MIN_ANSWER_TOKENS = int(os.environ.get("NNGPT_MIN_ANSWER_TOKENS", "1280") or 1280)


def warn_if_truncated(full_out, max_new_tokens, idx, tokenizer=None):
    """Report a generation that ran out of budget instead of finishing.

    This was called in the delta path but never defined or imported anywhere, so
    every delta generation raised NameError the moment it reached this line. It
    is defined here and used on both generation paths, because a truncated reply
    is the single most common way a cycle loses a model: the code block is cut
    mid-class, extraction finds no closing tag, and the model is counted as a
    failure with no indication of why.

    Returns True when the output looks truncated.
    """
    if not isinstance(full_out, str) or not full_out.strip():
        return False

    reasons = []
    if tokenizer is not None and max_new_tokens:
        try:
            produced = len(tokenizer(full_out, add_special_tokens=False)["input_ids"])
            if produced >= max_new_tokens - 2:
                reasons.append(f"hit the {max_new_tokens}-token generation budget")
        except Exception:
            pass
    # An unclosed block is only evidence of truncation when the reply stops
    # inside it. The model also just forgets the closing tag and carries on:
    # measured on one cycle, three replies held a finished file, a closed <tr>
    # and a closed <reason> and were reported as cut off over one absent
    # </nn>. If any block closes after this one opened, the reply reached the
    # end of its answer and the tag was merely omitted -- code_edit's
    # block_bodies reads it anyway, so there is nothing to report.
    for tag in ("nn", "tr", "hp"):
        opened = [m.end() for m in re.finditer(rf"<\s*{tag}\b[^>]*>", full_out)]
        if not opened or re.search(rf"<\s*/\s*{tag}\s*>", full_out):
            continue
        if re.search(r"<\s*/\s*(?:nn|tr|hp|edit|reason)\s*>", full_out[opened[-1]:]):
            continue
        reasons.append(f"<{tag}> was opened and never closed")
    if full_out.count("```") % 2:
        reasons.append("an unterminated code fence")

    if reasons:
        print(f"[TRUNCATED] B{idx}: {'; '.join(reasons)}. The reply is cut off, so "
              f"extraction will find less than was asked for. Lower the prompt size "
              f"or raise max_seq_length so the answer has room.", flush=True)
        return True
    return False


#: Set to 0 to skip the repair pass.
#: Off by default. The repair turn is the only thing that smoke-tests a
#: candidate *during generation*, and generation is the one moment the host
#: cannot afford it: the vLLM engine is resident and the trainer is parked in
#: RAM, which left 6 to 7 GiB free on a 31 GiB machine. Every candidate was
#: already being tested a second time at the validity gate, which runs after
#: the engine is torn down and has the memory for it, so the work was
#: duplicated and only the duplicate was dangerous. Removing it makes every
#: cycle behave identically -- the first one included, so a single cycle's log
#: shows whether the problem is gone.
#:
#: Set NNGPT_REPAIR_TURN=1 to have it back. It is cheap now (a build check, no
#: backward pass) and bounded by the host-headroom guard, but it still runs in
#: the tight window, and a cycle that skips it loses one retry, not a model.
REPAIR_ENV = "NNGPT_REPAIR_TURN"


def _repair_targets(models_dir, queue):
    """Written candidates that cannot be built, with the reason.

    The smoke test exists and runs -- but in the validity gate, after
    generation, when the model is parked and nothing can be asked. Running it
    here, while the engine is still up, is what makes a repair turn possible at
    all. In one measured run 218 evaluation failures were a single generated
    helper unpacking a 4-tuple in_shape into five names, regenerated every
    cycle because the model was never shown the traceback.
    """
    from ab.gpt.util.eval.smoke import smoke_check_dir
    out = []
    for idx, origdf, prompt_text in queue:
        for d in sorted(Path(models_dir).glob(f"B{idx}") ) + sorted(
                Path(models_dir).glob(f"B{idx}_*")):
            if not (d / new_nn_file).is_file():
                continue
            verdict, reason = smoke_check_dir(d, light=True)
            if verdict is False:
                out.append((d, origdf, prompt_text, reason))
    return out


REPAIR_PROMPT = """\
The edits you returned were applied, and the resulting file does not run. It \
was built exactly as the evaluation builds it -- Net(in_shape, out_shape, prm, \
device) with a 4-tuple in_shape, then train_setup, then one batch through \
forward and learn -- and it raised:

{reason}

Return the same design again, corrected, as <edit> blocks against the ORIGINAL \
baseline file you were given (not against your previous answer). Keep the \
improvement you intended; change only what is needed to make it run."""


def _run_repair_pass(models_dir, queue, chat_bot, seen, max_new_tokens,
                     save_llm_output):
    """One more turn for candidates that do not build, with their traceback.

    A second chance, not a loop: if the repair also fails to build, the
    validity gate discards it as before. Costs one generation round per cycle
    against the ten minutes of GPU time each broken candidate would otherwise
    take to fail in exactly the way already known here.
    """
    if not queue:
        return 0
    if not _env_on(REPAIR_ENV, default=False):
        print(f"[repair] off; candidates go straight to the validity gate, "
              f"which smoke-tests them once the engine is down "
              f"({REPAIR_ENV}=1 to re-enable)", flush=True)
        return 0
    try:
        targets = _repair_targets(models_dir, queue)
    except Exception as exc:
        print(f"[repair] skipped: {type(exc).__name__}: {exc}", flush=True)
        return 0
    if not targets:
        print("[repair] every generated candidate builds; nothing to repair",
              flush=True)
        return 0

    try:
        from ab.gpt.util.eval.smoke import _host_available_gib, HOST_FLOOR_GIB
        print(f"[repair] host has {_host_available_gib():.1f} GiB available "
              f"(floor {HOST_FLOOR_GIB:.0f} GiB)", flush=True)
    except Exception:
        pass
    print(f"[repair] {len(targets)} candidate(s) do not build (build check "
          f"only: construct, train_setup and one forward without gradients); "
          f"asking once more with the traceback", flush=True)
    prompts, meta = [], []
    for d, origdf, prompt_text, reason in targets:
        prompts.append(prompt_text + "\n\n"
                       + REPAIR_PROMPT.format(reason=reason))
        meta.append((d, origdf))
    try:
        if hasattr(chat_bot, "chat_batch"):
            outputs = chat_bot.chat_batch(
                prompts, engineer_prompt=False, max_new_tokens=max_new_tokens)
        else:
            outputs = [chat_bot.chat(p, engineer_prompt=False,
                                     max_new_tokens=max_new_tokens)
                       for p in prompts]
    except Exception as exc:
        print(f"[repair] generation failed: {type(exc).__name__}: {exc}",
              flush=True)
        return 0

    fixed = 0
    for (d, origdf), output in zip(meta, outputs):
        full_out = output[3] if isinstance(output, (list, tuple)) else output
        if not isinstance(full_out, str) or not full_out.strip():
            continue
        # Replaces the broken candidate in place, under its own name, so the
        # cycle's slot count and its seed pairing are unchanged.
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass
        idx = d.name[1:]
        written = _write_edit_candidates(Path(models_dir), idx, origdf,
                                        full_out, seen, save_llm_output)
        if written:
            fixed += 1
            try:
                create_file(Path(models_dir) / f"B{idx}", "repaired.txt",
                            "rewritten after a failed build\n")
            except Exception:
                pass
    print(f"[repair] {fixed} of {len(targets)} rewritten", flush=True)
    return fixed


def _best_of_n() -> int:
    """How many samples to draw per seed."""
    try:
        return max(1, int(os.environ.get("NNGPT_BEST_OF_N", "1") or 1))
    except ValueError:
        return 1


def _elite_block() -> str:
    """The run's best-so-far, as prompt context. Empty before the first one."""
    if not _env_on("NNGPT_ELITE_IN_PROMPT", default=True):
        return ""
    try:
        from ab.gpt.util.data import elite
        return elite.prompt_block(nngpt_dir)
    except Exception as exc:
        print(f"[elite] not shown in the prompt: {exc}", flush=True)
        return ""


def _edit_seen_path():
    return nngpt_dir / "edit_seen.json"


def _load_seen_code():
    seen = set()
    init = os.environ.get("NNGPT_EDIT_KNOWN_UIDS", "").strip()
    if init and Path(init).is_file():
        try:
            seen.update(json.loads(Path(init).read_text(encoding="utf-8")))
        except Exception as exc:
            print(f"[edit] could not read {init}: {exc}", flush=True)
    path = _edit_seen_path()
    if path.is_file():
        try:
            seen.update(json.loads(path.read_text()))
        except Exception as exc:
            print(f"[edit] could not read {path}: {exc}", flush=True)
    return seen


def _save_seen_code(seen):
    try:
        _edit_seen_path().parent.mkdir(parents=True, exist_ok=True)
        _edit_seen_path().write_text(json.dumps(sorted(seen)))
    except Exception as exc:
        print(f"[edit] could not persist seen code: {exc}", flush=True)


#: Transform names this cycle reused from the dataset, and minted itself.
_transform_counts = {"reused": set(), "new": set()}


def reset_transform_counts():
    _transform_counts["reused"].clear()
    _transform_counts["new"].clear()


#: Set to 1 to make a generated <tr> module usable by evaluation.
#:
#: ``prm["transform"]`` is resolved by NAME out of the installed ab/nn/transform
#: package, so a module that exists only as tr.py in the generation directory is
#: written, submitted -- and never trained with. Installing it under its
#: content-addressed name is what closes that gap. Off by default because it
#: writes a file into the installed dataset package; with it off the transform is
#: still saved and still shipped in the submission, and evaluation uses the
#: baseline's pipeline.
INSTALL_TRANSFORMS_ENV = "NNGPT_INSTALL_GENERATED_TRANSFORMS"


def _ab_nn_transform_dir():
    """Where the installed LEMUR package keeps its transform modules."""
    import importlib.util
    spec = importlib.util.find_spec("ab.nn")
    if not spec or not spec.origin:
        return None
    d = Path(spec.origin).parent / "transform"
    return d if d.is_dir() else None


def _check_transform(code):
    """``(ok, reason)``; None when the check could not run, so the block stays."""
    if not _env_on("NNGPT_TRANSFORM_SMOKE", default=True):
        return None, "transform check disabled"
    try:
        from ab.gpt.util.eval.smoke import transform_check
        return transform_check(code)
    except Exception as exc:
        return None, f"transform check unavailable: {type(exc).__name__}: {exc}"


def _install_generated_transform(code):
    """Make a generated transform resolvable by name. Returns the name or None.

    Checks the uid against what already exists first. A model very often
    proposes a pipeline the dataset already has -- crop, flip, ToTensor,
    Normalize is the obvious CIFAR answer -- and installing it again under a
    generated name produced several files for one transform, none of them
    connected to the dataset's own name for it.
    """
    if not _env_on(INSTALL_TRANSFORMS_ENV):
        return None
    from ab.gpt.util.Util import resolve_transform_name
    name, is_new = resolve_transform_name(code)
    if not is_new:
        print(f"[edit] transform already in the dataset as {name}; reusing it",
              flush=True)
        return name
    target_dir = _ab_nn_transform_dir()
    if target_dir is None:
        print("[edit] cannot install the generated transform: ab/nn/transform "
              "not found; evaluation will use the baseline pipeline", flush=True)
        return None
    path = target_dir / f"{name}.py"
    try:
        if not path.exists():
            path.write_text(code, encoding="utf-8")
        return name
    except Exception as exc:
        print(f"[edit] could not install transform {name}: {exc}", flush=True)
        return None


#: Estimated Jaccard above which a new candidate counts as a restatement of
#: one already accepted this cycle.
#:
#: model_identity is a graph hash plus lr/momentum, so a candidate that swaps
#: one backbone name is a different architecture by that measure and passes.
#: Measured: all 60 valid generations of one run shared a class skeleton and
#: differed by about twelve lines, and every one of them passed. 0.95 rejects a
#: token swap (measured 0.953 on a one-token change) while leaving a genuine
#: edit (0.0-0.8) alone.
NEAR_DUPLICATE_JACCARD = float(
    os.environ.get("NNGPT_NEAR_DUPLICATE_JACCARD", "0.95") or 0.95)

#: Signatures accepted in this cycle, so the check is intra-cycle only; across
#: cycles the graph-identity check and known_uids already apply.
_cycle_signatures: list = []


def reset_cycle_signatures():
    _cycle_signatures.clear()


def _too_similar(code, idx, n):
    """Why this candidate is a restatement of an accepted one, or None."""
    if NEAR_DUPLICATE_JACCARD >= 1.0:
        return None
    try:
        from ab.gpt.util.data.minhash_clusters import estimate_jaccard, signature
        sig = signature(code)
    except Exception:
        return None
    if sig is None:
        return None
    for other_name, other in _cycle_signatures:
        j = estimate_jaccard(sig, other)
        if j >= NEAR_DUPLICATE_JACCARD:
            return (f"a restatement of {other_name} (estimated Jaccard "
                    f"{j:.3f} >= {NEAR_DUPLICATE_JACCARD:.2f})")
    _cycle_signatures.append((f"B{idx}" + (f"_{n}" if n else ""), sig))
    return None


def _write_edit_candidates(models_dir, idx, origdf, full_out, seen,
                           save_llm_output):
    """Apply the reply's <edit> blocks to the baseline and save the result.

    Unlike the design path there is no closed set to validate against, so the
    guards are the ones that work for any architecture: the edit has to match
    exactly one place in the file, the result has to parse, it has to keep the
    LEMUR interface, and it has to be a file nothing has produced before.
    """
    from ab.gpt.util.data.code_edit import (
        apply_edit_sets, derive_prm, drop_noop_edits, hp_text, model_identity,
        uses_pretrained_weights,
        parse_edits, parse_full, parse_hp, parse_tr, validate_model)
    import hashlib

    base_dir = models_dir / f"B{idx}"
    makedirs(base_dir, exist_ok=True)
    create_file(base_dir, new_out_file, full_out or "")
    if origdf is not None:
        try:
            origdf.to_pickle(base_dir / "dataframe.df")
        except Exception as exc:
            print(f"[edit] B{idx}: could not save the baseline row: {exc}", flush=True)
    if origdf is None:
        print(f"[edit] B{idx}: no baseline row", flush=True)
        return 0

    baseline_code = origdf.get("nn_code", "") or ""
    hp = parse_hp(full_out)
    # The reply's input pipeline, if it offered one. Extracted here because the
    # edit path never looked for it: of 210 generations in one run the transform
    # slot was read 0 times, so the data-augmentation axis -- the cheapest
    # accuracy on a one-epoch CIFAR budget -- was never searched at all.
    tr_code = parse_tr(full_out)
    if tr_code:
        # Built and applied to one image before it is installed anywhere. A
        # broken pipeline used to reach the trainer and kill every evaluation
        # that used it; refusing it here costs the transform and keeps the
        # architecture edit, which is the part worth measuring.
        ok, why = _check_transform(tr_code)
        if ok is False:
            print(f"[edit] B{idx}: transform rejected ({why}); keeping the "
                  f"baseline pipeline", flush=True)
            tr_code = None
    tr_name = _install_generated_transform(tr_code) if tr_code else None
    if tr_code:
        print(f"[edit] B{idx}: transform offered"
              + (f", installed as {tr_name}" if tr_name
                 else " (saved and submitted; evaluation keeps the baseline "
                      "pipeline -- set NNGPT_INSTALL_GENERATED_TRANSFORMS=1 to "
                      "train with it)"), flush=True)
    raw_edits = parse_edits(full_out)
    edits = drop_noop_edits(raw_edits)
    if raw_edits and not edits:
        print(f"[edit] B{idx}: all {len(raw_edits)} edit(s) return the text "
              f"unchanged", flush=True)
    elif len(raw_edits) != len(edits):
        print(f"[edit] B{idx}: {len(raw_edits) - len(edits)} of {len(raw_edits)} "
              f"edit(s) change nothing and were dropped", flush=True)

    # Candidates come from the edit sets *and* from any whole file in the same
    # reply. A reply that opens with edits and then thinks better of it and
    # rewrites the file is common, and taking the edits merely because they
    # came first threw the rewrite away.
    candidates = [(code, f"{applied} edit(s)"
                   + (f", {skipped} skipped" if skipped else ""))
                  for code, applied, skipped in apply_edit_sets(baseline_code, edits)]
    whole = parse_full(full_out)
    if whole:
        candidates.append((whole, "whole file"))
    if not candidates:
        head = " ".join((full_out or "").split())[:160]
        print(f"[edit] B{idx}: nothing applied; reply starts: {head!r}", flush=True)
        return 0

    written = 0
    for n, (code, how) in enumerate(candidates):
        bad = validate_model(code)
        if bad:
            print(f"[edit] B{idx} candidate {n}: rejected ({bad})", flush=True)
            continue
        # LEMUR's own identity, which ignores comments, whitespace and local
        # names. Checking it here rejects a duplicate before it costs an hour
        # of training instead of after, when copy_to_lemur refuses the insert.
        prm = derive_prm(code, origdf.get("prm") or {}, hp)
        key = model_identity(code, prm)
        if key == model_identity(baseline_code, origdf.get("prm") or {}):
            print(f"[edit] B{idx} candidate {n}: same architecture as the "
                  f"baseline", flush=True)
            continue
        near = _too_similar(code, idx, n)
        if near is not None:
            print(f"[edit] B{idx} candidate {n}: {near}", flush=True)
            try:
                create_file(base_dir, "duplicate.txt", f"{near}\n")
            except Exception:
                pass
            continue
        if key in seen:
            print(f"[edit] B{idx} candidate {n}: architecture already produced",
                  flush=True)
            # Leave a reason behind. classify_failures() reads error.txt and
            # falls through to "truncated" when there is nothing to read, so a
            # cycle that lost 76 candidates to the duplicate check reported
            # them as truncated replies and the budget was raised instead of
            # the sampler being fixed.
            try:
                create_file(base_dir, "duplicate.txt",
                            f"architecture already produced: {key}\n")
            except Exception:
                pass
            continue
        seen.add(key)
        model_dir = base_dir if written == 0 else models_dir / f"B{idx}_{written}"
        makedirs(model_dir, exist_ok=True)
        if save_llm_output and model_dir is not base_dir:
            create_file(model_dir, new_out_file, full_out)
            origdf.to_pickle(model_dir / "dataframe.df")
        create_file(model_dir, new_nn_file, code)
        if tr_code:
            create_file(model_dir, transformer_file, tr_code)
            if tr_name:
                prm["transform"] = tr_name
                _transform_counts["reused" if not tr_name.startswith("tr-")
                                  else "new"].add(tr_name)
        with open(model_dir / hp_file, "w+") as f:
            json.dump(prm, f)
        create_file(model_dir, f"original_{origdf['nn']}.py", baseline_code)
        print(f"[edit] {model_dir.name}: {how}"
              + (f", hp {hp}" if hp else "")
              + f", {len(code.splitlines())} lines",
              flush=True)
        written += 1
    if not written:
        print(f"[ERROR] No code generated for model B{idx}", flush=True)
    return written


def _seed_pool_size(test_nn):
    """Number of seed rows to draw so test_nn prompts survive the length filter."""
    if not test_nn or test_nn < 1:
        return test_nn
    return max(int(test_nn * SEED_POOL_FACTOR), test_nn + SEED_POOL_MIN_SURPLUS)


def _prompt_token_length(chat_bot, prompt_text):
    """Rendered prompt length in tokens, or None when it cannot be measured."""
    try:
        if hasattr(chat_bot, "token_length"):          # vLLM sleep-mode engine
            return chat_bot.token_length(prompt_text)
        rendered = chat_bot._templated_from_messages(
            chat_bot._build_messages(prompt_text))
        return len(chat_bot.tokenizer(
            rendered, add_special_tokens=False)["input_ids"])
    except Exception:
        return None


def _prompt_budget(chat_bot, in_len, max_new_tokens, unsloth_max_input_length):
    """New-token budget left after an in_len-token prompt, or None to drop it.

    A budget below MIN_ANSWER_TOKENS counts as no budget at all. The underlying
    floor is 256 tokens, which is enough to start a reply but nowhere near
    enough to finish one, so prompts that squeezed past it were answered with a
    truncated block that extraction then rejected -- the epoch paid full
    generation cost for a model it could never evaluate.
    """
    if in_len is None:
        return max_new_tokens
    if unsloth_max_input_length and in_len > unsloth_max_input_length:
        return None
    try:
        if hasattr(chat_bot, "output_budget"):         # vLLM sleep-mode engine
            budget = chat_bot.output_budget(in_len, max_new_tokens)
        else:
            from ab.gpt.util.llm.Chatbot import _context_limit, _output_budget
            ctx = _context_limit(chat_bot.model, chat_bot.tokenizer)
            budget = _output_budget(ctx, in_len, max_new_tokens)
    except Exception:
        return max_new_tokens
    if budget is not None and budget < MIN_ANSWER_TOKENS:
        return None
    return budget


def _select_fitting_prompts(prompts, chat_bot, test_nn, max_new_tokens,
                            unsloth_max_input_length):
    """Take prompts in order until test_nn of them fit the context window.

    Returns ``(accepted, dropped)``. Dropping happens here, before anything is
    sent and before a model directory is created, so an oversized prompt leaves
    no empty full_output.txt behind and does not consume one of the test_nn
    slots -- the next candidate does.
    """
    accepted, dropped = [], 0
    wanted = test_nn if (test_nn and test_nn > 0) else len(prompts)
    for prompt_data in prompts:
        if len(accepted) >= wanted:
            break
        in_len = _prompt_token_length(chat_bot, prompt_data[1])
        if _prompt_budget(chat_bot, in_len, max_new_tokens,
                          unsloth_max_input_length) is None:
            dropped += 1
            print(f"[gen] Prompt dropped before sending: input_tokens={in_len} "
                  f"leaves under {MIN_ANSWER_TOKENS} tokens for the answer; "
                  f"refilling from the seed pool.", flush=True)
            continue
        accepted.append(prompt_data)
    if dropped:
        print(f"[gen] {dropped} oversized prompt(s) skipped; "
              f"{len(accepted)}/{wanted} prompts accepted for generation.",
              flush=True)
    if len(accepted) < wanted:
        print(f"[WARNING] Only {len(accepted)} of {wanted} prompts fit the context "
              f"window; the seed pool is exhausted. Raise NNGPT_SEED_POOL_FACTOR "
              f"or lower max_new_tokens / nn_code_max_chars.", flush=True)
    return accepted, dropped


def nn_gen(
    epoch,
    out_path,
    chat_bot,
    conf_keys,
    nn_train_epochs,
    prompt_dict,
    test_nn,
    max_new_tokens,
    save_llm_output,
    nn_name_prefix,
    unsloth_max_input_length,
    prompt_batch,
    use_backbone=False,
    sft_nn_prefixes=None,
    sft_dataset=None,
):
    print("Preparing prompts for generation, this might take a while...")

    from ab.gpt.util.data.code_edit import hp_text, uses_pretrained_weights

    edit_mode = _env_on("NNGPT_EDIT_MODE")
    seen_code = _load_seen_code() if edit_mode else None
    seed_context = {}
    _ctx_path = os.environ.get("NNGPT_EDIT_SEED_CONTEXT", "").strip()
    if edit_mode and _ctx_path and Path(_ctx_path).is_file():
        try:
            seed_context = json.loads(Path(_ctx_path).read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[edit] could not read {_ctx_path}: {exc}", flush=True)
    if edit_mode:
        print(f"[edit] edit mode on; {len(seen_code)} files already produced",
              flush=True)

    use_delta = nn_name_prefix == "delta"
    if not use_delta and isinstance(prompt_dict, dict) and conf_keys:
        first_key = conf_keys[0] if isinstance(
            conf_keys, (list, tuple)) else conf_keys
        key_config = prompt_dict.get(first_key, {})
        if isinstance(key_config, dict):
            use_delta = key_config.get(
                "use_delta", False) or "delta" in str(first_key).lower()

    prompts = []
    for key in conf_keys:
        key_config = prompt_dict[key]
        system_text = "\n".join(key_config.get("system", []))
        prompt = ""
        for pr in key_config["prompt"]:
            prompt += pr + "\n"

        num_joint_nns = key_config.get("num_joint_nns", 1)
        use_join = num_joint_nns >= 2
        # Pin generation seeds to the configured LEMUR corpus so the LLM is
        # conditioned on the intended architectures. Configurable via the
        # prompt-config JSON; defaults to the shared DEFAULT_DATASET /
        # DEFAULT_NN_PREFIXES (same corpus the pipeline curates) when unspecified.
        gen_dataset = key_config.get("dataset", DEFAULT_DATASET)
        gen_nn_prefixes = tuple(key_config.get("nn_prefixes") or DEFAULT_NN_PREFIXES)
        if use_join:
            from ab.nn.util.db.Query import JoinConf
            from ab.gpt.util.data.lemur_enrichment import patch_join_nn_query, enrich_dataframe
            patch_join_nn_query()
            data = lemur.data(
                only_best_accuracy=True,
                task=key_config["task"],
                dataset=gen_dataset,
                nn_prefixes=gen_nn_prefixes,
                epoch=key_config.get("epoch"),
                sql=JoinConf(
                    num_joint_nns=num_joint_nns,
                    same_columns=tuple(key_config.get("keep_same", [])),
                    diff_columns=tuple(key_config.get("no_repeat", [])),
                    enhance_nn=key_config.get("improve", False),
                ),
            )[:test_nn]
            if key_config.get("output_type") == "classification":
                enrich_dataframe(data)
            addon_data = None
        else:
            data_kwargs = {"only_best_accuracy": True, "task": key_config["task"]}
            if use_backbone and sft_nn_prefixes:
                data_kwargs["nn_prefixes"] = sft_nn_prefixes
            if use_backbone and sft_dataset:
                data_kwargs["dataset"] = sft_dataset
            if not use_backbone:
                # Pin generation seeds to the configured corpus (default cifar-10 / ga-)
                data_kwargs["dataset"] = gen_dataset
                data_kwargs["nn_prefixes"] = gen_nn_prefixes
            gen_epoch = key_config.get("epoch")
            if gen_epoch is not None:
                data_kwargs["epoch"] = gen_epoch
            data = lemur.data(**data_kwargs)
            if data.empty or "nn" not in data.columns:
                raise ValueError(
                    "No NN seed rows matched the generation filters: "
                    f"{data_kwargs}"
                )
            # LLMatic MAP-Elites seed selection (opt-in via key_config["llmatic"]).
            # Replaces random sampling with archive-driven mutation / periodic
            # crossover. Any failure returns None -> fall back to random sampling,
            # so seed selection can never break generation.
            llmatic_seeds = None
            if key_config.get("llmatic") and not use_backbone and not use_delta:
                from ab.gpt.util.method.llmatic.seeds import build_archive, select_seed_rows, cycle_from_path
                _archive = build_archive(key_config)
                if _archive is not None:
                    llmatic_seeds = select_seed_rows(
                        _archive, data, key_config, cycle_from_path(out_path), test_nn, out_path)
            if llmatic_seeds is not None:
                data = llmatic_seeds
            else:
                from ab.gpt.util.eval.Seed import pin_generation_seeds
                data = pin_generation_seeds(
                    data, _seed_pool_size(test_nn), out_path, epoch=epoch)
            if use_backbone:
                datasets = sorted(data["dataset"].dropna().unique().tolist()) if "dataset" in data else []
                print(
                    f"[TUNE] Backbone generation seed rows={len(data)} "
                    f"datasets={datasets} nn_prefixes={sft_nn_prefixes} sft_dataset={sft_dataset}"
                )
            addon_task = key_config.get("addon_task")
            addon_data = lemur.data(
                only_best_accuracy=True, task=addon_task) if addon_task else None

        output_type = key_config.get("output_type", "code")
        nn_code_max_chars = key_config.get("nn_code_max_chars")

        for _, row in data.iterrows():
            para_dict = {}
            for it in key_config["input_list"]:
                para_dict[it["para"]] = row[it["value"]]
            if use_backbone:
                target_pattern = None
                if "nn_code" in row and isinstance(row["nn_code"], str):
                    target_pattern = SFTUtil.extract_target_pattern_from_code(row["nn_code"])
                target_pattern = target_pattern or SFTUtil.available_patterns[len(prompts) % len(SFTUtil.available_patterns)]
                para_dict["target_pattern"] = target_pattern
                para_dict["backbone_prompt"] = SFTUtil.format_backbone_prompt(
                    accuracy=para_dict.get("accuracy", row.get("accuracy", "")),
                    target_pattern=target_pattern,
                )
            if key_config.get("shrink_nn_code") and "nn_code" in para_dict and isinstance(para_dict["nn_code"], str):
                # Show the LLM only the LLR-relevant slice of the baseline
                # (train_setup/learn + headers). The delta is still applied
                # to the FULL baseline from origdf['nn_code'] below.
                from ab.gpt.util.nn.DeltaUtil import shrink_nn_code_for_prompt
                para_dict["nn_code"] = shrink_nn_code_for_prompt(para_dict["nn_code"])
            if nn_code_max_chars and "nn_code" in para_dict and isinstance(para_dict["nn_code"], str):
                para_dict["nn_code"] = para_dict["nn_code"][:nn_code_max_chars]

            if addon_data is not None and not addon_data.empty:
                available_addon = addon_data.loc[addon_data.nn != row["nn"]]
                if not available_addon.empty:
                    addon_row = available_addon.sample(n=1).iloc[0]
                    if key_config.get("addon_list"):
                        for it in key_config["addon_list"]:
                            para_dict[it["para"]] = addon_row[it["value"]]

            # LLMatic crossover supplies a fully-rendered two-parent prompt that
            # the single-row template cannot express; use it verbatim when present.
            if edit_mode:
                # The edit prompt names the accuracy to beat. It is not a
                # column on the row, so it is injected here the way the design
                # path injects its own facts.
                # Pretrained and from-scratch baselines are scored against
                # their own group's ceiling, the same split the corpus used; a
                # scratch model asked to beat a pretrained one has no reachable
                # target. Falls back to the shared ceiling when only one is set.
                _shared = os.environ.get("NNGPT_DESIGN_CEILING", "")
                _grouped = os.environ.get(
                    "NNGPT_DESIGN_CEILING_PRETRAINED"
                    if uses_pretrained_weights(str(para_dict.get("nn_code") or ""))
                    else "NNGPT_DESIGN_CEILING_SCRATCH", "")
                try:
                    _ceiling = float(_grouped or _shared or 0.0)
                except ValueError:
                    _ceiling = 0.0
                para_dict.setdefault("target", max(
                    _ceiling, float(para_dict.get("accuracy") or 0.0)))
                # The same neighbour examples the corpus was rendered with. An
                # empty string keeps the prompt valid when a baseline has none.
                # The neighbour examples, plus the best result this run has
                # produced so far. Without the latter the prompt names a
                # ceiling computed once at cycle 0 and never tells the model
                # what is actually reachable -- and the run's own best was
                # never beaten after cycle 0.
                para_dict.setdefault(
                    "examples", seed_context.get(str(row.get("nn", "")), "")
                    + _elite_block())
                # The baseline's own hyperparameters. The corpus prompt states
                # them, and without them here the model cannot tell an
                # architectural gain from a tuning one, nor aim an <hp> block.
                para_dict.setdefault("hyper", hp_text(row.get("prm")))
                # ...and its input pipeline, for the same reason: the corpus
                # prompt shows it, and a <tr> block cannot keep an entry point
                # the model was never shown.
                from ab.gpt.util.data.edit_pairs import transform_prompt_block
                from ab.gpt.util.data.edit_pairs import Record as _Rec
                para_dict.setdefault("transform", transform_prompt_block(_Rec(
                    str(row.get("nn", "")), "", 0.0, 1, {},
                    str(row.get("transform") or ""),
                    str(row.get("transform_code") or ""))))

            llmatic_prompt = row.get("__llmatic_prompt__") if hasattr(row, "get") else None
            if isinstance(llmatic_prompt, str) and llmatic_prompt:
                prompt_text = llmatic_prompt
            else:
                try:
                    prompt_text = (
                        para_dict["backbone_prompt"]
                        if use_backbone
                        else prompt.format(**para_dict)
                    )
                except KeyError as exc:
                    # A placeholder the prompt config asks for that nothing
                    # supplies. Raised from deep inside the generator this was
                    # a bare KeyError naming only the key; say which prompt and
                    # what *is* available, and drop the seed rather than
                    # killing the run.
                    print(f"[gen] prompt key {exc} is not available for seed "
                          f"{row.get('nn', '?')}; have "
                          f"{sorted(para_dict)}. Skipping this seed.", flush=True)
                    continue
            prompts.append((system_text, prompt_text, row, output_type))

    # Length filter runs once, over the whole over-provisioned pool, so the
    # epoch still produces test_nn models when some prompts are too long.
    prompts, _ = _select_fitting_prompts(
        prompts, chat_bot, test_nn, max_new_tokens, unsloth_max_input_length)

    # Best-of-N: ask each seed N times and keep whichever answer measures
    # best. The objective is the maximum of a cycle, and the maximum of N
    # samples from one prompt beats the maximum of one sample from N prompts
    # when the variance is in the sampling rather than in the seed. Off by
    # default because it multiplies the EVALUATION budget, not just the
    # generation budget: at N=2 a cycle trains twice as many models, so raise
    # NNGPT_VALIDITY_KEEP with it or the gate throws the extra ones away.
    best_of = _best_of_n()
    if best_of > 1 and prompts:
        prompts = [p for p in prompts for _ in range(best_of)]
        print(f"[gen] best-of-{best_of}: {len(prompts)} prompt(s) from "
              f"{len(prompts) // best_of} seed(s). Evaluation budget scales "
              f"with this; NNGPT_VALIDITY_KEEP is "
              f"{os.environ.get('NNGPT_VALIDITY_KEEP', 'unset')}", flush=True)

    reset_cycle_signatures()
    reset_transform_counts()
    models_dir = synth_dir(out_path)

    if use_delta:
        for idx, prompt_data in tqdm(enumerate(prompts)):
            model_dir = models_dir / f"B{idx}"
            system_text, prompt_text, origdf, output_type = prompt_data
            chat_bot.system_prompt = system_text or None

            seed = epoch * 10000 + idx
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # Length filtering already happened over the whole seed pool in
            # _select_fitting_prompts(); everything that reaches here fits.

            baseline_code = origdf.get(
                'nn_code', '') if origdf is not None else ''

            _, hp, tr, full_out = chat_bot.chat(
                prompt_text, engineer_prompt=False, max_new_tokens=max_new_tokens)

            if use_backbone:
                from ab.gpt.util.llm.SFTUtil import skeleton_code
                from ab.gpt.util.Util import extract_str
                import textwrap
                block_code = extract_str(full_out, '<block>', '</block>')
                init_code = extract_str(full_out, '<init>', '</init>')
                forward_code = extract_str(full_out, '<forward>', '</forward>')
                if block_code and init_code and forward_code:
                    code = skeleton_code
                    sig_block = "def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):"
                    code = code.replace(sig_block, textwrap.dedent(block_code))
                    sig_init = "    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:"
                    code = code.replace(sig_init, textwrap.indent(
                        textwrap.dedent(init_code), "    "))
                    sig_forward = "    def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:"
                    code = code.replace(sig_forward, textwrap.indent(
                        textwrap.dedent(forward_code), "    "))
                else:
                    code = extract_code(full_out)
                if code is None:
                    print(f'[ERROR] No code generated for model B{idx}')
                    continue  # Skip if no code is generated at all

            makedirs(model_dir, exist_ok=True)
            if save_llm_output:
                create_file(model_dir, new_out_file, full_out)

            code = None
            current_out = full_out
            current_prompt = prompt_text

            for attempt in range(_MAX_DELTA_RETRIES + 1):
                if attempt > 0:
                    _, _, _, current_out = chat_bot.chat(
                        current_prompt, engineer_prompt=False, max_new_tokens=max_new_tokens
                    )

                delta = extract_delta(current_out)
                if not delta:
                    error_msg = 'No <delta>...</delta> block found in output.'
                elif not validate_delta(delta):
                    error_msg = 'Delta format is invalid (must be unified diff with --- / +++ headers and @@ hunks).'
                else:
                    applied = apply_delta(
                        baseline_code, delta) if baseline_code else None
                    if applied:
                        code = applied
                        print(
                            f'[INFO] Applied delta for B{idx} (attempt {attempt + 1})')
                        break
                    else:
                        error_msg = 'Delta patch failed to apply to the baseline code.'

                if attempt < _MAX_DELTA_RETRIES:
                    print(
                        f'[WARNING] Delta attempt {attempt + 1} failed for B{idx}: {error_msg} Retrying with feedback...')
                    current_prompt = (
                        prompt_text
                        + f'\n\n[SYSTEM FEEDBACK - Attempt {attempt + 1} failed]: {error_msg}'
                        + '\nPlease correct the delta and output it again.'
                    )

            if code is None:
                print(
                    f'[WARNING] All delta attempts failed for B{idx}. Trying syntax repair on extracted code.')
                raw_code = extract_code(full_out)
                if raw_code:
                    repaired = repair_code(raw_code)
                    if repaired:
                        code = repaired
                        print(
                            f'[INFO] Used syntax-repaired code fallback for B{idx}')

            hp_str = extract_hyperparam(full_out)
            tr_str = extract_transform(full_out)

            # Guard against NN-model code misrouted into the <tr> transform slot.
            # The reference model in the prompt is shown in <tr>/<nn> tags and the
            # LLM sometimes emits its full model inside <tr> (or echoes the
            # reference), which extract_transform would otherwise save as tr.py and
            # the model would never be evaluated. If the transform slot holds a
            # full NN model, promote it to new_nn.py when we have no NN code yet,
            # and never persist a model as a transform.
            if tr_str and is_nn_model_code(tr_str):
                print('[ROUTING] <tr> transform slot contains a full NN model.')
                if not (code and code.strip()):
                    repaired = repair_code(tr_str)
                    code = repaired or tr_str
                    print(f'[ROUTING] Promoted misrouted NN model from <tr> to new_nn.py for B{idx}')
                tr_str = None

            try:
                print(f'Generated params: {hp_str}')
                if hp_str and hp_str.strip():
                    hp_obj = json.loads(hp_str.replace("'", '"'))
                    with open(model_dir / hp_file, 'w+') as f:
                        json.dump(hp_obj, f)
                else:
                    print('[WARNING] No hyperparameters generated, skipping hp file')
            except Exception as e:
                print(f"[WARNING] Error processing hyperparameters: {e}")

            try:
                print(f'Generated transformer:\n\n{tr_str}\n----\n')
                if tr_str and tr_str.strip():
                    create_file(model_dir, transformer_file, tr_str)
                else:
                    print('[WARNING] No transformer code generated')
            except Exception as e:
                print(f'[WARNING] Error saving transformer: {e}')

            if code and code.strip():
                create_file(model_dir, new_nn_file, code)
                print(f'[INFO] Saved code to {model_dir / new_nn_file}')
            else:
                print(f'[ERROR] No code generated for model B{idx}')
                continue

            warn_if_truncated(full_out, max_new_tokens, idx,
                              getattr(chat_bot, "tokenizer", None))
            create_file(model_dir, new_out_file, full_out)
            df_file = model_dir / 'dataframe.df'
            if origdf is None:
                if isfile(df_file):
                    os.remove(df_file)
                    print(f'[DEBUG]Removed unmatched file: {df_file}')
            else:
                create_file(
                    model_dir, f"original_{origdf['nn']}.py", origdf['nn_code'])
                origdf.to_pickle(df_file)

    else:
        pending = []
        for idx, prompt_data in tqdm(enumerate(prompts)):
            system_text, prompt_text, origdf, output_type = prompt_data
            chat_bot.system_prompt = system_text or None

            # Everything in `prompts` has already passed the context-window
            # filter, so nothing is dropped here and `pending` always holds the
            # full test_nn set.

            pending.append(
                (idx, system_text, prompt_text, origdf, output_type))

        if prompt_batch < 1:
            prompt_batch = 1
        # vLLM schedules its own continuous batch and pages the KV cache, so it
        # wants every prompt at once rather than pairs. That is also what makes
        # 4-bit inference affordable: each dequantized weight tile is reused
        # across all sequences in flight instead of being rebuilt per pair, so
        # the dequantization cost is amortised instead of paid ~8 times over.
        if hasattr(chat_bot, "llm") and pending:
            prompt_batch = len(pending)
            print(f'[INFO] vLLM continuous batching: submitting all '
                  f'{prompt_batch} prompts together')
        elif prompt_batch > 1:
            print(
                f'[INFO] Batch generation enabled: prompt_batch={prompt_batch}')

        repair_queue = []
        for start in range(0, len(pending), prompt_batch):
            batch = pending[start: start + prompt_batch]
            chat_bot.system_prompt = batch[0][1] or None
            batch_prompts = [item[2] for item in batch]

            if prompt_batch > 1 and hasattr(chat_bot, 'chat_batch'):
                batch_outputs = chat_bot.chat_batch(
                    batch_prompts, engineer_prompt=False, max_new_tokens=max_new_tokens)
            else:
                batch_outputs = [chat_bot.chat(
                    p, engineer_prompt=False, max_new_tokens=max_new_tokens) for p in batch_prompts]

            for (idx, system_text, prompt_text, origdf, output_type), output in zip(batch, batch_outputs):
                model_dir = models_dir / f"B{idx}"
                code, hp, tr, full_out = output
                if edit_mode:
                    cut = warn_if_truncated(full_out, max_new_tokens, idx,
                                            getattr(chat_bot, "tokenizer", None))
                    written = _write_edit_candidates(
                        models_dir, idx, origdf, full_out, seen_code,
                        save_llm_output)
                    # Record the verdict instead of leaving the statistics to
                    # re-derive it from the reply text. classify_failures()
                    # could not tell a truncated reply from edits that did not
                    # apply, so one run reported 71 truncations and the token
                    # budget was blamed for a matching problem.
                    if cut and not written:
                        try:
                            create_file(models_dir / f"B{idx}", "truncated.txt",
                                        f"hit the {max_new_tokens}-token budget\n")
                        except Exception:
                            pass
                    if written:
                        repair_queue.append((idx, origdf, prompt_text))
                    continue
                if use_backbone:
                    code = SFTUtil.assemble_backbone_xml_completion(full_out)
                    if code is None:
                        print(f'[ERROR] Missing backbone XML tags for model B{idx}')

                makedirs(model_dir, exist_ok=True)
                if output_type == "classification":
                    create_file(model_dir, new_out_file, full_out)
                    if origdf is not None:
                        origdf.to_pickle(model_dir / "dataframe.df")
                    continue
                if save_llm_output:
                    create_file(model_dir, new_out_file, full_out)

                # A cut-off reply is the commonest way a cycle loses a model:
                # extraction finds no closing tag and reports "no code" without
                # saying the budget ran out. Say so.
                warn_if_truncated(full_out, max_new_tokens, idx,
                                  getattr(chat_bot, "tokenizer", None))

                try:
                    print(f'Generated params: {hp}')
                    if hp and hp.strip():
                        hp = json.loads(hp.replace("'", '"'))
                        with open(model_dir / hp_file, 'w+') as f:
                            json.dump(hp, f)
                    else:
                        print(
                            '[WARNING] No hyperparameters generated, skipping hp file')
                except Exception as e:
                    print(f'[WARNING] Error processing hyperparameters: {e}')

                try:
                    print(f'Generated transformer:\n\n{tr}\n----\n')
                    if tr and tr.strip():
                        create_file(model_dir, transformer_file, tr)
                    else:
                        print('[WARNING] No transformer code generated')
                except Exception as e:
                    print(f'[WARNING] Error saving transformer: {e}')

                if code and code.strip():
                    create_file(model_dir, new_nn_file, code)
                    print(f'[INFO] Saved code to {model_dir / new_nn_file}')
                else:
                    print(f'[ERROR] No code generated for model B{idx}')
                    continue

                create_file(model_dir, new_out_file, full_out)
                df_file = model_dir / 'dataframe.df'
                if origdf is None:
                    if isfile(df_file):
                        os.remove(df_file)
                        print(f'[DEBUG]Removed unmatched file: {df_file}')
                else:
                    create_file(
                        model_dir, f"original_{origdf['nn']}.py", origdf['nn_code'])
                    origdf.to_pickle(df_file)

    if edit_mode:
        _run_repair_pass(models_dir, repair_queue, chat_bot, seen_code,
                         max_new_tokens, save_llm_output)
        _save_seen_code(seen_code)
        print(f"[edit] transforms: {len(_transform_counts['reused'])} reused "
              f"from the dataset, {len(_transform_counts['new'])} new "
              f"({', '.join(sorted(_transform_counts['new'])) or 'none'})",
              flush=True)

    # NOTE: epoch statistics used to be written here. nn_gen() runs BEFORE the
    # models are evaluated, so the accuracy it read out of cycle_results.json
    # was always the PREVIOUS epoch's -- every row was one epoch late. The
    # record is now produced at the end of _evaluate_epoch(), from this epoch's
    # own artifacts. See ab/gpt/util/data/EpochStats.py.

    print('[DEBUG] Release memory.')
    release_memory()


def trans_gen(epoch, out_path, chat_bot, conf_keys, nn_train_epochs, prompt_dict_global, test_nn, max_new_tokens, save_llm_output, nn_name_prefix):
    """
    Transform Script Generation
    """
    print('Running Transform Generation...')

    out_gen_dir = str(TRANSFORM_OUT_DIR)
    result_gen_dir = str(TRANSFORM_RES_DIR)

    prompts = []

    all_data = load_data_from_folders(
        out_gen_dir, result_gen_dir, only_best_accuracy=True)
    if len(all_data) == 0:
        print("Warning: No data loaded from folders for generation. Skipping.", flush=True)
        return

    for key in conf_keys:
        prompt_config = prompt_dict_global[key]
        prompt = ''
        for pr in prompt_config['prompt']:
            prompt += pr + '\n'

        if len(all_data) < test_nn:
            print(
                f"Warning: Requested {test_nn} samples, but only {len(all_data)} available. Using all.", flush=True)
            data_sample = all_data.sample(n=len(all_data))
        else:
            data_sample = all_data.sample(n=test_nn)

        addon_data = all_data

        for _, row in data_sample.iterrows():
            para_dict = {}
            row_dict = row.to_dict()
            for it in prompt_config['input_list']:
                para_dict[it['para']] = row_dict.get(it['value'])

            # Avoid sampling the same transform
            filtered_addon_data = addon_data.loc[addon_data.id_name !=
                                                 row['id_name']]
            if len(filtered_addon_data) > 0:
                addon_row = filtered_addon_data.sample(n=1).iloc[0].to_dict()
                if prompt_config.get('addon_list'):
                    for it in prompt_config['addon_list']:
                        para_dict[it['para']] = addon_row.get(it['value'])
                prompts.append((prompt.format(**para_dict), row))
            else:
                print(
                    f"Warning: Could not find addon data for {row['id_name']}. Skipping prompt.", flush=True)

    models_dir = synth_dir(out_path)

    for idx, prompt_data in tqdm(enumerate(prompts)):
        model_dir = models_dir / f'B{idx}'
        prompt_text, origdf = prompt_data

        code, hp, tr, full_out = chat_bot.chat(
            prompt_text, engineer_prompt=False, max_new_tokens=max_new_tokens)

        makedirs(model_dir, exist_ok=True)
        if save_llm_output:
            create_file(model_dir, new_out_file, full_out)

        if tr and tr.strip():
            print(f'Generated transformer:\n\n{tr}\n----\n')
            create_file(model_dir, transformer_file, tr)
        else:
            print(f"[ERROR] No code generated for model B{idx}")
            continue

        df_file = model_dir / 'dataframe.df'
        if origdf is None:
            if isfile(df_file):
                os.remove(df_file)
        else:
            create_file(
                model_dir, f"original_{origdf['id_name']}.py", origdf['transform_code'])
            origdf.to_pickle(df_file)

    print('[DEBUG] Release memory.')
    release_memory()


# ============================================================
# SINGLE SOURCE OF TRUTH: STEP WRAPPERS
# These are what the AGENTS call (NOT reimplementing anything)
# ============================================================

def _has_generated_nn_code(out_path) -> bool:
    """Returns True if at least one synthesized model directory B*/ contains new_nn.py."""
    models_dir = synth_dir(out_path)
    if not exists(models_dir):
        return False
    for bdir in glob.glob(str(models_dir / "B*")):
        if isfile(os.path.join(bdir, new_nn_file)):
            return True
    return False


def _has_generated_output(out_path) -> bool:
    """Returns True if at least one synthesized model directory B*/ contains full_output.txt."""
    models_dir = synth_dir(out_path)
    if not exists(models_dir):
        return False
    for bdir in glob.glob(str(models_dir / "B*")):
        if isfile(os.path.join(bdir, new_out_file)):
            return True
    return False


def _vllm_peft_warning(vllm_engine, llm_path, init_adapter):
    """Warning for vLLM generation with --peft but without NNGPT_CARRY_ADAPTER=1, else None.

    vLLM loads the base weights by name and applies an adapter from disk; an adapter from --peft that
    was merged into the in-memory model never reaches the engine. Behaviour is unchanged, only reported.
    """
    if vllm_engine is not None and llm_path and init_adapter is None:
        return ("[vLLM] WARNING: --peft without NNGPT_CARRY_ADAPTER=1: the merged adapter is not visible to "
                "the vLLM engine, which generates with the base weights at epoch 0. Set "
                "NNGPT_CARRY_ADAPTER=1 to generate with the adapter.")
    return None


def generate_step(state: AgentState) -> dict:
    epoch = state["current_epoch"]
    skip_epoch = state.get("skip_epoch", 0)
    out_path = epoch_dir(epoch)

    # If generation is skipped, there is nothing new to predict on.
    if epoch < skip_epoch:
        print(f"[INFO] Skipped generation at epoch {epoch}")
        return {"next_action": "finetune"}

    print(f"[INFO] Generation at epoch {epoch}")

    vllm_engine = state.get("vllm_engine")
    if vllm_engine is not None:
        vllm_engine.fallback = state.get("chat_bot_hf")
        vllm_engine.prepare(
            epoch_dir(epoch - 1) / state["base_model_name"] if epoch > 0 else state.get("vllm_init_adapter"),
            state.get("model"),
        )

    try:
        if state.get("trans_mode", False):
            trans_gen(
                epoch,
                out_path,
                state["chat_bot"],
                state["conf_keys"],
                state["nn_train_epochs"],
                state["prompt_dict"],
                state["test_nn"],
                state["max_new_tokens"],
                state["save_llm_output"],
                state.get("nn_name_prefix"),
            )
        else:
            nn_gen(
                epoch,
                out_path,
                state["chat_bot"],
                state["conf_keys"],
                state["nn_train_epochs"],
                state["prompt_dict"],
                state["test_nn"],
                state["max_new_tokens"],
                state["save_llm_output"],
                state.get("nn_name_prefix"),
                state.get("unsloth_max_input_length"),
                state.get("prompt_batch", 1),
                use_backbone=state.get("use_backbone",False),
                sft_nn_prefixes=state.get("sft_nn_prefixes"),
                sft_dataset=state.get("sft_dataset"),
            )

    finally:
        if vllm_engine is not None:
            vllm_engine.release(state.get("model"))

    # Classification prompts may intentionally emit labels or structured output
    # without generating a runnable new_nn.py file.
    classification_mode = state.get("classification_mode", False)
    has_output = _has_generated_output(
        out_path) if classification_mode else _has_generated_nn_code(out_path)
    if not has_output:
        print(
            f"[INFO] No code generated at epoch {epoch}, skipping evaluation")
        # Still record the epoch. Skipping evaluation used to skip the report
        # as well, so a collapsed epoch vanished from epoch_tracker.json
        # entirely instead of being visible as 0 valid out of N attempts --
        # exactly the epochs worth seeing.
        try:
            from ab.gpt.util.data.EpochStats import update_epoch_report
            update_epoch_report(synth_dir(out_path), epoch, nngpt_dir)
        except Exception as exc:
            print(f"[EPOCH STATS] Could not record skipped epoch {epoch}: {exc}")
        if state.get("final_eval"):
            return {"next_action": "end", "final_eval_done": True}
        return {"next_action": "finetune"}

    return {"next_action": "evaluate"}


def _rebuild_design_corpus():
    """Re-render the SFT corpus from LEMUR, including this run's own results.

    Without this the trainer sees the same rows every cycle: training loss goes
    to ~0.02, the model memorises its own corpus and every cycle regenerates
    the designs it has already produced. Evaluated models are written back to
    LEMUR under the corpus prefix, so rebuilding here is what turns the loop
    into a loop instead of eight passes over one dataset.

    The pinned test baselines are held out by name inside build(), so growth
    cannot leak the test set into training. Never raises: a rebuild failure
    leaves the previous corpus in place and the cycle continues on it.
    """
    edit_path = os.environ.get("NNGPT_EDIT_REBUILD", "").strip()
    args_path = edit_path or os.environ.get("NNGPT_DESIGN_REBUILD", "").strip()
    if not args_path or not Path(args_path).is_file():
        return None
    try:
        if edit_path:
            from ab.gpt.util.data import edit_pairs as builder
        else:
            from ab.gpt.util.data import design_pairs as builder
        args = json.loads(Path(args_path).read_text(encoding="utf-8"))
        lemur.data.cache_clear()
        stats = builder.build(**args)
        print(f"[corpus] rebuilt: records={stats['records']} "
              f"pairs={stats['pairs']} train={stats['train_examples']} "
              f"observed_max={stats['observed_max']}", flush=True)
        return stats
    except Exception as exc:
        print(f"[design-corpus] rebuild skipped ({type(exc).__name__}: {exc}); "
              f"continuing on the previous corpus", flush=True)
        return None


def _park_trainer_for_eval(model):
    """Take the trainer off the card for the duration of evaluation.

    Generation parks it and then restores it before _evaluate_epoch runs, so
    the evaluator started with ~7.3 GiB already resident and lost candidates to
    CUDA OOM -- silently, because an OOM looks like any other failed model.
    """
    if model is None or not _env_on("NNGPT_PARK_TRAINER_FOR_EVAL"):
        return None
    try:
        from ab.gpt.util.llm.VLLMSleep import park_trainer
        return park_trainer(model)
    except Exception as exc:
        print(f"[eval] could not park the trainer ({type(exc).__name__}: {exc}); "
              f"evaluating with it resident", flush=True)
        return None


def _unpark_trainer_after_eval(model, parked):
    if not parked:
        return
    try:
        from ab.gpt.util.llm.VLLMSleep import unpark_trainer
        unpark_trainer(model, parked)
    except Exception as exc:
        print(f"[eval] could not restore the trainer: {exc}", flush=True)


def _evaluate_epoch(
    epoch,
    out_path,
    nn_name_prefix,
    nn_train_epochs,
    trans_mode,
    classification_mode=False,
    custom_synth_dir=None,
    trainer_model=None,
):
    """
    Single source of truth for one evaluation epoch.
    Runs NNEval (trains generated NNs for nn_train_epochs and records accuracy).
    Called by both the classic for-loop and the agent evaluator node.
    Returns a dict with accuracy results that the predictor can read.
    """
    models_dir = synth_dir(out_path)
    results = {"epoch": epoch}
    parked = _park_trainer_for_eval(trainer_model)

    try:
        return _evaluate_epoch_inner(
            epoch, out_path, nn_name_prefix, nn_train_epochs, trans_mode,
            classification_mode, custom_synth_dir, models_dir, results)
    finally:
        _unpark_trainer_after_eval(trainer_model, parked)


def _evaluate_epoch_inner(epoch, out_path, nn_name_prefix, nn_train_epochs,
                          trans_mode, classification_mode, custom_synth_dir,
                          models_dir, results):
    if exists(models_dir):
        release_memory()
        # Repair generated models that almost follow the LEMUR interface (class
        # rename to Net, in_shape unpack, F import, learn method, hyperparam strip)
        # before evaluation so near-miss candidates are not lost.
        if not trans_mode:
            try:
                from ab.gpt.util.nn.PostprocessNN import postprocess_directory
                postprocess_directory(models_dir)
            except Exception as exc:
                print(f'[WARN] postprocess_nn skipped: {exc}', flush=True)

        # Static validity gate, after the repair pass so near-misses already
        # fixed are not discarded. Eval.evaluate() rejects broken code before
        # training anyway, so this is not about saving GPU time -- it is about
        # sample size: an invalid generation produces no paired observation,
        # and without a cap on the valid ones an oversampled cycle would train
        # two to three times the intended number of models.
        # check_static_validity already existed; nothing called it here.
        if not trans_mode and os.environ.get(
                "NNGPT_VALIDITY_GATE", "").strip().lower() not in ("", "0", "false", "no"):
            try:
                from ab.gpt.util.eval.validity_gate import gate_directory
                _keep = os.environ.get("NNGPT_VALIDITY_KEEP", "").strip()
                gate_directory(models_dir, keep=int(_keep) if _keep else None)
            except Exception as exc:
                print(f'[WARN] validity gate skipped: {exc}', flush=True)

        if classification_mode:
            from ab.gpt.act.tune.classification.Eval import evaluate_epoch as cls_eval

            cls_result = cls_eval(models_dir)
            results[f"epoch_{epoch + 1}_accuracy"] = cls_result["accuracy"]
        elif trans_mode:
            try:
                run_eval(epoch_num=epoch, FT_MODE=True)
                print('[DEBUG] Release_memory.')
            except Exception as e:
                print(f"Error running evaluation main(): {e}", flush=True)
            print('Folder data reload will occur next epoch.')
        else:
            eval_cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
            if eval_cuda_visible_devices:
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = eval_cuda_visible_devices
                env.setdefault("NNGPT_NNEVAL_USE_ALL_VISIBLE_GPUS", "0")
                cmd = [
                    sys.executable,
                    "-m",
                    "ab.gpt.act.eval.Eval",
                    "--nn_train_epochs",
                    str(nn_train_epochs),
                    "--only_epoch",
                    str(epoch),
                ]
                if custom_synth_dir:
                    cmd.extend(["--custom_synth_dir", str(custom_synth_dir)])
                if nn_name_prefix:
                    cmd.extend(["--nn_name_prefix", str(nn_name_prefix)])
                print(
                    f"[TUNE] Running NNEval subprocess with "
                    f"CUDA_VISIBLE_DEVICES={eval_cuda_visible_devices} "
                    f"NNGPT_NNEVAL_USE_ALL_VISIBLE_GPUS={env.get('NNGPT_NNEVAL_USE_ALL_VISIBLE_GPUS')} "
                    f"custom_synth_dir={custom_synth_dir or ''}"
                )
                subprocess.run(cmd, check=True, env=env)
            else:
                NNEval.main(
                    nn_name_prefix=nn_name_prefix,
                    nn_train_epochs=nn_train_epochs,
                    only_epoch=epoch,
                    custom_synth_dir=custom_synth_dir,
                )
            print('[DEBUG] Release_memory.')
            release_memory()

            generated_count = len(glob.glob(str(models_dir / "B*" / new_nn_file)))
            artifact_count = len(glob.glob(str(models_dir / "B*" / "eval_info.json")))
            artifact_count += len(glob.glob(str(models_dir / "B*" / "error.txt")))
            if generated_count > 0 and artifact_count == 0:
                raise RuntimeError(
                    "NNEval produced no per-model artifacts after generated code was found: "
                    f"models_dir={models_dir}, generated={generated_count}. "
                    "Check custom_synth_dir and epoch_root settings."
                )

        print('Clear LEMUR query cache.')
        lemur.data.cache_clear()
        print('The cache has been cleared.')

    # Read accuracy from cycle_results.json (written by NNEval after evaluation)
    cycle_file = out_path.parent / "cycle_results.json"
    if cycle_file.is_file():
        try:
            with open(cycle_file) as f:
                cycle_data = json.load(f)
            best_acc = (
                cycle_data.get("evaluation", {}).get("best_accuracy")
                or cycle_data.get("best_accuracy")
                or cycle_data.get("accuracy")
            )
            if best_acc is not None:
                results[f"epoch_{epoch + 1}_accuracy"] = float(best_acc)
        except Exception:
            pass

    # Collect all predictor inputs from the first successful model's files.
    # Classic for-loop ignores these extra keys — agent evaluate_step passes them to state.
    if exists(models_dir):
        for bdir in sorted(glob.glob(str(models_dir / "B*"))):
            eval_info_path = os.path.join(bdir, "eval_info.json")
            df_path = os.path.join(bdir, "dataframe.df")
            nn_path = os.path.join(bdir, new_nn_file)
            tr_path = os.path.join(bdir, transformer_file)

            if not isfile(eval_info_path):
                continue
            try:
                with open(eval_info_path) as f:
                    eval_info = json.load(f)
                cli = eval_info.get("cli_args", {})
                args = eval_info.get("eval_args", {})
                # use exact DB column names so predictor can use them directly
                results["task"] = cli.get("task", "")
                results["dataset"] = cli.get("dataset", "")
                results["metric"] = cli.get("metric", "")
                results["prm"] = args if args else {}
                if isfile(nn_path):
                    with open(nn_path) as f:
                        results["nn_code"] = f.read()
                if isfile(tr_path):
                    with open(tr_path) as f:
                        results["transform_code"] = f.read()
                # fallback: read extra fields from dataframe.df
                if isfile(df_path):
                    try:
                        origdf = pd.read_pickle(df_path)
                        if not results.get("transform_code"):
                            results["transform_code"] = origdf.get(
                                "transform_code", "")
                        if not results.get("task"):
                            results["task"] = origdf.get("task", "")
                        if not results.get("dataset"):
                            results["dataset"] = origdf.get("dataset", "")
                        if not results.get("metric"):
                            results["metric"] = origdf.get("metric", "")
                        if not results.get("prm"):
                            results["prm"] = origdf.get("prm", {})
                        # nn name (used by predictor to look up DB IDs)
                        results["nn"] = origdf.get("nn", "")
                    except Exception:
                        pass
                break  # first successful model is enough
            except Exception:
                continue

    # Stage 2: re-measure the best few candidates before the cycle's headline
    # number is taken from a single training run. Does nothing unless
    # NNGPT_STAGE2_TOPK is set. Runs before the statistics so the summary can be
    # carried in them.
    # The run's own champions, recorded before the statistics so the next
    # cycle's prompts can name the number to beat.
    if exists(models_dir) and not trans_mode:
        try:
            from ab.gpt.util.data import elite
            elite.update(models_dir, nngpt_dir, epoch=epoch)
        except Exception as exc:
            print(f'[elite] archive not updated: {exc}', flush=True)

    stage2_summary = None
    if exists(models_dir) and not trans_mode:
        try:
            from ab.gpt.util.eval import stage2
            stage2_summary = stage2.run(
                models_dir, epoch, nn_name_prefix=nn_name_prefix,
                nn_train_epochs=nn_train_epochs)
        except Exception as exc:
            print(f'[WARN] stage 2 re-measurement skipped: {exc}', flush=True)

    # Epoch statistics: valid-generation rate with a Wilson interval, accuracy
    # of the generated models (mean with 95% CI, and max) and the accuracy the
    # seed models had before modification. Written after evaluation so the row
    # describes THIS epoch, then the plot is refreshed. Never raises.
    if exists(models_dir):
        from ab.gpt.util.data.EpochStats import update_epoch_report
        epoch_stats = update_epoch_report(models_dir, epoch, nngpt_dir)
        if epoch_stats:
            if stage2_summary:
                epoch_stats["stage2"] = stage2_summary
            results["epoch_stats"] = epoch_stats
        if stage2_summary:
            results["stage2"] = stage2_summary

    return results


def evaluate_step(state: AgentState) -> dict:
    """Thin agent wrapper — all logic lives in _evaluate_epoch()."""
    epoch = state["current_epoch"]
    out_path = epoch_dir(epoch)
    print(f"[INFO] Evaluating at epoch {epoch}")

    results = _evaluate_epoch(
        epoch,
        out_path,
        state.get("nn_name_prefix"),
        state["nn_train_epochs"],
        state.get("trans_mode", False),
        state.get("classification_mode", False),
        trainer_model=state.get("model"),
    )

    updates = {}

    # Count actual evaluations that produced results (not epoch numbers)
    # epoch_1_accuracy = first real evaluation, epoch_2_accuracy = second, epoch_3_accuracy = third
    # This works correctly with skip_epoch — epoch 0 skips generation so produces no accuracy
    has_epoch1_in_state = state.get("epoch_1_accuracy") is not None
    has_epoch2_in_state = state.get("epoch_2_accuracy") is not None
    has_epoch3_in_state = state.get("epoch_3_accuracy") is not None

    acc_key = f"epoch_{epoch + 1}_accuracy"
    best_acc = results.get(acc_key)

    if best_acc is not None:
        if not has_epoch1_in_state:
            updates["epoch_1_accuracy"] = best_acc
        elif not has_epoch2_in_state:
            updates["epoch_2_accuracy"] = best_acc
        elif not has_epoch3_in_state:
            updates["epoch_3_accuracy"] = best_acc

    # Pass all predictor inputs to state — names match exact DB column names
    for field in ["nn_code", "prm", "task", "dataset", "metric", "transform_code", "nn"]:
        if field in results:
            updates[field] = results[field]

    # Route to predictor only if enabled AND we have 3 epochs of results
    use_predictor = state.get("use_predictor", False)
    has_epoch1 = has_epoch1_in_state or "epoch_1_accuracy" in updates
    has_epoch2 = has_epoch2_in_state or "epoch_2_accuracy" in updates
    has_epoch3 = has_epoch3_in_state or "epoch_3_accuracy" in updates

    if state.get("final_eval"):
        # Closing pass: the adapter has now been validated and the epoch
        # finalised, so the run stops here instead of fine-tuning again.
        updates["next_action"] = "end"
        updates["final_eval_done"] = True
    elif use_predictor and has_epoch1 and has_epoch2 and has_epoch3:
        updates["next_action"] = "predict"
    else:
        updates["next_action"] = "finetune"

    return updates


def _completions_only_default(use_backbone):
    """Whether SFT masks the prompt, from NNGPT_TRAIN_ON_COMPLETIONS_ONLY.

    Unset keeps the historical behaviour (``use_backbone``). Routed through the
    environment rather than a tune() keyword because ab/gpt/act/tune/Tune.py
    dispatches to tune() implementations in Tune, Tune_Onnx and Tune_prun that
    do not share a signature, so a new keyword would raise TypeError for
    whichever backend was not updated. That module already routes seed pinning
    this way and documents the reason.
    """
    raw = os.environ.get("NNGPT_TRAIN_ON_COMPLETIONS_ONLY", "").strip().lower()
    if raw in ("", "auto"):
        return use_backbone
    return raw not in ("0", "false", "no", "off")


def _finetune_epoch(
    epoch, out_path, model, tokenizer, model_loader, lora_tuner,
    context_length, use_unsloth, unsloth_max_input_length,
    train_config_path, only_best_accuracy, max_prompts,
    max_new_tokens, base_model_name, trans_mode,
    temperature=1.0, top_k=50, top_p=0.9,
    resume_trainer_checkpoint=None,
    use_backbone=False,
    sft_nn_prefixes=None,
    sft_dataset=None,
    data_dir=None,
    budget_tokens=None,
    suppress_thinking=False,
    reset_previous_results=True,
):
    """
    Single source of truth for one finetune epoch.
    Called by both the classic for-loop and the agent finetuner node.
    Returns (model, chat_bot) with the newly fine-tuned model.
    """
    # Which rows this cycle trains on. Without it every cycle samples the same
    # subset and a corpus larger than max_prompts is mostly unused.
    os.environ["NNGPT_SFT_SAMPLE_SEED"] = str(epoch)

    if trans_mode:
        data_processor = TransformGenPrompt(
            context_length if context_length else model_loader.get_max_length(),
            tokenizer,
            train_config_path,
            TRANSFORM_OUT_DIR,
            TRANSFORM_RES_DIR,
        )
    elif use_backbone:
        from ab.gpt.util.prompt.SFTGenPrompt import SFTGenPrompt
        data_processor = SFTGenPrompt(
            context_length if context_length else model_loader.get_max_length(),
            tokenizer,
            nn_prefixes=sft_nn_prefixes,
            dataset=sft_dataset,
        )
    else:
        length = (
            unsloth_max_input_length if (use_unsloth and unsloth_max_input_length)
            else context_length if context_length
            else model_loader.get_max_length()
        )
        data_processor = NNGenPrompt(length, tokenizer, train_config_path, data_dir=data_dir)

    # Seed comes from the prompt config when set; otherwise the Trainer's own
    # default. hasattr guards data processors that don't define get_config_seed().
    config_seed = data_processor.get_config_seed() if hasattr(data_processor, 'get_config_seed') else None
    seed = config_seed if config_seed is not None else lora_tuner.training_args.seed
    dataset = data_processor.get_dataset(
        only_best_accuracy,
        seed=seed,
        max_prompts=max_prompts,
        max_new_tokens=max_new_tokens,
    )

    print("Dataset length:", len(dataset))
    model.train()
    model = lora_tuner.train(
        dataset,
        tokenizer,
        out_path / base_model_name,
        # Loss over the prompt as well as the completion is the historical
        # default, and for an improvement task it is actively harmful: the
        # baseline file appears verbatim in the prompt and again, nearly
        # unchanged, in the target, so most of the gradient goes into
        # reproducing text the model was handed rather than into the few
        # decisions that differ.
        train_on_completions_only=_completions_only_default(use_backbone),
        resume_from_checkpoint=resume_trainer_checkpoint,
        checkpoint_label="trainer",
    )

    del dataset
    release_memory()
    _report_host_memory("after fine-tuning")

    chat_bot = ChatBot(
        model, tokenizer, temperature=temperature, top_k=top_k, top_p=top_p,
        budget_tokens=budget_tokens, suppress_thinking=suppress_thinking)
    return model, chat_bot


def _report_host_memory(when):
    try:
        import resource
        from ab.gpt.util.eval.smoke import _host_available_gib
        fields = {}
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                name = line.split(":")[0]
                if name in ("Dirty", "Writeback", "Cached", "MemFree"):
                    fields[name] = int(line.split()[1]) / 2**20
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
        print(f"[mem] {when}: this process peak RSS {rss:.1f} GiB, host "
              f"available {_host_available_gib():.1f} GiB, free "
              f"{fields.get('MemFree', 0):.1f} GiB, cached "
              f"{fields.get('Cached', 0):.1f} GiB, dirty "
              f"{fields.get('Dirty', 0):.2f} GiB, writeback "
              f"{fields.get('Writeback', 0):.2f} GiB", flush=True)
    except Exception as exc:
        print(f"[mem] {when}: unavailable ({exc})", flush=True)


def finetune_step(state: AgentState) -> dict:
    """Thin agent wrapper — all logic lives in _finetune_epoch()."""
    epoch = state["current_epoch"]
    out_path = epoch_dir(epoch)
    print(f"[DEBUG] Perform finetune at epoch {epoch}")
    _report_host_memory("before fine-tuning")

    _prev_bot = state.get("chat_bot")
    _rebuild_design_corpus()
    vllm_engine = state.get("vllm_engine")
    if vllm_engine is not None:
        vllm_engine.ensure_asleep()
    model, chat_bot = _finetune_epoch(
        epoch, out_path,
        state["model"], state["tokenizer"], state["model_loader"], state["lora_tuner"],
        state.get("context_length"), state.get("use_unsloth", False),
        state.get("unsloth_max_input_length"),
        state["train_config_path"], state["only_best_accuracy"],
        state.get("max_prompts"), state["max_new_tokens"],
        state["base_model_name"], state.get("trans_mode", False),
        state.get("temperature", 1.0), state.get(
            "top_k", 50), state.get("top_p", 0.9),
        state.get("trainer_resume_checkpoint"),
        state.get("use_backbone", False),
        state.get("sft_nn_prefixes"),
        state.get("sft_dataset"),
        state.get("data_dir"),
        budget_tokens=getattr(_prev_bot, "budget_tokens", None),
        suppress_thinking=getattr(_prev_bot, "suppress_thinking", False),
    )

    if vllm_engine is not None:
        vllm_engine.fallback = chat_bot

    return {
        "model": model,
        "chat_bot": vllm_engine if vllm_engine is not None else chat_bot,
        "chat_bot_hf": chat_bot,
        "current_epoch": epoch + 1,
        "next_action": "generate",
        "trainer_resume_checkpoint": None,
    }


# ============================================================
# MAIN: tune()
# ============================================================


def _resolve_tune_resume_trainer_checkpoint(initial_adapter_path) -> Optional[str]:
    # Classic mode and agent mode both consume the same one-shot trainer resume path.
    resume_spec = TrainingRuntime.resolve_resume_spec(
        trainer_env="NNGPT_TRAIN_RESUME_TRAINER_CHECKPOINT",
        initial_adapter_active=bool(initial_adapter_path),
        initial_adapter_label="llm_path/--peft",
    )
    if resume_spec.trainer_checkpoint is None:
        return None
    return str(resume_spec.trainer_checkpoint)


def _config_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean config value: {value}")


def tune(
    test_nn,
    nn_train_epochs,
    skip_epoch,
    llm_path,
    llm_tune_conf,
    nn_gen_conf,
    conf_keys,
    llm_conf,
    training_args,
    peft_config,
    max_prompts=None,
    save_llm_output=True,
    max_new_tokens=16 * 1024,
    nn_name_prefix=None,
    temperature=1.0,
    top_k=50,
    top_p=0.9,
    test_metric=None,
    onnx_run=False,
    trans_mode=False,
    prompt_batch=1,
    use_agents=False,
    use_predictor=False,
    use_unsloth=False,
    enable_merge=False,
    classification_mode=False,
    use_backbone=False,
    sft_nn_prefixes=None,
    sft_dataset=None,
    num_cycles=None,
    context_length=None,
    max_input_length=None,
    only_best_accuracy=False,
    load_in_4bit=None,
    epoch_root=None,
    data_dir=None,
    llmatic=None,
    budget_tokens=None,
    skip_lm_finetune=False,
    suppress_thinking=False,
    dataset=None,
    nn_prefixes=None,
    reset_previous_results=True
):
    if not isinstance(conf_keys, (list, tuple)):
        conf_keys = (conf_keys,)

    with open(conf_llm_dir / llm_conf) as f:
        config = json.load(f)
    assert isinstance(config, dict)

    base_model_name = config["base_model_name"]
    merged_candidate = nngpt_upload / Path(base_model_name).name

    if merged_candidate.exists():
        print(f"[EVOLUTION] Using merged model: {merged_candidate}")
        base_model_name = str(merged_candidate)
    else:
        print(f"[EVOLUTION] Using base model from config: {base_model_name}")

    llm_tune_epochs = int(num_cycles) if num_cycles is not None else 100
    if context_length is None:
        context_length = config.get("default_context_length")
    unsloth_max_input_length = max_input_length
    unsloth_load_in_4bit = _config_bool(
        load_in_4bit, _config_bool(config.get("load_in_4bit"), True)
    )
    if "force_direct_generate" in config:
        os.environ["NNGPT_FORCE_DIRECT_GENERATE"] = (
            "1" if _config_bool(config.get("force_direct_generate"), False) else "0"
        )
    use_deepspeed = False
    chat_template_path = config.get("chat_template_path")
    # budget_tokens (from --max_output_tokens) must win over the config value
    if budget_tokens is not None:
        max_new_tokens = budget_tokens
        print(f"[BUDGET] max_new_tokens overridden by --max_output_tokens: {max_new_tokens}")
    access_token = None

    print(
        f'[DEBUG]Argument Information:\nSkip generation until Epoch: {skip_epoch}\nPath to saved LoRA Layers: {llm_path}')

    train_config_path = conf_train_dir / llm_tune_conf
    epoch_root_path = Path(epoch_root).expanduser() if epoch_root else epoch_dir()
    print(f"[EVOLUTION] Epoch root: {epoch_root_path}")

    def run_epoch_dir(*parts):
        out = epoch_root_path
        for part in parts:
            out = out / f"A{part}"
        return out

    with open(conf_test_dir / nn_gen_conf) as prompt_file:
        prompt_dict = json.load(prompt_file)
    assert isinstance(prompt_dict, dict)

    # Enable LLMatic MAP-Elites seed selection by injecting the config into each
    # generation key. nn_gen reads key_config["llmatic"]; see ab/gpt/util/method/llmatic/.
    if llmatic:
        for _key in conf_keys:
            if isinstance(prompt_dict.get(_key), dict):
                prompt_dict[_key]["llmatic"] = dict(llmatic)

    # Seed generation from the given corpus unless the prompt key sets its own;
    # nn_gen reads key_config["dataset"] / ["nn_prefixes"].
    for _key in conf_keys:
        if isinstance(prompt_dict.get(_key), dict):
            if dataset:
                prompt_dict[_key].setdefault("dataset", dataset)
            if nn_prefixes:
                prompt_dict[_key].setdefault("nn_prefixes", list(nn_prefixes))

    from ab.gpt.util.llm.LLM import LLM

    model_loader = LLM(
        base_model_name,
        quantization_config_4bit if unsloth_load_in_4bit else None,
        access_token=access_token,
        use_deepspeed=use_deepspeed,
        context_length=context_length,
        training_args=training_args,
        use_unsloth=use_unsloth,
        load_in_4bit=unsloth_load_in_4bit,
        chat_template_path=chat_template_path,
    )

    model = model_loader.get_model()
    tokenizer = model_loader.get_tokenizer()
    trainer_resume_checkpoint = _resolve_tune_resume_trainer_checkpoint(
        llm_path)

    init_adapter = None
    if llm_path:
        print(f'Load saved LoRA layer from path: {llm_path}')
        if _env_on("NNGPT_CARRY_ADAPTER"):
            # Merging into the 4-bit base re-quantizes the weights and erases the
            # adapter's update. Keep the base frozen; LoRA() reloads this
            # adapter as trainable and continues training it.
            if not Path(llm_path, "adapter_config.json").exists():
                raise FileNotFoundError(f"NNGPT_CARRY_ADAPTER: no adapter_config.json in {llm_path}")
            init_adapter = llm_path
        else:
            model = PeftModel.from_pretrained(model, llm_path, is_trainable=True)
            model = model.merge_and_unload()

    if use_deepspeed:
        deepspeed.initialize(model=model, config_params=ds_conf)

    lora_tuner = LoRA(
        model,
        tokenizer,
        training_args=training_args,
        access_token=access_token,
        peft_config=peft_config,
        use_unsloth=use_unsloth,
        init_adapter=init_adapter,
    )

    print('Using Max Length:', model_loader.get_max_length())

    if os.environ.get("NNGPT_CARRY_PROBE"):
        # Loss of the model exactly as this cycle will generate with it.
        _probe_len = (unsloth_max_input_length if (use_unsloth and unsloth_max_input_length)
                      else context_length if context_length else model_loader.get_max_length())
        _carry_probe(model, tokenizer, _probe_len, epoch_root_path.parent)

    chat_bot = ChatBot(
        model, tokenizer, temperature=temperature, top_k=top_k, top_p=top_p,
        budget_tokens=budget_tokens, suppress_thinking=suppress_thinking)

    # vLLM sleep-mode generation backend (NNGPT_VLLM_SLEEP=1 / --use_vllm).
    from ab.gpt.util.llm.VLLMSleep import sleep_mode_enabled, VLLMGeneration
    _gen_len = (unsloth_max_input_length if (use_unsloth and unsloth_max_input_length)
                else context_length if context_length else model_loader.get_max_length())
    vllm_engine = VLLMGeneration(
        base_model_name, tokenizer, _gen_len,
        temperature=temperature, top_k=top_k, top_p=top_p,
    ) if sleep_mode_enabled() else None
    _hf_chat_bot = chat_bot
    _warning = _vllm_peft_warning(vllm_engine, llm_path, init_adapter)
    if _warning:
        print(_warning, flush=True)
    if vllm_engine is not None:
        vllm_engine.fallback = _hf_chat_bot
        chat_bot = vllm_engine

    state = AgentState(
        experiment_id=nn_name_prefix or "exp_default",
        nn_name_prefix=nn_name_prefix,
        current_epoch=0,
        llm_tune_epochs=llm_tune_epochs,
        skip_epoch=skip_epoch,
        next_action="generate",
        status="pending",

        model=model,
        tokenizer=tokenizer,
        model_loader=model_loader,
        lora_tuner=lora_tuner,
        chat_bot=chat_bot,
        vllm_engine=vllm_engine,
        vllm_init_adapter=str(init_adapter) if init_adapter else None,
        chat_bot_hf=_hf_chat_bot,

        prompt_dict=prompt_dict,
        conf_keys=conf_keys,
        test_nn=test_nn,
        nn_train_epochs=nn_train_epochs,
        max_new_tokens=max_new_tokens,
        save_llm_output=save_llm_output,
        prompt_batch=prompt_batch,

        context_length=context_length,
        use_unsloth=use_unsloth,
        unsloth_max_input_length=unsloth_max_input_length,
        train_config_path=train_config_path,
        only_best_accuracy=only_best_accuracy,
        base_model_name=base_model_name,
        trans_mode=trans_mode,
        max_prompts=max_prompts,

        temperature=temperature,
        top_k=top_k,
        top_p=top_p,

        use_predictor=use_predictor,
        use_backbone=use_backbone,
        sft_nn_prefixes=sft_nn_prefixes,
        sft_dataset=sft_dataset,
        # Without this the agent path silently ignores an on-disk SFT corpus.
        # finetune_step() rebuilds the _finetune_epoch() call from the state,
        # and anything missing there falls back to the parameter default. The
        # classic loop passes data_dir explicitly, so the two paths disagreed
        # and only the agent one -- which is the default -- dropped it, falling
        # back to a LEMUR query against the train prompt config.
        data_dir=data_dir,
        trainer_resume_checkpoint=trainer_resume_checkpoint,
        enable_merge=enable_merge,
        classification_mode=classification_mode,
    )

    # Statistics files from a previous run are cleared by the caller
    # (ab/gpt/act/tune/Tune.py) before this function is entered; see
    # EpochStats.reset_run_artifacts. Doing it there keeps every tune()
    # backend -- Tune, Tune_Onnx, Tune_prun -- on the same signature.
    shutil.rmtree(epoch_root_path, ignore_errors=True)

    if use_agents:
        from ab.gpt.act.agents.run_agent import run_agent_controller
        return run_agent_controller(state)

    for epoch in range(llm_tune_epochs):
        print(f'[INFO]Start Epoch {epoch}')
        out_path = run_epoch_dir(epoch)
        if epoch < skip_epoch:
            print(f'Skipped generation at epoch {epoch}')
        else:
            if vllm_engine is not None:
                # Epoch 0 generates with the carried-over adapter (NNGPT_CARRY_ADAPTER), if any;
                # later epochs with the adapter trained in the previous epoch.
                vllm_engine.prepare(
                    run_epoch_dir(epoch - 1) / base_model_name if epoch > 0
                    else (str(init_adapter) if init_adapter else None),
                    model)
            if trans_mode:
                trans_gen(epoch, out_path, chat_bot, conf_keys, nn_train_epochs,
                          prompt_dict, test_nn, max_new_tokens, save_llm_output, nn_name_prefix)
            else:
                nn_gen(epoch, out_path, chat_bot, conf_keys, nn_train_epochs, prompt_dict, test_nn, max_new_tokens, save_llm_output, nn_name_prefix, unsloth_max_input_length, prompt_batch, use_backbone=use_backbone, sft_nn_prefixes=sft_nn_prefixes, sft_dataset=sft_dataset)

            if vllm_engine is not None:
                vllm_engine.release(model)

            _evaluate_epoch(
                epoch,
                out_path,
                nn_name_prefix,
                nn_train_epochs,
                trans_mode,
                classification_mode,
                custom_synth_dir=synth_dir(out_path),
                trainer_model=model,
            )

        if skip_lm_finetune:
            print(f'[INFO] skip_lm_finetune=True — skipping LLM LoRA update at epoch {epoch}.')
        else:
            print(f'[DEBUG]Perform finetune at epoch {epoch}.')
            _rebuild_design_corpus()
            if vllm_engine is not None:
                vllm_engine.ensure_asleep()
            model, chat_bot = _finetune_epoch(
                epoch, out_path, model, tokenizer, model_loader, lora_tuner,
                context_length, use_unsloth, unsloth_max_input_length,
                train_config_path, only_best_accuracy, max_prompts,
                max_new_tokens, base_model_name, trans_mode,
                temperature, top_k, top_p,
                trainer_resume_checkpoint,
                use_backbone=use_backbone,
                sft_nn_prefixes=sft_nn_prefixes,
                sft_dataset=sft_dataset,
                data_dir=data_dir,
                budget_tokens=budget_tokens,
                suppress_thinking=suppress_thinking,
            )
            trainer_resume_checkpoint = None
            if vllm_engine is not None:
                vllm_engine.fallback = chat_bot
                chat_bot = vllm_engine

        # One cycle is the natural place to give host memory back: generation
        # has torn its engine down, evaluation's workers have exited and the
        # trainer is finished with this cycle's corpus and optimizer. What the
        # process still holds at this point it will hold for the rest of the
        # run, so anything retained here accumulates cycle on cycle -- which is
        # what ends a long run on a machine that had memory to spare at the
        # start. The logged figures say whether the run can keep going: a
        # resident set that comes back to roughly the same value each cycle can
        # continue indefinitely, one that climbs names the cycle where
        # something started being kept.
        release_host_memory(label=f"cycle {epoch}")

    # Closing validation pass for the classic loop, mirroring the agent path:
    # the loop fine-tunes after evaluating, so without this the final adapter
    # is never generated from, never validated and never recorded.
    # NNGPT_SKIP_CLOSING_EVAL=1 skips it (the iterative pipeline evaluates the
    # adapter in the next cycle's generation anyway).
    skip_closing = _env_on("NNGPT_SKIP_CLOSING_EVAL")
    if skip_closing and not skip_lm_finetune and llm_tune_epochs > 0:
        print(f'[INFO] Closing validation pass skipped (NNGPT_SKIP_CLOSING_EVAL=1)')
    if not skip_lm_finetune and llm_tune_epochs > 0 and not skip_closing:
        final_epoch = llm_tune_epochs
        print(f'[INFO] Closing validation pass for the final adapter (epoch {final_epoch})')
        out_path = run_epoch_dir(final_epoch)
        if vllm_engine is not None:
            vllm_engine.prepare(
                run_epoch_dir(final_epoch - 1) / base_model_name, model)
        if trans_mode:
            trans_gen(final_epoch, out_path, chat_bot, conf_keys, nn_train_epochs,
                      prompt_dict, test_nn, max_new_tokens, save_llm_output, nn_name_prefix)
        else:
            nn_gen(final_epoch, out_path, chat_bot, conf_keys, nn_train_epochs, prompt_dict,
                   test_nn, max_new_tokens, save_llm_output, nn_name_prefix,
                   unsloth_max_input_length, prompt_batch, use_backbone=use_backbone,
                   sft_nn_prefixes=sft_nn_prefixes, sft_dataset=sft_dataset)
        if vllm_engine is not None:
            vllm_engine.release(model)

        _evaluate_epoch(
            final_epoch,
            out_path,
            nn_name_prefix,
            nn_train_epochs,
            trans_mode,
            classification_mode,
            custom_synth_dir=synth_dir(out_path),
            trainer_model=model,
        )

    if vllm_engine is not None:
        vllm_engine.shutdown()
