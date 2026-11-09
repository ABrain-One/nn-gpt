"""Mutate a real LEMUR model *source string* with the brute AST operators.

`mutate_source(name, source)` applies ONE random mutation (skip / activation /
kernel_size / dimension / scale / layer_type / stride, per config weights) to a
real seed's source and returns the mutated source string, or None if no valid
mutation was produced. This is the GA mutation operator over *actual seed
architectures* (as opposed to the parametric template in denoise_template.py).

Adapted from mutator.orchestrator.run_single_mutation, trimmed to: plan -> apply
-> emit -> validate-emitted, and returning the source instead of writing a .py /
touching the global dedup trackers (the GA does its own dedup via a fitness
cache). Must run with the mutator package importable and CWD under nn-dataset.
"""
import ast
import collections
import os
import sys

import torch
import torch.nn as nn

# npu_gate lives beside this file; the GA runs with a pod-local LEMUR root on sys.path but not
# necessarily this directory, so make the import work either way. It pulls in stdlib only.
try:
    from npu_gate import unconvertible
except ImportError:  # pragma: no cover - path shape depends on the launcher
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from npu_gate import unconvertible

from mutator import config  # noqa: F401  (imported for weight side-effects/env overrides)
from mutator.planning import ModelPlanner
from mutator.execution.code_mutator import CodeMutator
from mutator.utils.source_tracer import ModuleSourceTracer
from mutator.lemur_io import load_lemur_model, cleanup_temp_module, get_dataset_params

_DENOISE = get_dataset_params("denoise")  # spatial (256,256), output_size None
_H, _W = _DENOISE["spatial_dims"]

# COUNT WHAT THE GATE THROWS AWAY. mutate_source returns None for half a dozen different reasons
# and the GA only sees "no variant", so a new rejection cause is indistinguishable from a mutation
# operator that has run out of ideas. Callers should print this at the end of a run; a rising
# count here with a falling yield means the ban is fighting the mutator, not helping it.
_REJECTED_UNCONVERTIBLE = collections.Counter()


def rejected_unconvertible():
    """{rule_ids: count} of mutations dropped for introducing an unconvertible construct."""
    return dict(_REJECTED_UNCONVERTIBLE)


def _cleanup(model):
    info = getattr(model, "_temp_module_info", None)
    if info:
        try:
            cleanup_temp_module(*info)
        except Exception:
            pass


def _schedule(code_mutator, plan, original_model):
    """Translate a plan into scheduled CodeMutator edits (copied dispatch)."""
    for full_name, details in plan.items():
        loc = details.get("source_location")
        if not loc:
            continue
        mtype = details.get("mutation_type", "dimension")
        module = None if mtype == "skip" else original_model.get_submodule(full_name)

        if mtype == "dimension":
            arg = None
            if isinstance(module, nn.Linear):
                arg = "out_features" if details.get("new_out") is not None else "in_features"
            elif isinstance(module, nn.Conv2d):
                arg = "out_channels" if details.get("new_out") is not None else "in_channels"
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                arg = "num_features"
            if details.get("symbolic") and details.get("symbolic_expression") and arg:
                code_mutator.schedule_symbolic_modification(loc, arg, details["symbolic_expression"])
            else:
                val = details.get("new_out") or details.get("new_in")
                if arg and val is not None:
                    code_mutator.schedule_modification(loc, arg, val)
        elif mtype == "activation":
            if details.get("new_activation"):
                code_mutator.schedule_activation_modification(loc, details["new_activation"])
        elif mtype == "layer_type":
            if details.get("new_layer_type"):
                code_mutator.schedule_layer_type_modification(
                    loc, details["new_layer_type"], details.get("mutation_params", {}))
        elif mtype == "kernel_size":
            if details.get("new_kernel_size"):
                code_mutator.schedule_kernel_size_modification(loc, details["new_kernel_size"])
        elif mtype == "stride":
            if details.get("new_stride"):
                code_mutator.schedule_stride_modification(loc, details["new_stride"])
        elif mtype == "scale":
            arg_name, val = details.get("arg_name"), details.get("new_value")
            if arg_name and val is not None:
                code_mutator.schedule_modification(loc, arg_name, val)
        elif mtype == "skip":
            action, input_arg = details.get("action"), details.get("input_arg")
            if action and input_arg:
                code_mutator.schedule_skip_modification(loc, action, input_arg)


def _valid_denoise_source(source):
    """Instantiate the emitted source and check output shape == input shape."""
    model = None
    try:
        model = load_lemur_model(source)
        dev = next(model.parameters()).device
        x = torch.randn(2, 3, _H, _W, device=dev)
        model.eval()
        with torch.no_grad():
            out = model(x)
        return tuple(out.shape) == tuple(x.shape)
    except Exception:
        return False
    finally:
        if model is not None:
            _cleanup(model)


def _introduces_unconvertible(parent_src, mutated_src):
    """New unconvertible-op rules the mutation ADDED, relative to the parent. [] if none.

    MONOTONIC, NOT ABSOLUTE, AND THE DIFFERENCE MATTERS. Rejecting any mutated source that trips
    a ban would also reject every mutation of a seed that ALREADY trips one -- 97 of the 1,582
    corpus models carry AdaptiveMaxPool2d -- so those lineages would return None forever. The GA
    cannot distinguish that from "no distinct variant exists": it would log `no distinct variant
    in 8 tries`, burn its whole retry budget on the seed, and quietly stop exploring it, which is
    exactly how the dnseed2 arm's mutation operator starved at generation 30.

    So the rule enforced here is only that a mutation must not make deployability WORSE. An
    already-unconvertible seed stays explorable and its descendants can still be repaired; a
    convertible seed can never be mutated into an unconvertible child.
    """
    before = {rid for rid, _, _ in unconvertible(parent_src)}
    return sorted({rid for rid, _, _ in unconvertible(mutated_src)} - before)


def mutate_source(name, source, mutation_type=None):
    """Return a mutated source string (one random op), or None on failure.

    If mutation_type is given (e.g. 'skip'), the plan's operator type is checked
    and non-matching plans are rejected (used to force skip-only mutations).

    A mutation that introduces an unconvertible construct is rejected as invalid: the child
    would train for ~50 GPU-minutes and then be undeployable, which the fitness function cannot
    see because MACs and PSNR both look fine (see _introduces_unconvertible).
    """
    original = None
    try:
        if not isinstance(source, str) or not source.strip():
            return None
        with ModuleSourceTracer(source) as tracer:
            original = load_lemur_model(source)
        source_map = tracer.create_source_map(original)

        planner = ModelPlanner(original, source_map=source_map,
                               search_depth=config.PRODUCER_SEARCH_DEPTH,
                               source_code=source)
        plan = planner.plan_random_mutation()
        if not plan:
            return None
        if mutation_type is not None:
            types = {d.get("mutation_type") for d in plan.values()}
            if mutation_type not in types:
                return None

        planner.apply_plan()  # in-memory validation (raises on bad plan)

        cm = CodeMutator(source)
        _schedule(cm, plan, original)
        mutated = cm.get_modified_code()

        if not isinstance(mutated, str) or mutated == source:
            return None
        try:
            ast.parse(mutated)
        except SyntaxError:
            return None
        # Cheap AST check before the expensive build+forward validation below.
        added = _introduces_unconvertible(source, mutated)
        if added:
            _REJECTED_UNCONVERTIBLE[",".join(added)] += 1
            return None
        if not _valid_denoise_source(mutated):
            return None
        return mutated
    except Exception:
        return None
    finally:
        if original is not None:
            _cleanup(original)
