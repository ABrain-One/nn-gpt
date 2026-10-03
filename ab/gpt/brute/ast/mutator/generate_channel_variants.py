#!/usr/bin/env python3
"""
Generate valid channel-mutated variants of a local model file using the
brute/ast mutation system, in either of its two channel modes:

  bracket  Channel-configuration mode. Mutates one entry of a channel-config
           container inside the Net class, e.g.
               channels = [64, 128, 256, 512]  ->  [64, 192, 256, 512]
               init_block_channels = 64        ->  96
           The value flows through the builder loop to exactly one stage, so
           every consumer (residual downsample, batch norm, classifier) stays
           consistent. This is the mode that works for AirNet.

  direct   Direct individual-channel mode. Mutates the channel literal at the
           layer instantiation site (nn.Conv2d / nn.Linear). This is the
           original brute/ast dimension mutation and is what works for
           flattened models such as AlexNet.

Every candidate is built, forward/backward-checked, and identified with the
pipeline's canonical uuid4 (LEMUR architecture certificate). Output files are
named  ast-<mutation_type>-<model>-<checksum>.py .

Usage (run from the ast directory or anywhere):
    python generate_channel_variants.py --model /path/AirNet.py --name AirNet \
        --mode bracket --count 1112 --out ./generated
"""

import argparse
import importlib.util
import json
import os
import platform
import re
import resource
import shutil
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
AST_DIR = os.path.dirname(HERE)
if AST_DIR not in sys.path:
    sys.path.insert(0, AST_DIR)

import torch
import torch.nn as nn

from mutator import config
from mutator.utils.source_tracer import ModuleSourceTracer
from ab.nn.util.Util import uuid4
from mutator.planning import ModelPlanner
from mutator.execution.code_mutator import CodeMutator


def _configure_mode(mode: str) -> None:
    weights = {k: 0.0 for k in config.MUTATION_TYPE_WEIGHTS}
    if mode == "bracket":
        weights["channel_config"] = 1.0
    elif mode == "direct":
        weights["dimension"] = 1.0
    else:
        raise SystemExit(f"Unknown mode: {mode!r} (expected 'bracket' or 'direct')")
    config.MUTATION_TYPE_WEIGHTS = weights


def _load_net(source_path: str, tag: str, h: int, w: int, classes: int):
    spec = importlib.util.spec_from_file_location(tag, source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Net((2, 3, h, w), (classes,), {"lr": 0.01, "momentum": 0.9}, torch.device("cpu"))


def _instantiate_from_code(code: str, source_path: str, h: int, w: int, classes: int):
    tmpdir = tempfile.mkdtemp(prefix="ast_channel_")
    path = os.path.join(tmpdir, "candidate.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(code)
    try:
        spec = importlib.util.spec_from_file_location("ast_channel_candidate", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Net((2, 3, h, w), (classes,), {"lr": 0.01, "momentum": 0.9}, torch.device("cpu"))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _schedule_plan(code_mutator: CodeMutator, plan: dict, original_model) -> None:
    for module_name, details in plan.items():
        mtype = details.get("mutation_type", "dimension")
        location = details.get("source_location")
        if not location:
            continue

        if mtype == "channel_config":
            code_mutator.schedule_channel_config_modification(
                location, details.get("index"), details.get("new_value"), details.get("name", "")
            )
            continue

        module = original_model.get_submodule(module_name)
        if mtype == "dimension":
            if isinstance(module, nn.Linear):
                arg = "out_features" if details.get("new_out") is not None else "in_features"
            elif isinstance(module, nn.Conv2d):
                arg = "out_channels" if details.get("new_out") is not None else "in_channels"
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                arg = "num_features"
            else:
                continue
            value = details.get("new_out") or details.get("new_in")
            if value is not None:
                code_mutator.schedule_modification(location, arg, value)
        elif mtype == "activation" and details.get("new_activation"):
            code_mutator.schedule_activation_modification(location, details["new_activation"])
        elif mtype == "layer_type" and details.get("new_layer_type"):
            code_mutator.schedule_layer_type_modification(location, details["new_layer_type"], details.get("mutation_params", {}))
        elif mtype == "kernel_size" and details.get("new_kernel_size"):
            code_mutator.schedule_kernel_size_modification(location, details["new_kernel_size"])
        elif mtype == "stride" and details.get("new_stride"):
            code_mutator.schedule_stride_modification(location, details["new_stride"])


def _validate(model, h: int, w: int, classes: int) -> bool:
    try:
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(2, 3, h, w))
        if tuple(out.shape) != (2, classes):
            return False
        model.train()
        x = torch.randn(2, 3, h, w).requires_grad_(True)
        out = model(x)
        out.sum().backward()
        for _, param in model.named_parameters():
            if param.grad is None:
                return False
        torch.optim.SGD(model.parameters(), lr=0.01).step()
        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate channel-mutated model variants via brute/ast")
    ap.add_argument("--model", required=True, help="Path to the model .py file")
    ap.add_argument("--name", default=None, help="Model name for filenames (default: file stem)")
    ap.add_argument("--mode", choices=["bracket", "direct"], default="bracket")
    ap.add_argument("--count", type=int, default=1112, help="Target number of unique valid models")
    ap.add_argument("--out", default=os.path.join(AST_DIR, "generated"), help="Output directory")
    ap.add_argument("--max-attempts", type=int, default=200000)
    ap.add_argument("--input-size", type=int, default=32)
    ap.add_argument("--classes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--exclude-dir", action="append", default=[],
                    help="Directory of previously generated ast-dimension-<name>-<hash>.py to exclude (repeatable)")
    args = ap.parse_args()

    if args.seed is not None:
        import random
        random.seed(args.seed)

    _configure_mode(args.mode)
    config.DEBUG_MODE = False

    source_path = os.path.abspath(args.model)
    model_name = args.name or os.path.splitext(os.path.basename(source_path))[0]
    source = open(source_path, encoding="utf-8").read()
    os.makedirs(args.out, exist_ok=True)
    h = w = args.input_size
    classes = args.classes

    # Search-space size for the chosen mode (used for coverage reporting).
    # Bracket mode: every (container, element) x (alternative width) is one point.
    from mutator.planning.channel_config_planner import ChannelConfigPlanner
    _probe = ChannelConfigPlanner.__new__(ChannelConfigPlanner)
    _probe.model_planner = None
    _probe.source_code = source
    import ast as _ast
    try:
        _probe.tree = _ast.parse(source)
    except SyntaxError:
        _probe.tree = None
    candidate_count = len(_probe._find_candidates()) if _probe.tree is not None else 0
    alternatives_per_candidate = max(len(config.VALID_CHANNEL_SIZES) - 1, 0)
    search_space_size = candidate_count * alternatives_per_candidate if args.mode == "bracket" else None

    seen = set()
    for exclude_dir in args.exclude_dir:
        for fname in os.listdir(exclude_dir):
            m = re.match(r"^ast-dimension-.*-([0-9a-f]{32})\.py$", fname)
            if m:
                seen.add(m.group(1))
    if seen:
        print(f"Excluding {len(seen)} previously generated architectures")
    saved = 0
    attempts = 0
    stats = {"no_plan": 0, "build_fail": 0, "invalid": 0, "duplicate": 0}
    start = time.time()
    cpu_start = time.process_time()

    while saved < args.count and attempts < args.max_attempts:
        attempts += 1
        tag = f"ast_channel_src_{attempts}"
        try:
            with ModuleSourceTracer(source) as tracer:
                net = _load_net(source_path, tag, h, w, classes)
            source_map = tracer.create_source_map(net)
            planner = ModelPlanner(net, source_map=source_map, search_depth=config.PRODUCER_SEARCH_DEPTH,
                                   source_code=source)
            plan = planner.plan_random_mutation()
        except Exception:
            stats["build_fail"] += 1
            continue

        if not plan:
            stats["no_plan"] += 1
            continue

        mtype = next((d.get("mutation_type", "dimension") for d in plan.values()), "unknown")

        try:
            code_mutator = CodeMutator(source)
            _schedule_plan(code_mutator, plan, net)
            modified_code = code_mutator.get_modified_code()
            if mtype == "channel_config":
                model = _instantiate_from_code(modified_code, source_path, h, w, classes)
            else:
                model = planner.apply_plan()
        except Exception:
            stats["build_fail"] += 1
            continue

        if not _validate(model, h, w, classes):
            stats["invalid"] += 1
            continue
        if mtype != "channel_config":
            # For direct mode the saved code must also build and run.
            try:
                if not _validate(_instantiate_from_code(modified_code, source_path, h, w, classes), h, w, classes):
                    stats["invalid"] += 1
                    continue
            except Exception:
                stats["invalid"] += 1
                continue

        checksum = uuid4(modified_code)
        if checksum in seen:
            stats["duplicate"] += 1
            continue
        seen.add(checksum)
        # Both channel modes are written with the uniform 'ast-dimension' label.
        fname = f"ast-dimension-{model_name}-{checksum}.py"
        with open(os.path.join(args.out, fname), "w", encoding="utf-8") as fh:
            fh.write(modified_code)
        saved += 1

        if not args.quiet and saved % 100 == 0:
            rate = saved / max(attempts, 1)
            print(f"  saved={saved}/{args.count} attempts={attempts} unique={len(seen)} "
                  f"yield={rate:.3f} elapsed={time.time() - start:.0f}s {stats}", flush=True)

    wall_s = time.time() - start
    cpu_s = time.process_time() - cpu_start
    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    def _per(n):
        return (wall_s / n) if n else None

    report = {
        "model": model_name,
        "model_file": source_path,
        "mode": args.mode,
        "seed": args.seed,
        "target": args.count,
        "saved_unique_valid": saved,
        "attempts": attempts,
        "stats": stats,
        "wall_seconds": round(wall_s, 3),
        "cpu_seconds": round(cpu_s, 3),
        "cpu_percent": round(100.0 * cpu_s / wall_s, 2) if wall_s else None,
        "peak_rss_mb": round(peak_rss_kb / 1024.0, 2),
        "throughput_attempts_per_s": round(attempts / wall_s, 4) if wall_s else None,
        "throughput_valid_per_s": round(saved / wall_s, 4) if wall_s else None,
        "wall_seconds_per_attempt": round(_per(attempts), 6) if _per(attempts) else None,
        "wall_seconds_per_valid_model": round(_per(saved), 6) if _per(saved) else None,
        "cpu_seconds_per_attempt": round(cpu_s / attempts, 6) if attempts else None,
        "attempts_per_valid_model": round(attempts / saved, 4) if saved else None,
        "valid_yield": round(saved / attempts, 4) if attempts else None,
        "search_space_size": search_space_size,
        "search_space_containers": candidate_count,
        "alternatives_per_candidate": alternatives_per_candidate,
        "coverage_of_search_space": round(saved / search_space_size, 5) if search_space_size else None,
        "distinct_checksums": len(seen),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "python": sys.version.split()[0],
        },
        "completed": saved >= args.count,
    }
    report_path = os.path.join(args.out, "compute_budget.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"\n[{args.mode}] model={model_name} saved={saved} unique={len(seen)} attempts={attempts} "
          f"wall={wall_s:.1f}s cpu={cpu_s:.1f}s peak_rss={peak_rss_kb/1024.0:.0f}MB stats={stats}")
    print(f"compute budget written to {report_path}")
    if saved < args.count:
        print(f"[warn] reached only {saved} unique valid models (target {args.count}). "
              f"The reachable valid space for this model/mode may be smaller.")


if __name__ == "__main__":
    main()
