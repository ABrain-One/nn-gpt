"""High-level orchestration for fetching models and running mutations.

This module contains the end-to-end driver that was previously in main.py.
It exposes a main() entrypoint and helpers for fetching models and running
mutations. Behavior is preserved; only the structure is modularized.
"""

from collections import Counter
import multiprocessing
import os
import sys
import time
import traceback
from typing import Dict, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

import ab.nn.api as nn_dataset

from mutator import config
from mutator.utils.file_utils import save_plan_to_file
from mutator.utils.source_tracer import ModuleSourceTracer
from mutator.planning import ModelPlanner
from mutator.execution.code_mutator import CodeMutator
from mutator.tracking import get_mutation_tracker, get_plan_tracker
from mutator.lemur_io import (
    load_lemur_model,
    cleanup_temp_module,
    get_dataset_params,
    load_constructor_from_string_via_file,
)


def _schedule_plan_on_mutator(code_mutator: CodeMutator, plan: dict, original_model) -> None:
    """Translate a mutation plan into CodeMutator scheduling calls.

    Handles both per-module mutations (dimension/activation/...) and the
    source-level channel-configuration (bracket/scalar) mutation.
    """
    for full_module_name, details in plan.items():
        mutation_type = details.get("mutation_type", "dimension")

        if mutation_type == "channel_config":
            location = details.get("source_location")
            if location:
                code_mutator.schedule_channel_config_modification(
                    location,
                    details.get("index"),
                    details.get("new_value"),
                    details.get("name", ""),
                )
                if config.DEBUG_MODE:
                    print(f"Scheduled channel-config modification for {details.get('name')}[{details.get('index')}] = {details.get('new_value')}")
            continue

        location = details.get("source_location")
        if not location:
            continue

        module = original_model.get_submodule(full_module_name)

        if mutation_type == "dimension":
            arg_to_change = None
            if isinstance(module, nn.Linear):
                arg_to_change = 'out_features' if details.get('new_out') is not None else 'in_features'
            elif isinstance(module, nn.Conv2d):
                arg_to_change = 'out_channels' if details.get('new_out') is not None else 'in_channels'
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                arg_to_change = 'num_features'

            if details.get('symbolic') and details.get('symbolic_expression') and arg_to_change:
                code_mutator.schedule_symbolic_modification(location, arg_to_change, details['symbolic_expression'])
                if config.DEBUG_MODE:
                    print(f"Scheduled symbolic modification for {full_module_name}: {arg_to_change} = {details['symbolic_expression']}")
            else:
                new_value = details.get('new_out') or details.get('new_in')
                if arg_to_change and new_value is not None:
                    code_mutator.schedule_modification(location, arg_to_change, new_value)
                    if config.DEBUG_MODE:
                        print(f"Scheduled fixed-value modification for {full_module_name}: {arg_to_change} = {new_value}")

        elif mutation_type == "activation":
            new_activation = details.get('new_activation')
            if new_activation:
                code_mutator.schedule_activation_modification(location, new_activation)

        elif mutation_type == "layer_type":
            new_layer_type = details.get('new_layer_type')
            mutation_params = details.get('mutation_params', {})
            if new_layer_type:
                code_mutator.schedule_layer_type_modification(location, new_layer_type, mutation_params)

        elif mutation_type == "kernel_size":
            new_kernel_size = details.get('new_kernel_size')
            if new_kernel_size:
                code_mutator.schedule_kernel_size_modification(location, new_kernel_size)

        elif mutation_type == "stride":
            new_stride = details.get('new_stride')
            if new_stride:
                code_mutator.schedule_stride_modification(location, new_stride)


def _instantiate_from_code(modified_code: str, h: int, w: int, out_classes: int):
    """Build a model from mutated source code (source-level mutations)."""
    constructor, temp_module_name, temp_module_path = load_constructor_from_string_via_file(modified_code)
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = constructor((2, 3, h, w), (out_classes,), {'lr': 0.01, 'momentum': 0.9}, device)
    finally:
        cleanup_temp_module(temp_module_name, temp_module_path)
    return model


def fetch_base_model_sources() -> Tuple[Dict[str, str], list]:
    """Fetch model source codes from LEMUR, filtered to base models only.

    Returns a tuple of (model_sources, empty_or_problem_models).
    """
    # Fetch ALL models from LEMUR database
    data = nn_dataset.data(only_best_accuracy=False)
    all_model_names = data['nn'].unique().tolist()

    # Filter to only BASE models (exclude generated variants with UUIDs)
    import re
    uuid_pattern = re.compile(r'-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')
    base_model_names = [name for name in all_model_names if not uuid_pattern.search(name)]

    # Further filter models based on config if specified
    if config.SPECIFIC_MODELS:
        model_names = [name for name in base_model_names if name in config.SPECIFIC_MODELS]
    else:
        model_names = base_model_names

    model_sources: Dict[str, str] = {}
    empty_models = []

    with tqdm(model_names, desc="Fetching models") as pbar:
        for name in pbar:
            model_data = data[data['nn'] == name]
            if not model_data.empty:
                # Get the best accuracy version
                best_model = model_data.loc[model_data['accuracy'].idxmax()]
                code = best_model['nn_code']
                if code and code.strip():
                    model_sources[name] = code
                    pbar.set_postfix_str(f"✓ {name}")
                else:
                    empty_models.append(name)
                    pbar.set_postfix_str(f"⚠ Empty: {name}")
            else:
                empty_models.append(name)
                pbar.set_postfix_str(f"⚠ No data: {name}")

    return model_sources, empty_models


def run_single_mutation(worker_args):
    """Run a single mutation attempt for a given (model_name, model_source)."""
    model_name, model_source = worker_args
    plan, original_model = {}, None
    try:
        if not isinstance(model_source, str):
            return 'skipped_not_string', None

        # Determine dataset for this model
        dataset = "ImageNet"  # Default to ImageNet
        # Try to get dataset from LEMUR data
        try:
            df = nn_dataset.data(only_best_accuracy=False)
            rows = df[df['nn_code'] == model_source]
            if not rows.empty:
                best_row = rows.loc[rows['accuracy'].idxmax()]
                dataset = best_row.get('dataset', "ImageNet")
        except Exception:
            dataset = "ImageNet"

        # Get dataset-specific parameters
        dataset_params = get_dataset_params(dataset)
        h, w = dataset_params["spatial_dims"]
        expected_out_classes = dataset_params["output_size"][0]

        # Use LEMUR loader with specialized parameters
        with ModuleSourceTracer(model_source) as tracer:
            original_model = load_lemur_model(model_source)
        # Extract temp module info for cleanup
        temp_module_info = getattr(original_model, '_temp_module_info', None)
        source_map = tracer.create_source_map(original_model)

        planner = ModelPlanner(original_model, source_map=source_map, search_depth=config.PRODUCER_SEARCH_DEPTH, source_code=model_source)
        plan = planner.plan_random_mutation()

        if not plan:
            save_plan_to_file(model_name, 'skipped_no_plan', {}, {"reason": "Planner could not find a valid mutation."})
            return 'skipped_no_plan', None

        # Check if this mutation plan is unique
        plan_tracker = get_plan_tracker()
        if not plan_tracker.is_unique_plan(plan):
            if config.DEBUG_MODE:
                print(f"  -> Skipping duplicate mutation plan")
            save_plan_to_file(model_name, 'skipped_duplicate_plan', plan, {"reason": "Duplicate mutation plan"})
            return 'skipped_duplicate_plan', None
        else:
            # Register this unique plan
            plan_tracker.register_plan(plan)

        # Determine the dominant mutation type of this plan.
        plan_mutation_type = next(
            (d.get("mutation_type", "dimension") for d in plan.values()), "dimension"
        )

        code_mutator = CodeMutator(model_source)

        if plan_mutation_type == "channel_config":
            # Source-level mutation: apply it to the code, then build the model
            # from the mutated code so verification matches what is saved.
            _schedule_plan_on_mutator(code_mutator, plan, original_model)
            modified_code = code_mutator.get_modified_code()
            mutated_model = _instantiate_from_code(modified_code, h, w, expected_out_classes)
        else:
            mutated_model = planner.apply_plan()

        # Enhanced model verification
        try:
            # Test with dataset-specific input resolution
            test_input = torch.randn(2, 3, h, w)

            # Forward pass
            mutated_model.eval()
            with torch.no_grad():
                output = mutated_model(test_input)

            # Check output shape with dataset-specific class count
            if output.shape[0] != 2 or output.shape[1] != expected_out_classes:
                raise RuntimeError(f"Output shape {output.shape} not as expected for input {h}x{w} (expected {expected_out_classes} classes)")

            # Backward pass test
            mutated_model.train()
            test_input.requires_grad = True
            output = mutated_model(test_input)
            loss = output.sum()
            loss.backward()

            # Check gradients
            for name, param in mutated_model.named_parameters():
                if param.grad is None:
                    raise RuntimeError(f"No gradient for parameter {name}")

            # Simple learning capability check
            optimizer = torch.optim.SGD(mutated_model.parameters(), lr=0.01)
            mutated_model.train()
            test_input = torch.randn(2, 3, 224, 224)
            output = mutated_model(test_input)
            loss = output.sum()
            loss.backward()
            optimizer.step()

        except Exception as e:
            raise RuntimeError(f"Model verification failed: {str(e)}")

        if plan_mutation_type != "channel_config":
            # Channel-config mutations were already applied to code_mutator
            # above (before building the model).
            _schedule_plan_on_mutator(code_mutator, plan, original_model)
            modified_code = code_mutator.get_modified_code()

        # Save to nn-dataset repository
        from ab.nn.util.Util import uuid4  # use canonical hashing
        checksum = uuid4(modified_code)
        timestamp = int(time.time() * 1000)

        # Extract mutation type from plan for filename prefix
        mutation_type = "unknown"
        if plan:
            for details in plan.values():
                if "mutation_type" in details:
                    mutation_type = details["mutation_type"]
                    break

        # Check if this mutation is unique using the tracker
        mutation_tracker = get_mutation_tracker()
        if not mutation_tracker.is_unique_mutation(checksum):
            if config.DEBUG_MODE:
                print(f"  -> Skipping duplicate mutation with checksum: {checksum}")
            # cleanup temp module for duplicate mutation
            if hasattr(original_model, '_temp_module_info'):
                tn, tp = getattr(original_model, '_temp_module_info')
                cleanup_temp_module(tn, tp)
            return 'skipped_duplicate', None

        # Register this unique mutation
        mutation_tracker.register_mutation(checksum)

        # Use configurable output root from config
        model_dir = os.path.join(config.MUTATED_MODELS_OUTPUT_ROOT, model_name)
        os.makedirs(model_dir, exist_ok=True)
        # Channel-config and direct channel mutations share the uniform
        # 'ast-dimension' label in the filename.
        filename_label = "dimension" if mutation_type in ("dimension", "channel_config") else mutation_type
        model_path = os.path.join(model_dir, f"ast-{filename_label}-{model_name}-{checksum}.py")

        with open(model_path, 'w', encoding='utf-8') as f:
            f.write(modified_code)

        if config.DEBUG_MODE:
            print(f"  -> Saved unique mutation: {model_path}")
            print(f"  -> Total unique mutations so far: {mutation_tracker.get_unique_count()}")

        # Optionally save mutated model to LEMUR DB (nn-dataset)
        if getattr(config, 'SAVE_MUTATED_TO_DB', False):
            try:
                # Build DB params: start from discovered/supported prm if possible
                db_prm = {}
                # Merge required training params from config
                db_prm.update(getattr(config, 'DB_TRAIN_PRM', {}))
                # Ensure minimal required keys exist
                required_keys = {'batch', 'epoch', 'transform'}
                missing = [k for k in required_keys if k not in db_prm]
                if missing:
                    raise ValueError(f"Missing required DB training params: {missing}. Set config.DB_TRAIN_PRM.")
                # Call nn-dataset to train-eval and save
                model_name_db, acc, a2t, code_score = nn_dataset.check_nn(
                    modified_code,
                    task=getattr(config, 'DB_TASK', None),
                    dataset=getattr(config, 'DB_DATASET', None),
                    metric=getattr(config, 'DB_METRIC', None),
                    prm=db_prm,
                    save_to_db=True,
                    prefix=getattr(config, 'DB_MODEL_PREFIX', 'mutated'),
                    save_path=None,
                    export_onnx=False,
                    epoch_limit_minutes=getattr(config, 'DB_EPOCH_LIMIT_MINUTES', 10)
                )
                if config.DEBUG_MODE:
                    print(f"  -> Saved to DB as: {model_name_db} (acc={acc:.4f})")
            except Exception as e:
                if config.DEBUG_MODE:
                    print(f"[WARN] Failed to save mutated model to DB: {e}")

        # Convert absolute model path to relative path for the plan file
        try:
            relative_model_path = os.path.relpath(model_path, "f:/mutator_env")
        except ValueError:
            # Fallback to absolute path if relative path conversion fails
            relative_model_path = model_path

        # Save plan for successful mutation as well
        try:
            save_plan_to_file(model_name, 'success', plan, {"checksum": checksum, "model_path": relative_model_path})
        except Exception:
            if config.DEBUG_MODE:
                print("[WARN] Failed to save success plan file")
        # cleanup temp module after successful mutation
        if hasattr(original_model, '_temp_module_info'):
            tn, tp = getattr(original_model, '_temp_module_info')
            cleanup_temp_module(tn, tp)
        return 'success', {"path": model_path, "checksum": checksum}

    except Exception as e:
        error_str = f"{repr(e)}\n{traceback.format_exc()}"
        status = 'fail_worker_uncaught_error'
        if 'original_model' not in locals() or not original_model: status = 'fail_instantiation'
        elif not plan: status = 'fail_planning'
        save_plan_to_file(model_name, status, plan, {"error": error_str})
        # cleanup temp module on failure
        if 'original_model' in locals() and original_model and hasattr(original_model, '_temp_module_info'):
            tn, tp = getattr(original_model, '_temp_module_info')
            cleanup_temp_module(tn, tp)

        # Ensure temp files are cleaned up in all error cases
        if 'temp_module_info' in locals() and temp_module_info:
            tn, tp = temp_module_info
            cleanup_temp_module(tn, tp)

        return status, None


def run_all_models() -> Tuple[int, int]:
    """Fetch models and run mutations according to config; returns (success, failed)."""
    # Fetch sources
    model_sources, empty_models = fetch_base_model_sources()

    print(f"\nSuccessfully fetched {len(model_sources)} models from LEMUR")
    if empty_models:
        print(f"Models with issues: {', '.join(empty_models)}")

    if not model_sources:
        print("Error: No models were fetched from LEMUR. Exiting.")
        return 0, 0

    # Add summary header
    print("\n" + "=" * 80)
    print(f"STARTING MUTATION PROCESS: {len(model_sources)} MODELS")
    print("=" * 80 + "\n")

    total_success = 0
    total_failed = 0

    with tqdm(model_sources.items(), desc="Mutating models", total=len(model_sources)) as model_pbar:
        for model_name, source in model_pbar:
            model_pbar.set_postfix_str(f"Model: {model_name}")
            num_processes = min(config.NUM_WORKERS, config.MAX_CORES_TO_USE)

            if config.DEBUG_MODE:
                print("\n" + "#" * 80)
                print(f"# Mutating Model: {model_name} with {num_processes} workers")
                print("#" * 80)

            worker_args = [(model_name, source)] * config.NUM_ATTEMPTS_PER_MODEL
            stats = Counter()
            start_time = time.time()

            if config.DEBUG_MODE:
                results = [run_single_mutation(arg) for arg in tqdm(worker_args, desc=f"Mutating {model_name}")]
            else:
                with multiprocessing.Pool(processes=num_processes) as pool:
                    results = list(tqdm(pool.imap_unordered(run_single_mutation, worker_args),
                                        total=len(worker_args),
                                        desc=f"Mutating {model_name}"))

            for status, data in results:
                stats[status] += 1
                if status == 'success' and config.DEBUG_MODE:
                    print(f"  -> Mutated model saved to: {data['path']}")

            end_time = time.time()
            total = sum(stats.values())
            failed = stats.get('fail_worker_uncaught_error', 0) + stats.get('fail_instantiation', 0) + stats.get('fail_planning', 0)

            print(f"{model_name}: {stats['success']}✓ {failed}✗ in {end_time - start_time:.2f}s")
            model_pbar.set_postfix_str(f"{model_name}: ✓{stats['success']} ✗{failed}")

            total_success += stats['success']
            total_failed += failed

    return total_success, total_failed


def main():
    multiprocessing.freeze_support()
    print("Starting LEMUR-based model mutation system")
    try:
        success, failed = run_all_models()
    except Exception as e:
        print(f"Error running orchestrator: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 80)
    print(f"TOTAL MUTATION SUMMARY: {success} successful mutations, {failed} failed mutations")
    print("=" * 80)
