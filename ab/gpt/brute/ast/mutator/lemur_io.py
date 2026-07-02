"""LEMUR I/O and model loading utilities."""

import os
import sys
import time
import inspect
import importlib
from typing import Tuple

import torch
import ab.nn.api as nn_dataset

# Dataset parameters used throughout the system
DATASET_PARAMS = {
    "ImageNet": {
        "spatial_dims": (224, 224),
        "output_size": (1000,)
    },
    "CIFAR10": {
        "spatial_dims": (32, 32),
        "output_size": (10,)
    },
    "CIFAR100": {
        "spatial_dims": (32, 32),
        "output_size": (100,)
    },
    "MNIST": {
        "spatial_dims": (28, 28),
        "output_size": (10,)
    },
    "denoise": {
        # Image-to-image task (mobile denoising). output_size=None signals that the
        # network output must match the input shape rather than emit a class count.
        "spatial_dims": (256, 256),
        "output_size": None,
    }
}


def get_dataset_params(dataset_name: str) -> dict:
    """Get parameters for a specific dataset."""
    return DATASET_PARAMS.get(dataset_name, DATASET_PARAMS["ImageNet"])


# Default parameters for LEMUR models
DEFAULT_IN_SHAPE = (1, 3, 224, 224)  # (batch, channels, height, width)
DEFAULT_OUT_SHAPE = (1000,)  # Output shape for classification
DEFAULT_PRM = {
    'lr': 0.01,
    'momentum': 0.9,
    'dropout': 0.5
}

# Common safe defaults to fill supported hyperparameters when nn-dataset prm is unavailable
COMMON_PRM_DEFAULTS = {
    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'dropout': 0.5,
    'betas': (0.9, 0.999),
    'eps': 1e-8,
    'alpha': 0.99,
}


def load_constructor_from_string_via_file(code_string: str) -> Tuple[object, str, str]:
    """Write code to a uniquely named temp module file and import it.
    Returns (constructor, module_name, module_path).
    """
    pid = os.getpid(); timestamp = time.time_ns()
    temp_module_name = f"_temp_model_{pid}_{timestamp}"
    temp_module_path = f"{temp_module_name}.py"
    with open(temp_module_path, 'w', encoding='utf-8') as f:
        f.write(code_string)
    spec = importlib.util.spec_from_file_location(temp_module_name, temp_module_path)  # type: ignore[attr-defined]
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[temp_module_name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    constructor = getattr(module, 'Net')
    return constructor, temp_module_name, temp_module_path


def cleanup_temp_module(temp_module_name: str, temp_module_path: str) -> None:
    try:
        if os.path.exists(temp_module_path):
            os.remove(temp_module_path)
    except OSError:
        pass
    pycache_dir = os.path.join(os.path.dirname(temp_module_path), '__pycache__')
    if os.path.isdir(pycache_dir):
        for fname in os.listdir(pycache_dir):
            if fname.startswith(temp_module_name):
                try:
                    os.remove(os.path.join(pycache_dir, fname))
                except OSError:
                    pass
    sys.modules.pop(temp_module_name, None)


def load_lemur_model(model_source: str):
    """Load LEMUR model with proper parameters.
    - Prefer prm from LEMUR dataset for this exact code (best-accuracy row).
    - Filter prm to the model's supported_hyperparameters() if provided.
    - Fallback to safe defaults for any supported keys not present.
    """
    constructor, temp_module_name, temp_module_path = load_constructor_from_string_via_file(model_source)

    # LEMUR models require specific parameters
    sig = inspect.signature(constructor)
    params = {}

    # Discover the module that owns the constructor to query supported_hyperparameters()
    supported_params = None
    try:
        model_module = importlib.import_module(constructor.__module__)
        if hasattr(model_module, 'supported_hyperparameters'):
            maybe_supported = model_module.supported_hyperparameters()
            if isinstance(maybe_supported, (set, list, tuple)):
                supported_params = set(maybe_supported)
        # Also check class-level attribute for completeness
        if supported_params is None and hasattr(constructor, 'supported_hyperparameters'):
            maybe_supported = constructor.supported_hyperparameters()
            if isinstance(maybe_supported, (set, list, tuple)):
                supported_params = set(maybe_supported)
    except Exception:
        supported_params = None

    # Try to get prm from nn-dataset for this exact code
    dataset_prm = None
    try:
        df = nn_dataset.data(only_best_accuracy=False)
        rows = df[df['nn_code'] == model_source]
        if not rows.empty:
            best_row = rows.loc[rows['accuracy'].idxmax()]
            dataset_prm = best_row.get('prm', None)
    except Exception:
        dataset_prm = None

    if "in_shape" in sig.parameters:
        params["in_shape"] = DEFAULT_IN_SHAPE
    if "out_shape" in sig.parameters:
        params["out_shape"] = DEFAULT_OUT_SHAPE
    if "prm" in sig.parameters:
        # Start from dataset prm when available
        if isinstance(dataset_prm, dict) and dataset_prm:
            if supported_params:
                prm = {k: v for k, v in dataset_prm.items() if k in supported_params}
                # Fill any missing supported keys with safe defaults
                for k in supported_params:
                    if k not in prm and (k in COMMON_PRM_DEFAULTS or k in DEFAULT_PRM):
                        prm[k] = COMMON_PRM_DEFAULTS.get(k, DEFAULT_PRM.get(k))
            else:
                prm = dataset_prm
        else:
            # Build from supported set if present, otherwise fall back to legacy DEFAULT_PRM
            if supported_params:
                prm = {}
                for k in supported_params:
                    if k in COMMON_PRM_DEFAULTS:
                        prm[k] = COMMON_PRM_DEFAULTS[k]
                    elif k in DEFAULT_PRM:
                        prm[k] = DEFAULT_PRM[k]
                if not prm:
                    prm = DEFAULT_PRM
            else:
                prm = DEFAULT_PRM
        params["prm"] = prm
    if "device" in sig.parameters:
        params["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_instance = constructor(**params)
    # store temp module info for later cleanup
    setattr(model_instance, '_temp_module_info', (temp_module_name, temp_module_path))
    return model_instance


def fetch_model_from_lemur(model_name: str):
    """Fetch model code from LEMUR database by name."""
    try:
        data = nn_dataset.data(only_best_accuracy=True)
        model_data = data[data['nn'] == model_name]
        if not model_data.empty:
            return model_data.iloc[0]['nn_code']
        return None
    except Exception as e:
        print(f"Error fetching model {model_name}: {str(e)}")
        return None
