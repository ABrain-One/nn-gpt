"""
Edge deployment metrics for LLM-generated neural network models.

Measures, per generated model:
- parameter count with a hard budget gate (default 6M, matching the generation prompt)
- TFLite INT8 export (deployability check; the .tflite artifact feeds the
  on-device benchmarking pipeline)
- inference latency in milliseconds: TFLite CPU interpreter when the litert
  stack is installed, PyTorch CPU forward pass as fallback
- efficiency score = accuracy / latency_ms (the thesis metric)

All measurements run on CPU at the model's serving resolution (norm_256_flip
=> 256x256), batch size 1.
"""

import importlib.util
import statistics
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

PARAM_LIMIT_DEFAULT = 6_000_000
INPUT_SIZE_DEFAULT = 256
LATENCY_WARMUP = 5
LATENCY_RUNS = 20

DATASET_CLASSES = {
    'cifar-10': 10,
    'cifar-100': 100,
    'imagenette': 10,
    'svhn': 10,
    'mnist': 10,
}

# Same defaults the generation step injects into hp.txt.
HP_DEFAULTS = {
    'batch': 16, 'dropout': 0.2, 'epoch': 1,
    'lr': 0.01, 'momentum': 0.9, 'transform': 'norm_256_flip',
}


def load_net(nn_file: Path, prm: Optional[Dict[str, Any]] = None,
             input_size: int = INPUT_SIZE_DEFAULT, num_classes: int = 10) -> torch.nn.Module:
    """Import a generated model file and instantiate Net on CPU in eval mode."""
    prm = {**HP_DEFAULTS, **(prm or {})}
    in_shape = (1, 3, input_size, input_size)
    out_shape = (num_classes,)

    spec = importlib.util.spec_from_file_location(f"edge_nn_{uuid.uuid4().hex}", str(nn_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    net_cls = getattr(module, 'Net')
    model = net_cls(in_shape, out_shape, prm, torch.device('cpu'))
    model.to('cpu')
    model.eval()
    return model


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def torch_cpu_latency_ms(model: torch.nn.Module, input_size: int = INPUT_SIZE_DEFAULT,
                         warmup: int = LATENCY_WARMUP, runs: int = LATENCY_RUNS) -> float:
    """Median CPU forward-pass latency, batch 1."""
    x = torch.randn(1, 3, input_size, input_size)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def export_tflite_int8(model: torch.nn.Module, tflite_path: Path,
                       input_size: int = INPUT_SIZE_DEFAULT) -> Tuple[bool, Optional[str]]:
    """
    Convert to INT8-quantized TFLite via the ai-edge/litert converter with a
    representative dataset (post-training quantization). Returns (success, error).
    """
    try:
        import numpy as np
        import tensorflow as tf
        try:
            import litert_torch as ai_edge_torch
        except ImportError:
            import ai_edge_torch
    except Exception as e:
        return False, f"tflite deps missing: {e}"

    try:
        dummy_input = (torch.randn(1, 3, input_size, input_size),)

        def representative_dataset():
            for _ in range(50):
                yield [np.random.randn(1, 3, input_size, input_size).astype(np.float32)]

        converter_flags = {
            "optimizations": [tf.lite.Optimize.DEFAULT],
            "representative_dataset": representative_dataset,
            "target_spec": {"supported_ops": [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]},
        }
        try:
            converted = ai_edge_torch.convert(model, dummy_input,
                                              _ai_edge_converter_flags=converter_flags)
        except TypeError:
            converted = ai_edge_torch.convert(model, dummy_input)

        tflite_path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(converted, 'export'):
            converted.export(str(tflite_path))
        elif isinstance(converted, (bytes, bytearray)):
            tflite_path.write_bytes(bytes(converted))
        else:
            return False, f"unexpected converter output type: {type(converted)}"
        return True, None
    except Exception as e:
        return False, str(e)


def tflite_cpu_latency_ms(tflite_path: Path, input_size: int = INPUT_SIZE_DEFAULT,
                          warmup: int = LATENCY_WARMUP, runs: int = LATENCY_RUNS) -> Optional[float]:
    """Median latency of the TFLite model on the host CPU (XNNPACK kernels)."""
    try:
        import numpy as np
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
    except Exception:
        return None

    try:
        interpreter = Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()[0]
        x = np.random.randn(*inp['shape']).astype(inp['dtype'])

        for _ in range(warmup):
            interpreter.set_tensor(inp['index'], x)
            interpreter.invoke()
        times = []
        for _ in range(runs):
            interpreter.set_tensor(inp['index'], x)
            t0 = time.perf_counter()
            interpreter.invoke()
            times.append((time.perf_counter() - t0) * 1000.0)
        return statistics.median(times)
    except Exception:
        return None


def benchmark(nn_file: Path,
              prm: Optional[Dict[str, Any]] = None,
              accuracy: Optional[float] = None,
              dataset: str = 'cifar-10',
              input_size: int = INPUT_SIZE_DEFAULT,
              param_limit: int = PARAM_LIMIT_DEFAULT,
              tflite_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Full edge benchmark for one generated model file.

    Returns a dict with: params, param_gate_ok, tflite_ok, tflite_error,
    latency_ms, latency_source, torch_cpu_latency_ms, efficiency, error.
    """
    result: Dict[str, Any] = {
        'nn_file': str(nn_file),
        'dataset': dataset,
        'input_size': input_size,
        'accuracy': accuracy,
        'params': None,
        'param_gate_ok': None,
        'param_limit': param_limit,
        'torch_cpu_latency_ms': None,
        'tflite_ok': False,
        'tflite_error': None,
        'tflite_path': None,
        'latency_ms': None,
        'latency_source': None,
        'efficiency': None,
        'error': None,
        'torch_num_threads': torch.get_num_threads(),
    }

    num_classes = DATASET_CLASSES.get(dataset, 10)

    try:
        model = load_net(nn_file, prm=prm, input_size=input_size, num_classes=num_classes)
    except Exception as e:
        result['error'] = f"model load failed: {e}"
        return result

    try:
        result['params'] = count_params(model)
        result['param_gate_ok'] = result['params'] <= param_limit
    except Exception as e:
        result['error'] = f"param count failed: {e}"
        return result

    try:
        result['torch_cpu_latency_ms'] = round(torch_cpu_latency_ms(model, input_size), 3)
    except Exception as e:
        result['error'] = f"torch latency failed: {e}"
        return result

    if tflite_path is not None:
        ok, err = export_tflite_int8(model, tflite_path, input_size)
        result['tflite_ok'] = ok
        result['tflite_error'] = err
        if ok:
            result['tflite_path'] = str(tflite_path)
            lat = tflite_cpu_latency_ms(tflite_path, input_size)
            if lat is not None:
                result['latency_ms'] = round(lat, 3)
                result['latency_source'] = 'tflite_cpu'

    if result['latency_ms'] is None:
        result['latency_ms'] = result['torch_cpu_latency_ms']
        result['latency_source'] = 'torch_cpu'

    if accuracy is not None and result['latency_ms'] and result['latency_ms'] > 0:
        result['efficiency'] = round(float(accuracy) / result['latency_ms'], 6)

    return result
