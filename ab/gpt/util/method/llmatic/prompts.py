"""Prompt templates for LLMatic-style variation operators.

Two operators:

* **Mutation** -- "improve one architecture while keeping a similar size". This
  is essentially the pipeline's current behavior, so :func:`build_mutation_prompt`
  reproduces the existing single-seed ``NN_gen`` prompt. In practice the patch
  can instead reuse the config-driven prompt unchanged and only swap *which*
  seed is chosen (via the archive); this builder exists for completeness and
  standalone testing.

* **Crossover** -- "combine the strengths of two architectures into a new one".
  This is the genuinely new operator (:func:`build_crossover_prompt`).

Both keep the SAME system message, LEMUR ``Net`` interface rules, and
``<hp>/<tr>/<nn>`` output contract as the live config, so downstream postprocess
fixers and the evaluator work unchanged. Both accept an optional
``nn_code_max_chars`` so two parent bodies still fit a short-context model's
window (the reason ``NN_gen_shortctx`` exists).

A ``seed`` / ``parent`` is a dict with keys: ``nn_code``, ``accuracy``,
``dataset``, ``task``, ``metric``, ``epoch``, ``prm``, ``transform_code``,
``metric_code`` (the same columns ``nn_gen`` reads from a corpus row).
"""

from __future__ import annotations

from typing import Optional


# System message: identical in spirit to conf/prompt/.../NN_gen.json so the
# model's global behavior does not change between operators.
LEMUR_SYSTEM = "\n".join([
    "You are an expert PyTorch neural network architect.",
    "Your task is to generate improved neural network architectures, training "
    "hyperparameters, and data transforms that maximise performance on computer "
    "vision tasks.",
    "Always follow LEMUR NN Dataset conventions: class name 'Net', a standalone "
    "function 'supported_hyperparameters' (outside the class) returning a set of "
    "hyperparameter key names, and required methods __init__, forward, "
    "train_setup, and learn with their exact signatures.",
    "Output only the requested XML tags (<hp>, <tr>, <nn>) in the correct order "
    "— no explanation, no markdown, no extra text.",
])

# Compact LEMUR interface contract. Condensed from the live NN_gen config -- the
# rules the evaluator actually enforces. The 7 postprocess fixers repair the rest.
INTERFACE_RULES = """The neural network code inside <nn>...</nn> MUST follow this exact LEMUR interface:
  1. A module-level function BEFORE the class: def supported_hyperparameters(): return {'lr', 'momentum'}
     Return a set of ONLY the keys you access via prm['key'] inside the class. Undeclared-but-used OR declared-but-unused keys cause failure. Safe default: {'lr', 'momentum'}.
  2. The main model class MUST be named exactly 'Net' and inherit from nn.Module (never from torchvision models). Helper blocks may have any name.
  3. Net.__init__ signature exactly: (self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device). Store self.device = device. in_shape is (batch, channels, H, W) -- use in_shape[1] for channels; use out_shape[0] for num classes.
  4. Methods train_setup(self, prm) and learn(self, train_data) inside Net. train_setup sets self.optimizer and self.criteria (does not return them). learn iterates `for inputs, labels in train_data:`.
  5. Always `import torch` and `import torch.nn as nn` (and `import torch.nn.functional as F` if you use F.*).
  6. Avoid hardcoded spatial dims in Linear layers; use nn.AdaptiveAvgPool2d((1,1)) before flattening.
  7. Every attribute used in forward() must be defined in __init__()."""

OUTPUT_RULES = """Provide a single COMPLETE Python file. Do not truncate, do not use '...' as a placeholder, do not leave any method body empty.
Output only three XML tags in this exact order: <hp>...</hp>, <tr>...</tr>, and <nn>...</nn>. No JSON, no markdown fences, no explanation."""


def _get(d: dict, key: str, default: str = "") -> str:
    v = d.get(key, default)
    return default if v is None else v


def _clip(code: str, max_chars: Optional[int]) -> str:
    if not isinstance(code, str):
        return ""
    if max_chars is not None and len(code) > max_chars:
        return code[:max_chars] + "\n# ... (truncated for context budget) ...\n"
    return code


def build_mutation_prompt(seed: dict, nn_code_max_chars: Optional[int] = None) -> tuple:
    """Return ``(system, prompt)`` for a single-seed mutation (improve + keep size).

    Mirrors the existing NN_gen behavior: condition on one reference model and
    ask for a novel, similarly-sized improvement.
    """
    nn_code = _clip(_get(seed, "nn_code"), nn_code_max_chars)
    prompt = f"""Provide training hyperparameters (JSON between <hp> and </hp>), transform code (Python between <tr> and </tr>) and a neural network model (Python between <nn> and </nn>) that maximises the metric '{_get(seed, 'metric')}' after the first epoch of training on the '{_get(seed, 'dataset')}' dataset for the task '{_get(seed, 'task')}'.

Improve the reference architecture below while keeping a SIMILAR parameter count and depth -- refine it, do not radically resize it. Make the result differ from every LEMUR model by adding/removing/adjusting layers, reordering, or changing dimensions.

The reference LEMUR model achieved metric '{_get(seed, 'metric')}' = {_get(seed, 'accuracy')} at epoch {_get(seed, 'epoch')} on '{_get(seed, 'dataset')}' with hyperparameters <hp>{_get(seed, 'prm')}</hp>:
<tr>{_get(seed, 'transform_code')}</tr>
<metric>{_get(seed, 'metric_code')}</metric>
<nn>{nn_code}</nn>

{INTERFACE_RULES}

{OUTPUT_RULES}"""
    return LEMUR_SYSTEM, prompt


def build_crossover_prompt(
    parent_a: dict,
    parent_b: dict,
    nn_code_max_chars: Optional[int] = 1200,
) -> tuple:
    """Return ``(system, prompt)`` that combines two parents into a new model.

    ``nn_code_max_chars`` defaults to 1200 *per parent* so both bodies plus the
    interface rules fit a short-context (4k) model. Set to None to send full code.
    """
    dataset = _get(parent_a, "dataset") or _get(parent_b, "dataset")
    task = _get(parent_a, "task") or _get(parent_b, "task")
    metric = _get(parent_a, "metric") or _get(parent_b, "metric")
    metric_code = _get(parent_a, "metric_code") or _get(parent_b, "metric_code")
    code_a = _clip(_get(parent_a, "nn_code"), nn_code_max_chars)
    code_b = _clip(_get(parent_b, "nn_code"), nn_code_max_chars)

    prompt = f"""Provide training hyperparameters (JSON between <hp> and </hp>), transform code (Python between <tr> and </tr>) and a neural network model (Python between <nn> and </nn>) that maximises the metric '{metric}' after the first epoch of training on the '{dataset}' dataset for the task '{task}'.

Your task is CROSSOVER: combine the complementary strengths of the TWO reference architectures below into a single NEW architecture. Take the best structural ideas from each parent -- e.g. one parent's feature-extraction blocks with the other's classifier or connectivity pattern, or a blend of their depth/width choices -- and synthesise a coherent model that is DIFFERENT from both parents and from every LEMUR model. Do not simply concatenate them; integrate their ideas into one clean design.

PARENT A -- metric '{metric}' = {_get(parent_a, 'accuracy')} at epoch {_get(parent_a, 'epoch')}, hyperparameters <hp>{_get(parent_a, 'prm')}</hp>:
<nn>{code_a}</nn>

PARENT B -- metric '{metric}' = {_get(parent_b, 'accuracy')} at epoch {_get(parent_b, 'epoch')}, hyperparameters <hp>{_get(parent_b, 'prm')}</hp>:
<nn>{code_b}</nn>

<metric>{metric_code}</metric>

{INTERFACE_RULES}

{OUTPUT_RULES}"""
    return LEMUR_SYSTEM, prompt
