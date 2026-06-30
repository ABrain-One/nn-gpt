"""
EdgeEfficiencyPrompt.py — Sarmad's MSc thesis contribution.

Adds the training `duration` of each reference model to the curriculum prompt
packing, enabling efficiency-ranked (accuracy / duration) prompts for
edge-deployable model generation.

This module does NOT modify any lab file on disk. Importing it installs an
in-memory wrapper around NNGenPrompt._pack_k_models that appends
duration_{i} for each reference model. The lab's source remains untouched.
"""

from ab.gpt.util.prompt.NNGenPromptCurriculum import NNGenPrompt

# Keep a reference to the lab's original packing function.
_original_pack_k_models = NNGenPrompt._pack_k_models


def _pack_k_models_with_duration(rows, k, cfg=None):
    packed = _original_pack_k_models(rows, k, cfg)
    for i, row in enumerate(rows, start=1):
        packed[f"duration_{i}"] = row.get("duration")
    return packed


def install():
    """Install the duration-aware packing (idempotent)."""
    if getattr(NNGenPrompt._pack_k_models, "_edge_patched", False):
        return
    _pack_k_models_with_duration._edge_patched = True
    NNGenPrompt._pack_k_models = staticmethod(_pack_k_models_with_duration)


# Auto-install on import.
install()
