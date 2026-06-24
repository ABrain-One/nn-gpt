"""
EdgeEfficiencyPrompt.py — Sarmad's MSc thesis contribution.

Self-contained extension of the curriculum prompt builder that exposes the
training `duration` of each reference model, enabling efficiency-ranked
(accuracy / duration) prompts for edge-deployable model generation.

Does NOT modify any lab file. Overrides only the model packing step to add
duration_{i} for each reference model.
"""

from ab.gpt.util.prompt.NNGenPromptCurriculum import NNGenPrompt


class EdgeEfficiencyPrompt(NNGenPrompt):
    @staticmethod
    def _pack_k_models(rows, k, cfg=None):
        packed = NNGenPrompt._pack_k_models(rows, k, cfg)
        for i, row in enumerate(rows, start=1):
            packed[f"duration_{i}"] = row.get("duration")
        return packed
