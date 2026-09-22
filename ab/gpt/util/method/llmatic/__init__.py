"""LLMatic-style MAP-Elites quality-diversity layer for the iterative pipeline.

Enabled by the native CLI flags ``--llmatic`` and ``--llmatic_crossover_every``,
which ``TuneNNGen.py`` wires through the iterative generation pipeline. See
``archive.py`` for the MAP-Elites archive, ``descriptors.py`` for pre-eval
behavioral-descriptor extraction, ``seeds.py`` for seed selection, and
``prompts.py`` for the mutation / crossover prompt templates.
"""
