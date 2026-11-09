"""Denoise genetic-algorithm search (skip-focused).

Standalone reuse of the generic GA core from ``ga/modular`` (engine, population,
selection, crossover, mutation) plus a denoise-specific fitness function and a
skip-configurable model template. Kept separate from ``modular`` to avoid its
LLM/fractal coupling (``modular/__init__.py`` imports the LLM mutation strategy).
"""
