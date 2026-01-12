"""
Iterative Fine-Tuning Pipeline for Neural Architecture Generation

This package contains the main pipeline orchestration and supporting modules
for the iterative fine-tuning approach.
"""

# Note: IterativeFinetuner is NOT imported here to avoid circular imports.
# It should be imported directly from ab.gpt.iterative_finetune when needed.
from ab.gpt.iterative_pipeline.novelty_checker import NoveltyChecker
from ab.gpt.iterative_pipeline.training_data_manager import TrainingDataManager
from ab.gpt.iterative_pipeline.structural_reranker import StructuralReranker
from ab.gpt.iterative_pipeline.pipeline_validation import (
    PipelineValidator, RetryHandler, StageValidator, ErrorRecovery
)
from ab.gpt.iterative_pipeline.gpu_memory_manager import (
    ensure_gpu_memory, clear_gpu_cache, get_gpu_memory_info, check_gpu_memory,
    kill_gpu_processes
)

__all__ = [
    # IterativeFinetuner removed to avoid circular import - import directly from ab.gpt.iterative_finetune
    'NoveltyChecker',
    'TrainingDataManager',
    'StructuralReranker',
    'PipelineValidator',
    'RetryHandler',
    'StageValidator',
    'ErrorRecovery',
    'ensure_gpu_memory',
    'clear_gpu_cache',
    'get_gpu_memory_info',
    'check_gpu_memory',
    'kill_gpu_processes',
]

