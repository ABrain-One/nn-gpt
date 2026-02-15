"""
Learning Rate Scheduler Configuration & Generation Module

This package provides:
- Scheduler strategy definitions and generation
- Model configuration and selection
- Database integration for metrics storage
- Per-class accuracy tracking
- Fine-tuning workflow support

Key Components:
- const.py: Configuration constants and model definitions
- util.py: Utility functions for NN filtering and database operations
- schedulers.py: Main scheduler generation script

Usage:
    python schedulers.py  # Generate all scheduler variants
    
    from ab.gpt.brute.lr.util import unique_nn_cls, get_active_model
    models = unique_nn_cls(epoch_max=90)  # Query best models
"""

from .const import (
    ACTIVE_MODEL,
    IMAGE_CLASSIFICATION_MODELS,
    DEFAULT_HYPERPARAMS,
    CIFAR10_CLASSES,
)

from .util import (
    unique_nn,
    unique_nn_cls,
    get_active_model,
    set_active_model,
    init_database,
    save_scheduler_result,
    get_best_schedulers,
    get_class_accuracies,
    generate_class_data_for_epoch,
)

__all__ = [
    # Constants
    'ACTIVE_MODEL',
    'IMAGE_CLASSIFICATION_MODELS',
    'DEFAULT_HYPERPARAMS',
    'CIFAR10_CLASSES',
    # Functions
    'unique_nn',
    'unique_nn_cls',
    'get_active_model',
    'set_active_model',
    'init_database',
    'save_scheduler_result',
    'get_best_schedulers',
    'get_class_accuracies',
    'generate_class_data_for_epoch',
]

__version__ = '1.0.0'
__author__ = 'NNGpt Team'
