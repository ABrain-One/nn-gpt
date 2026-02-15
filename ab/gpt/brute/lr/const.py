"""
Learning Rate Scheduler Constants & Configuration Module

This module defines:
- Model configurations for scheduler fine-tuning
- Image classification models list
- Database and prompt parameters
- Scheduler strategy definitions
"""

# ============================================================================
# Model Configuration
# ============================================================================

# Currently active model for fine-tuning
ACTIVE_MODEL = "ResNet18"

# Image Classification Models - Core Model List
# These are the primary models supported for LR scheduler experimentation
# Prefix Model Logic: Each model can be prefixed with model_ for instance tracking
IMAGE_CLASSIFICATION_MODELS = [
    'AlexNet',
    'BagNet',
    'ConvNeXt',
    'ConvNeXtTransformer',
    'DarkNet',
    'DenseNet',
    'EfficientNet',
    'FractalNet',
    'GoogLeNet',
    'ICNet',
    'InceptionV3',
    'MNASNet',
    'MaxVit',
    'MobileNetV2',
    'MobileNetV3',
    'RegNet',
    'ResNet',           # Primary focus (generic)
    'ResNet18',         # Specific variant - CURRENT FOCUS
    'ResNet34',         # Specific variant
    'ResNet50',         # Specific variant
    'ResNet101',        # Specific variant
    'ShuffleNet',
    'SqueezeNet',
    'SwinTransformer',
    'VGG',
    'VisionTransformer',
]

# Models planned for future support (Phase 2)
ADDITIONAL_MODELS = [
    'AirNet',
    'AirNext',
    'BayesianNet',
    'ComplexNet',
    'DPN68',
    'DPN107',
    'DPN131',
    'Diffuser',
    'EfficientNetV2',
    'MobileNetV4',
    'NASNet',
    'Xception',
]

# All supported models
ALL_MODELS = IMAGE_CLASSIFICATION_MODELS + ADDITIONAL_MODELS

# Unique Neural Network Functions by Model Type
# Maps model names to their unique characteristics/functions
UNIQUE_NN_FUNCTIONS = {
    'ResNet18': {
        'depth': 18,
        'blocks': [2, 2, 2, 2],
        'bottleneck': False,
        'description': 'Shallow residual network with 18 layers',
        'num_params_millions': 11.2,
        'inference_speed': 'fast',
        'use_cases': ['edge_devices', 'real_time', 'mobile'],
    },
    'ResNet34': {
        'depth': 34,
        'blocks': [3, 4, 6, 3],
        'bottleneck': False,
        'description': 'Medium residual network with 34 layers',
        'num_params_millions': 21.8,
        'inference_speed': 'moderate',
        'use_cases': ['general_purpose', 'research'],
    },
    'ResNet50': {
        'depth': 50,
        'blocks': [3, 4, 6, 3],
        'bottleneck': True,
        'description': 'Deep residual network with bottleneck blocks',
        'num_params_millions': 25.6,
        'inference_speed': 'moderate',
        'use_cases': ['production', 'large_scale'],
    },
    'ResNet101': {
        'depth': 101,
        'blocks': [3, 4, 23, 3],
        'bottleneck': True,
        'description': 'Very deep residual network',
        'num_params_millions': 44.5,
        'inference_speed': 'slow',
        'use_cases': ['research', 'high_accuracy'],
    },
    'MobileNetV2': {
        'depth': 19,
        'blocks': 'inverted_residual',
        'bottleneck': True,
        'description': 'Lightweight network for mobile devices',
        'num_params_millions': 3.5,
        'inference_speed': 'very_fast',
        'use_cases': ['mobile', 'embedded', 'edge'],
    },
    'EfficientNet': {
        'depth': 'variable',
        'blocks': 'mobile_inverted_bottleneck',
        'bottleneck': True,
        'description': 'Scaled efficiency-optimized network',
        'num_params_millions': 5.3,
        'inference_speed': 'very_fast',
        'use_cases': ['mobile', 'resource_constrained'],
    },
    'VGG': {
        'depth': 16,
        'blocks': 'sequential',
        'bottleneck': False,
        'description': 'Traditional sequential CNN architecture',
        'num_params_millions': 138.4,
        'inference_speed': 'slow',
        'use_cases': ['baseline', 'feature_extraction'],
    },
    'DenseNet': {
        'depth': 121,
        'blocks': 'dense_blocks',
        'bottleneck': True,
        'description': 'Dense connections between layers',
        'num_params_millions': 7.0,
        'inference_speed': 'moderate',
        'use_cases': ['efficient_deep', 'research'],
    },
}

# Model Priority for Focus
MODEL_PRIORITY = {
    'tier_1_primary': ['ResNet18', 'ResNet34', 'ResNet50'],  # Current focus
    'tier_2_extended': ['MobileNetV2', 'EfficientNet', 'DenseNet'],  # Phase 2
    'tier_3_future': ['SwinTransformer', 'VisionTransformer'],  # Phase 3
}

# ============================================================================
# Learning Rate Scheduler Types
# ============================================================================

SCHEDULER_TYPES = {
    'step_based': [
        'StepLR',
        'MultiStepLR',
        'ExponentialLR',
    ],
    'polynomial': [
        'PolynomialLR_quadratic',
        'PolynomialLR_cubic',
    ],
    'cosine': [
        'CosineAnnealingLR',
        'CosineAnnealingWarmRestarts',
    ],
    'cyclic': [
        'CyclicLR_triangular',
        'CyclicLR_triangular2',
        'CyclicLR_exp_range',
        'OneCycleLR',
    ],
    'warmup': [
        'LinearLR_warmup',
        'ConstantLR_warmup',
        'HF_linear_warmup',
        'HF_cosine_warmup',
    ],
    'lambda': [
        'LambdaLR_linear_decay',
        'LambdaLR_power_decay',
    ],
    'multiplicative': [
        'MultiplicativeLR_exponential',
    ],
}

# ============================================================================
# Database Configuration
# ============================================================================

DB_CONFIG = {
    'path': 'db/ab.nn.db',
    'use_cache': True,
    'auto_refresh': True,
}

# ============================================================================
# CIFAR-10/CIFAR-20 Dataset Classes Configuration
# ============================================================================

# Standard CIFAR-10 classes (initial 10 classes) - Currently Supported
CIFAR10_CLASSES = [
    'airplane',      # 0
    'automobile',    # 1
    'bird',          # 2
    'cat',           # 3
    'deer',          # 4
    'dog',           # 5
    'frog',          # 6
    'horse',         # 7
    'ship',          # 8
    'truck',         # 9
]

CIFAR10_CLASSES_COUNT = len(CIFAR10_CLASSES)

# Extended CIFAR-10 classes (10 additional classes for CIFAR-20 concept)
# These are semantic extensions for broader classification coverage
CIFAR10_EXTENDED_CLASSES = [
    'aircraft',      # 10 - Related to airplane
    'bus',           # 11 - Related to automobile
    'train',         # 12 - Related to truck
    'boat',          # 13 - Related to ship
    'motorcycle',    # 14 - Related to automobile
    'bicycle',       # 15 - Related to vehicle
    'pet',           # 16 - Related to dog/cat
    'wild_animal',   # 17 - Related to deer/bird
    'building',      # 18 - Related to structures
    'vehicle_other', # 19 - Related to vehicles
]

# Combined 20-class dataset (for future expansion)
CIFAR20_CLASSES = CIFAR10_CLASSES + CIFAR10_EXTENDED_CLASSES
CIFAR20_CLASSES_COUNT = len(CIFAR20_CLASSES)

# Class Groups for Analysis
CLASS_GROUPS = {
    'vehicles': {
        'classes': ['airplane', 'automobile', 'bird', 'truck', 'ship'],
        'indices': [0, 1, 8, 9],
        'description': 'Vehicle-related classes'
    },
    'animals': {
        'classes': ['cat', 'dog', 'bird', 'deer', 'frog', 'horse'],
        'indices': [2, 3, 4, 5, 6, 7],
        'description': 'Animal-related classes'
    },
    'transport': {
        'classes': ['airplane', 'automobile', 'truck', 'ship'],
        'indices': [0, 1, 8, 9],
        'description': 'Transportation classes'
    },
    'living': {
        'classes': ['cat', 'dog', 'bird', 'deer', 'frog', 'horse'],
        'indices': [2, 3, 4, 5, 6, 7],
        'description': 'Living creatures'
    },
}

# Class-specific Metrics Tracking
CLASS_METRICS_CONFIG = {
    'per_class_accuracy': True,
    'per_class_precision': True,
    'per_class_recall': True,
    'per_class_f1': True,
    'confusion_matrix': True,
    'per_epoch_tracking': True,
    'initial_classes': 10,  # Start with CIFAR-10
    'extended_classes': 10, # Add CIFAR extended classes in Phase 2
}

# ============================================================================
# Prompt Configuration
# ============================================================================

# Prompt Configuration for Model Detection & LR Extraction
# These prompts help identify which model is active and extract hyperparameters
PROMPT_CONFIG = {
    # Model Detection Prompt - Identify which model is currently active
    'model_detection': {
        'prefix': 'model_',
        'active_indicator': '<<<ACTIVE>>>',
        'detection_keywords': [
            'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101',
            'MobileNetV2', 'EfficientNet', 'DenseNet', 'VGG'
        ],
        'prompt_template': (
            "Which prefix model is currently active for fine-tuning?\n"
            "Format: {prefix}_{model_name}\n"
            "Example: model_ResNet18 <<<ACTIVE>>>"
        ),
        'response_format': 'model_<NAME> <<<ACTIVE>>>',
    },
    
    # Learning Rate Extraction - Extract LR from model configuration
    'lr_extraction': {
        'keywords': ['learning_rate', 'lr', 'learning rate', 'base_lr', 'initial_lr'],
        'default_lr': 0.01,  # Conservative default
        'lr_range': [0.0001, 1.0],
        'extraction_sources': [
            'model_hyperparameters',
            'training_config',
            'scheduler_config',
            'from_database'
        ],
        'prompt_template': (
            "What is the learning rate for {model_name}?\n"
            "Expected range: {lr_range}\n"
            "Return format: lr={value}"
        ),
        'response_validation': {
            'type': 'float',
            'min_value': 0.0001,
            'max_value': 1.0,
        },
    },
    
    # Database Parameter Extraction
    'database_extraction': {
        'query_prefix': 'SELECT * FROM scheduler_results WHERE',
        'storage_prefix': 'INSERT INTO scheduler_results',
        'key_parameters': [
            'model_name',
            'scheduler_type',
            'learning_rate',
            'epoch',
            'accuracy',
            'loss'
        ],
        'prompt_template': (
            "Extract scheduler metrics for model {model_name} from database.\n"
            "Required parameters: {key_parameters}\n"
            "Filter by epoch <= {epoch_max}"
        ),
    },
    
    # Fine-tuning Workflow Detection
    'finetune_workflow': {
        'workflow_steps': [
            'load_pretrained_model',
            'freeze_backbone',
            'add_classification_head',
            'set_learning_rate',
            'initialize_scheduler',
            'train_with_validation',
            'save_checkpoint',
            'evaluate_metrics',
        ],
        'statistics_collection': [
            'training_accuracy',
            'validation_accuracy',
            'training_loss',
            'validation_loss',
            'learning_rate_progression',
            'per_class_accuracy',
        ],
        'prompt_template': (
            "Fine-tune {model_name} with {scheduler_type} scheduler.\n"
            "Learning rate: {learning_rate}\n"
            "Steps: {workflow_steps}\n"
            "Collect statistics: {statistics_collection}"
        ),
    },
}

# ============================================================================
# Hyperparameter Defaults
# ============================================================================

DEFAULT_HYPERPARAMS = {
    'learning_rate': 0.1,
    'momentum': 0.9,
    'dropout': 0.2,
    'weight_decay': 1e-4,
    'epoch_max': 90,
    'batch_size': 128,
    'warmup_epochs': 0.1,
    'step_size': 30,
    'gamma': 0.1,
    'T_0': 10,
    'T_mult': 2,
    'eta_min': 0.0,
    'min_lr': 0.01,
}

# ============================================================================
# Output Configuration
# ============================================================================

OUTPUT_DIRS = {
    'base': 'out/nngpt/llm/epoch/A0/synth_nn',
    'models': 'out/nngpt/llm/epoch/A0/synth_nn/models',
    'results': 'out/nngpt/llm/epoch/A0/results',
    'logs': 'out/nngpt/llm/epoch/A0/logs',
}

# ============================================================================
# File Naming Conventions
# ============================================================================

FILE_CONVENTIONS = {
    'model_prefix': 'A0_',
    'model_file': 'new_nn.py',
    'hp_file': 'hp.txt',
    'results_file': 'results.json',
    'log_file': 'training.log',
}

# ============================================================================
# Feature Flags - Control System Behavior
# ============================================================================

FEATURES = {
    # Scheduler Features
    'enable_huggingface_schedulers': True,
    'enable_pytorch_schedulers': True,
    
    # Model Support Features
    'enable_multi_model_support': True,      # Now enabled for ResNet variants
    'enable_model_prefix_logic': True,       # Support model_XXX prefix tracking
    'enable_model_fine_tuning': True,        # Enable fine-tuning workflow
    'enable_additional_models': False,       # Phase 2: Additional models
    
    # Class/Dataset Features
    'enable_extended_classes': False,        # Phase 2: Support 20 classes
    'enable_per_class_metrics': True,        # Track per-class accuracy
    'enable_class_groups': True,             # Enable class grouping analysis
    
    # Database Features
    'enable_database_caching': True,
    'enable_class_data_storage': True,
    'enable_fine_tune_tracking': True,
    
    # Analysis Features
    'enable_convergence_analysis': True,
    'enable_scheduler_comparison': True,
    'enable_hyperparameter_optimization': False,  # Phase 2
    
    # Logging & Reporting
    'enable_detailed_logging': True,
    'enable_model_summaries': True,
}
