import multiprocessing
import os

# List of specific models to mutate (leave empty to mutate all)
SPECIFIC_MODELS = ["AlexNet"]

DEBUG_MODE = False
# When True, keeps temporary model source files for debugging.
KEEP_TEMP_MODEL_FILES = False
NUM_ATTEMPTS_PER_MODEL = 2000
PRODUCER_SEARCH_DEPTH = 10
PLANS_OUTPUT_DIR = "mutation_plans"
NUM_WORKERS = multiprocessing.cpu_count()
MAX_CORES_TO_USE = 8
VALID_CHANNEL_SIZES = [n for n in range(4, 1025)]

# Root folder where mutated models are written (can be changed by the user)
# Default: within this repository under mutator/nn-dataset/mutated_models
# To write into the top-level nn-dataset repo instead, set this to, e.g.:
#   os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'nn-dataset', 'mutated_models'))
MUTATED_MODELS_OUTPUT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'nn-dataset', 'mutated_models')
)

# --- OPTIONAL: SAVE MUTATED MODELS TO LEMUR DB ---
# Toggle DB saving for each successfully generated mutated model
SAVE_MUTATED_TO_DB = False
# Required identifiers when saving to DB (set these to match your dataset)
DB_TASK = None          # e.g., 'img-classification'
DB_DATASET = None       # e.g., 'CIFAR10'
DB_METRIC = None        # e.g., 'acc'
# A prefix added to model name before uuid hash, resulting name: f"{prefix}-{uuid4(nn_code)}"
DB_MODEL_PREFIX = "mutated"
# Minimal training parameters required by nn-dataset during check_nn/train_new
# Ensure 'batch', 'epoch', and 'transform' are provided and valid for your dataset
DB_TRAIN_PRM = {
    # 'batch': 64,
    # 'epoch': 1,
    # 'transform': 'CIFAR10-Default',
    # 'num_workers': 2,
}
# Optional: cap per-epoch runtime during DB check (minutes)
DB_EPOCH_LIMIT_MINUTES = 10

# --- MUTATION TYPE CONFIGURATIONS ---
# Activation function mutation mappings
ACTIVATION_MUTATIONS = {
    'ReLU': ['GELU', 'ELU', 'LeakyReLU', 'SiLU'],  # Using SiLU instead of Swish
    'GELU': ['ReLU', 'ELU', 'SiLU'],
    'ELU': ['ReLU', 'GELU', 'LeakyReLU', 'SiLU'],
    'LeakyReLU': ['ReLU', 'GELU', 'ELU', 'SiLU'],
    'SiLU': ['ReLU', 'GELU', 'ELU'],  # SiLU is PyTorch's Swish
    'Tanh': ['ReLU', 'GELU', 'SiLU'],
    'Sigmoid': ['ReLU', 'GELU', 'Tanh']
}

# Layer type mutation mappings  
LAYER_TYPE_MUTATIONS = {
    'BatchNorm2d': ['GroupNorm', 'LayerNorm', 'InstanceNorm2d'],
    'GroupNorm': ['BatchNorm2d', 'LayerNorm', 'InstanceNorm2d'],
    'LayerNorm': ['BatchNorm2d', 'GroupNorm'],
    'InstanceNorm2d': ['BatchNorm2d', 'GroupNorm'],
    'MaxPool2d': ['AvgPool2d', 'AdaptiveMaxPool2d', 'AdaptiveAvgPool2d'],
    'AvgPool2d': ['MaxPool2d', 'AdaptiveMaxPool2d', 'AdaptiveAvgPool2d'],
}

# Mutation type weights (probability distribution)
#
# There are two channel mutation modes:
#   'dimension'       - DIRECT mode: edits the channel literal at the layer
#                       instantiation site (nn.Conv2d / nn.Linear argument).
#                       Works for flattened models (e.g. AlexNet) and for
#                       loops whose constructor call already carries a literal.
#   'channel_config'  - BRACKET mode: edits one entry inside a channel
#                       configuration container (e.g. channels = [64,128,256,512]
#                       or init_block_channels = 64) that is then consumed by a
#                       loop / builder function (e.g. AirNet.build_features).
#                       This is the only mode that produces independent,
#                       per-stage channel changes for models whose Conv2d calls
#                       live inside reusable block classes.
MUTATION_TYPE_WEIGHTS = {
    'dimension': 1.00,       # 100% - direct channel/feature size edits
    'channel_config': 0.0,   # 0%  - bracket channel-configuration edits
    'activation': 0.0,      # 0% - no activation function mutations
    'layer_type': 0.0,      # 0% - no layer type mutations
    'kernel_size': 0.0,     # 0% - no kernel size mutations
    'stride': 0.0,          # 0% - no stride mutations
    'architectural': 0.0    # 0% - no architectural mutations
}

# --- CHANNEL CONFIGURATION (BRACKET) MUTATION ---
# Names that identify a channel configuration container in the Net class.
# A candidate is:
#   * an assignment to one of these names whose value is a list/tuple of ints
#     (e.g. channels = [64, 128, 256, 512]); each list entry is mutable, or
#   * an assignment to one of these names whose value is a single int
#     (e.g. init_block_channels = 64).
# Matching is case-insensitive substring matching against the assignment target.
CHANNEL_CONFIG_NAME_PATTERNS = [
    'channel', 'channels', 'plane', 'planes', 'width', 'widths',
    'filter', 'filters', 'feature', 'features', 'dim', 'dims',
    'depth', 'stage', 'block_setting', 'block_spec', 'cfg',
    'hidden', 'out_channels', 'in_channels', 'init_block',
]

# --- DIMENSION MUTATION STRATEGY ---
# Probability distribution for choosing forward (producer-led) vs.
# backward (consumer-led) dimension propagation.
PROPAGATION_DIRECTION_WEIGHTS = {
    'forward': 0.5,  # 70% chance for forward propagation
    'backward': 0.5   # 30% chance for backward propagation
}

# Kernel size mutations for Conv2d layers
KERNEL_SIZE_MUTATIONS = {
    'Conv2d': {
        1: [3, 5],
        3: [1, 5, 7],
        5: [3, 7],
        7: [3, 5, 9],
        9: [7, 11],
        11: [7, 9, 13]
    }
}

# Stride mutations for Conv2d layers
STRIDE_MUTATIONS = {
    'Conv2d': {
        1: [2],
        2: [1, 3],
        3: [2, 4],
        4: [2, 3]
    }
}

# Padding strategies
PADDING_STRATEGIES = ['same', 'valid'] # 'custom' might be too complex for now

# --- HELPER FUNCTION MUTATION CONTROL ---
# Controls whether helper function calls (like conv3x3()) should be mutation targets
# True: Mutate helper function calls (current behavior) - allows indirect mutations
# False: Only mutate direct nn.Module instantiations - more semantically correct
ALLOW_HELPER_FUNCTION_MUTATIONS = False

# Helper function patterns to detect (used when ALLOW_HELPER_FUNCTION_MUTATIONS = False)
HELPER_FUNCTION_PATTERNS = [
    'conv1x1', 'conv3x3', 'conv5x5', 'conv7x7',      # Convolution helpers
    'make_layer', 'make_block', 'make_stage',          # Layer builders  
    'build_', 'create_', 'get_',                       # Factory functions
    'downsample', 'upsample',                          # Sampling helpers
    'Block', 'Bottleneck', 'BasicBlock', 'DPNBlock',   # Common block patterns
]

# Top-level class patterns to identify Net classes for fixed-number mutations
TOP_LEVEL_CLASS_PATTERNS = ['Net']

# Mutation mode configuration
# Options: 'auto' (context-aware), 'always_symbolic', 'always_fixed'
MUTATION_MODE = 'always_fixed'

# Symbolic mutation weights (probability distribution for symbolic vs fixed mutations)
# Only used when MUTATION_MODE is 'auto'
# For your use case, set symbolic weight to 0.9 to ensure most mutations use symbolic expressions
# This will prevent dimensional mismatches in helper blocks like BasicBlock
SYMBOLIC_MUTATION_WEIGHTS = {
    'symbolic': 1.0,   # 100% chance of symbolic mutations
    'fixed': 0.0       # 0% chance of fixed-number mutations
}

# Available symbolic operations for parameter-based mutations
SYMBOLIC_OPERATIONS = ['*', '//', '+', '-', '<<', '>>']  # Added bit shifts for more variety
SYMBOLIC_OPERANDS = list(range(1, 65))  # Expanded to include more values for free combinations

# --- CONVNEXT COMPATIBILITY SETTINGS ---
# Modules that are problematic for torch.fx symbolic tracing
FX_INCOMPATIBLE_MODULES = [
    'StochasticDepth', 'LayerNorm2d', 'Permute'
]

# Alternative modules for FX-incompatible ones during mutation
FX_COMPATIBLE_REPLACEMENTS = {
    'StochasticDepth': 'Dropout',           # Replace with standard Dropout
    'LayerNorm2d': 'BatchNorm2d',          # Replace with BatchNorm2d
    'Permute': 'Identity',                 # Replace with Identity (no-op)
}

# ConvNeXT-specific mutation patterns
CONVNEXT_MUTATIONS = {
    # Depth-wise convolution mutations
    'depthwise_conv': {
        'kernel_sizes': [3, 5, 7],         # Alternative kernel sizes
        'group_ratios': [1, 0.5, 1.0],     # groups=1 (standard), groups=dim//2, groups=dim
    },
    # MLP expansion ratios
    'mlp_expansion': [2, 4, 6, 8],         # Alternative expansion ratios
}

# --- ARCHITECTURAL MUTATION SETTINGS ---
# Prioritize high-level architectural changes in Net class
PRIORITIZE_ARCHITECTURAL_MUTATIONS = True

# High-level architectural patterns to target
ARCHITECTURAL_MUTATIONS = {
    # ConvNeXT block configurations
    'convnext_block_settings': {
        # [input_channels, output_channels, num_layers] alternatives
        'stage_configs': [
            # Original: [96, 192, 3], [192, 384, 3], [384, 768, 9], [768, None, 3]
            # Variant 1: Smaller model
            [[64, 128, 2], [128, 256, 2], [256, 512, 6], [512, None, 2]],
            # Variant 2: Different layer depths
            [[96, 192, 2], [192, 384, 4], [384, 768, 12], [768, None, 2]],
            # Variant 3: More stages
            [[48, 96, 2], [96, 192, 3], [192, 384, 6], [384, 768, 6], [768, None, 3]],
            # Variant 4: Wider channels
            [[128, 256, 3], [256, 512, 3], [512, 1024, 9], [1024, None, 3]],
        ]
    },
    # Fixed parameter mutations
    'fixed_parameters': {
        'stochastic_depth_prob': [0.05, 0.1, 0.15, 0.2],  # Alternative dropout rates
        'layer_scale': [1e-6, 1e-5, 1e-4, 1e-7],          # Alternative layer scales
        'kernel_sizes': [3, 4, 5, 6],                      # Alternative stem kernel sizes
        'strides': [2, 4, 6],                              # Alternative stem strides
    }
}