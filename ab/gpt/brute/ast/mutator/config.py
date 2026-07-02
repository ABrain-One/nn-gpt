import multiprocessing
import os

# List of specific models to mutate (leave empty to mutate all)
# Denoise NAS seeds — GOOD models (all have global residual + decent PSNR), free from the
# existing pool. 7F best, 9F 2nd, 5F tiny/efficient, 8F mobile, 3F/4F/10F mid, LiteDenoiseNet
# the group's strongest (37.66 dB). WEAK seeds 1F/2F/6F (no global residual) are intentionally
# EXCLUDED until the skip-connection operator exists — they're reserved for the
# skip before/after experiment (see plan), since the current operators can't add the residual.
SPECIFIC_MODELS = [
    "DenoiseUnet7F", "DenoiseUnet9F", "DenoiseUnet5F", "DenoiseUnet8F",
    "DenoiseUnet3F", "DenoiseUnet4F", "DenoiseUnet10F", "LiteDenoiseNet",
]

DEBUG_MODE = True  # smoke run: single-process + verbose so rejection reasons are visible
# When True, keeps temporary model source files for debugging.
KEEP_TEMP_MODEL_FILES = False
NUM_ATTEMPTS_PER_MODEL = 10  # smoke run; scale to 50-200 once pipeline is verified
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
# Denoise NAS: generation is decoupled from eval (our search driver calls check_nn),
# so SAVE_MUTATED_TO_DB stays False; these are set for when/if DB-eval is enabled.
DB_TASK = "img-denoising"
DB_DATASET = "denoise"
DB_METRIC = "psnr"
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
# Phase 1 (component mutations): biased toward shape-preserving ops (activation, norm
# swaps) so fully-convolutional denoise nets yield valid candidates; dimension/stride
# often break skip-connection shape matching and get rejected by the validator (expected).
# architectural stays 0.0 — those operators are ConvNeXT-only (Phase 2 adds denoise topology).
MUTATION_TYPE_WEIGHTS = {
    'dimension': 0.10,
    'activation': 0.15,
    'layer_type': 0.15,
    'kernel_size': 0.10,
    'stride': 0.10,
    'architectural': 0.0,   # ConvNeXT-specific; not used for denoise
    'scale': 0.40,          # Phase 2: grow sub-net width/depth (capacity/topology + bigger models)
    'skip': 0.0,            # Phase 2: toggle global input->output residual in forward().
                            #   Kept 0.0 for the general search; raise it (e.g. via the CLI
                            #   `--only skip` or `--weights skip:1.0`) to search skip connections.
}

# Human-readable operator catalog — used by `cli.py --list-operators` and the README.
# When you add an operator (see README "Adding an operator"), add a line here too.
OPERATOR_DESCRIPTIONS = {
    'dimension':    'Change channel/feature widths of a conv/linear/norm (constructor kwarg).',
    'activation':   'Swap an activation module (ReLU/GELU/SiLU/ELU/...).',
    'layer_type':   'Swap a norm/pool layer type (BatchNorm/GroupNorm/LayerNorm, Max/AvgPool).',
    'kernel_size':  'Change a Conv2d kernel size.',
    'stride':       'Change a Conv2d stride.',
    'architectural':'ConvNeXT-only block/stage restructuring (not used for denoise).',
    'scale':        'Grow a sub-network width/depth constructor kwarg.',
    'skip':         'Toggle a global input->output residual in forward() (dataflow operator).',
}

# --- RUNTIME OVERRIDES (env vars) -------------------------------------------
# Let the CLI (cli.py) and batch/SLURM jobs configure a search WITHOUT editing this
# file. Because env vars are inherited by child processes, these overrides also reach
# multiprocessing workers (which re-import config) — so they work in parallel mode too,
# unlike assigning config attributes at runtime. Each applies only if its var is set.
import os as _os

def _env_list(name):
    v = _os.environ.get(name)
    return [s.strip() for s in v.split(',') if s.strip()] if v else None

def _env_int(name):
    v = _os.environ.get(name)
    return int(v) if v not in (None, '') else None

_v = _env_list('BRUTE_MODELS')
if _v is not None:
    SPECIFIC_MODELS = _v

_v = _env_int('BRUTE_ATTEMPTS')
if _v is not None:
    NUM_ATTEMPTS_PER_MODEL = _v

_v = _env_int('BRUTE_WORKERS')
if _v is not None:
    MAX_CORES_TO_USE = _v

if _os.environ.get('BRUTE_OUTPUT'):
    MUTATED_MODELS_OUTPUT_ROOT = _os.path.abspath(_os.environ['BRUTE_OUTPUT'])

if _os.environ.get('BRUTE_DEBUG') is not None:
    DEBUG_MODE = _os.environ['BRUTE_DEBUG'] not in ('0', '', 'false', 'False')

# BRUTE_ONLY=<op> isolates a single operator (weight 1.0, all others 0.0).
_only = _os.environ.get('BRUTE_ONLY')
if _only:
    MUTATION_TYPE_WEIGHTS = {k: (1.0 if k == _only else 0.0) for k in MUTATION_TYPE_WEIGHTS}
    MUTATION_TYPE_WEIGHTS.setdefault(_only, 1.0)

# BRUTE_WEIGHTS="skip:1.0,activation:0.5" overrides individual operator weights.
_weights = _os.environ.get('BRUTE_WEIGHTS')
if _weights:
    for _pair in _weights.split(','):
        if ':' in _pair:
            _k, _val = _pair.split(':', 1)
            MUTATION_TYPE_WEIGHTS[_k.strip()] = float(_val)

# --- PHASE 2: ARCHITECTURAL SCALE MUTATION (denoise capacity/topology) ---
# Grows scalar architectural hyperparameters of a sub-network constructor call.
SCALE_MUTATIONS = {
    # width-like params: multiplied by a width factor
    'width_params': ['num_filters', 'base_channels', 'base_ch', 'dim', 'width',
                     'embed_dim', 'hidden_dim', 'channels', 'nf', 'features'],
    'width_factors': [1.5, 2.0],
    'width_max': 256,
    # depth-like params: incremented
    'depth_params': ['depth', 'blocks_per_level', 'mid_blocks', 'num_blocks',
                     'n_blocks', 'num_layers', 'num_res_blocks', 'stages'],
    'depth_deltas': [1, 2],
    'depth_max': 10,
    'min_value': 1,
}

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