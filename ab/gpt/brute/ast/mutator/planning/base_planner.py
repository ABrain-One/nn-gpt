"""
Base model planner containing core planning logic and the main ModelPlanner class.
"""

import random
import hashlib
import os
import json
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Any
import operator

import torch
import torch.nn as nn
import torch.fx as fx

from mutator import config
from mutator.utils import (
    ModuleSourceTracer,
    get_available_parameters,
    read_source_file,
    find_call_node_at_line,
)
from .dimension_planner import DimensionPlanner
from .channel_config_planner import ChannelConfigPlanner
from .activation_planner import ActivationPlanner
from .layer_planner import LayerTypePlanner
from .spatial_planner import SpatialPlanner
from .architectural_planner import ArchitecturalPlanner
from .fallback_planner import FallbackPlanner
from mutator.utils.fx_graph_utils import get_detailed_graph


class ModelPlanner:
    """
    Main model planner class that coordinates different types of mutations.
    
    This class serves as the central coordinator for planning various types of
    neural network mutations including dimension changes, activation swaps,
    layer type changes, and architectural modifications.
    """
    
    MUTABLE_MODULES = (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm)
    ACTIVATION_MODULES = (nn.ReLU, nn.GELU, nn.ELU, nn.LeakyReLU, nn.Tanh, nn.Sigmoid, nn.SiLU)
    
    def __init__(self, model: nn.Module, source_map: dict = None, search_depth: int = 3,
                 source_code: Optional[str] = None):
        """
        Initialize the model planner.
        
        Args:
            model: The PyTorch model to plan mutations for
            source_map: Optional mapping of module names to source locations
            search_depth: Depth for graph traversal operations
            source_code: Optional full source of the model, required for
                channel-configuration (bracket) mutations
        """
        self.original_model = model
        self.source_map = source_map or {}
        self.search_depth = search_depth
        self.source_code = source_code
        self.plan = {}
        
        # Correctly load VALID_CHANNEL_SIZES from config.py
        self.VALID_CHANNEL_SIZES = config.VALID_CHANNEL_SIZES
        
        # Initialize specialized planners
        self.dimension_planner = DimensionPlanner(self)
        self.channel_config_planner = ChannelConfigPlanner(self, source_code=source_code)
        self.activation_planner = ActivationPlanner(self)
        self.layer_planner = LayerTypePlanner(self)
        self.spatial_planner = SpatialPlanner(self)
        self.architectural_planner = ArchitecturalPlanner(self)
        self.fallback_planner = FallbackPlanner(self)
        
        # Initialize model analysis
        self._initialize_model_analysis()
    
    def _initialize_model_analysis(self):
        """Initialize model analysis components."""
        try:
            # Use our custom utility to get a detailed graph
            self.graph = get_detailed_graph(self.original_model)
            self.submodules = dict(self.original_model.named_modules())
            self.fx_compatible = True
            
            if config.DEBUG_MODE:
                print("[ModelPlanner] Successfully created FX graph for model analysis")
                
        except Exception as e:
            # Fallback for FX-incompatible models
            if config.DEBUG_MODE:
                print(f"[ModelPlanner] FX tracing failed: {e}. Using fallback mode.")
            self.graph = None
            self.submodules = dict(self.original_model.named_modules())
            self.fx_compatible = False
        
        # Initialize spatial tracking if input shape is available
        self.spatial_tracker = {}
        if hasattr(config, 'INPUT_SHAPE') and config.INPUT_SHAPE:
            self.input_shape = config.INPUT_SHAPE
        else:
            # Default input shape for common vision models
            self.input_shape = (3, 224, 224)

    def _validate_spatial_change(self, module_name: str, new_params: dict) -> bool:
        """Validate if mutation maintains valid spatial dimensions."""
        if module_name not in self.spatial_tracker:
            return True  # Skip validation if no dimension info

        current_h, current_w = self.spatial_tracker[module_name]
        input_h, input_w = current_h, current_w

        module = self.submodules.get(module_name)
        if module is None:
            return True

        # Extract current params from module
        kernel_size = getattr(module, 'kernel_size', 1)
        stride = getattr(module, 'stride', 1)
        padding = getattr(module, 'padding', 0)
        dilation = getattr(module, 'dilation', 1)

        # Override with proposed new params
        if 'kernel_size' in new_params:
            kernel_size = new_params['kernel_size']
        if 'stride' in new_params:
            stride = new_params['stride']
        if 'padding' in new_params:
            padding = new_params['padding']
        if 'dilation' in new_params:
            dilation = new_params['dilation']

        # Normalize to tuples
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        new_h = (input_h + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
        new_w = (input_w + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1

        return new_h >= 1 and new_w >= 1

    def _compute_spatial_dimensions(self):
        """Compute spatial dimensions for all layers in the network."""
        current_h, current_w = self.input_shape[1], self.input_shape[2]

        for name, module in self.original_model.named_modules():
            if name == '':
                continue

            # Store current dimensions
            self.spatial_tracker[name] = (current_h, current_w)

            # Update dimensions based on layer type
            if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
                kernel_size = module.kernel_size
                stride = module.stride
                padding = module.padding
                dilation = module.dilation

                if isinstance(kernel_size, int):
                    kernel_size = (kernel_size, kernel_size)
                if isinstance(stride, int):
                    stride = (stride, stride)
                if isinstance(padding, int):
                    padding = (padding, padding)
                if isinstance(dilation, int):
                    dilation = (dilation, dilation)

                current_h = (current_h + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
                current_w = (current_w + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1

                current_h = max(1, current_h)
                current_w = max(1, current_w)
            elif isinstance(module, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
                if isinstance(module.output_size, int):
                    current_h = module.output_size
                    current_w = module.output_size
                else:
                    current_h, current_w = module.output_size
            elif isinstance(module, nn.Linear):
                current_h, current_w = 1, 1

            self.spatial_tracker[name] = (current_h, current_w)

    def plan_random_mutation(self) -> Dict[str, Any]:
        """
        Plan a random mutation for the model.
        
        Returns:
            Dictionary containing the mutation plan
        """
        if config.DEBUG_MODE:
            print("[ModelPlanner] Planning random mutation...")
        
        # Check if model is FX-compatible for advanced mutations
        if not self.fx_compatible:
            if config.DEBUG_MODE:
                print("[ModelPlanner] Using fallback mutation planning for FX-incompatible model")
            return self.fallback_planner.plan_fallback_mutation()
        
        # Choose mutation type based on configuration weights
        mutation_types = list(config.MUTATION_TYPE_WEIGHTS.keys())
        weights = list(config.MUTATION_TYPE_WEIGHTS.values())
        
        # Filter out types that aren't applicable to this model
        applicable_types = []
        applicable_weights = []
        
        for mut_type, weight in zip(mutation_types, weights):
            if self._is_mutation_type_applicable(mut_type):
                applicable_types.append(mut_type)
                applicable_weights.append(weight)
        
        if not applicable_types:
            if config.DEBUG_MODE:
                print("[ModelPlanner] No applicable mutation types found")
            return {}
        
        # Select mutation type using weighted random choice
        selected_type = random.choices(applicable_types, weights=applicable_weights, k=1)[0]
        
        if config.DEBUG_MODE:
            print(f"[ModelPlanner] Selected mutation type: {selected_type}")
        
        # Delegate to appropriate specialized planner
        if selected_type == 'dimension':
            return self.dimension_planner.plan_dimension_mutation()
        elif selected_type == 'channel_config':
            return self.channel_config_planner.plan_channel_config_mutation()
        elif selected_type == 'activation':
            return self.activation_planner.plan_activation_mutation()
        elif selected_type == 'layer_type':
            return self.layer_planner.plan_layer_type_mutation()
        elif selected_type == 'kernel_size':
            return self.spatial_planner.plan_kernel_size_mutation()
        elif selected_type == 'stride':
            return self.spatial_planner.plan_stride_mutation()
        elif selected_type == 'architectural':
            return self.architectural_planner.plan_architectural_mutation()
        else:
            # Fallback to dimension mutation
            return self.dimension_planner.plan_dimension_mutation()
    
    def _is_mutation_type_applicable(self, mutation_type: str) -> bool:
        """
        Check if a mutation type is applicable to the current model.
        
        Args:
            mutation_type: Type of mutation to check
            
        Returns:
            True if the mutation type can be applied to this model
        """
        if mutation_type == 'dimension':
            # Check for mutable modules
            return any(isinstance(module, self.MUTABLE_MODULES) 
                      for module in self.original_model.modules())

        elif mutation_type == 'channel_config':
            # Applicable when the source exposes a channel-config container
            return self.channel_config_planner.has_candidates()

        elif mutation_type == 'activation':
            # Check for activation modules
            return any(isinstance(module, self.ACTIVATION_MODULES)
                      for module in self.original_model.modules())
        
        elif mutation_type == 'layer_type':
            # Check for normalization/pooling layers
            target_types = ['BatchNorm2d', 'LayerNorm', 'GroupNorm', 'MaxPool2d', 'AvgPool2d']
            return any(type(module).__name__ in target_types
                      for module in self.original_model.modules())
        
        elif mutation_type in ['kernel_size', 'stride']:
            # Check for Conv2d layers
            return any(isinstance(module, nn.Conv2d)
                      for module in self.original_model.modules())
        
        elif mutation_type == 'architectural':
            # Architectural mutations are always applicable
            return True
        
        return False

    def apply_plan(self) -> nn.Module:
        """
        Apply the current mutation plan to create a new model.
        
        Returns:
            New model with mutations applied
            
        Raises:
            ValueError: If no mutation plan exists
        """
        if not self.plan:
            raise ValueError("No mutation plan exists. Please run 'plan_random_mutation()' first.")
        
        new_model = deepcopy(self.original_model)
        
        for name, details in self.plan.items():
            try:
                original_module = new_model.get_submodule(name)
                mutation_type = details.get("mutation_type", "dimension")  # backward compatibility
                
                if mutation_type == "dimension":
                    mutated_copy = self._create_mutated_copy(
                        original_module, 
                        details["new_in"], 
                        details["new_out"]
                    )
                elif mutation_type == "activation":
                    mutated_copy = self._create_activation_mutation(
                        original_module, 
                        details["new_activation"]
                    )
                elif mutation_type == "layer_type":
                    mutated_copy = self._create_layer_type_mutation(
                        original_module, 
                        details["new_layer_type"], 
                        details["mutation_params"]
                    )
                else:
                    continue  # skip unknown mutation types
                    
                self._set_nested_attr(new_model, name, mutated_copy)
                
            except AttributeError:
                continue
                
        return new_model

    def clear_plan(self):
        """Clear the current mutation plan."""
        self.plan = {}

    # --- Helper wrappers used by planners ---
    def _get_source_code_for_location(self, source_location: dict) -> str:
        """Return source code string for a given source location dict.

        Expects a mapping like { 'filename': <path>, 'lineno': <int>, ... }.
        """
        try:
            filename = source_location.get('filename') if isinstance(source_location, dict) else None
            if not filename:
                return ""
            return read_source_file(filename)
        except Exception:
            return ""

    def _find_call_node_at_line(self, source_code: str, lineno: int):
        """Delegate to utils to find the AST Call node at a given line."""
        return find_call_node_at_line(source_code, lineno)

    @classmethod
    def _create_mutated_copy(cls, module: nn.Module, new_in_channels, new_out_channels):
        """Create a mutated copy of a module with new dimensions."""
        if not isinstance(module, cls.MUTABLE_MODULES): 
            return deepcopy(module)
        
        if isinstance(module, nn.Conv2d):
            old_out, old_in = module.out_channels, module.in_channels
            new_in = new_in_channels or old_in
            new_out = new_out_channels or old_out
            groups = module.groups
            
            if (new_in != old_in or new_out != old_out) and groups > 1:
                if new_in % groups != 0 or new_out % groups != 0: 
                    groups = 1
            
            new_module = nn.Conv2d(
                in_channels=new_in, 
                out_channels=new_out, 
                kernel_size=module.kernel_size, 
                stride=module.stride, 
                padding=module.padding, 
                dilation=module.dilation, 
                groups=groups, 
                bias=module.bias is not None
            )
            
            min_out, min_in = min(old_out, new_out), min(old_in, new_in)
            copy_in_channels = min_in // (module.groups // new_module.groups)
            
            new_module.weight.data.zero_()
            new_module.weight.data[:min_out, :copy_in_channels, ...] = module.weight.data[:min_out, :copy_in_channels, ...]
            
            if module.bias is not None: 
                new_module.bias.data.zero_()
                new_module.bias.data[:min_out] = module.bias.data[:min_out]
                
        elif isinstance(module, nn.Linear):
            old_out, old_in = module.out_features, module.in_features
            new_in = new_in_channels or old_in
            new_out = new_out_channels or old_out
            
            new_module = nn.Linear(
                in_features=new_in, 
                out_features=new_out, 
                bias=module.bias is not None
            )
            
            min_out, min_in = min(old_out, new_out), min(old_in, new_in)
            
            new_module.weight.data.zero_()
            new_module.weight.data[:min_out, :min_in] = module.weight.data[:min_out, :min_in]
            
            if module.bias is not None: 
                new_module.bias.data.zero_()
                new_module.bias.data[:min_out] = module.bias.data[:min_out]
                
        elif isinstance(module, nn.BatchNorm2d):
            old_feats = module.num_features
            new_feats = new_in_channels or old_feats
            
            new_module = nn.BatchNorm2d(
                num_features=new_feats, 
                eps=module.eps, 
                momentum=module.momentum, 
                affine=module.affine, 
                track_running_stats=module.track_running_stats
            )
            
            min_feats = min(old_feats, new_feats)
            
            if new_module.track_running_stats:
                new_module.running_mean.data.zero_()
                new_module.running_var.data.fill_(1)
                new_module.running_mean.data[:min_feats] = module.running_mean.data[:min_feats]
                new_module.running_var.data[:min_feats] = module.running_var.data[:min_feats]
                
            if new_module.affine:
                new_module.weight.data.fill_(1)
                new_module.bias.data.zero_()
                new_module.weight.data[:min_feats] = module.weight.data[:min_feats]
                new_module.bias.data[:min_feats] = module.bias.data[:min_feats]
                
        elif isinstance(module, nn.LayerNorm):
            old_feats = module.normalized_shape[0]
            new_feats = new_in_channels or old_feats
            
            new_module = nn.LayerNorm(
                normalized_shape=[new_feats], 
                eps=module.eps, 
                elementwise_affine=module.elementwise_affine
            )
            
            min_feats = min(old_feats, new_feats)
            
            if new_module.elementwise_affine:
                new_module.weight.data.fill_(1)
                new_module.bias.data.zero_()
                new_module.weight.data[:min_feats] = module.weight.data[:min_feats]
                new_module.bias.data[:min_feats] = module.bias.data[:min_feats]
                
        return new_module

    def _create_activation_mutation(self, module: nn.Module, new_activation: str) -> nn.Module:
        """Create a new activation module with the specified type."""
        # Preserve common parameters where possible
        inplace = getattr(module, 'inplace', True)
        
        if new_activation == 'ReLU':
            return nn.ReLU(inplace=inplace)
        elif new_activation == 'GELU':
            return nn.GELU()
        elif new_activation == 'ELU':
            return nn.ELU(inplace=inplace)
        elif new_activation == 'LeakyReLU':
            return nn.LeakyReLU(inplace=inplace)
        elif new_activation == 'SiLU':
            return nn.SiLU(inplace=inplace)
        elif new_activation == 'Tanh':
            return nn.Tanh()
        elif new_activation == 'Sigmoid':
            return nn.Sigmoid()
        else:
            return nn.ReLU(inplace=inplace)  # fallback

    def _create_layer_type_mutation(self, module: nn.Module, new_layer_type: str, params: dict) -> nn.Module:
        """Create a new layer module with the specified type."""
        if new_layer_type == 'BatchNorm2d':
            return nn.BatchNorm2d(num_features=params['num_features'])
        elif new_layer_type == 'GroupNorm':
            return nn.GroupNorm(num_groups=params['num_groups'], num_channels=params['num_channels'])
        elif new_layer_type == 'LayerNorm':
            return nn.LayerNorm(normalized_shape=params.get('normalized_shape', [params['num_features']]))
        elif new_layer_type == 'InstanceNorm2d':
            return nn.InstanceNorm2d(num_features=params['num_features'])
        elif new_layer_type == 'MaxPool2d':
            return nn.MaxPool2d(
                kernel_size=params['kernel_size'],
                stride=params['stride'],
                padding=params['padding']
            )
        elif new_layer_type == 'AvgPool2d':
            return nn.AvgPool2d(
                kernel_size=params['kernel_size'],
                stride=params['stride'],
                padding=params['padding']
            )
        elif new_layer_type == 'AdaptiveMaxPool2d':
            return nn.AdaptiveMaxPool2d(output_size=params['output_size'])
        elif new_layer_type == 'AdaptiveAvgPool2d':
            return nn.AdaptiveAvgPool2d(output_size=params['output_size'])
        else:
            return deepcopy(module)  # fallback

    @staticmethod
    def _set_nested_attr(obj: nn.Module, name: str, value: nn.Module):
        """Set a nested attribute on an object."""
        parts = name.split('.')
        parent = obj
        for part in parts[:-1]: 
            parent = getattr(parent, part)
        setattr(parent, parts[-1], value)

    @staticmethod
    def get_model_checksum(model: nn.Module) -> str:
        """Generate a checksum for the model structure."""
        try:
            if isinstance(model, fx.GraphModule): 
                graph_repr = model.graph.print_tabular()
            else: 
                graph_repr = fx.symbolic_trace(model).graph.print_tabular()
            return hashlib.sha256(graph_repr.encode()).hexdigest()
        except: 
            return os.urandom(16).hex()

    @staticmethod
    def get_model_parameter_checksum(model: nn.Module) -> str:
        """Generate checksum based on model parameters for FX-incompatible models."""
        try:
            # Collect parameter shapes and names to create a structural signature
            param_info = []
            for name, param in model.named_parameters():
                param_info.append(f"{name}:{param.shape}")
            
            # Also collect module types and names for additional structural info
            module_info = []
            for name, module in model.named_modules():
                if len(name) > 0:  # Skip root module
                    module_info.append(f"{name}:{type(module).__name__}")
            
            # Create combined signature
            signature = "|".join(param_info) + "||" + "|".join(module_info)
            return hashlib.sha256(signature.encode()).hexdigest()
        except Exception:
            return os.urandom(16).hex()