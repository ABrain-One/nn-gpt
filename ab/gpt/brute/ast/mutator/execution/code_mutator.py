"""
CodeMutator class for applying AST modifications to neural network source code.

This module provides the CodeMutator class which can apply various types of
modifications to PyTorch neural network source code through AST manipulation.
"""

import ast
from typing import Dict, Any, List, Optional

from mutator import config
from mutator.execution.constants import ARG_TO_POS_MAP
from mutator.execution.apply_dimension import apply_dimension_modification
from mutator.execution.apply_channel_config import apply_channel_config_modification
from mutator.execution.apply_symbolic import apply_symbolic_modification
from mutator.execution.apply_activation import apply_activation_modification
from mutator.execution.apply_layer_type import apply_layer_type_modification
from mutator.execution.apply_architectural import apply_architectural_modification
from mutator.execution.apply_spatial import apply_kernel_size_modification, apply_stride_modification


class CodeMutator(ast.NodeTransformer):
    """
    Applies AST modifications to neural network source code.
    
    This class can apply various types of modifications including:
    - Dimension modifications (channel/feature size changes)
    - Symbolic expression modifications
    - Activation function changes
    - Layer type conversions
    - Architectural modifications
    - Kernel size changes
    - Stride modifications
    """
    
    ARG_TO_POS_MAP = ARG_TO_POS_MAP

    def __init__(self, code_string: str):
        """
        Initialize the CodeMutator with source code.
        
        Args:
            code_string: The source code to modify
        """
        self.tree = ast.parse(code_string)
        self.modifications = []
        if config.DEBUG_MODE:
            print("[CodeMutator] Initialized.")

    def _validate_conv_params(self, arg_name: str, new_value: int, node: ast.Call) -> Dict[str, Any]:
        """
        Validate convolution parameters to prevent invalid depthwise config.
        
        Args:
            arg_name: The parameter name being modified
            new_value: The new value for the parameter
            node: The AST call node being modified
            
        Returns:
            Dictionary of additional modifications needed (e.g., groups parameter)
        """
        current_params = {}
        
        # Collect current parameters
        for kw in node.keywords:
            current_params[kw.arg] = getattr(kw.value, 'value', None)
        for i, arg in enumerate(node.args):
            if i == 0: current_params['in_channels'] = getattr(arg, 'value', None)
            if i == 1: current_params['out_channels'] = getattr(arg, 'value', None)
            if i == 5: current_params['groups'] = getattr(arg, 'value', 1)
        
        # Handle depthwise convolution constraint
        if (arg_name == 'out_channels' 
                and current_params.get('groups', 1) == current_params.get('in_channels')
                and new_value != current_params.get('in_channels')):
            return {'groups': 1}  # Convert to standard convolution
        return {}

    def schedule_modification(self, location: Dict[str, int], arg_name: str, new_value: Any) -> None:
        """
        Schedule a dimension-based modification (backward compatibility).
        
        Args:
            location: Source location information (lineno, col_offset)
            arg_name: Parameter name to modify
            new_value: New value for the parameter
        """
        if location and new_value is not None:
            mod = {
                'type': 'dimension',
                'location': location,
                'arg_name': arg_name,
                'new_value': new_value
            }
            self.modifications.append(mod)
            if config.DEBUG_MODE:
                print(f"[CodeMutator] Scheduled dimension modification: {mod}")

    def schedule_channel_config_modification(self, location: Dict[str, int], index: Optional[int], new_value: Any, name: str = "") -> None:
        """
        Schedule a channel-configuration (bracket / scalar) modification.

        Args:
            location: Source location of the container (list/tuple) or the
                scalar value node.
            index: Element index inside a list/tuple, or None for a scalar.
            new_value: New integer width.
            name: Optional container variable name (for diagnostics).
        """
        if location and new_value is not None:
            mod = {
                'type': 'channel_config',
                'location': location,
                'index': index,
                'new_value': new_value,
                'name': name,
            }
            self.modifications.append(mod)
            if config.DEBUG_MODE:
                print(f"[CodeMutator] Scheduled channel-config modification: {mod}")

    def schedule_symbolic_modification(self, location: Dict[str, int], arg_name: str, symbolic_expression: str) -> None:
        """
        Schedule a symbolic expression modification.
        
        Args:
            location: Source location information (lineno, col_offset)
            arg_name: Parameter name to modify
            symbolic_expression: Symbolic expression to apply
        """
        if location and symbolic_expression:
            mod = {
                'type': 'symbolic',
                'location': location,
                'arg_name': arg_name,
                'symbolic_expression': symbolic_expression
            }
            self.modifications.append(mod)
            if config.DEBUG_MODE:
                print(f"[CodeMutator] Scheduled symbolic modification: {mod}")

    def schedule_activation_modification(self, location: Dict[str, int], new_activation: str) -> None:
        """
        Schedule an activation function modification.
        
        Args:
            location: Source location information (lineno, col_offset)
            new_activation: New activation function name
        """
        if location and new_activation:
            mod = {
                'type': 'activation',
                'location': location,
                'new_activation': new_activation
            }
            self.modifications.append(mod)
            if config.DEBUG_MODE:
                print(f"[CodeMutator] Scheduled activation modification: {mod}")

    def schedule_layer_type_modification(self, location: Dict[str, int], new_layer_type: str, params: Dict[str, Any]) -> None:
        """
        Schedule a layer type modification.
        
        Args:
            location: Source location information (lineno, col_offset)
            new_layer_type: New layer type name
            params: Additional parameters for the layer type conversion
        """
        if location and new_layer_type:
            mod = {
                'type': 'layer_type',
                'location': location,
                'new_layer_type': new_layer_type,
                'params': params
            }
            self.modifications.append(mod)
            if config.DEBUG_MODE:
                print(f"[CodeMutator] Scheduled layer type modification: {mod}")

    def schedule_architectural_modification(self, location: Dict[str, int], architectural_type: str, params: Dict[str, Any]) -> None:
        """
        Schedule an architectural modification for high-level network structure.
        
        Args:
            location: Source location information (lineno, col_offset)
            architectural_type: Type of architectural modification
            params: Parameters for the architectural modification
        """
        if location and architectural_type:
            mod = {
                'type': 'architectural',
                'location': location,
                'architectural_type': architectural_type,
                'params': params
            }
            self.modifications.append(mod)
            if config.DEBUG_MODE:
                print(f"[CodeMutator] Scheduled architectural modification: {mod}")

    def schedule_kernel_size_modification(self, location: Dict[str, int], new_kernel_size: int) -> None:
        """
        Schedule a kernel size modification.
        
        Args:
            location: Source location information (lineno, col_offset)
            new_kernel_size: New kernel size value
        """
        if location and new_kernel_size:
            mod = {
                'type': 'kernel_size',
                'location': location,
                'new_kernel_size': new_kernel_size
            }
            self.modifications.append(mod)
            if config.DEBUG_MODE:
                print(f"[CodeMutator] Scheduled kernel size modification: {mod}")

    def schedule_stride_modification(self, location: Dict[str, int], new_stride: int) -> None:
        """
        Schedule a stride modification.
        
        Args:
            location: Source location information (lineno, col_offset)
            new_stride: New stride value
        """
        if location and new_stride:
            mod = {
                'type': 'stride',
                'location': location,
                'new_stride': new_stride
            }
            self.modifications.append(mod)
            if config.DEBUG_MODE:
                print(f"[CodeMutator] Scheduled stride modification: {mod}")

    def visit_Call(self, node: ast.Call) -> ast.Call:
        """
        Visit AST Call nodes and apply scheduled modifications.
        
        Args:
            node: AST call node to visit
            
        Returns:
            Modified AST call node
        """
        self.generic_visit(node)
        
        for mod in self.modifications:
            loc = mod['location']
            if (hasattr(node, 'lineno') and node.lineno == loc['lineno'] and
                    hasattr(node, 'col_offset') and node.col_offset == loc['col_offset']):
                
                if config.DEBUG_MODE:
                    print(f"[CodeMutator] Found AST Call node at Line {loc['lineno']}, Col {loc['col_offset']} for {mod['type']} modification.")

                if mod['type'] == 'dimension':
                    apply_dimension_modification(node, mod)
                elif mod['type'] == 'symbolic':
                    apply_symbolic_modification(node, mod)
                elif mod['type'] == 'activation':
                    apply_activation_modification(node, mod)
                elif mod['type'] == 'layer_type':
                    apply_layer_type_modification(node, mod)
                elif mod['type'] == 'architectural':
                    apply_architectural_modification(node, mod)
                elif mod['type'] == 'kernel_size':
                    apply_kernel_size_modification(node, mod)
                elif mod['type'] == 'stride':
                    apply_stride_modification(node, mod)

        return node

    def visit_List(self, node: ast.List) -> ast.List:
        """
        Visit List nodes to handle architectural mutations like block_setting.
        
        Args:
            node: AST list node to visit
            
        Returns:
            Modified AST list node
        """
        self.generic_visit(node)
        
        for mod in self.modifications:
            loc = mod['location']
            matches = (hasattr(node, 'lineno') and node.lineno == loc['lineno'] and
                       hasattr(node, 'col_offset') and node.col_offset == loc['col_offset'])

            if mod['type'] == 'architectural' and matches:
                if config.DEBUG_MODE:
                    print(f"[CodeMutator] Found AST List node at Line {loc['lineno']}, Col {loc['col_offset']} for architectural modification.")
                self._apply_architectural_modification(node, mod)
            elif mod['type'] == 'channel_config' and matches:
                if config.DEBUG_MODE:
                    print(f"[CodeMutator] Found AST List node at Line {loc['lineno']}, Col {loc['col_offset']} for channel-config modification.")
                apply_channel_config_modification(node, mod)
        
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        """Handle scalar channel-configuration modifications (e.g. init_block_channels = 64)."""
        self.generic_visit(node)

        for mod in self.modifications:
            if mod['type'] != 'channel_config':
                continue
            loc = mod['location']
            value = node.value
            if (hasattr(value, 'lineno') and value.lineno == loc['lineno'] and
                    hasattr(value, 'col_offset') and value.col_offset == loc['col_offset'] and
                    isinstance(value, ast.Constant)):
                if config.DEBUG_MODE:
                    print(f"[CodeMutator] Found scalar channel-config at Line {loc['lineno']} for modification.")
                apply_channel_config_modification(value, mod)

        return node

    def get_modified_code(self) -> str:
        """
        Apply all scheduled modifications and return the modified source code.
        
        Returns:
            Modified source code as a string
        """
        if config.DEBUG_MODE:
            print("[CodeMutator] Applying all scheduled modifications to AST.")
        modified_tree = self.visit(self.tree)
        return ast.unparse(modified_tree)
