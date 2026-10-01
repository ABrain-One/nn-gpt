#!/usr/bin/env python3
"""
Novelty Checker for Generated Neural Networks

Uses AST-based structural analysis to determine if a generated model
is structurally unique compared to previously seen models.

Structural hash is computed from:
- Layer types and sequences
- Connection patterns (skip connections, parallel paths)
- Block structures
- Parameter shapes (relative, not absolute)
"""

import ast
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict

from ab.nn.util.Const import out_dir


class StructuralHasher:
    """Extract structural signature from PyTorch model code."""
    
    def __init__(self):
        self.layer_types = []
        self.connections = []
        self.block_names = set()
        
    def extract_from_code(self, code: str) -> Dict[str, Any]:
        """Extract structural features from code."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {"error": "syntax_error"}
        
        # Extract class definitions
        classes = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes[node.name] = self._analyze_class(node)
        
        # Build structural signature
        signature = {
            "classes": list(classes.keys()),
            "blocks": [name for name in classes.keys() if name != "Net"],
            "layer_sequence": self._extract_layer_sequence(classes),
            "connections": self._extract_connections(classes),
            "depth": self._estimate_depth(classes),
        }
        
        return signature
    
    def _analyze_class(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Analyze a single class definition."""
        info = {
            "name": node.name,
            "bases": [self._get_base_name(b) for b in node.bases],
            "layers": [],
            "forward_ops": [],
        }
        
        # Find __init__ method
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                info["layers"] = self._extract_layers(item)
            elif isinstance(item, ast.FunctionDef) and item.name == "forward":
                info["forward_ops"] = self._extract_forward_ops(item)
        
        return info
    
    def _get_base_name(self, base) -> str:
        """Extract base class name."""
        if isinstance(base, ast.Attribute):
            return base.attr
        elif isinstance(base, ast.Name):
            return base.id
        return "Unknown"
    
    def _extract_layers(self, init_func: ast.FunctionDef) -> List[str]:
        """Extract layer types from __init__ method."""
        layers = []
        for node in ast.walk(init_func):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    # nn.Conv2d, nn.Linear, etc.
                    layers.append(node.func.attr)
                elif isinstance(node.func, ast.Name):
                    # Custom blocks
                    layers.append(node.func.id)
        return layers
    
    def _extract_forward_ops(self, forward_func: ast.FunctionDef) -> List[str]:
        """Extract operations from forward method."""
        ops = []
        for node in ast.walk(forward_func):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    ops.append(node.func.attr)
                elif isinstance(node.func, ast.Name):
                    ops.append(node.func.id)
            elif isinstance(node, ast.BinOp):
                ops.append("add")  # Residual connection
        return ops
    
    def _extract_layer_sequence(self, classes: Dict) -> List[str]:
        """Extract the sequence of layers across all classes."""
        sequence = []
        if "Net" in classes:
            sequence.extend(classes["Net"]["layers"])
        for name, info in classes.items():
            if name != "Net":
                sequence.extend(info["layers"])
        return sequence
    
    def _extract_connections(self, classes: Dict) -> List[str]:
        """Extract connection patterns (residuals, concatenations, etc.)."""
        connections = []
        for name, info in classes.items():
            forward_ops = info.get("forward_ops", [])
            if "add" in forward_ops:
                connections.append("residual")
            if "cat" in forward_ops or "concat" in forward_ops:
                connections.append("concat")
        return list(set(connections))
    
    def _estimate_depth(self, classes: Dict) -> int:
        """Estimate network depth."""
        if "Net" in classes:
            return len(classes["Net"]["layers"])
        return 0
    
    def compute_hash(self, signature: Dict[str, Any]) -> str:
        """Compute structural hash from signature."""
        # Normalize signature for consistent hashing
        normalized = json.dumps(signature, sort_keys=True)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]


class NoveltyChecker:
    """Track and check novelty of generated models."""
    
    def __init__(self, cache_file: Optional[Path] = None):
        self.seen_hashes: Set[str] = set()
        self.seen_signatures: Dict[str, Dict] = {}
        self.cache_file = cache_file or out_dir / 'iterative_cycles' / 'seen_models.json'
        self.hasher = StructuralHasher()
        
        # Load existing cache if available
        if self.cache_file.exists():
            self.load_cache()
    
    def is_novel(self, code: str, model_id: str = None) -> bool:
        """Check if a model is structurally novel (does NOT add to seen set)."""
        signature = self.hasher.extract_from_code(code)
        
        if "error" in signature:
            return False  # Syntax errors are not novel
        
        struct_hash = self.hasher.compute_hash(signature)
        
        if struct_hash in self.seen_hashes:
            return False
        
        # DO NOT add to seen models here - only check
        # Models should be added via mark_as_seen() or add_training_data() 
        # only when actually selected for training
        return True
    
    def mark_as_seen(self, code: str, model_id: str = None, source: str = "generated"):
        """Explicitly mark a model as seen (for selected models that will be added to training)."""
        signature = self.hasher.extract_from_code(code)
        
        if "error" in signature:
            return False  # Cannot mark invalid code as seen
        
        struct_hash = self.hasher.compute_hash(signature)
        
        # Add to seen models
        self.seen_hashes.add(struct_hash)
        self.seen_signatures[struct_hash] = {
            "signature": signature,
            "model_id": model_id,
            "source": source,
            "hash": struct_hash,
        }
        
        return True
    
    def add_training_data(self, code: str, source: str = "training"):
        """Add a model from training data to the seen set."""
        signature = self.hasher.extract_from_code(code)
        if "error" not in signature:
            struct_hash = self.hasher.compute_hash(signature)
            self.seen_hashes.add(struct_hash)
            self.seen_signatures[struct_hash] = {
                "signature": signature,
                "source": source,
                "hash": struct_hash,
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about seen models."""
        return {
            "total_seen": len(self.seen_hashes),
            "unique_blocks": len(set(
                tuple(sig["signature"].get("blocks", []))
                for sig in self.seen_signatures.values()
                if "signature" in sig
            )),
        }
    
    def save_cache(self):
        """Save seen models to cache file."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "seen_hashes": list(self.seen_hashes),
            "signatures": self.seen_signatures,
        }
        self.cache_file.write_text(json.dumps(data, indent=2))
    
    def load_cache(self):
        """Load seen models from cache file."""
        try:
            data = json.loads(self.cache_file.read_text())
            self.seen_hashes = set(data.get("seen_hashes", []))
            self.seen_signatures = data.get("signatures", {})
            print(f"[INFO] Loaded {len(self.seen_hashes)} seen models from cache")
        except Exception as e:
            print(f"[WARN] Failed to load cache: {e}")


def main():
    """Test novelty checker."""
    checker = NoveltyChecker()
    
    # Example code 1
    code1 = """
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return self.bn(self.conv(x))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.block1 = ConvBlock(3, 64)
        self.block2 = ConvBlock(64, 128)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x
"""
    
    # Example code 2 (different structure)
    code2 = """
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.block1 = ResBlock(64)
        
    def forward(self, x):
        return self.block1(x)
"""
    
    print("Testing novelty checker...")
    print(f"Code 1 is novel: {checker.is_novel(code1, 'test1')}")
    # Mark as seen after checking
    checker.mark_as_seen(code1, 'test1', source='test')
    print(f"Code 1 again is novel: {checker.is_novel(code1, 'test1_dup')}")
    print(f"Code 2 is novel: {checker.is_novel(code2, 'test2')}")
    # Mark as seen
    checker.mark_as_seen(code2, 'test2', source='test')
    print(f"Stats: {checker.get_stats()}")
    
    # Save cache
    checker.save_cache()
    print(f"Cache saved to {checker.cache_file}")


if __name__ == "__main__":
    main()



