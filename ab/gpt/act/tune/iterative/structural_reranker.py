#!/usr/bin/env python3
"""
Structural Reranker for Generated Neural Networks

Scores generated models based on AST-level structural patterns known to perform
well in first-epoch training (residual connections, normalization, GAP, etc.).
Helps prioritize evaluation of high-potential candidates.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json


class StructuralReranker:
    """Score and rank generated models by structural quality."""
    
    def __init__(self):
        self.pattern_weights = {
            "residual_connections": 3.0,      # x + F(x) patterns
            "batch_norm": 2.0,                # BatchNorm2d
            "group_norm": 2.0,                # GroupNorm
            "global_avg_pool": 2.5,           # AdaptiveAvgPool2d or GAP
            "depthwise_separable": 2.0,       # groups=in_channels pattern
            "squeeze_excitation": 1.5,        # SE block patterns
            "progressive_channels": 1.5,      # Good channel expansion
            "strided_conv": 1.0,              # Strided convs for downsampling
            "reasonable_depth": 1.0,          # 10-30 layers
            "penalties": {
                "large_fc": -2.0,             # Large flatten + FC layers
                "no_normalization": -3.0,     # No BN/GN at all
                "too_shallow": -1.5,          # < 5 layers
                "too_deep": -1.0,             # > 50 layers
            }
        }
    
    def score_model(self, code: str, model_id: str = None) -> Dict[str, Any]:
        """
        Score a generated model based on structural patterns.
        
        Args:
            code: Python code string
            model_id: Optional identifier for logging
        
        Returns:
            Dictionary with score, breakdown, and detected patterns
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {
                "model_id": model_id,
                "score": 0.0,
                "error": "syntax_error",
                "patterns": {}
            }
        
        patterns = self._detect_patterns(code, tree)
        score = self._compute_score(patterns)
        
        return {
            "model_id": model_id,
            "score": score,
            "patterns": patterns,
            "breakdown": self._score_breakdown(patterns)
        }
    
    def _detect_patterns(self, code: str, tree: ast.AST) -> Dict[str, Any]:
        """Detect structural patterns in code."""
        patterns = {
            "residual_connections": 0,
            "batch_norm": 0,
            "group_norm": 0,
            "global_avg_pool": 0,
            "depthwise_separable": 0,
            "squeeze_excitation": 0,
            "conv_layers": 0,
            "fc_layers": 0,
            "has_flatten": False,
            "channel_progression": [],
        }
        
        # Regex patterns for quick detection
        residual_pattern = r'(\w+\s*\+\s*\w+|return\s+\w+\s*\+|\+=)'
        bn_pattern = r'nn\.BatchNorm2d'
        gn_pattern = r'nn\.GroupNorm'
        gap_pattern = r'(nn\.AdaptiveAvgPool2d|F\.adaptive_avg_pool2d|\.mean\(\[.*2.*3.*\]\))'
        dw_pattern = r'groups\s*=\s*(\w+|self\.\w+)'
        se_pattern = r'(\.squeeze\(|\.view\(.*1.*1\)|channel.*attention)'
        
        # Count pattern occurrences
        patterns["residual_connections"] = len(re.findall(residual_pattern, code))
        patterns["batch_norm"] = len(re.findall(bn_pattern, code))
        patterns["group_norm"] = len(re.findall(gn_pattern, code))
        patterns["global_avg_pool"] = len(re.findall(gap_pattern, code))
        patterns["depthwise_separable"] = len(re.findall(dw_pattern, code))
        patterns["squeeze_excitation"] = len(re.findall(se_pattern, code))
        
        # Count layers via AST
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    attr_name = node.func.attr
                    if 'Conv' in attr_name:
                        patterns["conv_layers"] += 1
                        # Try to extract out_channels for progression tracking
                        if len(node.args) >= 2:
                            try:
                                if isinstance(node.args[1], ast.Constant):
                                    patterns["channel_progression"].append(node.args[1].value)
                            except:
                                pass
                    elif 'Linear' in attr_name:
                        patterns["fc_layers"] += 1
            elif isinstance(node, ast.Name):
                if node.id in ['flatten', 'Flatten']:
                    patterns["has_flatten"] = True
        
        return patterns
    
    def _compute_score(self, patterns: Dict[str, Any]) -> float:
        """Compute overall score from detected patterns."""
        score = 0.0
        
        # Positive contributions
        if patterns["residual_connections"] > 0:
            score += self.pattern_weights["residual_connections"] * min(patterns["residual_connections"], 5)
        
        if patterns["batch_norm"] > 0:
            score += self.pattern_weights["batch_norm"] * min(patterns["batch_norm"], 10)
        
        if patterns["group_norm"] > 0:
            score += self.pattern_weights["group_norm"] * min(patterns["group_norm"], 10)
        
        if patterns["global_avg_pool"] > 0:
            score += self.pattern_weights["global_avg_pool"]
        
        if patterns["depthwise_separable"] > 0:
            score += self.pattern_weights["depthwise_separable"] * min(patterns["depthwise_separable"], 3)
        
        if patterns["squeeze_excitation"] > 0:
            score += self.pattern_weights["squeeze_excitation"] * min(patterns["squeeze_excitation"], 2)
        
        # Channel progression bonus (smooth expansion)
        channels = patterns["channel_progression"]
        if len(channels) >= 3:
            is_progressive = all(channels[i] <= channels[i+1] * 2 for i in range(len(channels)-1))
            if is_progressive:
                score += self.pattern_weights["progressive_channels"]
        
        # Depth analysis
        total_layers = patterns["conv_layers"] + patterns["fc_layers"]
        if 10 <= total_layers <= 30:
            score += self.pattern_weights["reasonable_depth"]
        elif total_layers < 5:
            score += self.pattern_weights["penalties"]["too_shallow"]
        elif total_layers > 50:
            score += self.pattern_weights["penalties"]["too_deep"]
        
        # Penalties
        if patterns["has_flatten"] and patterns["fc_layers"] > 1:
            # Penalize flatten + large FC head (slow convergence)
            score += self.pattern_weights["penalties"]["large_fc"]
        
        if patterns["batch_norm"] == 0 and patterns["group_norm"] == 0:
            # No normalization at all
            score += self.pattern_weights["penalties"]["no_normalization"]
        
        return max(score, 0.0)  # Ensure non-negative
    
    def _score_breakdown(self, patterns: Dict[str, Any]) -> Dict[str, float]:
        """Generate detailed score breakdown."""
        breakdown = {}
        
        if patterns["residual_connections"] > 0:
            breakdown["residual_connections"] = self.pattern_weights["residual_connections"] * min(patterns["residual_connections"], 5)
        
        if patterns["batch_norm"] > 0:
            breakdown["batch_norm"] = self.pattern_weights["batch_norm"] * min(patterns["batch_norm"], 10)
        
        if patterns["group_norm"] > 0:
            breakdown["group_norm"] = self.pattern_weights["group_norm"] * min(patterns["group_norm"], 10)
        
        if patterns["global_avg_pool"] > 0:
            breakdown["global_avg_pool"] = self.pattern_weights["global_avg_pool"]
        
        if patterns["depthwise_separable"] > 0:
            breakdown["depthwise_separable"] = self.pattern_weights["depthwise_separable"] * min(patterns["depthwise_separable"], 3)
        
        if patterns["squeeze_excitation"] > 0:
            breakdown["squeeze_excitation"] = self.pattern_weights["squeeze_excitation"] * min(patterns["squeeze_excitation"], 2)
        
        total_layers = patterns["conv_layers"] + patterns["fc_layers"]
        if 10 <= total_layers <= 30:
            breakdown["reasonable_depth"] = self.pattern_weights["reasonable_depth"]
        elif total_layers < 5:
            breakdown["too_shallow_penalty"] = self.pattern_weights["penalties"]["too_shallow"]
        elif total_layers > 50:
            breakdown["too_deep_penalty"] = self.pattern_weights["penalties"]["too_deep"]
        
        if patterns["has_flatten"] and patterns["fc_layers"] > 1:
            breakdown["large_fc_penalty"] = self.pattern_weights["penalties"]["large_fc"]
        
        if patterns["batch_norm"] == 0 and patterns["group_norm"] == 0:
            breakdown["no_normalization_penalty"] = self.pattern_weights["penalties"]["no_normalization"]
        
        return breakdown
    
    def rank_models(
        self,
        model_files: List[Path],
        top_k: int = None
    ) -> List[Tuple[Path, float, Dict[str, Any]]]:
        """
        Rank a list of model files by structural score.
        
        Args:
            model_files: List of paths to model .py files
            top_k: Return only top K models (None = all)
        
        Returns:
            List of (file_path, score, details) tuples, sorted by score descending
        """
        scored_models = []
        
        for model_file in model_files:
            try:
                code = model_file.read_text()
                result = self.score_model(code, str(model_file.name))
                scored_models.append((model_file, result["score"], result))
            except Exception as e:
                print(f"[WARN] Failed to score {model_file}: {e}")
                scored_models.append((model_file, 0.0, {"error": str(e)}))
        
        # Sort by score descending
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            scored_models = scored_models[:top_k]
        
        return scored_models
    
    def save_rankings(
        self,
        rankings: List[Tuple[Path, float, Dict[str, Any]]],
        output_file: Path
    ):
        """Save rankings to JSON file."""
        rankings_data = [
            {
                "model_file": str(path),
                "score": score,
                "details": details
            }
            for path, score, details in rankings
        ]
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(rankings_data, f, indent=2)
        
        print(f"[INFO] Saved rankings to {output_file}")


def main():
    """Example usage and testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Rank generated models by structural quality")
    parser.add_argument("--models_dir", type=str, required=True, help="Directory containing generated .py files")
    parser.add_argument("--top_k", type=int, default=None, help="Return only top K models")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for rankings")
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"[ERROR] Models directory not found: {models_dir}")
        return
    
    # Find all .py files
    model_files = list(models_dir.glob("*.py"))
    print(f"[INFO] Found {len(model_files)} model files")
    
    # Rank models
    reranker = StructuralReranker()
    rankings = reranker.rank_models(model_files, top_k=args.top_k)
    
    # Print top models
    print(f"\n{'='*80}")
    print(f"TOP {len(rankings)} MODELS BY STRUCTURAL SCORE")
    print(f"{'='*80}")
    for i, (path, score, details) in enumerate(rankings, 1):
        print(f"\n{i}. {path.name}")
        print(f"   Score: {score:.2f}")
        if "breakdown" in details:
            print(f"   Breakdown: {details['breakdown']}")
    
    # Save if requested
    if args.output:
        output_file = Path(args.output)
        reranker.save_rankings(rankings, output_file)


if __name__ == "__main__":
    main()






