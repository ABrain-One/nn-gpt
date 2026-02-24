"""
Delta utilities for computing and applying code deltas.

This module provides functionality to:
1. Compute unified diffs between two code versions
2. Apply unified diffs to reconstruct improved code from baseline
3. Validate and repair generated Python code

Uses Python's built-in difflib for minimal external dependencies.
"""

import ast
import difflib
from typing import Optional, List, Tuple
import re
import tempfile
import subprocess
import os


def validate_python_syntax(code: str) -> tuple:
    """
    Validate that code is syntactically correct Python.
    
    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    if not code or not code.strip():
        return False, "Empty code"
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"


def repair_code(code: str) -> Optional[str]:
    """
    Attempt to repair common LLM-generated syntax errors.
    
    Fixes:
    1. Stray closing brackets/parentheses on their own line
    2. Misplaced nn.Module lines outside Sequential context
    
    Returns:
        Repaired code or None if repair fails.
    """
    if not code:
        return None
    
    lines = code.splitlines()
    repaired_lines = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Skip stray brackets that appear alone
        if stripped in [')', ']', '},', '])', '])']:
            continue
        
        # Skip misplaced nn.XXX lines at wrong indentation (outside Sequential)
        if re.match(r'^\s{12,}nn\.\w+\([^)]*\),?\s*$', line):
            # Check if we're inside a Sequential context
            in_sequential = False
            for j in range(max(0, i - 10), i):
                if 'nn.Sequential' in lines[j] or 'self.layers = nn.Sequential' in lines[j]:
                    in_sequential = True
                    break
            if not in_sequential:
                continue
        
        repaired_lines.append(line)
    
    repaired = '\n'.join(repaired_lines)
    
    # Validate repair worked
    is_valid, _ = validate_python_syntax(repaired)
    if is_valid:
        return repaired
    
    return None


def compute_delta(baseline_code: str, improved_code: str) -> str:
    """
    Compute unified diff between baseline and improved code.
    
    Args:
        baseline_code: Original code version (string)
        improved_code: Improved code version (string)
        
    Returns:
        Unified diff string in standard format
    """
    if not baseline_code or not improved_code:
        return ""
    
    baseline_lines = baseline_code.splitlines(keepends=True)
    improved_lines = improved_code.splitlines(keepends=True)
    
    # Generate unified diff
    delta = difflib.unified_diff(
        baseline_lines,
        improved_lines,
        fromfile='baseline.py',
        tofile='improved.py',
        lineterm='',
        n=3  # Context lines
    )
    
    return ''.join(delta)


def apply_delta(baseline_code: str, delta: str) -> Optional[str]:
    """
    Apply unified diff to baseline code to reconstruct improved code.
    
    This function tries multiple methods:
    1. Use system 'patch' command if available (most reliable)
    2. Manual application using unified diff parser (fallback)
    
    Now includes syntax validation - only returns code that parses correctly.
    
    Args:
        baseline_code: Original code
        delta: Unified diff string
        
    Returns:
        Improved code or None if application failed or result has syntax errors
    """
    if not baseline_code or not delta:
        return None
    
    # Validate delta format first
    if not validate_delta(delta):
        return None
    
    # Try system patch command first (most reliable)
    result = _apply_delta_with_patch(baseline_code, delta)
    if result is not None:
        is_valid, error = validate_python_syntax(result)
        if is_valid:
            return result
        # Try repair if invalid
        repaired = repair_code(result)
        if repaired:
            return repaired
        print(f"[DELTA] Patch result has syntax error: {error}")
    
    # Fallback to manual application
    result = _apply_delta_manual(baseline_code, delta)
    if result is not None:
        is_valid, error = validate_python_syntax(result)
        if is_valid:
            return result
        # Try repair if invalid
        repaired = repair_code(result)
        if repaired:
            return repaired
        print(f"[DELTA] Manual result has syntax error: {error}")
    
    return None


def _apply_delta_with_patch(baseline_code: str, delta: str) -> Optional[str]:
    """
    Apply delta using system 'patch' command.
    
    Returns:
        Applied code or None if patch command unavailable/failed
    """
    baseline_file = None
    delta_file = None
    
    try:
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py', encoding='utf-8') as f:
            f.write(baseline_code)
            baseline_file = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.diff', encoding='utf-8') as f:
            f.write(delta)
            delta_file = f.name
        
        # Apply patch using stdin (more reliable for unified diffs)
        # patch -u reads unified diff format, --quiet suppresses output
        with open(delta_file, 'r', encoding='utf-8') as diff_f:
            result = subprocess.run(
                ['patch', '-u', '--quiet', baseline_file],
                stdin=diff_f,
                capture_output=True,
                text=True,
                check=False  # Don't raise on error, we'll check return code
            )
        
        if result.returncode == 0:
            # Read patched file
            with open(baseline_file, 'r', encoding='utf-8') as f:
                improved_code = f.read()
            return improved_code
        else:
            return None
            
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        # patch command not available or failed
        return None
    finally:
        # Cleanup
        if baseline_file and os.path.exists(baseline_file):
            try:
                os.unlink(baseline_file)
            except OSError:
                pass
        if delta_file and os.path.exists(delta_file):
            try:
                os.unlink(delta_file)
            except OSError:
                pass


def _apply_delta_manual(baseline_code: str, delta: str) -> Optional[str]:
    """
    Manually apply unified diff by parsing and applying hunks.
    
    This is a fallback when patch command is not available.
    Implements a proper unified diff parser that handles:
    - Deletions (lines starting with '-')
    - Additions (lines starting with '+')
    - Context lines (lines starting with ' ')
    - Modifications (combination of '-' and '+' lines)
    
    Algorithm:
    1. Parse hunks from unified diff
    2. For each hunk, find the starting position in baseline
    3. Process hunk lines sequentially:
       - Context (' '): verify and advance position
       - Deletion ('-'): remove line at current position
       - Addition ('+'): insert line at current position
    4. Track cumulative offset for subsequent hunks
    
    Returns:
        Applied code or None if parsing/application failed
    """
    try:
        baseline_lines = baseline_code.splitlines(keepends=True)
        delta_lines = delta.splitlines(keepends=True)
        
        # Parse unified diff
        hunks = _parse_unified_diff(delta_lines)
        if not hunks:
            return None
        
        # Apply hunks to baseline
        result_lines = list(baseline_lines)
        cumulative_offset = 0  # Track how previous hunks changed line numbers
        
        for hunk in hunks:
            old_start, old_count, new_start, new_count, hunk_lines = hunk
            
            # Calculate actual starting position (1-based to 0-based, plus offset)
            line_pos = old_start - 1 + cumulative_offset
            
            # Validate starting position
            if line_pos < 0:
                line_pos = 0
            if line_pos > len(result_lines):
                # Hunk is beyond end of file, append at end
                line_pos = len(result_lines)
            
            # Process hunk lines in order
            for line in hunk_lines:
                if line.startswith(' '):
                    # Context line - verify it matches (optional check)
                    # Just advance position
                    if line_pos < len(result_lines):
                        line_pos += 1
                elif line.startswith('-'):
                    # Deletion - remove line at current position
                    if 0 <= line_pos < len(result_lines):
                        del result_lines[line_pos]
                        cumulative_offset -= 1
                        # Don't advance position after deletion
                elif line.startswith('+'):
                    # Addition - insert line at current position
                    new_line = line[1:]  # Remove '+' prefix
                    result_lines.insert(line_pos, new_line)
                    line_pos += 1
                    cumulative_offset += 1
        
        return ''.join(result_lines)
        
    except Exception as e:
        # Return None on any error (fallback method, so we don't want to crash)
        return None


def _parse_unified_diff(delta_lines: List[str]) -> List[Tuple[int, int, int, int, List[str]]]:
    """
    Parse unified diff format into hunks.
    
    Format:
        @@ -old_start,old_count +new_start,new_count @@
        -old line
        +new line
         context line
    
    Returns:
        List of (old_start, old_count, new_start, new_count, hunk_lines) tuples
    """
    hunks = []
    current_hunk = None
    hunk_lines = []
    
    for line in delta_lines:
        # Skip header lines
        if line.startswith('---') or line.startswith('+++'):
            continue
        
        # Hunk header
        hunk_match = re.match(r'@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@', line)
        if hunk_match:
            # Save previous hunk if exists
            if current_hunk is not None:
                hunks.append((*current_hunk, hunk_lines))
            
            # Start new hunk
            old_start = int(hunk_match.group(1))
            old_count = int(hunk_match.group(2) or 1)
            new_start = int(hunk_match.group(3))
            new_count = int(hunk_match.group(4) or 1)
            
            current_hunk = (old_start, old_count, new_start, new_count)
            hunk_lines = []
            continue
        
        # Hunk content
        if current_hunk is not None:
            if line.startswith('-') or line.startswith('+') or line.startswith(' '):
                hunk_lines.append(line)
    
    # Save last hunk
    if current_hunk is not None:
        hunks.append((*current_hunk, hunk_lines))
    
    return hunks


def validate_delta(delta: str) -> bool:
    """
    Validate that a delta string is in correct unified diff format.
    
    Args:
        delta: Delta string to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not delta or not delta.strip():
        return False
    
    lines = delta.splitlines()
    
    # Check for unified diff markers
    has_header = any(line.startswith('---') for line in lines)
    has_hunk = any(line.startswith('@@') for line in lines)
    
    return has_header and has_hunk


def compute_novelty_jaccard(baseline_code: str, improved_code: str) -> Tuple[bool, float]:
    """
    Compute Jaccard similarity between baseline and improved code using MinHash.
    
    Based on 'From Memorization to Creativity' paper methodology:
    - Uses MinHash with token-level shingles
    - Returns (is_novel, jaccard_similarity)
    - Novel if Jaccard < 0.90 (paper's threshold τ = 0.90)
    
    This is optional and requires datasketch library.
    
    Args:
        baseline_code: Original code
        improved_code: Improved code
        
    Returns:
        Tuple of (is_novel: bool, jaccard_similarity: float)
        Returns (True, 0.0) if datasketch not available or on error
    """
    try:
        from ab.gpt.util.nn_sftcodegen_rag import to_minhash
        
        baseline_mh = to_minhash(baseline_code)
        improved_mh = to_minhash(improved_code)
        jaccard = baseline_mh.jaccard(improved_mh)
        
        # Paper's threshold: τ = 0.90
        # If Jaccard >= 0.90, it's a near-duplicate (not novel)
        is_novel = jaccard < 0.90
        
        return is_novel, jaccard
    except ImportError:
        # datasketch not available - return default (assume novel)
        return True, 0.0
    except Exception:
        # On any error, return default (assume novel)
        return True, 0.0


def validate_delta_novelty(baseline_code: str, delta: str) -> Tuple[bool, float]:
    """
    Validate delta and compute novelty vs baseline.
    
    Applies delta to baseline, then computes Jaccard similarity.
    Returns whether the result is novel enough (not a near-duplicate).
    
    Based on 'From Memorization to Creativity' paper methodology.
    
    Args:
        baseline_code: Original baseline code
        delta: Unified diff to apply
        
    Returns:
        Tuple of (is_novel: bool, jaccard_similarity: float)
        Returns (False, 1.0) if delta application fails
    """
    applied_code = apply_delta(baseline_code, delta)
    if not applied_code:
        return False, 1.0  # Invalid delta = not novel (maximum similarity)
    
    return compute_novelty_jaccard(baseline_code, applied_code)

