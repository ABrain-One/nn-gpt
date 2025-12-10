"""
Delta utilities for computing and applying code deltas.

This module provides functionality to:
1. Compute unified diffs between two code versions
2. Apply unified diffs to reconstruct improved code from baseline

Uses Python's built-in difflib for minimal external dependencies.
"""

import difflib
from typing import Optional, List, Tuple
import re
import tempfile
import subprocess
import os


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
    
    Args:
        baseline_code: Original code
        delta: Unified diff string
        
    Returns:
        Improved code or None if application failed
    """
    if not baseline_code or not delta:
        return None
    
    # Validate delta format first
    if not validate_delta(delta):
        return None
    
    # Try system patch command first (most reliable)
    result = _apply_delta_with_patch(baseline_code, delta)
    if result is not None:
        return result
    
    # Fallback to manual application
    result = _apply_delta_manual(baseline_code, delta)
    if result is not None:
        return result
    
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

