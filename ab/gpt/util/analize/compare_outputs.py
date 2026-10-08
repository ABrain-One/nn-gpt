#!/usr/bin/env python3
"""
Compare two output files side-by-side

Usage:
    python3 compare_outputs.py <file1> <file2> [--show-diff]
"""

import sys
import os
import re
from pathlib import Path

def extract_sections(text):
    """Extract code sections from output"""
    sections = {}
    
    # Reasoning (before <hp>)
    hp_idx = text.find('<hp>')
    sections['reasoning'] = text[:hp_idx] if hp_idx > 0 else ""
    
    # Hyperparameters
    hp_match = re.search(r'<hp>(.*?)</hp>', text, re.DOTALL)
    sections['hp'] = hp_match.group(1).strip() if hp_match else ""
    
    # Transform
    tr_match = re.search(r'<tr>(.*?)</tr>', text, re.DOTALL)
    sections['tr'] = tr_match.group(1).strip() if tr_match else ""
    
    # Model
    nn_match = re.search(r'<nn>(.*?)</nn>', text, re.DOTALL)
    sections['nn'] = nn_match.group(1).strip() if nn_match else ""
    
    return sections

def analyze_text(text):
    """Analyze text for metrics"""
    return {
        "chars": len(text),
        "lines": len(text.split('\n')),
        "tokens": len(text) // 4,  # Estimate
    }

def compare_files(file1, file2, show_diff=False):
    """Compare two output files"""
    
    # Read files
    try:
        with open(file1, 'r', encoding='utf-8', errors='ignore') as f:
            text1 = f.read()
    except Exception as e:
        print(f"[ERROR] Could not read {file1}: {e}")
        return
    
    try:
        with open(file2, 'r', encoding='utf-8', errors='ignore') as f:
            text2 = f.read()
    except Exception as e:
        print(f"[ERROR] Could not read {file2}: {e}")
        return
    
    # Extract sections
    sections1 = extract_sections(text1)
    sections2 = extract_sections(text2)
    
    # Analyze each
    metrics1 = {k: analyze_text(v) for k, v in sections1.items()}
    metrics2 = {k: analyze_text(v) for k, v in sections2.items()}
    
    total1 = analyze_text(text1)
    total2 = analyze_text(text2)
    
    # Print comparison
    print("\n" + "="*90)
    print("OUTPUT SIZE COMPARISON")
    print("="*90)
    print(f"\nFile 1: {file1}")
    print(f"File 2: {file2}\n")
    
    print(f"{'SECTION':<20} {'FILE 1':<25} {'FILE 2':<25} {'CHANGE':<15}")
    print(f"{'-'*90}")
    
    # Total comparison
    char_diff = total2['chars'] - total1['chars']
    char_pct = (char_diff / total1['chars'] * 100) if total1['chars'] > 0 else 0
    token_diff = total2['tokens'] - total1['tokens']
    token_pct = (token_diff / total1['tokens'] * 100) if total1['tokens'] > 0 else 0
    line_diff = total2['lines'] - total1['lines']
    line_pct = (line_diff / total1['lines'] * 100) if total1['lines'] > 0 else 0
    
    print(f"{'TOTAL (chars)':<20} {total1['chars']:>10} chars       {total2['chars']:>10} chars       {char_pct:>+6.1f}%")
    print(f"{'TOTAL (tokens)':<20} {total1['tokens']:>10} tokens      {total2['tokens']:>10} tokens      {token_pct:>+6.1f}%")
    print(f"{'TOTAL (lines)':<20} {total1['lines']:>10} lines       {total2['lines']:>10} lines       {line_pct:>+6.1f}%")
    print(f"{'-'*90}")
    
    # Section-by-section
    for section in ['reasoning', 'hp', 'tr', 'nn']:
        m1 = metrics1.get(section, analyze_text(""))
        m2 = metrics2.get(section, analyze_text(""))
        
        diff = m2['chars'] - m1['chars']
        pct = (diff / m1['chars'] * 100) if m1['chars'] > 0 else 0
        
        print(f"{section.upper():<20} {m1['chars']:>10} chars       {m2['chars']:>10} chars       {pct:>+6.1f}%")
    
    print(f"{'='*90}\n")
    
    # Show reduction summary
    if total2['chars'] < total1['chars']:
        print("✓ OUTPUT SIZE REDUCED")
        print(f"  Characters: {total1['chars']} → {total2['chars']} ({char_pct:.1f}% reduction)")
        print(f"  Tokens:     {total1['tokens']} → {total2['tokens']} ({token_pct:.1f}% reduction)")
        print(f"  Lines:      {total1['lines']} → {total2['lines']} ({line_pct:.1f}% reduction)")
    else:
        print("✗ OUTPUT SIZE INCREASED")
        print(f"  Characters: {total1['chars']} → {total2['chars']} ({char_pct:+.1f}%)")
        print(f"  Tokens:     {total1['tokens']} → {total2['tokens']} ({token_pct:+.1f}%)")
        print(f"  Lines:      {total1['lines']} → {total2['lines']} ({line_pct:+.1f}%)")
    
    print()
    
    # Show section changes
    print("Section Changes:")
    reasoning_diff = metrics2['reasoning']['chars'] - metrics1['reasoning']['chars']
    reasoning_pct = (reasoning_diff / metrics1['reasoning']['chars'] * 100) if metrics1['reasoning']['chars'] > 0 else 0
    print(f"  Reasoning: {reasoning_pct:+.1f}% (should be negative for Solution 3)")
    
    code_diff = (metrics2['hp']['chars'] + metrics2['tr']['chars'] + metrics2['nn']['chars']) - \
                (metrics1['hp']['chars'] + metrics1['tr']['chars'] + metrics1['nn']['chars'])
    code_total1 = metrics1['hp']['chars'] + metrics1['tr']['chars'] + metrics1['nn']['chars']
    code_pct = (code_diff / code_total1 * 100) if code_total1 > 0 else 0
    print(f"  Code:      {code_pct:+.1f}% (should be small or negative due to Solution 5)")
    
    print(f"\n{'='*90}\n")
    
    # Show diff if requested
    if show_diff:
        print("REASONING SECTION COMPARISON")
        print("="*90)
        r1_lines = sections1['reasoning'].split('\n')
        r2_lines = sections2['reasoning'].split('\n')
        
        print(f"File 1 reasoning: {len(r1_lines)} lines")
        print(f"File 2 reasoning: {len(r2_lines)} lines")
        print(f"Difference: {len(r2_lines) - len(r1_lines)} lines")
        print("\nFirst 20 lines of File 1 reasoning:")
        print('\n'.join(r1_lines[:20]))
        print("\nFirst 20 lines of File 2 reasoning:")
        print('\n'.join(r2_lines[:20]))

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 compare_outputs.py <file1> <file2> [--show-diff]")
        print("\nExample:")
        print("  python3 compare_outputs.py B0/full_output.txt B1/full_output.txt")
        sys.exit(1)
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    show_diff = '--show-diff' in sys.argv
    
    if not os.path.exists(file1):
        print(f"[ERROR] File not found: {file1}")
        sys.exit(1)
    
    if not os.path.exists(file2):
        print(f"[ERROR] File not found: {file2}")
        sys.exit(1)
    
    compare_files(file1, file2, show_diff=show_diff)

if __name__ == "__main__":
    main()
