# ab/gpt/util/Util.py

import os
import os.path
import re
import shutil
import ast
import json
from pathlib import Path
from ab.gpt.util.Const_Onnx import new_lemur_nn_dir, new_nn_file, new_lemur_stat_dir
from ..util.Code import *

def nn_accepted(nn_dir):
    accepted = True
    return accepted

def verify_nn_code(nn_dir, nn_file):
    verified = True
    error_message = ''
    if not verified:
        with open(nn_dir / f"error_code_verification.txt", "w+") as error_file:
            error_file.write(f"Code verification failed: {error_message}")
    return verified

def exists(f):
    return f and os.path.exists(f)

def create_symlink(src, dst):
    """
    Create a symbolic link from src to dst.
    If dst already exists (as file or link), do nothing.
    """
    dst = Path(dst)
    src = Path(src)
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst)
    except OSError as e:
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

def create_file(directory, filename, content):
    """
    Create a file with given content in the specified directory.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def preprocess_llm_output(txt):
    """
    Preprocess LLM output to handle common formatting issues.
    Removes '### Response:' prefix, markdown code blocks, and XML declarations.
    """
    if not txt:
        return txt
    
    # Remove '### Response:' prefix
    txt = re.sub(r'^###\s*Response:\s*\n?', '', txt.strip())
    
    # Remove markdown code blocks (``````xml, ```
    txt = re.sub(r'```(?:python|xml|json)?\s*\n?', '', txt)
    
    # Remove XML declarations
    txt = re.sub(r'<\?xml[^?]*\?>\s*', '', txt)
    
    # Remove stray backticks
    txt = re.sub(r'`+', '', txt)
    
    return txt

def extract_str(s: str, start: str, end: str):
    """
    Extract text between start and end markers.
    """
    try:
        # Preprocess to remove common decorators
        s = preprocess_llm_output(s)
        
        # Find last occurrence of end marker
        end_idx = s.rindex(end)
        s = s[:end_idx]
        
        # Split by start marker and take last occurrence
        spl = s.split(start)
        if len(spl) > 1:
            s = spl[-1]
            
            # Split by end marker and take first occurrence
            spl = s.split(end)
            if len(spl) > 1:
                s = spl[0]
            
            return s.strip()
    except:
        pass
    
    return None

def extract_code(txt):
    """
    Extract neural network code from <nn>...</nn> tags.
    Handles incomplete generations by checking for closing tag.
    """
    # Preprocess
    txt = preprocess_llm_output(txt)
    
    # Clean up spacing variations
    txt = txt.replace('< nn >', '<nn>').replace('<.nn>', '</nn>').replace(' nn >', '</nn>')
    
    # Check if <nn> section is complete
    if '<nn>' in txt and '</nn>' not in txt:
        print("[EXTRACT] !! NN section incomplete (missing </nn>)")
        print("[EXTRACT]   Likely cause: max_new_tokens too low")
        return None
    
    # Try to extract <nn> content
    patterns = [
        ('<nn>', '</nn>'),
        ('<nn>', '</nn>'),
        ('<nn>', '</nn>'),
        ('', '')
    ]
    
    extracted = next(filter(None, map(lambda l: extract_str(txt, l[0], l[1]), patterns)), '')
    
    if extracted:
        print(f"[EXTRACT] ✓ Found NN code: {len(extracted)} chars")
        return improve_code(extracted)
    else:
        print("[EXTRACT] ✗ No NN code found")
        return None

def extract_hyperparam(txt):
    """
    Extract hyperparameters from <hp>...</hp> tags.
    Returns JSON string that can be parsed.
    """
    # Preprocess
    txt = preprocess_llm_output(txt)
    
    # Clean up spacing variations
    txt = txt.replace('< hp >', '<hp>').replace('<.hp>', '</hp>').replace(' hp >', '</hp>')
    
    # Try to extract <hp> content
    patterns = [
        ('<hp>', '</hp>'),
        ('<hp>', '</hp>'),
        ('<hp>', '</hp>'),
        ('', '')
    ]
    
    extracted = next(filter(None, map(lambda l: extract_str(txt, l[0], l[1]), patterns)), '')
    
    if extracted:
        # Validate JSON
        try:
            # Clean up common JSON issues
            cleaned = re.sub(r',\s*}', '}', extracted)  # Remove trailing commas
            cleaned = re.sub(r',\s*]', ']', cleaned)
            
            # Try to parse to validate
            json.loads(cleaned)
            print(f"[EXTRACT] ✓ Found HP (valid JSON): {len(cleaned)} chars")
            return improve_code(cleaned)
        except json.JSONDecodeError as e:
            print(f"[EXTRACT] ✗ HP JSON invalid: {e}")
            print(f"[EXTRACT]   Raw: {extracted[:200]}")
            return None
    else:
        print("[EXTRACT] ✗ No HP tags found")
        return None

def extract_transform(txt):
    """
    Extract transformer code from <tr>...</tr> tags.
    """
    # Preprocess
    txt = preprocess_llm_output(txt)
    
    # Clean up spacing variations
    txt = txt.replace('< tr >', '<tr>').replace('<.tr>', '</tr>').replace(' tr >', '</tr>')
    
    # Check if <tr> section is complete
    if '<tr>' in txt and '</tr>' not in txt:
        print("[EXTRACT] ⚠ TR section incomplete (missing </tr>)")
        return None
    
    # Try to extract <tr> content
    patterns = [
        ('<tr>', '</tr>'),
        ('<tr>', '</tr>'),
        ('<tr>', '</tr>'),
        ('', '')
    ]
    
    extracted = next(filter(None, map(lambda l: extract_str(txt, l[0], l[1]), patterns)), '')
    
    if extracted:
        print(f"[EXTRACT] ✓ Found TR code: {len(extracted)} chars")
        return improve_code(extracted)
    else:
        print("[EXTRACT] ✗ No TR code found")
        return None

def extract_delta(txt):
    """
    Extract delta (unified diff) from text.
    Looks for:
    1. <delta>...</delta> XML tags
    2. Unified diff format (lines starting with ---, +++, @@)
    
    Args:
        txt: Text containing delta
        
    Returns:
        Delta string or None if not found
    """
    # Try XML tags first
    delta = extract_str(txt.replace('< delta >', '<delta>').replace('<.delta>', '</delta>').replace(' delta >', '</delta>'),
                       '<delta>', '</delta>')
    if delta:
        return delta
    
    # Try to extract unified diff format
    lines = txt.splitlines()
    delta_lines = []
    in_diff = False
    
    for line in lines:
        if line.startswith('---') or line.startswith('+++') or line.startswith('@@'):
            in_diff = True
            delta_lines.append(line)
        elif in_diff:
            if line.startswith('-') or line.startswith('+') or line.startswith(' '):
                delta_lines.append(line)
            elif line.strip() and not line.startswith('diff'):
                # End of diff block
                break
    
    if delta_lines:
        return '\n'.join(delta_lines)
    
    return None

def copy_to_lemur(gen_nn_dir, name, task, dataset, metric):
    Path(new_lemur_nn_dir).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(gen_nn_dir / new_nn_file, new_lemur_nn_dir / f'{name}.py')
    dr_nm = new_lemur_stat_dir / f"{task}_{dataset}_{metric}_{name}"
    Path(dr_nm).mkdir(parents=True, exist_ok=True)
    for f_nm in [f for f in os.listdir(gen_nn_dir) if re.match(r'[0-9]+\.json', f)]:
        shutil.copyfile(gen_nn_dir / f_nm, dr_nm / f_nm)
