"""
•  Prepends each retrieved *block* verbatim to the LLM-generated wrapper so the
   saved file is self-contained.
•  Removes unused imports that reference packages **not installed** in the
   current Python environment (and never crashes on malformed source).
•  Normalises stray top-level indentation in both block and wrapper.
•  Retry feedback loop: gate traceback is fed back to the LLM; retries use
   sampling so the model can self-correct.
•  Hard 2 000 000 parameter budget to avoid OOMs at training time.
•  Smart prompt selection: automatically chooses specialized prompts based on
   block type (padding, attention, transformer, loss, complex architectures).
"""

from __future__ import annotations

import ast
import importlib.util
import json
import random
import shutil
import sys
import textwrap
import tokenize
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ab.rag.extract_blocks import BlockExtractor
from ..Const import (
    conf_test_dir,
    epoch_dir,
    synth_dir,
    new_out_file,
    new_nn_file,
)
from .Code import *
from ..Util import extract_code
from ab.nn.util.Util import create_file

# ────────────────────────────────────────────────────────────────
MAX_TRIES           = 3
LLM_MAX_TOKENS      = 3_500
PARAM_BUDGET        = 2_000_000
CONTEXT_WINDOW_LIMIT = 4_000  # Conservative limit for most models
ERROR_MSG_TRUNC_LEN = 200     # Truncate error messages for retry feedback
random.seed(42)

# Set up blocks directory and lazily fetch code blocks only if not cached
blocks_dir = Path(__file__).resolve().parents[3] / "blocks"
blocks_dir.mkdir(exist_ok=True)

# If the directory is empty (no cached blocks), run BlockExtractor to populate it; otherwise skip.
if any(blocks_dir.glob("*.py")):
    print("[INFO] Existing blocks detected – skipping extraction")
else:
    print("[INFO] No cached blocks found – extracting with BlockExtractor …")
    extractor = BlockExtractor()
    extractor.extract_blocks_from_file()

# Refresh list of available block files from the blocks directory
available_blocks = [f.stem for f in blocks_dir.glob("*.py") if f.is_file()]

# ────────────────────────────────────────────────────────────────
# Block type detection and prompt selection
# ────────────────────────────────────────────────────────────────
def _detect_block_type(block_code: str) -> str:
    """
    Intelligently analyze the block code structure and functionality to determine the appropriate prompt key.
    """
    block_lower = block_code.lower()
    
    # Parse the code to understand its structure
    try:
        tree = ast.parse(block_code)
    except SyntaxError:
        # If parsing fails, fall back to keyword matching
        return _fallback_detection(block_lower)
    
    # Analyze class definitions and their methods
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    
    for cls in classes:
        class_name = cls.name.lower()
        
        # Check for specific high-priority patterns first
        
        # 1. Padding blocks - look for padding-related methods and inheritance
        if any(keyword in class_name for keyword in ['pad', 'padding']):
            return "synthesize_padding_blocks"
        
        # 2. Loss blocks - look for loss-related patterns
        if any(keyword in class_name for keyword in ['loss', 'criterion', 'cost']):
            return "synthesize_loss_blocks"
        
        # 3. Attention blocks - look for attention mechanisms
        if any(keyword in class_name for keyword in ['attention', 'attn', 'selfattn', 'crossattn']):
            return "synthesize_attention_blocks"
        
        # 4. Transformer blocks - look for transformer components
        if any(keyword in class_name for keyword in ['transformer', 'vit', 'bert', 'encoder', 'decoder']):
            return "synthesize_transformer_blocks"
        
        # 5. Complex architectures - look for complete models
        if any(keyword in class_name for keyword in ['net', 'model', 'backbone', 'head', 'classifier']):
            return "synthesize_complex_architectures"
    
    # Analyze method signatures and operations
    methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    
    for method in methods:
        method_name = method.name.lower()
        
        # Check for attention-related operations
        if any(keyword in method_name for keyword in ['attention', 'attn', 'qkv', 'multihead']):
            return "synthesize_attention_blocks"
        
        # Check for loss-related operations
        if any(keyword in method_name for keyword in ['loss', 'forward_loss', 'compute_loss']):
            return "synthesize_loss_blocks"
        
        # Check for padding operations
        if any(keyword in method_name for keyword in ['pad', 'padding', 'pad_sequence']):
            return "synthesize_padding_blocks"
    
    # Analyze imports to understand functionality
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    
    for imp in imports:
        if isinstance(imp, ast.ImportFrom) and imp.module:
            module_name = imp.module.lower()
            
            # Check for attention-related imports
            if any(keyword in module_name for keyword in ['attention', 'transformer', 'vit']):
                return "synthesize_attention_blocks"
            
            # Check for loss-related imports
            if any(keyword in module_name for keyword in ['loss', 'criterion']):
                return "synthesize_loss_blocks"
    
    # Analyze string literals and comments for hints
    strings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Str):
            strings.append(node.s)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.append(node.value)
    
    for string in strings:
        string_lower = string.lower()
        
        # Check for attention-related descriptions
        if any(keyword in string_lower for keyword in ['attention', 'self-attention', 'multi-head', 'cross-attention', 'qkv', 'scaled dot product']):
            return "synthesize_attention_blocks"
        
        # Check for loss-related descriptions
        if any(keyword in string_lower for keyword in ['loss', 'criterion', 'loss function', 'cross entropy', 'focal loss']):
            return "synthesize_loss_blocks"
        
        # Check for padding-related descriptions
        if any(keyword in string_lower for keyword in ['padding', 'pad', 'boundary', 'circular', 'reflection']):
            return "synthesize_padding_blocks"
        
        # Check for normalization descriptions
        if any(keyword in string_lower for keyword in ['normalization', 'batch norm', 'layer norm', 'group norm']):
            return "synthesize_classification_rag"
        
        # Check for convolution descriptions
        if any(keyword in string_lower for keyword in ['convolution', 'conv', 'depthwise', 'separable']):
            return "synthesize_classification_rag"
    
    # Analyze variable names and assignments for additional hints
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
    for assign in assignments:
        for target in assign.targets:
            if isinstance(target, ast.Name):
                var_name = target.id.lower()
                if any(keyword in var_name for keyword in ['attention', 'attn', 'qkv']):
                    return "synthesize_attention_blocks"
                if any(keyword in var_name for keyword in ['loss', 'criterion']):
                    return "synthesize_loss_blocks"
                if any(keyword in var_name for keyword in ['pad', 'padding']):
                    return "synthesize_padding_blocks"
    
    # Fallback to keyword-based detection
    return _fallback_detection(block_lower)


def _fallback_detection(block_lower: str) -> str:
    """
    Fallback detection using keyword matching when AST parsing fails or doesn't provide clear signals.
    """
    # High-priority patterns that are very specific
    if any(keyword in block_lower for keyword in ['_circularpadnd', '_constantpadnd', '_reflectionpadnd', '_replicationpadnd']):
        return "synthesize_padding_blocks"
    
    if any(keyword in block_lower for keyword in ['xca', 'xcablock', 'windowattention', 'swinselfattention']):
        return "synthesize_attention_blocks"
    
    if any(keyword in block_lower for keyword in ['vit', 'transformer', 'bert']):
        return "synthesize_transformer_blocks"
    
    if any(keyword in block_lower for keyword in ['siouloss', 'tverskyloss', 'varifocalloss', 'wingloss']):
        return "synthesize_loss_blocks"
    
    if any(keyword in block_lower for keyword in ['xception', 'vgg', 'resnet', 'densenet']):
        return "synthesize_complex_architectures"
    
    # Medium-priority patterns
    if any(keyword in block_lower for keyword in ['attention', 'attn']):
        return "synthesize_attention_blocks"
    
    if any(keyword in block_lower for keyword in ['loss', 'criterion']):
        return "synthesize_loss_blocks"
    
    if any(keyword in block_lower for keyword in ['pad', 'padding']):
        return "synthesize_padding_blocks"
    
    if any(keyword in block_lower for keyword in ['transformer', 'encoder', 'decoder']):
        return "synthesize_transformer_blocks"
    
    if any(keyword in block_lower for keyword in ['net', 'model', 'backbone']):
        return "synthesize_complex_architectures"
    
    # Default to general RAG prompt
    return "synthesize_classification_rag"


def _get_prompt_for_block(template: dict, block_code: str) -> List[str]:
    """
    Get the appropriate prompt based on the block type.
    """
    block_type = _detect_block_type(block_code)
    
    # Try to get the specific prompt for this block type
    if block_type in template:
        prompt = template[block_type].get("prompt")
        if prompt:
            print(f"[PROMPT] Using {block_type} for this block")
            return prompt
    
    # Fallback to the general RAG prompt
    prompt = template.get("synthesize_classification_rag", {}).get("prompt")
    if prompt:
        print(f"[PROMPT] Using synthesize_classification_rag (fallback)")
        return prompt
    
    # Last resort: look for any prompt in the template
    for key, value in template.items():
        if isinstance(value, dict) and "prompt" in value:
            print(f"[PROMPT] Using {key} (last resort)")
            return value["prompt"]
    
    raise KeyError("No suitable prompt found in template")


# ────────────────────────────────────────────────────────────────
# Helper utilities
# ────────────────────────────────────────────────────────────────
def _load_block(name: str) -> str | None:
    """Fetch a code block from the blocks directory."""
    p = blocks_dir / f"{name}.py"
    if p.is_file():
        print(f"[LOAD] {name}")
        return p.read_text()
    print(f"[MISS] {name} not found in blocks directory")
    return None


def _escape_braces(s: str) -> str:
    """Double every literal brace except {block} placeholder."""
    sent = "<<BLOCK>>"
    return s.replace("{block}", sent).replace("{", "{{").replace("}", "}}").replace(
        sent, "{block}"
    )


# ────────────────────────────────────────────────────────────────
#  1) strip unused imports of pkgs not installed
#  2) never crash on malformed blocks (TokenError, etc.)
# ────────────────────────────────────────────────────────────────
def _strip_unavailable_imports(src: str) -> tuple[str, bool]:
    """
    returns (clean, False) … nothing stripped / parse failed
            (clean, True)  … at least one dead import removed
            ("",    None)  … unavailable pkg is *used* → skip block
    """
    try:
        tree = ast.parse(src)
    except (SyntaxError, IndentationError, tokenize.TokenError):
        return src, False  # leave unchanged – later gates will decide

    dead, alias2pkg = {}, {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for al in node.names:
                top = al.name.split(".")[0]
                if importlib.util.find_spec(top) is None:
                    dead[node.lineno] = al.asname or top
                    alias2pkg[al.asname or top] = top
        elif isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            if importlib.util.find_spec(top) is None:
                for al in node.names:
                    dead[node.lineno] = al.asname or al.name
                    alias2pkg[al.asname or al.name] = top

    if not dead:
        return src, False

    used = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        root = node
        if isinstance(node, ast.Attribute):
            while isinstance(root, ast.Attribute):
                root = root.value
        if isinstance(root, ast.Name) and root.id in alias2pkg:
            used.add(root.id)

    if used:
        return "", None  # real dependency on missing pkg

    cleaned = [
        ln for i, ln in enumerate(src.splitlines(True), 1) if i not in dead
    ]
    return "".join(cleaned), True


# ────────────────────────────────────────────────────────────────
# Main pipeline
# ────────────────────────────────────────────────────────────────
def alter(epochs: int, test_conf: str, llm_name: str) -> None:
    """Generate altered NNs using one building block per wrapper."""

    with open(conf_test_dir / test_conf) as f:
        template = json.load(f)

    print("Loading DeepSeek-Coder-7B-Instruct …")
    tok = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        llm_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    print("Ready ✔")

    shutil.rmtree(epoch_dir(), ignore_errors=True)
    kept = tried = 0

    for ep in range(epochs):
        queue = available_blocks.copy()
        base_out = epoch_dir(ep)
        idx = 0

        while queue:
            name = queue.pop(random.randrange(len(queue)))
            raw_block = _load_block(name)
            if raw_block is None:
                continue

            block, flag = _strip_unavailable_imports(raw_block)
            if flag is None:
                print(f"[SKIP] {name}: needs unavailable package")
                continue
            if flag:
                print(f"[STRIP] {name}: removed unused imports of unavailable packages")

            block = normalize_top_indent(textwrap.dedent(block).lstrip("\n"))

            # Get the appropriate prompt for this block type
            prompt_tpl = _get_prompt_for_block(template, block)
            user_prompt = _escape_braces("\n".join(prompt_tpl)).format(block=block)
            if len(tok.tokenize(user_prompt)) + 50 > CONTEXT_WINDOW_LIMIT:
                print("[WARN] prompt near context window")

            prev_error = ""
            for attempt in range(MAX_TRIES):
                tried += 1
                sys_msg = f"{llm_name} – retry={attempt}"
                if attempt and prev_error:
                    sys_msg += f" | previous error: {prev_error}"

                messages = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_prompt},
                ]
                inp = tok.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt"
                ).to(model.device)

                # Guard against CUDA OOM: attempt the generation; if memory error occurs
                # clear the cache and skip this block instead of crashing the whole run.
                try:
                    gen = model.generate(
                        inp,
                        max_new_tokens=LLM_MAX_TOKENS,
                        temperature=(0.2 if attempt == 0 else 0.7),
                        top_p=0.9,
                        do_sample=(attempt > 0),
                        eos_token_id=tok.eos_token_id,
                    )
                except (torch.cuda.OutOfMemoryError, RuntimeError) as oom_err:
                    # Some backends raise RuntimeError with "out of memory" in the message
                    if "out of memory" in str(oom_err).lower():
                        print("[OOM] Skipping block due to CUDA memory exhaustion")
                        torch.cuda.empty_cache()
                        prev_error = "CUDA OOM"
                        break  # exit retry loop and move to next block
                    else:
                        raise

                raw = tok.decode(gen[0][len(inp[0]):], skip_special_tokens=True)
                wrapper = textwrap.dedent(
                    (extract_code(raw) or "").replace("```python", "").replace("```", "")
                ).strip()

                full_code = dedup_imports(block + "\n\n" + wrapper)

                try:
                    compiled = compile(full_code, "<string>", "exec")
                    ns: dict[str, object] = {}
                    exec(compiled, ns)
                    Net = ns.get("Net")
                    shp = ns.get("supported_hyperparameters")
                    if Net is None or shp is None:
                        raise ValueError("missing Net or supported_hyperparameters")

                    dummy = {"lr": 0.01, "momentum": 0.9}
                    net = Net((3, 32, 32), (10,), dummy, torch.device("cpu"))
                    if sum(p.numel() for p in net.parameters()) > PARAM_BUDGET:
                        raise ValueError("param budget exceeded")

                except Exception as e:
                    prev_error = str(e)[:ERROR_MSG_TRUNC_LEN]
                    print(f"[GATE] {name} failed ({prev_error})")
                    continue  # retry

                break  # success
            else:
                print(f"[FAIL] giving up on {name}")
                continue

            out_dir = synth_dir(base_out) / f"B{idx}"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / new_nn_file).write_text(full_code)
            create_file(out_dir, new_out_file, raw)
            print(f"[INFO] ✔ {name} → {out_dir / new_nn_file}")

            idx += 1
            kept += 1

        print(f"\n[STATS] kept {kept} / {tried} generations ({kept/tried:.1%})")
