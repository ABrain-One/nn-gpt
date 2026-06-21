"""
RAG-based SFT code generation utility.
Generates NNEval-compatible PyTorch models using finetuned LLMs with novelty filtering.
"""
import json, re, sys, hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from ab.gpt.util.Const import (
    conf_llm_dir,
    conf_test_dir,
    conf_prompt_dir,
    epoch_dir,
    synth_dir,
    new_nn_file,
)

try:
    from datasketch import MinHash, MinHashLSH
    _HAS_DATASKETCH = True
except Exception:
    _HAS_DATASKETCH = False

FENCE_PY = re.compile(r"```python\s*(.*?)```", re.S | re.M)
FENCE_ANY = re.compile(r"```\s*(.*?)```", re.S | re.M)
TOKEN_RE = re.compile(r"[A-Za-z_]\w*|[^\s]")


def extract_python_block(s: str) -> str:
    """Extract largest fenced code block or return raw string."""
    m = FENCE_PY.findall(s)
    if not m:
        m = FENCE_ANY.findall(s)
    if not m:
        return s.strip()
    return max(m, key=len).strip()


def has_nn_module_subclass(src: str) -> bool:
    """Check if source defines nn.Module subclass."""
    try:
        import ast
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for b in node.bases:
                    if getattr(b, "attr", "") == "Module" or getattr(b, "id", "") == "Module":
                        return True
        return False
    except Exception:
        return False


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dictionaries."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


# Text novelty via MinHash-LSH
def _tokenize(code: str) -> List[str]:
    return TOKEN_RE.findall(code)


def _shingles(tokens: List[str], n: int = 7) -> List[str]:
    if len(tokens) < n:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def to_minhash(code: str, num_perm: int = 128, n: int = 7):
    """Compute MinHash for code string."""
    mh = MinHash(num_perm=num_perm)
    for sh in _shingles(_tokenize(code), n=n):
        mh.update(sh.encode("utf-8"))
    return mh


def build_train_lsh(train_jsonl: Path, threshold: float = 0.85, num_perm: int = 128):
    """Build LSH index from training data for novelty detection."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    id2mh = {}
    rows = load_jsonl(train_jsonl)
    for i, row in enumerate(rows):
        msgs = row.get("messages", [])
        asst = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
        code = extract_python_block(asst) if asst else ""
        if not code:
            continue
        mh = to_minhash(code, num_perm=num_perm)
        key = row.get("id", f"train_{i}")
        lsh.insert(key, mh)
        id2mh[key] = mh
    return lsh, id2mh


def nearest_jaccard(lsh, id2mh, mh) -> float:
    """Find maximum Jaccard similarity to indexed training samples."""
    cands = lsh.query(mh)
    best = 0.0
    for k in cands:
        j = mh.jaccard(id2mh[k])
        if j > best:
            best = j
    return best


# Code validation for NNEval compatibility
def validate_code_for_nneval(code: str) -> tuple:
    """Validate code structure for NNEval.py compatibility. Returns (is_valid, error_msg)."""
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return (False, f"Syntax error: {e}")

    has_net = False
    net_methods = set()
    has_supported_hyperparams = False

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Net":
            has_net = True
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    net_methods.add(item.name)
        elif isinstance(node, ast.FunctionDef) and node.name == "supported_hyperparameters":
            has_supported_hyperparams = True

    if not has_net:
        return (False, "No Net class")
    if not has_supported_hyperparams:
        return (False, "Missing supported_hyperparameters()")

    required = {"__init__", "forward", "train_setup", "learn"}
    missing = required - net_methods
    if missing:
        return (False, f"Missing methods: {missing}")

    if "import torch" not in code:
        return (False, "Missing 'import torch'")
    if "import torch.nn" not in code and "from torch import nn" not in code:
        return (False, "Missing torch.nn import")

    return (True, "")


# Model loading and prompt rendering
def ensure_pad(tokenizer):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


def load_model_and_tokenizer(
    base_model: str,
    load_in_4bit: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    hf_token: Optional[str] = None,
    peft_checkpoint: Optional[str] = None,
):
    """
    Load model and tokenizer, optionally with PEFT adapters.
    
    Args:
        base_model: Base model name or path
        load_in_4bit: Whether to load in 4-bit quantization
        dtype: Data type for model
        hf_token: Hugging Face token
        peft_checkpoint: Optional path to PEFT checkpoint (if provided, will load adapters)
    """
    # Validate base_model is not None or empty (prevents 'NoneType' object has no attribute 'endswith' error)
    if base_model is None:
        raise ValueError("base_model cannot be None in load_model_and_tokenizer()")
    if not isinstance(base_model, str):
        raise TypeError(f"base_model must be a string, got {type(base_model)}: {base_model}")
    if not base_model.strip():
        raise ValueError(f"base_model cannot be empty string, got: '{base_model}'")
    
    # Check if path exists (for local paths) and validate
    base_model_path = Path(base_model)
    path_exists = base_model_path.exists()
    is_absolute = base_model_path.is_absolute()
    
    print(f"[DEBUG] Path validation: base_model='{base_model}', exists={path_exists}, is_absolute={is_absolute}")
    
    if path_exists:
        if not base_model_path.is_dir():
            raise ValueError(f"base_model path exists but is not a directory: {base_model}")
        # Ensure it's an absolute path for consistency
        if not is_absolute:
            base_model = str(base_model_path.resolve())
            print(f"[INFO] Converted relative path to absolute: {base_model}")
        else:
            base_model = str(base_model_path)  # Ensure it's a string
    elif not is_absolute:
        # Might be a relative path - try to resolve it
        try:
            resolved = base_model_path.resolve()
            if resolved.exists():
                base_model = str(resolved)
                print(f"[INFO] Resolved relative path to: {base_model}")
            else:
                # Might be a HuggingFace model identifier, which is fine
                print(f"[INFO] base_model '{base_model}' does not exist as local path, assuming HuggingFace model identifier")
        except (OSError, RuntimeError) as e:
            # Path resolution failed, might be a HuggingFace identifier
            print(f"[INFO] Could not resolve path '{base_model}', assuming HuggingFace model identifier: {e}")
    else:
        # Absolute path that doesn't exist - might be a HuggingFace model identifier
        print(f"[INFO] base_model '{base_model}' (absolute path) does not exist, assuming HuggingFace model identifier")
    
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    ) if load_in_4bit else None

    # Load tokenizer with better error handling
    # IMPORTANT: This is a known bug in Transformers v4.57.3 (issue #42583)
    # The error: vocab_file is None when transformers tries convert_slow_tokenizer(..., from_tiktoken=True)
    # Root cause: Fine-tune checkpoints often don't include complete tokenizer assets
    # Solution: Load tokenizer from base model (not checkpoint) and use slow tokenizer if fast fails
    # Reference: https://github.com/huggingface/transformers/issues/42583
    try:
        print(f"[DEBUG] Loading tokenizer from: {base_model}")
        
        # Verify tokenizer files exist (checkpoint dirs often lack complete tokenizer assets)
        # If missing, try loading from HuggingFace model ID instead
        tokenizer_path = Path(base_model)
        tokenizer_files = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]
        has_tokenizer_files = any((tokenizer_path / f).exists() for f in tokenizer_files)
        
        # Determine tokenizer source: prefer local if files exist, otherwise try HuggingFace model ID
        tokenizer_source = base_model
        if not has_tokenizer_files and tokenizer_path.exists() and tokenizer_path.is_dir():
            # Local path exists but missing tokenizer files - extract HuggingFace model ID from path
            # Path format: .../out/llm/ABrain/NNGPT-UniqueArch-Rag -> model ID: ABrain/NNGPT-UniqueArch-Rag
            path_parts = list(tokenizer_path.parts)
            # Look for the last two directory components that form a model ID (org/model)
            if len(path_parts) >= 2:
                # Try last two components: parent/name
                parent_name = path_parts[-2]
                model_name = path_parts[-1]
                potential_hf_id = f"{parent_name}/{model_name}"
                # Check if this looks like a HuggingFace model ID (not a local absolute path)
                if not Path(potential_hf_id).is_absolute() and '/' in potential_hf_id:
                    print(f"[INFO] Tokenizer files missing locally, trying HuggingFace model ID: {potential_hf_id}")
                    tokenizer_source = potential_hf_id
                else:
                    print(f"[WARN] Tokenizer files not found in {base_model}")
                    print(f"[WARN] Will attempt to load from local path anyway (may download from HuggingFace)")
            else:
                print(f"[WARN] Tokenizer files not found in {base_model}")
                print(f"[WARN] Will attempt to load from local path anyway (may download from HuggingFace)")
        elif not has_tokenizer_files:
            # Path doesn't exist as directory - might be HuggingFace model ID already
            print(f"[INFO] Path doesn't exist as directory, treating as HuggingFace model ID: {base_model}")
            tokenizer_source = base_model
        
        try:
            # Try fast tokenizer first (default behavior)
            print(f"[DEBUG] Loading tokenizer from: {tokenizer_source}")
            tok = AutoTokenizer.from_pretrained(tokenizer_source, token=hf_token, trust_remote_code=True, use_fast=True)
        except AttributeError as fast_error:
            if "'NoneType' object has no attribute 'endswith'" in str(fast_error):
                # Known bug in Transformers v4.57.3: vocab_file is None during fast tokenizer conversion
                # Fix: Use slow tokenizer class directly (bypasses the buggy conversion code)
                print(f"[WARN] Fast tokenizer conversion failed (Transformers v4.57.3 bug - vocab_file=None)")
                print(f"[WARN] Falling back to slow tokenizer to avoid the conversion bug")
                
                # Load config to determine tokenizer class
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(tokenizer_source, token=hf_token, trust_remote_code=True)
                model_type = getattr(config, 'model_type', '').lower()
                
                if 'llama' in model_type or 'deepseek' in model_type:
                    # Use slow LLaMA tokenizer directly (requires sentencepiece)
                    try:
                        from transformers import LlamaTokenizer
                        tok = LlamaTokenizer.from_pretrained(tokenizer_source, token=hf_token, trust_remote_code=True)
                        print(f"[INFO] Successfully loaded slow LlamaTokenizer")
                    except ImportError as e:
                        raise RuntimeError(
                            f"Slow tokenizer requires sentencepiece. Install with: pip install sentencepiece protobuf\n"
                            f"Original error: {e}"
                        ) from e
                else:
                    # For other model types, try AutoTokenizer with use_fast=False
                    # Note: This may still fail if the tokenizer config forces fast mode
                    print(f"[WARN] Model type '{model_type}' detected, trying AutoTokenizer with use_fast=False")
                    tok = AutoTokenizer.from_pretrained(
                        tokenizer_source, 
                        token=hf_token, 
        trust_remote_code=True,
                        use_fast=False
                    )
            else:
                raise
    except Exception as e:
        raise RuntimeError(
            f"Failed to load tokenizer from '{base_model}': {e}\n"
            f"Note: Fine-tune checkpoints often lack complete tokenizer assets.\n"
            f"Solution: Load tokenizer from the original base model path, not the checkpoint directory."
        ) from e
    
    ensure_pad(tok)
    print(f"[DEBUG] Tokenizer loaded and configured successfully", flush=True)
    
    # Load model with better error handling
    # First, verify the model directory structure
    model_path = Path(base_model)
    print(f"[DEBUG] About to load model. base_model='{base_model}', path exists: {model_path.exists()}", flush=True)
    
    if model_path.exists():
        required_files = ["config.json"]
        missing_files = [f for f in required_files if not (model_path / f).exists()]
        if missing_files:
            print(f"[WARN] Model directory missing files: {missing_files}")
        else:
            print(f"[DEBUG] Model directory structure verified: {base_model}")
    else:
        print(f"[WARN] Model path does not exist: {base_model}")
    
    try:
        print(f"[DEBUG] Loading model from: {base_model}", flush=True)
        print(f"[DEBUG] Using quantization_config: {bnb_cfg is not None}", flush=True)
        if bnb_cfg is not None:
            print(f"[DEBUG] Quantization config details: load_in_4bit={bnb_cfg.load_in_4bit}, compute_dtype={bnb_cfg.bnb_4bit_compute_dtype}", flush=True)
        print(f"[DEBUG] Using torch_dtype: {dtype}", flush=True)
        
        # Try loading with device_map="auto" if quantization is enabled (helps with path resolution)
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
        }
        
        if bnb_cfg is not None:
            load_kwargs["quantization_config"] = bnb_cfg
            load_kwargs["device_map"] = "auto"  # Helps with quantization path resolution
        
        mdl = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
        print(f"[DEBUG] Model loaded successfully")
    except AttributeError as e:
        if "'NoneType' object has no attribute 'endswith'" in str(e):
            # This is a known issue with transformers when path resolution fails
            # Often happens with quantization - try fallback without quantization
            import traceback
            tb_str = traceback.format_exc()
            
            if bnb_cfg is not None:
                print(f"[WARN] Model loading failed with quantization. Trying fallback without quantization...")
                try:
                    # Fallback: try loading without quantization
                    fallback_kwargs = {
                        "trust_remote_code": True,
                        "torch_dtype": dtype,
                        "device_map": "auto",
                    }
                    mdl = AutoModelForCausalLM.from_pretrained(base_model, **fallback_kwargs)
                    print(f"[INFO] Model loaded successfully without quantization (fallback)")
                except Exception as fallback_e:
                    # If fallback also fails, raise the original error with full context
                    raise ValueError(
                        f"Transformers library error: 'NoneType' object has no attribute 'endswith'. "
                        f"This typically occurs when transformers cannot resolve the model path internally. "
                        f"base_model='{base_model}' (exists: {model_path.exists()}), "
                        f"quantization_config={bnb_cfg is not None}. "
                        f"Fallback without quantization also failed: {fallback_e}. "
                        f"Full traceback:\n{tb_str}"
                    ) from e
            else:
                # No quantization, so this is a different issue
                raise ValueError(
                    f"Transformers library error: 'NoneType' object has no attribute 'endswith'. "
                    f"This typically occurs when transformers cannot resolve the model path internally. "
                    f"base_model='{base_model}' (exists: {model_path.exists()}), "
                    f"quantization_config={bnb_cfg is not None}. "
                    f"Full traceback:\n{tb_str}"
                ) from e
        raise
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        raise RuntimeError(
            f"Failed to load model from '{base_model}': {e}\n"
            f"Full traceback:\n{tb_str}"
        ) from e
    
    # Load PEFT adapters if checkpoint is provided
    if peft_checkpoint:
        try:
            from peft import PeftModel
            print(f"[INFO] Loading PEFT adapters from: {peft_checkpoint}")
            mdl = PeftModel.from_pretrained(mdl, peft_checkpoint, is_trainable=False)
            print(f"[INFO] PEFT adapters loaded successfully")
        except Exception as e:
            print(f"[WARN] Failed to load PEFT adapters from {peft_checkpoint}: {e}")
            print(f"[WARN] Continuing with base model only")
    
    return mdl, tok


def load_prompt_template(prompt_config_path: Path) -> Dict[str, str]:
    """Load prompt template from JSON config file."""
    try:
        with open(prompt_config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load prompt template: {e}")
        # Minimal but executable default with hyperparams
        return {
            "system_prompt_template": "",
            "prefix_code": (
                "```python\n"
                "import torch\n"
                "import torch.nn as nn\n\n"
                "def supported_hyperparameters():\n"
                "    return {\"lr\", \"momentum\", \"weight_decay\", \"dropout\"}\n\n"
            ),
        }


def render_prompt(tokenizer, messages: List[Dict[str, str]], prompt_template: Dict[str, str]) -> Dict[str, torch.Tensor]:
    """Render prompt with enhanced system message for NNEval compatibility."""
    msgs = [m for m in messages if m["role"] in ("system", "user")]

    enhanced_msgs = []
    for msg in msgs:
        if msg["role"] == "system":
            enhanced_system = msg["content"] + "\n\n" + prompt_template.get("system_prompt_template", "")
            enhanced_msgs.append({"role": "system", "content": enhanced_system})
        else:
            enhanced_msgs.append(msg)

    text = tokenizer.apply_chat_template(enhanced_msgs, tokenize=False, add_generation_prompt=True)
    toks = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return toks


# Known training blocks for rejection sampling
KNOWN_TRAINING_BLOCKS = {
    "DlaBasic",
    "InvertedResidual",
    "ConvBlock",
    "AirUnit",
    "AirInitBlock",
    "HardMish",
    "RMSNorm",
    "Distance",
    "ResBlock",
    "BasicBlock",
    "Bottleneck",
    "ConvNorm",
    "_InvertedResidual",
    "HardMishAutoFn",
}


def contains_known_blocks(code: str) -> bool:
    """Check if code contains known training block class names."""
    import ast
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                if class_name in KNOWN_TRAINING_BLOCKS:
                    return True
                for known_block in KNOWN_TRAINING_BLOCKS:
                    if known_block.lower() in class_name.lower() and class_name != "Net":
                        return True
    except Exception:
        for known_block in KNOWN_TRAINING_BLOCKS:
            if f"class {known_block}" in code or f"class {known_block}(" in code:
                return True
    return False


@torch.inference_mode()
def generate_code(
    model, tokenizer, toks, device,
    max_new_tokens: int = 2048,
    temperature: float = 0.2,
    top_p: float = 0.9,
    top_k: int = 50,
    prefix_code: Optional[str] = None,
) -> str:
    """Generate code from model with optional prefix for constrained decoding."""
    input_ids = toks["input_ids"].to(device)
    attn_mask = toks.get("attention_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)

    if prefix_code:
        prefix_tokens = tokenizer(prefix_code, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        input_ids = torch.cat([input_ids, prefix_tokens], dim=1)
        if attn_mask is not None:
            prefix_mask = torch.ones((attn_mask.shape[0], prefix_tokens.shape[1]), device=device, dtype=attn_mask.dtype)
            attn_mask = torch.cat([attn_mask, prefix_mask], dim=1)

    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
    )

    generated_tokens = gen[0][input_ids.shape[1]:]
    out = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    if prefix_code:
        out = prefix_code + out
    return out


# Main generation function
def main(
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    epoch: int = 0,
    config: str = "nngpt_unique_arch_rag.json",
    base_model: Optional[str] = None,
    load_in_4bit: bool = True,
    device: Optional[str] = None,
    max_items: int = 300,
    max_new_tokens: int = 2048,
    temperature: float = 0.20,
    top_p: float = 0.9,
    top_k: int = 50,
    rejection_sampling: bool = True,
    max_rejections: int = 5,
    hf_token: Optional[str] = None,
    novelty: bool = True,
    near_dup_threshold: float = 0.85,
    train_data_dir: Optional[str] = None,
    save_failures: bool = True,
    force_valid_structure: bool = True,
    prompt_template: str = "unique_rag_test_rules.json",
    samples_per_prompt: int = 5,
):
    """
    Generate NNEval-compatible PyTorch models using a finetuned LLM.
    
    Args:
        data_dir: Folder with NN_Rag_gen_test.jsonl (default: conf_test_dir/NN_Rag_gen_test.jsonl)
        output_dir: Output directory for results (default: out/nngpt/llm/epoch/A{epoch}/)
        epoch: Epoch number for model directory structure (default: 0)
        config: JSON config file in conf/llm/ directory holding base_model_name
        base_model: Base model name (overrides config file if provided)
        load_in_4bit: Load model in 4-bit quantization (default: True)
        device: Device to run generation on (default: cuda if available else cpu)
        max_items: Max number of prompts from NN_Rag_gen_test.jsonl to process
        max_new_tokens: Max new tokens to generate per sample
        temperature: Generation temperature (default: 0.20)
        top_p: Nucleus sampling threshold (default: 0.9)
        top_k: Top-k sampling (default: 50)
        rejection_sampling: Enable rejection sampling (default: True)
        max_rejections: Max rejection attempts before accepting anyway (default: 5)
        hf_token: Hugging Face token if needed for model access
        novelty: Enable text MinHash/LSH novelty scoring & rejection (default: True)
        near_dup_threshold: NN-Jaccard >= threshold => near-duplicate (default: 0.85)
        train_data_dir: Folder with NN_Rag_gen_train.jsonl for novelty check
        save_failures: Save failure code blocks (default: True)
        force_valid_structure: Use prefix_code to enforce NNEval structure (default: True)
        prompt_template: Prompt template config file in conf/prompt/test/ (default: unique_rag_test_rules.json)
        samples_per_prompt: How many architectures to generate per prompt (default: 5)
    """
    # Set device if not provided
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load prompt template from config (test dir holds your JSON prompt file)
    prompt_config_path = conf_test_dir / prompt_template
    prompt_tmpl = load_prompt_template(prompt_config_path)
    print(f"[INFO] Loaded prompt template from: {prompt_config_path}")

    # Set up model directory structure
    # If output_dir is provided (pipeline mode), use it for both models and results
    # Otherwise use default epoch structure (standalone mode)
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Pipeline mode: save models to output_dir/accepted_code
        models_base_dir = out_dir / "accepted_code"
        models_base_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Standalone mode: use traditional epoch structure
        current_epoch_path = epoch_dir(epoch)
        models_base_dir = synth_dir(current_epoch_path)
        models_base_dir.mkdir(parents=True, exist_ok=True)
        out_dir = current_epoch_path

    # Output subdirs
    fail_dir = out_dir / "code_fail"
    fail_dir.mkdir(exist_ok=True)
    raw_output_dir = out_dir / "raw_outputs"
    raw_output_dir.mkdir(exist_ok=True)
    raw_fail_dir = out_dir / "raw_failures"
    raw_fail_dir.mkdir(exist_ok=True)

    print(f"[INFO] Models will be saved to: {models_base_dir}")
    print(f"[INFO] Results will be saved to: {out_dir}")
    print(f"[INFO]   - results.jsonl")
    print(f"[INFO]   - summary.json")

    # Load base model config if not explicitly given
    if base_model is None:
        config_path = conf_llm_dir / config
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                base_model = cfg.get("base_model_name")
                if base_model:
                    print(f"[INFO] Loaded base_model '{base_model}' from config file: {config_path}")
                else:
                    raise ValueError(f"Config file {config_path} does not contain 'base_model_name'")
            except Exception as e:
                print(f"[ERROR] Failed to load config from {config_path}: {e}")
                sys.exit(1)
        else:
            print(f"[ERROR] Config file not found: {config_path}")
            sys.exit(1)
    else:
        print(f"[INFO] Using explicitly provided base_model: {base_model}")

    # Load data
    # Priority: 1) data_dir/test.jsonl (pipeline mode with cycle data)
    #           2) data_dir/NN_Rag_gen_test.jsonl (backward compatibility)
    #           3) conf_test_dir/NN_Rag_gen_test.jsonl (standalone/Sft_Rag.py mode)
    if data_dir:
        test_path = Path(data_dir) / "test.jsonl"
        if not test_path.exists():
            # Fallback to old naming convention
            test_path = Path(data_dir) / "NN_Rag_gen_test.jsonl"
    else:
        test_path = conf_test_dir / "NN_Rag_gen_test.jsonl"

    if not test_path.exists():
        print(f"[ERROR] Test data not found at {test_path}")
        print(f"[INFO] Expected: test.jsonl or NN_Rag_gen_test.jsonl in {Path(data_dir) if data_dir else conf_test_dir}")
        sys.exit(1)

    rows = load_jsonl(test_path)
    print(f"[INFO] Loaded {len(rows)} items from {test_path}")

    # Limit number of prompts processed
    num_prompts = min(max_items, len(rows))
    total_targets = num_prompts * samples_per_prompt

    # Validate base_model is not None
    if base_model is None:
        raise ValueError("base_model cannot be None. Please provide a model path or name.")

    # Check if base_model is actually a PEFT checkpoint directory
    # (In pipeline mode, checkpoint_path is passed as base_model)
    # (In standalone mode via Sft_Rag.py, base_model is a HuggingFace model name from config)
    peft_checkpoint = None
    base_model_name = base_model
    
    # Check if the path looks like a PEFT checkpoint (has adapter_config.json)
    # Note: HuggingFace model names (e.g., "deepseek-ai/deepseek-coder-1.3b") won't exist as paths,
    # so this check only runs for local file paths (pipeline mode with checkpoints)
    checkpoint_path = Path(base_model)
    if checkpoint_path.exists() and (checkpoint_path / "adapter_config.json").exists():
        # This is a PEFT checkpoint - need to load base model first, then adapters
        print(f"[INFO] Detected PEFT checkpoint at: {base_model}")
        
        # Try to get base model name from adapter config
        try:
            adapter_config = json.loads((checkpoint_path / "adapter_config.json").read_text())
            base_model_name = adapter_config.get("base_model_name_or_path")
            if not base_model_name:
                # Fallback: try to get from config.json if it exists
                config_path = checkpoint_path / "config.json"
                if config_path.exists():
                    config = json.loads(config_path.read_text())
                    base_model_name = config.get("_name_or_path") or config.get("base_model_name_or_path")
            
            if base_model_name and base_model_name.strip():
                print(f"[INFO] Base model from adapter config: {base_model_name}")
            else:
                raise ValueError(
                    f"Could not determine base model name from checkpoint. "
                    f"adapter_config.get('base_model_name_or_path') = {adapter_config.get('base_model_name_or_path')}"
                )
        except Exception as e:
            print(f"[ERROR] Could not read adapter config: {e}")
            print(f"[ERROR] Cannot load PEFT checkpoint without base model name")
            raise ValueError(f"Failed to determine base model from PEFT checkpoint: {e}")
        
        peft_checkpoint = str(checkpoint_path)
        print(f"[INFO] Will load PEFT adapters from: {peft_checkpoint}")
    elif checkpoint_path.exists():
        # Path exists but doesn't have adapter_config.json - might be a regular model directory
        # or a non-PEFT checkpoint. Just use it as-is (will load as base model).
        print(f"[INFO] Using provided path as base model: {base_model}")
        print(f"[INFO] (Not a PEFT checkpoint - no adapter_config.json found)")
    
    # Final validation: ensure base_model_name is not None or empty before loading
    if not base_model_name or not base_model_name.strip():
        raise ValueError(
            f"base_model_name is None or empty. Cannot load model. "
            f"base_model was: {base_model}, "
            f"base_model_name was: {base_model_name}, "
            f"checkpoint_path exists: {checkpoint_path.exists() if base_model else False}"
        )
    
    # Debug logging before calling load_model_and_tokenizer
    print(f"[DEBUG] About to load model with base_model_name='{base_model_name}' (type: {type(base_model_name)})")
    print(f"[DEBUG] peft_checkpoint={peft_checkpoint}")
    
    # Load model and tokenizer (with PEFT adapters if checkpoint detected)
    try:
        model, tokenizer = load_model_and_tokenizer(
            base_model_name,
            load_in_4bit=load_in_4bit,
            dtype=torch.bfloat16,
            hf_token=hf_token,
            peft_checkpoint=peft_checkpoint,
        )
    except AttributeError as e:
        if "'NoneType' object has no attribute 'endswith'" in str(e):
            # This error occurs inside transformers library, not because we passed None
            # It suggests transformers is trying to process a path that became None internally
            # This can happen if the model path doesn't exist or transformers can't resolve it
            import traceback
            tb_str = traceback.format_exc()
            raise ValueError(
                f"MODEL LOADING ERROR: Transformers library error: 'NoneType' object has no attribute 'endswith'. "
                f"This suggests an internal issue in transformers when processing the model path. "
                f"base_model={base_model}, base_model_name={base_model_name} (type: {type(base_model_name)}), "
                f"peft_checkpoint={peft_checkpoint}. "
                f"Path exists: {Path(base_model_name).exists() if base_model_name else 'N/A'}. "
                f"Full traceback:\n{tb_str}"
            ) from e
        raise
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        raise RuntimeError(
            f"Unexpected error loading model: {e}\n"
            f"base_model={base_model}, base_model_name={base_model_name}\n"
            f"Full traceback:\n{tb_str}"
        ) from e
    model.eval()
    device_obj = torch.device(device)
    model.to(device_obj)
    try:
        if getattr(model.config, "use_cache", True):
            model.config.use_cache = False
    except Exception:
        pass

    # Novelty indices
    text_lsh = None
    text_id2mh = None
    if novelty:
        if not _HAS_DATASKETCH:
            print("[NOVELTY] datasketch not installed; disabling text near-dup.")
            novelty = False
        else:
            # Determine training data directory for novelty check
            if train_data_dir:
                tdir = Path(train_data_dir)
            elif data_dir:
                tdir = Path(data_dir)
            else:
                tdir = conf_prompt_dir / "train"  # Default: conf/prompt/train/

            # Try train.jsonl first (pipeline mode), then NN_Rag_gen_train.jsonl (standalone mode)
            tpath = tdir / "train.jsonl"
            if not tpath.exists():
                tpath = tdir / "NN_Rag_gen_train.jsonl"
            
            if tpath.exists():
                print(f"[NOVELTY] indexing text MinHash on {tpath} (τ={near_dup_threshold}) ...")
                text_lsh, text_id2mh = build_train_lsh(tpath, threshold=near_dup_threshold)
            else:
                print(f"[NOVELTY] WARNING: train.jsonl or NN_Rag_gen_train.jsonl not found in {tdir}; text novelty vs train disabled.")
                text_lsh = None

    # For intra-run novelty among generated models
    generated_mhs: List[Any] = []  # store MinHashes of accepted models

    # Iterate over prompts and generate multiple samples per prompt
    results_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"
    done = ok = 0
    near_dup_train_hits = near_dup_train_total = 0
    near_dup_gen_hits = near_dup_gen_total = 0

    with results_path.open("w", encoding="utf-8") as fout:
        global_idx = 0  # index used for B{idx} folders etc.

        for prompt_idx, item in enumerate(rows[:num_prompts]):
            mid = item.get("id", f"prompt_{prompt_idx}")
            messages = item["messages"]
            user_txt = next((m["content"] for m in messages if m["role"] == "user"), "")

            toks = render_prompt(tokenizer, messages, prompt_tmpl)

            for sample_idx in range(samples_per_prompt):
                nn_j_train = None
                nn_j_gen = None
                near_dup_train_flag = None
                near_dup_gen_flag = None
                raw = None
                rejection_count = 0
                current_temp = temperature
                mh = None  # MinHash for the accepted candidate

                try:
                    # Rejection sampling loop:
                    # 1) structure validation
                    # 2) novelty rejection vs train + generated
                    # 3) known-block rejection
                    candidate_mh = None
                    candidate_nn_j_train = None
                    candidate_nn_j_gen = None
                    candidate_near_dup_train = None
                    candidate_near_dup_gen = None

                    while True:
                        prefix_code = (
                            prompt_tmpl.get("prefix_code", None)
                            if force_valid_structure
                            else None
                        )

                        raw = generate_code(
                            model,
                            tokenizer,
                            toks,
                            device_obj,
                            max_new_tokens=max_new_tokens,
                            temperature=current_temp,
                            top_p=top_p,
                            top_k=top_k,
                            prefix_code=prefix_code,
                        )

                        code = extract_python_block(raw)

                        # 1) Validate code structure
                        is_valid, error_msg = validate_code_for_nneval(code)
                        if not is_valid:
                            if rejection_sampling:
                                rejection_count += 1
                                if rejection_count >= max_rejections:
                                    print(
                                        f"[WARN] gidx={global_idx}: Reached max rejections "
                                        f"({max_rejections}), accepting despite validation error: {error_msg}"
                                    )
                                    break
                                current_temp = min(1.0, current_temp + 0.1)
                                print(
                                    f"[REJECT-STRUCT] gidx={global_idx}: Attempt {rejection_count}/{max_rejections} "
                                    f"- Validation failed: {error_msg}, retrying with temp={current_temp:.2f}"
                                )
                                continue
                            else:
                                print(f"[WARN] gidx={global_idx}: Invalid structure but rejection_sampling disabled: {error_msg}")
                                break

                        # 2) Novelty-based rejection (vs train AND vs previously generated)
                        candidate_mh = None
                        candidate_nn_j_train = None
                        candidate_nn_j_gen = None
                        candidate_near_dup_train = None
                        candidate_near_dup_gen = None

                        if novelty:
                            candidate_mh = to_minhash(code)

                            # vs train set
                            if text_lsh is not None:
                                candidate_nn_j_train = nearest_jaccard(text_lsh, text_id2mh, candidate_mh)
                                candidate_near_dup_train = candidate_nn_j_train >= near_dup_threshold

                            # vs already generated models
                            if generated_mhs:
                                best_gen = 0.0
                                for prev_mh in generated_mhs:
                                    j = candidate_mh.jaccard(prev_mh)
                                    if j > best_gen:
                                        best_gen = j
                                candidate_nn_j_gen = best_gen
                                candidate_near_dup_gen = candidate_nn_j_gen >= near_dup_threshold
                            else:
                                candidate_nn_j_gen = 0.0
                                candidate_near_dup_gen = False

                            candidate_is_near_dup = bool(candidate_near_dup_train or candidate_near_dup_gen)

                            if candidate_is_near_dup and rejection_sampling:
                                rejection_count += 1
                                if rejection_count >= max_rejections:
                                    print(
                                        f"[WARN] gidx={global_idx}: Reached max rejections "
                                        f"({max_rejections}), accepting despite near-duplicate "
                                        f"(J_train={candidate_nn_j_train}, J_gen={candidate_nn_j_gen}, τ={near_dup_threshold})"
                                    )
                                    # accept this one (near-dup)
                                    break
                                current_temp = min(1.0, current_temp + 0.1)
                                print(
                                    f"[REJECT-NOVELTY] gidx={global_idx}: Attempt {rejection_count}/{max_rejections} "
                                    f"- J_train={candidate_nn_j_train}, J_gen={candidate_nn_j_gen} ≥ τ={near_dup_threshold}, "
                                    f"retrying with temp={current_temp:.2f}"
                                )
                                continue

                        # 3) Known-block rejection (secondary to novelty)
                        if rejection_sampling and contains_known_blocks(code):
                            rejection_count += 1
                            if rejection_count >= max_rejections:
                                print(
                                    f"[WARN] gidx={global_idx}: Reached max rejections "
                                    f"({max_rejections}), accepting despite known blocks"
                                )
                                break
                            current_temp = min(1.0, current_temp + 0.1)
                            print(
                                f"[REJECT-BLOCK] gidx={global_idx}: Attempt {rejection_count}/{max_rejections} "
                                f"- Known block detected, retrying with temp={current_temp:.2f}"
                            )
                            continue

                        # If we reached here, we accept this candidate
                        break

                    # Save raw output
                    raw_output_path = raw_output_dir / f"gen_{global_idx:04d}_raw.txt"
                    raw_output_path.write_text(raw, encoding="utf-8")

                    if not has_nn_module_subclass(code):
                        raise RuntimeError("No nn.Module subclass in code block.")

                    # Fix accepted novelty stats for this sample
                    if novelty and candidate_mh is not None:
                        mh = candidate_mh
                        nn_j_train = candidate_nn_j_train
                        nn_j_gen = candidate_nn_j_gen
                        near_dup_train_flag = candidate_near_dup_train
                        near_dup_gen_flag = candidate_near_dup_gen

                        # Track stats only for accepted models
                        # vs train
                        if text_lsh is not None and nn_j_train is not None:
                            near_dup_train_total += 1
                            if near_dup_train_flag:
                                near_dup_train_hits += 1

                        # vs generated set
                        near_dup_gen_total += 1
                        if near_dup_gen_flag:
                            near_dup_gen_hits += 1

                        # Add to generated set for intra-run novelty
                        generated_mhs.append(mh)

                    # Save code in NNEval-compatible structure:
                    #   out/nngpt/llm/epoch/A{epoch}/synth_nn/B{global_idx}/new_nn.py
                    model_dir = models_base_dir / f"B{global_idx}"
                    model_dir.mkdir(parents=True, exist_ok=True)
                    code_path = model_dir / new_nn_file
                    code_path.write_text(code, encoding="utf-8")

                    rec = {
                        "id": mid,
                        "prompt_index": prompt_idx,
                        "sample_index": sample_idx,
                        "global_index": global_idx,
                        "file": str(code_path),
                        "raw_output": str(raw_output_path),
                        "ok": True,
                        # Backwards-compatible fields (vs train)
                        "nn_jaccard": round(nn_j_train, 4) if nn_j_train is not None else None,
                        "near_dup_text": bool(near_dup_train_flag) if near_dup_train_flag is not None else None,
                        # Explicit split: train vs generated
                        "nn_jaccard_train": round(nn_j_train, 4) if nn_j_train is not None else None,
                        "near_dup_text_train": bool(near_dup_train_flag) if near_dup_train_flag is not None else None,
                        "nn_jaccard_gen": round(nn_j_gen, 4) if nn_j_gen is not None else None,
                        "near_dup_text_gen": bool(near_dup_gen_flag) if near_dup_gen_flag is not None else None,
                        "rejection_count": rejection_count if rejection_sampling else None,
                    }
                    ok += 1
                except Exception as e:
                    # Save raw output for failures
                    raw_fail_path: Optional[Path] = None
                    if raw is not None:
                        raw_fail_path = raw_fail_dir / f"gen_{global_idx:04d}_raw.txt"
                        raw_fail_path.write_text(raw, encoding="utf-8")

                    # Save code for failures (if requested)
                    fail_code_path: Optional[Path] = None
                    if save_failures:
                        fail_code_path = fail_dir / f"gen_{global_idx:04d}_fail.py"
                        try:
                            if "code" in locals() and code:
                                fail_code_path.write_text(code, encoding="utf-8")
                            elif raw is not None:
                                fail_code_path.write_text(
                                    f"# Raw LLM output (code extraction failed):\n"
                                    f"# {type(e).__name__}: {e}\n\n{raw}",
                                    encoding="utf-8",
                                )
                            else:
                                fail_code_path.write_text(
                                    f"# No code generated\n# Error: {type(e).__name__}: {e}",
                                    encoding="utf-8",
                                )
                        except Exception as write_err:
                            print(f"[WARN] Failed to save failure code for gidx={global_idx}: {write_err}")
                            fail_code_path = None

                    rec = {
                        "id": mid,
                        "prompt_index": prompt_idx,
                        "sample_index": sample_idx,
                        "global_index": global_idx,
                        "file": None,
                        "raw_output": str(raw_fail_path) if raw_fail_path is not None else None,
                        "fail_code": str(fail_code_path) if fail_code_path is not None else None,
                        "ok": False,
                        "error": f"{type(e).__name__}: {e}",
                        "nn_jaccard": None,
                        "near_dup_text": None,
                        "nn_jaccard_train": None,
                        "near_dup_text_train": None,
                        "nn_jaccard_gen": None,
                        "near_dup_text_gen": None,
                    }

                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                done += 1
                global_idx += 1

                if done % 10 == 0:
                    print(f"[{done}/{total_targets}] ok={ok}")

    # Summary
    text_near_dup_rate_train = (
        round(near_dup_train_hits / max(1, near_dup_train_total), 4)
        if near_dup_train_total
        else None
    )
    text_near_dup_rate_gen = (
        round(near_dup_gen_hits / max(1, near_dup_gen_total), 4)
        if near_dup_gen_total
        else None
    )

    summary = {
        "counts": {"done": done, "ok": ok, "fail": done - ok},
        "rates": {
            "executable_rate": round(ok / max(1, done), 4),
            # Backwards compatible: keep old key mapped to train-based novelty
            "text_near_dup_rate": text_near_dup_rate_train,
            "text_near_dup_rate_train": text_near_dup_rate_train,
            "text_near_dup_rate_gen": text_near_dup_rate_gen,
        },
        "paths": {
            "results": str(results_path),
            "dir_models": str(models_base_dir),
            "dir_raw_outputs": str(raw_output_dir),
            "dir_raw_failures": str(raw_fail_dir),
            "dir_failures": str(fail_dir),
        },
        "generation": {
            "num_prompts": num_prompts,
            "samples_per_prompt": samples_per_prompt,
            "total_requested_samples": total_targets,
        },
        "notes": [
            "Novelty vs train: MinHash/LSH against NN_Rag_gen_train.jsonl (if available).",
            "Novelty vs generated: MinHash Jaccard vs previously accepted models in this run.",
            "Near-duplicates rejected up to max_rejections before being optionally accepted.",
        ],
    }
    Path(summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DONE] {ok}/{done} succeeded. Results → {results_path}")
    if near_dup_train_total > 0:
        print(
            f"[NOVELTY-TRAIN] accepted near-dup hits: {near_dup_train_hits} / {near_dup_train_total}  "
            f"(τ={near_dup_threshold})"
        )
    if near_dup_gen_total > 0:
        print(
            f"[NOVELTY-GEN] accepted near-dup hits: {near_dup_gen_hits} / {near_dup_gen_total}  "
            f"(τ={near_dup_threshold})"
        )
    print(f"Summary → {summary_path}")
    print(f"Generated models (NNEval-compatible) → {models_base_dir}")
    print(f"Raw outputs → {raw_output_dir}")
    print(f"Raw failures → {raw_fail_dir}")
    print(f"Failure code → {fail_dir}")
