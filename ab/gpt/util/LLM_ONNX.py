from ab.nn.util.Const import out_dir
from ab.gpt.util.Const import llm_dir, llm_tokenizer_dir
from ab.gpt.util.LLMUtil import quantization_config_4bit
from ab.gpt.util.Util import exists
import os
import json
import tempfile
import shutil
import torch
import torch.cuda
from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    PreTrainedTokenizer,
    PreTrainedModel
)


# Endpoints 
ONNX_LOCAL_PATH = "./onnx_output_folder"  # Default ONNX folder -  LOCAL
ONNX_HF_REPO = "mahmoud-han/OlympicCoder-7B-ONNX-Int8" # ONLINE VERSION

# IMPORT OPTIMUM FOR ONNX
try:
    from optimum.onnxruntime import ORTModelForCausalLM
    OPTIMUM_AVAILABLE = True
except ImportError:
    OPTIMUM_AVAILABLE = False

class LLM:
    def __init__(self,
                 model_path: str,
                 bnb_config: BitsAndBytesConfig = None,
                 use_onnx_if_available: bool = True,  # NEW: explicit flag
                 max_memory: str = "24000MB",
                 access_token=None,
                 use_deepspeed=False,
                 base_path=out_dir,
                 context_length=None,
                 gguf_file=None):
        
        self.context_length = context_length
        
        # --- Determine ONNX vs PyTorch Path ---
        real_model_path = None
        is_onnx = False
        
        # Priority 1: Check if local ONNX folder exists and has model.onnx
        if use_onnx_if_available and os.path.exists(ONNX_LOCAL_PATH):
            onnx_model_file = os.path.join(ONNX_LOCAL_PATH, "model.onnx")
            if os.path.exists(onnx_model_file):
                print(f"✓ Found ONNX model at: {ONNX_LOCAL_PATH}")
                real_model_path = ONNX_LOCAL_PATH
                is_onnx = True
            else:
                print(f"⚠ ONNX folder exists but no model.onnx found inside")
        
        # Priority 2: Check if model_path itself is a directory
        if not is_onnx:
            if os.path.isdir(model_path):
                real_model_path = model_path
                # Check if it contains ONNX model
                if use_onnx_if_available and os.path.exists(os.path.join(model_path, "model.onnx")):
                    print(f"✓ Found ONNX model at: {model_path}")
                    is_onnx = True
            else:
                # Priority 3: Use internal directory structure (out/llm/...)
                real_model_path = llm_dir(base_path, model_path)
                if use_onnx_if_available and os.path.exists(os.path.join(real_model_path, "model.onnx")):
                    print(f"✓ Found ONNX model at: {real_model_path}")
                    is_onnx = True
        
        # --- Load Tokenizer ---
        tokenizer_path = llm_tokenizer_dir(base_path, model_path) if not os.path.isdir(model_path) else model_path
        tokenizer_exists = exists(tokenizer_path)
        
        print("Forcing tokenizer download from Hugging Face...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            trust_remote_code=True,
            padding_side="right"
        )
        
        self.tokenizer.add_eos_token = True
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        
        if tokenizer_exists:
            print(f"Loading Tokenizer from local files: {tokenizer_path}")
        else:
            print("Downloading Tokenizer...")
            if not os.path.isdir(model_path):
                self.tokenizer.save_pretrained(tokenizer_path, token=access_token)
                print(f"Tokenizer saved to: {tokenizer_path}")
        
        # --- Load Model ---
        if is_onnx:
            if not OPTIMUM_AVAILABLE:
                raise ImportError(
                    "Found an ONNX model but 'optimum' is not installed.\n"
                    "Run: pip install optimum[onnxruntime-gpu]"
                )
            
            print(f"Loading ONNX Model from: '{real_model_path}'")
            self.model = ORTModelForCausalLM.from_pretrained(
                real_model_path,
                provider="CUDAExecutionProvider",
                use_io_binding=True,
                provider_options={
                    "device_id": 0,
                    "arena_extend_strategy": "kSameAsRequested",
                }
            )
            
            if "CUDAExecutionProvider" not in self.model.model.get_providers():
                raise RuntimeError(
                    "CRITICAL: ONNX Runtime failed to load CUDA! Check your install."
                )
            print("ONNX model loaded successfully with CUDA")
        
        else:
            # Fallback - Standard PyTorch Loading
            print(f"Loading PyTorch Model from: '{real_model_path}'")
            config = self._get_safe_config(real_model_path, model_path, access_token)
            
            deepspeed_specific_prm = {} if use_deepspeed else {'device_map': "auto"}
            
            model_kwargs = dict(
                trust_remote_code=True,
                max_memory={i: max_memory for i in range(torch.cuda.device_count())},
                token=access_token,
                torch_dtype=torch.float16,
                gguf_file=gguf_file,
                config=config,
                **deepspeed_specific_prm
            )
            
            if bnb_config is not None:
                model_kwargs["quantization_config"] = bnb_config
            
            self.model = AutoModelForCausalLM.from_pretrained(real_model_path, **model_kwargs)
            print("PyTorch model loaded successfully ✓ ")
    
    def _get_safe_config(self, src_dir, model_path, access_token):
        # Helper to safely load config, removing incompatible keys like quantization_config if needed
        if src_dir and os.path.exists(os.path.join(src_dir, "config.json")):
            with open(os.path.join(src_dir, "config.json"), "r") as f:
                cfg_dict = json.load(f)
            
            if cfg_dict.get("quantization_config", "absent") is None:
                del cfg_dict["quantization_config"]
            
            tmp_cfg_dir = tempfile.mkdtemp(prefix="sanitized_cfg_")
            try:
                with open(os.path.join(tmp_cfg_dir, "config.json"), "w") as f:
                    json.dump(cfg_dict, f)
                return AutoConfig.from_pretrained(tmp_cfg_dir, trust_remote_code=True, token=access_token)
            finally:
                shutil.rmtree(tmp_cfg_dir, ignore_errors=True)
        
        return AutoConfig.from_pretrained(model_path, trust_remote_code=True, token=access_token)
    
    def get_model(self) -> PreTrainedModel:
        return self.model
    
    def get_tokenizer(self) -> PreTrainedTokenizer:
        return self.tokenizer
    
    def get_max_length(self) -> int:
        if self.context_length:
            return self.context_length
        
        for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
            max_length = getattr(self.model.config, length_setting, None)
            if max_length:
                break
        
        if not max_length:
            max_length = 4096
        
        return max_length
