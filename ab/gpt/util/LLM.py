# ab/gpt/util/LLM.py
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


class LLM:
    def __init__(self,
                 model_path: str,
                 bnb_config: BitsAndBytesConfig = None,
                 local_path=None,
                 max_memory: str = "24000MB",
                 access_token=None,
                 use_deepspeed=False,
                 base_path=out_dir,
                 context_length=None,
                 gguf_file=None,
                 training_args=None):
        # --- Tokenizer ---
        self.context_length = context_length
        tok_fl_nm = llm_tokenizer_dir(base_path, model_path)
        raw_fl_nm = llm_dir(base_path, model_path)
        tokenizer_exists = exists(tok_fl_nm)

        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_fl_nm if tokenizer_exists else model_path,
            trust_remote_code=True, token=access_token, gguf_file=gguf_file
        )
        self.tokenizer.add_eos_token = True
        if self.tokenizer.pad_token_id is None:
            # Map pad_token to eos_token to avoid accidental masking or unk-id behavior
            # This is safer for LLaMA-like models (e.g., DeepSeek-Coder)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        if tokenizer_exists:
            print("Loading Tokenizer from local files:", tok_fl_nm)
        else:
            print("Downloading Tokenizer...")
            self.tokenizer.save_pretrained(tok_fl_nm, access_token=access_token)
            print("Tokenizer saved to: ", tok_fl_nm)

        # --- Determine source dir for local model files (if any) ---
        src_dir = None
        if exists(local_path):
            src_dir = local_path
        elif exists(raw_fl_nm):
            src_dir = raw_fl_nm

        # --- Build a safe config without relying on from_dict/get_config_dict ---
        config = None
        if src_dir is not None and os.path.exists(os.path.join(src_dir, "config.json")):
            # Read local config, sanitize if needed, and load via a temporary folder
            with open(os.path.join(src_dir, "config.json"), "r") as f:
                cfg_dict = json.load(f)
            if cfg_dict.get("quantization_config", "absent") is None:
                # Remove null to prevent HF internals from calling .to_dict() on None
                del cfg_dict["quantization_config"]

            # Write sanitized config into a small temp dir and load from there
            tmp_cfg_dir = tempfile.mkdtemp(prefix="sanitized_cfg_")
            try:
                with open(os.path.join(tmp_cfg_dir, "config.json"), "w") as f:
                    json.dump(cfg_dict, f)
                config = AutoConfig.from_pretrained(
                    tmp_cfg_dir, trust_remote_code=True, token=access_token
                )
            finally:
                # We can keep or clean; keeping is usually fine, but we’ll clean to avoid clutter.
                shutil.rmtree(tmp_cfg_dir, ignore_errors=True)

        if config is None:
            # Remote (or no local config found) → normal path
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=True, token=access_token
            )

        # --- Model ---
        # Figure out if ZeRO-3 is enabled (deepspeed arg can be dict or path)
        # Check training_args.deepspeed first if available, otherwise use use_deepspeed boolean
        use_zero3 = False
        if training_args is not None:
            deepspeed_cfg = getattr(training_args, "deepspeed", None)
            use_zero3 = bool(deepspeed_cfg)
        elif use_deepspeed:
            # Fallback: if use_deepspeed is True, assume ZeRO-3 might be used
            use_zero3 = True
        
        # Build model kwargs (sanitize for ZeRO-3)
        deepspeed_specific_prm = {} if use_zero3 else {"device_map": "auto"}
        model_kwargs = dict(
            trust_remote_code=True,
            max_memory={i: max_memory for i in range(torch.cuda.device_count())},
            token=access_token,
            torch_dtype=torch.bfloat16,  # QLoRA compute
            gguf_file=gguf_file,
            config=config,
            **deepspeed_specific_prm
        )
        
        # Only pass quantization_config if explicitly provided (prevents conflicts)
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
        
        # --- ZeRO-3 guard: strip incompatible args ---
        # NOTE: Other files using device_map (RAG_AlterNN.py, TuneRL.py, MergeLLM.py, etc.)
        # are safe because they don't use DeepSpeed/ZeRO-3. Only this LLM class needs sanitization
        # when training_args.deepspeed is set.
        if use_zero3:
            # Absolutely no device_map / low_cpu_mem_usage on ZeRO-3
            model_kwargs.pop("device_map", None)
            model_kwargs.pop("low_cpu_mem_usage", None)
            # (optional) these can also trip sharding heuristics—keep it simple on ZeRO-3:
            model_kwargs.pop("max_memory", None)
            model_kwargs.pop("offload_folder", None)
        
        # Debug: verify nothing slipped through
        print("[DEBUG from_pretrained kwargs]", {k: ("***" if k == "token" else v) for k, v in model_kwargs.items()})
        
        base_model = local_path if exists(local_path) else raw_fl_nm if exists(raw_fl_nm) else model_path
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            **model_kwargs
        )
        if exists(local_path):
            print("Loading Model from local files:", "'" + local_path + "'")
        elif exists(raw_fl_nm):
            print(f"Loading Raw Model from local files: '{raw_fl_nm}'")
        else:
            self.model.save_pretrained(raw_fl_nm, access_token=access_token)
            print("Model saved to: ", raw_fl_nm)

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

