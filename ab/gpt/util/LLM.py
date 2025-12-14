from ab.nn.util.Const import out_dir
from ab.gpt.util.Const import llm_dir, llm_tokenizer_dir
from ab.gpt.util.LLMUtil import quantization_config_4bit
from ab.gpt.util.Util import exists

import os
import json
import tempfile
import shutil
import torch
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
                 gguf_file=None):
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
        
        # FIX: Ensure pad_token is set (critical for batching)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        # FIX: Force Left Padding for Batch Inference (generation)
        self.tokenizer.padding_side = "left"

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
                # We can keep or clean; keeping is usually fine.
                shutil.rmtree(tmp_cfg_dir, ignore_errors=True)

        if config is None:
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=True, token=access_token
            )

        # --- Model ---
        model_kwargs = dict(
            trust_remote_code=True,
            token=access_token,
            torch_dtype=torch.float16, # Use float16 for GPU efficiency
            gguf_file=gguf_file,
            config=config,
            # FIX: Use "auto" to load onto GPU, not None (CPU)
            device_map="auto" 
        )

        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config

        self.model = AutoModelForCausalLM.from_pretrained(
            local_path if exists(local_path) else raw_fl_nm if exists(raw_fl_nm) else model_path,
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
