"""
LLM Client for Qwen3 8B using Transformers library.
Provides a unified interface for text generation.
"""

import os
import json
import torch
import random
import numpy as np
import requests
from ab.gpt.util.LLM import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# Default seed for reproducibility
DEFAULT_SEED = 43


class LLMClient:
    """LLM Client that loads Qwen3 8B and provides generation interface."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", seed: int = DEFAULT_SEED, use_remote: bool = False, use_unsloth: bool = False, *args, **kwargs):
        """
        Initialize the LLM client.
        
        Args:
            model_name: HuggingFace model name or remote model ID. Default is Qwen2.5-7B-Instruct.
            seed: Random seed for reproducible generation. Default is 43.
            use_remote: Whether to use a remote API.
            use_unsloth: Whether to use Unsloth for optimization.
        """
        self.model_name = model_name
        self.seed = seed
        self.use_remote = use_remote
        self.use_unsloth = use_unsloth
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize call counter for deterministic seeding across runs
        self._call_counter = 0
        
        # Set initial seed
        self._set_seed_with_counter()
        
        if self.use_remote:
            self.api_key = os.environ.get("SiliconCloud_Key")
            if not self.api_key:
                raise ValueError("Environment variable 'SiliconCloud_Key' is not set for remote model usage.")
            self.api_url = "https://api.siliconflow.cn/v1/chat/completions"
            print(f"Initialized remote client for model {model_name}")
        else:
            model_tokenizer_loader = LLM(
                model_path=model_name,
                use_unsloth=self.use_unsloth,
            )
            self.model = model_tokenizer_loader.model
            self.tokenizer = model_tokenizer_loader.tokenizer
    
    def _set_seed_with_counter(self):
        """Set seed based on call counter for reproducibility across runs but diversity within run."""
        # Use base seed + call counter so each call gets a unique but reproducible seed
        current_seed = self.seed + self._call_counter
        self._call_counter += 1
        
        set_seed(current_seed)
        torch.manual_seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(current_seed)
        np.random.seed(current_seed)
        random.seed(current_seed)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        Generate text from the given prompt.
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            
        Returns:
            Generated text response
        """
        # Set seed based on counter to ensure diversity + reproducibility
        self._set_seed_with_counter()
        
        if self.use_remote:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
            try:
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=300)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"Error calling remote API: {e}")
                return ""
        else:
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Remove the input tokens from the output
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response

if __name__ == "__main__":
    # Test the LLM client
    client = LLMClient()
    response = client.generate("Hello, how are you?")
    print(f"Response: {response}")
