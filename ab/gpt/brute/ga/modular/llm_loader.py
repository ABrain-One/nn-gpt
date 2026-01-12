import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class LocalLLMLoader:
    def __init__(self, model_path, use_quantization=True):
        self.model_path = model_path
        
        print(f"Loading Model: {model_path}")
        print(f"Quantization: {use_quantization}")

        # Quantization Config
        bnb_config = None
        if use_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True
            )

        # Load Tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except:
             # Fallback to local path if simple name fails
             self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load Model
        # Force device map to cuda if available and not quantized (or auto if quantized)
        # For 1.3B, we can usually just load to CUDA:0
        device_map = "auto" 
        if not use_quantization and torch.cuda.is_available():
            device_map = None # Move manually
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        except OSError:
             # Fallback to local
             self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True,
                local_files_only=True,
                 torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )

        if not use_quantization and torch.cuda.is_available():
            self.model.to("cuda")

        self.model.eval()

    def generate(self, prompt, max_new_tokens=1024, temperature=0.8, top_k=50, top_p=0.9):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up prompt echo
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]
            
        return generated_text.strip()

