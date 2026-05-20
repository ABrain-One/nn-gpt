import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os

def _load_model_config():
    """Load model_config.json from the same directory. Raises error if missing."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[Config] model_config.json not found at {config_path}. Please create it.")
    with open(config_path, "r") as f:
        config = json.load(f)
    print(f"[Config] Loaded model_config.json  (context_length={config.get('context_length', 'N/A')})")
    return config

class LocalLLMLoader:
    def __init__(self, model_path=None, use_quantization=True, adapter_path=None):
        # Load centralised config
        self.config = _load_model_config()

        # If no model_path provided, use the one from config
        # self.model_path = model_path
        if model_path is None:
            model_path = self.config["base_model_name"]
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

        # Prepare for Training (k-bit)
        if use_quantization:
            self.model = prepare_model_for_kbit_training(self.model)

        # Initialize or Load LoRA
        # Check if adapter weight files actually exist (not just config/metadata)
        adapter_weights_exist = False
        if adapter_path and os.path.exists(adapter_path):
            weight_files = ["adapter_model.safetensors", "adapter_model.bin"]
            adapter_weights_exist = any(os.path.isfile(os.path.join(adapter_path, wf)) for wf in weight_files)

        if adapter_weights_exist:
            print(f"[LoRA] Loading existing adapters from {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=True, local_files_only=True)
        else:
            if adapter_path and os.path.exists(adapter_path):
                print(f"[LoRA] Adapter directory exists at {adapter_path} but no weight files found. Initializing fresh adapters...")
            else:
                print("[LoRA] No adapter directory found. Initializing fresh adapters...")
            # Target modules for DeepSeek
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, peft_config)
        
        self.model.print_trainable_parameters()

    def generate(self, prompt, max_new_tokens=1024, temperature=0.8, top_k=50, top_p=0.9):
        # Ensure model is in eval mode for generation
        self.model.eval()
        
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=self.config.get("context_length", 4096)
        )
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
        
        # Decode only the new tokens to avoid prompt echoing issues
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
        return generated_text.strip()

    def train_on_buffer(self, training_data, epochs=1):
        """
        Fine-tune on the collected buffer (Prompt + Completion).
        data format: [{'prompt': '...', 'completion': '...'}, ...]
        """
        if not training_data:
            return
            
        print(f"[LoRA] Training on {len(training_data)} examples...")
        self.model.train()
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-4) # Higher LR for quick adaptation?
        
        for epoch in range(epochs):
            total_loss = 0
            for item in training_data:
                # Format: "Prompt... \n Completion..."
                full_text = item['prompt'] + "\n" + item['completion']
                
                # inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)
                inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=self.config.get("context_length", 4096))
                if torch.cuda.is_available():
                    inputs = inputs.to("cuda")
                
                # Causal LM: Labels = Inputs
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(training_data)
            print(f"[LoRA] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            
    def save_adapters(self, save_path):
        # Save only adapters
        self.model.save_pretrained(save_path)

