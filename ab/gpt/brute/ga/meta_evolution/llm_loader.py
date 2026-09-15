import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os

def _load_model_config():
    """Load model_config.json from the pipeline directory. Raises error if missing."""
    pipeline_dir = os.environ.get("PIPELINE_DIR", os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(pipeline_dir, "model_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[Config] model_config.json not found at {config_path}. Please create it.")
    with open(config_path, "r") as f:
        config = json.load(f)
    config["context_length"] = config.get("context_length", 4096)
    print(f"[Config] Loaded model_config.json  (context_length={config['context_length']})")
    return config

def get_model_short_name():
    try:
        config = _load_model_config()
        full = config.get("base_model_name", "unknown").lower()
        if "qwen" in full: return "qwen"
        if "deepseek" in full: return "deepseek"
        if "mistral" in full: return "mistral"
        return full.split('/')[0]
    except Exception:
        return "unknown"

def get_dataset_name(script_path):
    filename = os.path.basename(script_path)
    if "cifar100" in filename:
        return "cifar100"
    elif "imagenet100" in filename:
        return "imagenet100"
    else:
        return "cifar10"


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
             self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=False)

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
                local_files_only=False,
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
            self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=True, local_files_only=False)
        else:
            if adapter_path and os.path.exists(adapter_path):
                print(f"[LoRA] Adapter directory exists at {adapter_path} but no weight files found. Initializing fresh adapters...")
            else:
                print("[LoRA] No adapter directory found. Initializing fresh adapters...")
            # Target modules for Qwen2.5 (same as DeepSeek — both Llama-style)
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
        
        messages = [
            {"role": "system", "content": "You are an elite AI Research Engineer and Evolutionary Computation Expert."},
            {"role": "user", "content": prompt}
        ]
        chat_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer(
            chat_text, return_tensors="pt", truncation=True,
            max_length=self.config.get("context_length", 4096)
        )
        
        # Add context length warning
        num_tokens = inputs.input_ids.shape[1]
        if num_tokens > 3500:
            print(f"[WARN] Prompt is very large ({num_tokens} tokens). Nearing 4096 limit, which may cause truncation and mode collapse.")
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
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5) # Lowered: 2e-4 was causing mode collapse
        
        accumulation_steps = min(4, len(training_data)) # Accumulate over up to 4 examples
        
        for epoch in range(epochs):
            total_loss = 0
            optimizer.zero_grad()
            
            for i, item in enumerate(training_data):
                messages = [
                    {"role": "system", "content": "You are an elite AI Research Engineer and Evolutionary Computation Expert."},
                    {"role": "user", "content": item['prompt']},
                    {"role": "assistant", "content": item['completion']}
                ]
                full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
                
                inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=self.config.get("context_length", 4096))
                if torch.cuda.is_available():
                    inputs = inputs.to("cuda")
                
                # --- Loss Masking: only compute loss on completion tokens ---
                prompt_msgs = [
                    {"role": "system", "content": "You are an elite AI Research Engineer and Evolutionary Computation Expert."},
                    {"role": "user", "content": item['prompt']}
                ]
                prompt_text = self.tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
                prompt_tokens = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=self.config.get("context_length", 4096))
                prompt_length = prompt_tokens["input_ids"].shape[1]
                
                labels = inputs["input_ids"].clone()
                # Safeguard: Don't mask out the entire sequence if the prompt got truncated
                mask_length = min(prompt_length, labels.shape[1] - 1)
                labels[0, :mask_length] = -100  # Mask prompt tokens from loss
                
                # # Causal LM: Labels = Inputs (old: trained on full prompt+completion)
                # outputs = self.model(**inputs, labels=inputs["input_ids"])
                outputs = self.model(**inputs, labels=labels)
                
                # Scale the loss since we are accumulating
                loss = outputs.loss / accumulation_steps
                
                # Safeguard: Skip if loss is NaN or Inf to prevent adapter corruption
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[WARN] Loss is {loss.item()}. Skipping gradient update for this example to prevent mode collapse.")
                    continue
                    
                loss.backward()
                
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(training_data):
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Re-scale loss for reporting total_loss
                total_loss += loss.item() * accumulation_steps
                
            avg_loss = total_loss / len(training_data)
            print(f"[LoRA] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            
    def save_adapters(self, save_path):
        # Save only adapters
        self.model.save_pretrained(save_path)

