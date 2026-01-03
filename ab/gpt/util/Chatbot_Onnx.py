# ab/gpt/util/Chatbot.py

from transformers import PreTrainedTokenizer, PreTrainedModel, pipeline
from ab.gpt.util.Util import extract_code, extract_hyperparam, extract_transform
import torch

extra_instructions = (
    " Use PyTorch for the implementation. Keep the code short. Name the main class of the model \"Net\"."
    " The model code must include default parameters for initialization in the constructor. "
    "Provide only the code. Don't provide any explanation. Remove any text from this reply. "
    "Don't include comments in the code."
)

example_prompt = (
    "Write PyTorch code for an efficient classification model that includes self-attention blocks."
    + extra_instructions
)

class ChatBot:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, keep_memory=False,
                 temperature=1.0, top_k=50, top_p=0.9):
        self.show_additional_info = False
        self.model = model
        self.tokenizer = tokenizer
        self.__keep_memory = keep_memory
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        
        # Check if model is ONNX (wrapped or direct ORTModel)
        self.is_onnx = (
            hasattr(model, 'ort_model') or  # Our OnnxCausalLMWrapper
            type(model).__name__ == 'ORTModelForCausalLM' or
            'ORTModel' in type(model).__name__
        )
        
        # Only create pipeline for PyTorch models
        if not self.is_onnx:
            try:
                self.__pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                )
                print("[INFO] Using Hugging Face pipeline for generation")
            except Exception as e:
                print(f"[WARN] Pipeline creation failed: {e}")
                print("[INFO] Falling back to direct generation")
                self.__pipeline = None
        else:
            print("[INFO] ONNX model detected, using direct generation (no pipeline)")
            self.__pipeline = None
        
        if self.__keep_memory:
            self.__messages = []

    def chat(self, prompt: str, max_len=None, max_new_tokens=None, engineer_prompt=True) -> tuple[str, str, str, str]:
        # Set model to eval mode (no-op for ONNX)
        if hasattr(self.model, "eval"):
            self.model.eval()
        
        if engineer_prompt:
            prompt += extra_instructions
        
        if self.__keep_memory:
            self.__messages.append({"role": "user", "content": prompt})
            in_next = self.__messages
        else:
            in_next = [{"role": "user", "content": prompt}]
        
        # Use pipeline if available (PyTorch path)
        if self.__pipeline is not None:
            try:
                out = self.__pipeline(
                    in_next,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    max_len=max_len,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                )[0]["generated_text"][-1]['content']
                
                assert isinstance(out, str)
                
                if self.__keep_memory:
                    self.__messages.append({"role": "assistant", "content": out})
                
                nn = extract_code(out)
                return nn, extract_hyperparam(out), extract_transform(out), out
                
            except Exception as e:
                print(f"[ERROR] Pipeline generation failed: {e}")
                print("[INFO] Falling back to direct generation")
        
        # Direct generation (ONNX or PyTorch fallback)
        return self._direct_generate(in_next, max_new_tokens, max_len)

    def _direct_generate(self, messages, max_new_tokens, max_len):
        """Direct model.generate() call without pipeline - works for ONNX and PyTorch"""
        try:
            # Apply chat template to format messages
            if hasattr(self.tokenizer, 'apply_chat_template'):
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback: concatenate messages
                formatted_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.tokenizer.model_max_length - (max_new_tokens or 4096)
            )
            
            # Move to appropriate device
            if hasattr(self.model, 'device'):
                device = self.model.device
            elif self.is_onnx:
                device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            else:
                device = next(self.model.parameters()).device
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # FIX: Store input length before generation
            input_length = inputs['input_ids'].shape[-1]  # Use shape[-1] for sequence length
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens or 4096,
                    max_length=max_len,
                    do_sample=True,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # FIX: Decode only the generated part (skip input prompt)
            generated_ids = outputs[0][input_length:]  # Use input_length, not shape[1]
            out = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            assert isinstance(out, str)
            
            if self.__keep_memory:
                self.__messages.append({"role": "assistant", "content": out})
            
            nn = extract_code(out)
            return nn, extract_hyperparam(out), extract_transform(out), out
            
        except Exception as e:
            print(f"[ERROR] Direct generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, ""
