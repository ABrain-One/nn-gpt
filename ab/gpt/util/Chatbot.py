# ab/gpt/util/Chatbot.py

from transformers import PreTrainedTokenizer, PreTrainedModel, pipeline
from ab.gpt.util.Util import extract_code, extract_hyperparam, extract_transform, extract_all_to_train
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

def _extract_generated_content(item):
    """Normalize HF pipeline outputs across single/batch return shapes."""
    cur = item
    if isinstance(cur, list):
        cur = cur[0] if cur else ""
    if isinstance(cur, dict):
        generated = cur.get("generated_text", "")
        if isinstance(generated, list):
            if not generated:
                return ""
            last = generated[-1]
            if isinstance(last, dict):
                return last.get("content", "")
            if isinstance(last, str):
                return last
            return str(last)
        if isinstance(generated, str):
            return generated
        return str(generated)
    if isinstance(cur, str):
        return cur
    return str(cur)


def _strip_prompt_prefix(text, prompt):
    """Remove echoed prompt text from generation output when pipeline returns full text."""
    if not isinstance(text, str):
        return text
    if isinstance(prompt, str) and prompt and text.startswith(prompt):
        return text[len(prompt):].lstrip()
    return text

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

    def _prepare_pipeline_input(self, prompt_text):
        """Build a pipeline-ready text prompt using chat template when available."""
        messages = [{"role": "user", "content": prompt_text}]
        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        return prompt_text

    def _direct_generate_batch(self, prompts, max_new_tokens=None, max_len=None):
        """Run true batched generation via model.generate and strip prompt prefixes by token length."""
        if hasattr(self.model, "eval"):
            self.model.eval()

        formatted_prompts = [self._prepare_pipeline_input(p) for p in prompts]
        tokenizer_max_len = getattr(self.tokenizer, "model_max_length", None)
        if tokenizer_max_len is None or tokenizer_max_len > 10**8:
            tokenizer_max_len = 4096
        token_budget = max_new_tokens or 4096
        max_input_len = max(1, tokenizer_max_len - token_budget)

        original_padding_side = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = "left"
        try:
            inputs = self.tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_len,
            )
        finally:
            self.tokenizer.padding_side = original_padding_side

        if 'input_ids' in inputs:
            input_ids = inputs['input_ids']
            vocab_size = self.tokenizer.vocab_size
            max_token_id = input_ids.max().item()
            if max_token_id >= vocab_size:
                print(f"[WARN] Invalid token IDs detected in batch: max_id={max_token_id}, vocab_size={vocab_size}")
                clamp_value = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else vocab_size - 1
                inputs['input_ids'] = torch.clamp(input_ids, max=clamp_value)

        if hasattr(self.model, 'device') and self.model.device is not None:
            device = self.model.device
        elif self.is_onnx:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        inputs = {k: v.to(device) for k, v in inputs.items()}
        if 'attention_mask' in inputs:
            input_lengths = inputs['attention_mask'].sum(dim=1).tolist()
        else:
            input_lengths = [inputs['input_ids'].shape[1]] * inputs['input_ids'].shape[0]

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

        results = []
        for i in range(outputs.shape[0]):
            generated_ids = outputs[i][int(input_lengths[i]):]
            generated = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            nn = extract_code(generated)
            results.append((nn, extract_hyperparam(generated), extract_transform(generated), generated))
        return results

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
                generation_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": True,
                    "max_len": max_len,
                    "temperature": self.temperature,
                    "top_k": self.top_k,
                    "top_p": self.top_p,
                }
                try:
                    out_item = self.__pipeline(
                        in_next,
                        return_full_text=False,
                        **generation_kwargs,
                    )[0]
                    out = _extract_generated_content(out_item)
                except TypeError:
                    out_item = self.__pipeline(in_next, **generation_kwargs)[0]
                    out = _extract_generated_content(out_item)
                
                assert isinstance(out, str)
                
                if self.__keep_memory:
                    self.__messages.append({"role": "assistant", "content": out})

                return (*extract_all_to_train(out), out)
                
            except Exception as e:
                print(f"[ERROR] Pipeline generation failed: {e}")
                print("[INFO] Falling back to direct generation")
        
        # Direct generation (ONNX or PyTorch fallback)
        return self._direct_generate(in_next, max_new_tokens, max_len)

    def chat_batch(self, prompts, max_len=None, max_new_tokens=None, engineer_prompt=True):
        """Batch generation for multiple prompts; falls back to per-prompt generation."""
        if not prompts:
            return []

        if self.__keep_memory:
            return [self.chat(p, max_len=max_len, max_new_tokens=max_new_tokens, engineer_prompt=engineer_prompt) for p in prompts]

        prepared_prompts = [p + extra_instructions if engineer_prompt else p for p in prompts]
        if self.__pipeline is not None or not self.is_onnx:
            try:
                return self._direct_generate_batch(prepared_prompts, max_new_tokens=max_new_tokens, max_len=max_len)
            except Exception as e:
                print(f"[WARN] Direct batch generation failed: {e}")
                print("[INFO] Falling back to per-prompt generation")

        return [self.chat(p, max_len=max_len, max_new_tokens=max_new_tokens, engineer_prompt=engineer_prompt) for p in prompts]

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


            # -- FIX 1: Validate token IDs before GPU move -- 

            if 'input_ids' in inputs:
                input_ids = inputs['input_ids']
                vocab_size = self.tokenizer.vocab_size
                max_token_id = input_ids.max().item()

            if max_token_id >= vocab_size:
                print(f"[WARN] Invalid token IDs detected: max_id={max_token_id}, vocab_size={vocab_size}")
                print(f"[WARN] Clamping to valid range [0, {vocab_size-1}]")

            clamp_value = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else vocab_size - 1
            input_ids = torch.clamp(input_ids, max=clamp_value)
            inputs['input_ids'] = input_ids
            print(f"[WARN] After clamping: max_id={input_ids.max().item()}")

            
            # Move to appropriate device
            # if hasattr(self.model, 'device'):
            #     device = self.model.device
            # elif self.is_onnx:
            #     device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            # else:
            #     device = next(self.model.parameters()).device
            
            if hasattr(self.model, 'device') and self.model.device is not None:
                device = self.model.device
            elif self.is_onnx:
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            else:
                try:
                    device = next(self.model.parameters()).device
                except StopIteration:
                    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


                    
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
            
            return (*extract_all_to_train(out), out)
            
        except Exception as e:
            print(f"[ERROR] Direct generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, ""
