# ab/gpt/util/Chatbot.py

import os
import re

from transformers import PreTrainedTokenizer, PreTrainedModel, pipeline
from ab.gpt.util.GenerationDType import align_generation_head_dtype, infer_generation_head_dtype
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


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _safe_tokenizer_max_length(tokenizer, fallback: int = 4096) -> int:
    tokenizer_max_len = getattr(tokenizer, "model_max_length", None)
    try:
        tokenizer_max_len = int(tokenizer_max_len)
    except (TypeError, ValueError, OverflowError):
        tokenizer_max_len = int(fallback)
    if tokenizer_max_len <= 0 or tokenizer_max_len > 10**8:
        tokenizer_max_len = int(fallback)
    return tokenizer_max_len


def _effective_vocab_size(tokenizer) -> int:
    try:
        return max(int(tokenizer.vocab_size), len(tokenizer))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return len(tokenizer)


def _strip_reasoning_output(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    first_tag_positions = [
        pos for pos in (
            cleaned.find("<block>"),
            cleaned.find("<nn>"),
            cleaned.find("```python"),
            cleaned.find("```"),
        )
        if pos >= 0
    ]
    if first_tag_positions:
        return cleaned[min(first_tag_positions):].lstrip()
    return cleaned

class ChatBot:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, keep_memory=False,
                 temperature=1.0, top_k=50, top_p=0.9, system_prompt: str = None):
        self.show_additional_info = False
        self.model = model
        self.tokenizer = tokenizer
        self.__keep_memory = keep_memory
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.system_prompt = system_prompt
        self.disable_chat_template = _env_flag("NNGPT_DISABLE_CHAT_TEMPLATE")
        self.strip_think_output = _env_flag("NNGPT_STRIP_THINK_OUTPUT")
        
        # Check if model is ONNX (wrapped or direct ORTModel)
        self.is_onnx = (
            hasattr(model, 'ort_model') or  # Our OnnxCausalLMWrapper
            type(model).__name__ == 'ORTModelForCausalLM' or
            'ORTModel' in type(model).__name__
        )

        if not self.is_onnx:
            generation_dtype = infer_generation_head_dtype(self.model)
            align_generation_head_dtype(
                self.model,
                generation_dtype,
                log_prefix="[ChatBot]",
            )
        
        # Only create pipeline for PyTorch models
        force_direct = os.getenv("NNGPT_FORCE_DIRECT_GENERATE", "").strip().lower() in {"1", "true", "yes", "on"}
        if force_direct:
            print("[INFO] NNGPT_FORCE_DIRECT_GENERATE set, using direct generation")
            self.__pipeline = None
        elif not self.is_onnx:
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

    def _build_messages(self, user_content: str) -> list:
        """Build a messages list with optional system role prepended."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages

    def _prepare_pipeline_input(self, prompt_text):
        """Build a pipeline-ready text prompt using chat template when available."""
        messages = self._build_messages(prompt_text)
        if not self.disable_chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        if self.system_prompt:
            return f"System: {self.system_prompt}\nUser: {prompt_text}\nAssistant:"
        return prompt_text

    def _direct_generate_batch(self, prompts, max_new_tokens=None, max_len=None):
        """Run true batched generation via model.generate and strip prompt prefixes by token length."""
        if hasattr(self.model, "eval"):
            self.model.eval()

        formatted_prompts = [self._prepare_pipeline_input(p) for p in prompts]
        tokenizer_max_len = _safe_tokenizer_max_length(self.tokenizer)
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
                add_special_tokens=False,  # chat template already includes BOS; prevents EOS being appended
            )
        finally:
            self.tokenizer.padding_side = original_padding_side

        if 'input_ids' in inputs:
            input_ids = inputs['input_ids']
            vocab_size = _effective_vocab_size(self.tokenizer)
            max_token_id = input_ids.max().item()
            if max_token_id >= vocab_size:
                print(f"[WARN] Invalid token IDs detected in batch: max_id={max_token_id}, vocab_size={vocab_size}")
                inputs['input_ids'] = torch.clamp(input_ids, max=vocab_size - 1)

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
        # With left-padding, all sequences in the batch share the same padded input length.
        # The generated tokens start at index padded_input_length in each output row.
        padded_input_length = inputs['input_ids'].shape[1]

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
            generated_ids = outputs[i][padded_input_length:]
            generated = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            if self.strip_think_output:
                generated = _strip_reasoning_output(generated)
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
            if not self.__messages and self.system_prompt:
                self.__messages.append({"role": "system", "content": self.system_prompt})
            self.__messages.append({"role": "user", "content": prompt})
            in_next = self.__messages
        else:
            in_next = self._build_messages(prompt)
        
        # Use pipeline if available (PyTorch path)
        if self.__pipeline is not None:
            try:
                generation_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": True,
                    "max_length": max_len,
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
                if self.strip_think_output:
                    out = _strip_reasoning_output(out)
                
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
            if not self.disable_chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback: concatenate messages
                formatted_prompt = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])
                formatted_prompt = f"{formatted_prompt}\nAssistant:"
            
            tokenizer_max_len = _safe_tokenizer_max_length(self.tokenizer)
            token_budget = max_new_tokens or 4096
            max_input_len = max(1, tokenizer_max_len - token_budget)

            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max(max_input_len, 128),
                add_special_tokens=False,  # chat template already includes BOS; prevents EOS being appended
            )


            # -- FIX 1: Validate token IDs before GPU move -- 
            if 'input_ids' in inputs:
                input_ids = inputs['input_ids']
                vocab_size = _effective_vocab_size(self.tokenizer)
                max_token_id = input_ids.max().item()

                if max_token_id >= vocab_size:
                    print(f"[WARN] Invalid token IDs detected: max_id={max_token_id}, vocab_size={vocab_size}")
                    print(f"[WARN] Clamping to valid range [0, {vocab_size-1}]")
                    input_ids = torch.clamp(input_ids, max=vocab_size - 1)
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
            if self.strip_think_output:
                out = _strip_reasoning_output(out)
            
            assert isinstance(out, str)
            
            if self.__keep_memory:
                self.__messages.append({"role": "assistant", "content": out})
            
            return (*extract_all_to_train(out), out)
            
        except Exception as e:
            print(f"[ERROR] Direct generation failed: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError("Direct generation failed") from e
