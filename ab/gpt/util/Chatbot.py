# ab/gpt/util/Chatbot.py

from transformers import PreTrainedTokenizer, PreTrainedModel, pipeline
from ab.gpt.util.Util import extract_code, extract_hyperparam, extract_transform
import torch

extra_instructions = (
    " IMPORTANT: Minimize thinking and reasoning. Go directly to the code."
    " Use PyTorch for the implementation. Keep the code short and efficient."
    " Name the main class of the model \"Net\"."
    " The model code must include default parameters for initialization in the constructor."
    " Structure: Brief task summary (1-2 lines) → Approach (1-2 lines) → Code only."
    " NO lengthy explanations, NO step-by-step thinking, NO alternatives discussion."
    " Provide ONLY: <hp>hyperparameters</hp>, <tr>transform code</tr>, <nn>model code</nn>."
    " Each section must be COMPLETE code with no explanations."
    " Don't include comments in the code. Remove any other text."
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
        
        # Enforce max_new_tokens limit to prevent truncation
        if max_new_tokens is not None and max_new_tokens > 0:
            # Ensure we don't exceed the limit
            max_new_tokens = min(max_new_tokens, 16384)  # Cap at 16k tokens max
            print(f"[INFO] Enforcing max_new_tokens={max_new_tokens} to prevent truncation")
        
        if self.__keep_memory:
            self.__messages.append({"role": "user", "content": prompt})
            in_text = self.__messages
        else:
            in_next = [{"role": "user", "content": prompt}]
        
        # Use pipeline if available (PyTorch path)
        if self.__pipeline is not None:
            try:
                out = self.__pipeline(
                    in_next,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,  # Allow Random answer
                    max_len=max_len,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                )[0]["generated_text"][-1]['content']
                
                assert isinstance(out, str)
                
                # Validate output size and truncate if needed
                out = self._validate_output_size(out, max_new_tokens)

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
            
            # Validate output size and truncate if needed
            out = self._validate_output_size(out, max_new_tokens)
            
            if self.__keep_memory:
                self.__messages.append({"role": "assistant", "content": out})
            
            nn = extract_code(out)
            return nn, extract_hyperparam(out), extract_transform(out), out
            
        except Exception as e:
            print(f"[ERROR] Direct generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, ""

    def _validate_output_size(self, output: str, max_new_tokens: int) -> str:
        """
        Validate and enforce output size within max_new_tokens limit.
        
        This prevents truncation by:
        1. Stop generation early when all tags close
        2. Intelligently truncate at meaningful boundaries
        3. Strip unnecessary whitespace/comments 
        4. Close any unclosed tags (</nn>, </hp>, </tr>)
        5. Ensure extracted code remains valid
        
        Args:
            output: Generated text to validate
            max_new_tokens: Maximum allowed tokens
            
        Returns:
            Validated/truncated output with closed tags and stripped whitespace
        """
        if max_new_tokens is None or max_new_tokens <= 0:
            cleaned = self._strip_unnecessary_content(output)
            return self._close_unclosed_tags(cleaned)
        
        # Solution 3: Check if generation is complete (all tags closed)
        if self._is_generation_complete(output):
            print("[INFO] Generation complete - all tags properly closed, truncating after last tag")
            truncated = self._truncate_after_tag_closure(output)
            cleaned = self._strip_unnecessary_content(truncated)
            return self._close_unclosed_tags(cleaned)
        
        # Tokenize to count tokens
        tokens = self.tokenizer.encode(output)
        token_count = len(tokens)
        
        if token_count <= max_new_tokens:
            # Within limit, apply Solution 5: strip unnecessary content
            cleaned = self._strip_unnecessary_content(output)
            return self._close_unclosed_tags(cleaned)
        
        # Exceeded limit - need to truncate intelligently
        print(f"[WARN] Output size ({token_count} tokens) exceeds max_new_tokens={max_new_tokens}")
        
        # Try to truncate at code boundary (look for class/function end)
        truncated = self._truncate_at_boundary(output, max_new_tokens)
        
        # Strip unnecessary content and close tags
        cleaned = self._strip_unnecessary_content(truncated)
        cleaned = self._close_unclosed_tags(cleaned)
        
        new_token_count = len(self.tokenizer.encode(cleaned))
        print(f"[INFO] Truncated from {token_count} to {new_token_count} tokens")
        
        return cleaned
    
    def _close_unclosed_tags(self, text: str) -> str:
        """
        Close any unclosed XML-like tags in the output.
        
        This ensures that truncated output has complete tag structure:
        - </nn> for <nn>...</nn> blocks
        - </hp> for <hp>...</hp> blocks  
        - </tr> for <tr>...</tr> blocks
        
        Args:
            text: Text potentially with unclosed tags
            
        Returns:
            Text with all tags properly closed
        """
        # List of tags to check and close
        tags_to_close = [('</nn>', '<nn>'), ('</hp>', '<hp>'), ('</tr>', '<tr>')]
        
        for closing_tag, opening_tag in tags_to_close:
            open_count = text.count(opening_tag)
            close_count = text.count(closing_tag)
            
            # If we have more opening tags than closing tags, add closing tags
            if open_count > close_count:
                missing_closes = open_count - close_count
                print(f"[INFO] Adding {missing_closes} missing {closing_tag} tags")
                text = text + (closing_tag + '\n') * missing_closes
        
        return text
    
    def _is_generation_complete(self, text: str) -> bool:
        """
        Check if generation is complete - all XML tags are properly paired.
        
        Solution 3: Stop early when all tags close to reduce output tokens.
        
        Args:
            text: Generated text
            
        Returns:
            True if all opening tags have corresponding closing tags
        """
        tags_to_check = [('</nn>', '<nn>'), ('</hp>', '<hp>'), ('</tr>', '<tr>')]
        
        for closing_tag, opening_tag in tags_to_check:
            open_count = text.count(opening_tag)
            close_count = text.count(closing_tag)
            
            # If any tag is not closed, generation is incomplete
            if open_count > 0 and open_count != close_count:
                return False
        
        return True
    
    def _truncate_after_tag_closure(self, text: str) -> str:
        """
        Truncate text immediately after the last complete XML tag closes.
        
        Solution 3: Remove any reasoning or output after the final </tr>, </hp>, or </nn>.
        
        Args:
            text: Text potentially with extra content after tag closure
            
        Returns:
            Text truncated after last tag closes
        """
        closing_tags = ['</nn>', '</hp>', '</tr>']
        last_pos = -1
        last_tag = None
        
        # Find the rightmost closing tag
        for tag in closing_tags:
            pos = text.rfind(tag)
            if pos > last_pos:
                last_pos = pos
                last_tag = tag
        
        if last_pos != -1 and last_tag:
            # Truncate right after the closing tag
            truncated = text[:last_pos + len(last_tag)]
            print(f"[INFO] Truncated after {last_tag} (position {last_pos})")
            return truncated
        
        return text
    
    def _strip_unnecessary_content(self, text: str) -> str:
        """
        Remove unnecessary whitespace, empty lines, and comments from output.
        
        Solution 5: Reduce output size by:
        1. Removing multiple consecutive blank lines (keep max 1)
        2. Stripping trailing whitespace from each line
        3. Removing lines that are only whitespace
        4. Reducing excessive indentation (preserve structure, remove excess)
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text with reduced whitespace
        """
        if not text:
            return text
        
        lines = text.split('\n')
        cleaned_lines = []
        prev_blank = False
        
        for line in lines:
            # Strip trailing whitespace
            stripped = line.rstrip()
            
            # Check if line is empty or only whitespace
            if not stripped:
                # Keep at most one blank line in a row
                if not prev_blank:
                    cleaned_lines.append('')
                    prev_blank = True
            else:
                # For non-empty lines, preserve leading whitespace but remove tabs if possible
                # Keep indentation for code structure
                leading_spaces = len(line) - len(line.lstrip())
                # Normalize: max 2 levels of indentation per line (8 spaces per level)
                normalized_indent = min(leading_spaces, 16)
                cleaned_line = ' ' * normalized_indent + stripped[leading_spaces:] if leading_spaces < len(line) else stripped
                cleaned_lines.append(cleaned_line)
                prev_blank = False
        
        # Join and remove trailing blank lines
        result = '\n'.join(cleaned_lines).rstrip()
        
        # Only log if we made significant changes
        original_lines = len(text.split('\n'))
        cleaned_line_count = len(result.split('\n'))
        if cleaned_line_count < original_lines:
            print(f"[INFO] Stripped unnecessary content: {original_lines} → {cleaned_line_count} lines")
        
        return result

    def _truncate_at_boundary(self, text: str, max_tokens: int) -> str:
        """
        Truncate text at meaningful boundary (end of class/function definition).
        
        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed
            
        Returns:
            Truncated text at safe boundary
        """
        # Binary search to find safe truncation point
        low, high = len(text) // 4, len(text)  # Start from 1/4 length
        best_truncation = text[:max_tokens*4]  # Fallback: rough character-to-token conversion
        
        # Look for natural boundaries (closing brackets, newlines)
        lines = text.split('\n')
        accumulated = ""
        
        for line in lines:
            test_text = accumulated + line + '\n'
            token_count = len(self.tokenizer.encode(test_text))
            
            if token_count <= max_tokens:
                accumulated = test_text
                best_truncation = accumulated
            else:
                # Exceeded limit, return previous good state
                break
        
        return best_truncation.rstrip()
