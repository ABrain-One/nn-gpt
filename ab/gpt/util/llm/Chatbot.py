# ab/gpt/util/llm/Chatbot.py

import os
import re

from transformers import PreTrainedModel, pipeline
from ab.gpt.util.llm.GenerationDType import align_generation_head_dtype, infer_generation_head_dtype
from ab.gpt.util.llm.LLMUtil import drop_vision_tower
from ab.gpt.util.Util import extract_code, extract_hyperparam, extract_transform, extract_all_to_train
from ab.gpt.util.prompt.Prompt import DEFAULT_CHAT_TEMPLATE
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


def _unwrap_tokenizer(tokenizer_or_processor):
    """Return the real tokenizer behind a multimodal processor.

    Thin wrapper over the shared implementation in LLMUtil: this file,
    VLLMSleep and Prompt each grew their own copy, and three copies of one
    rule is how they drift apart. The import is local so that nothing in the
    module import order depends on it.
    """
    from ab.gpt.util.llm.LLMUtil import unwrap_tokenizer
    return unwrap_tokenizer(tokenizer_or_processor)


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
    if tokenizer_max_len <= 0 or tokenizer_max_len > 10 ** 8:
        tokenizer_max_len = int(fallback)
    return tokenizer_max_len


# The prompt is never cut to make room for the output. The output budget shrinks
# to fit the real context window; a prompt that leaves fewer than this many tokens
# is skipped (empty output, counted as a failed generation by the caller).
# NNGPT_LEGACY_PROMPT_TRUNCATION=1 restores the old truncating behaviour.
N2_MIN_NEW_TOKENS = 256
SKIPPED_OUTPUT = (None, None, None, "")


def _legacy_truncation() -> bool:
    return _env_flag("NNGPT_LEGACY_PROMPT_TRUNCATION")


def _context_limit(model, tokenizer, fallback: int = 4096) -> int:
    """Smallest real context window any component will tolerate.

    config.max_position_embeddings alone is NOT safe. Unsloth (and some other
    wrappers) rebuild RoPE / attention buffers for the max_seq_length passed at
    load time, which is routinely far smaller than the architectural window:
    Qwen2.5-Coder reports 32768 in its config while FastLanguageModel may have
    been built for 4096. Feeding a longer prompt then fails deep inside
    generate() with "The size of tensor a (4096) must match the size of tensor
    b (7462) at non-singleton dimension 3", and every length guard that trusted
    the config number waved the prompt through. Take the minimum of everything
    that claims to be a limit.

    VL checkpoints (Qwen3.8-27B) nest the language-model settings under
    config.text_config and may expose nothing useful at the top level, so that
    sub-config is inspected too.
    """
    candidates = []
    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", None) if cfg is not None else None
    generation_cfg = getattr(model, "generation_config", None)
    for obj in (model, cfg, text_cfg, generation_cfg):
        if obj is None:
            continue
        for attr in ("max_seq_length", "max_position_embeddings", "n_positions", "seq_length"):
            n = getattr(obj, attr, None)
            try:
                n = int(n)
            except (TypeError, ValueError, OverflowError):
                continue
            if 0 < n <= 10 ** 7:
                candidates.append(n)
    if not candidates:
        return _safe_tokenizer_max_length(tokenizer, fallback)
    return min(candidates)


def _output_budget(ctx: int, in_len: int, max_new_tokens) -> "int | None":
    """New-token budget that fits after an in_len-token prompt, or None to skip it."""
    want = max_new_tokens or 4096
    room = ctx - in_len
    if room < min(N2_MIN_NEW_TOKENS, want):
        return None
    return min(want, room)


def _effective_vocab_size(tokenizer) -> "int | None":
    """Largest id the embedding table is known to accept, or None if unknown.

    Returns None instead of raising when neither .vocab_size nor __len__ nor
    get_vocab() is available; callers then skip the clamp rather than crashing.
    A processor is unwrapped first, so the old AttributeError/TypeError pair
    ('Qwen3VLProcessor' has no attribute 'vocab_size' / has no len()) cannot
    occur.
    """
    tok = _unwrap_tokenizer(tokenizer)
    sizes = []
    for getter in (
            lambda: int(tok.vocab_size),
            lambda: len(tok),
            lambda: len(tok.get_vocab()),
    ):
        try:
            n = getter()
        except (AttributeError, TypeError, ValueError, OverflowError):
            continue
        if isinstance(n, int) and n > 0:
            sizes.append(n)
    return max(sizes) if sizes else None


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
    MIN_PROMPT_TOKENS = 1024  # never starve the prompt below this

    def __init__(self, model: PreTrainedModel, tokenizer, keep_memory=False,
                 temperature=1.0, top_k=50, top_p=0.9, system_prompt: str = None,
                 budget_tokens=None, suppress_thinking=False):
        self.show_additional_info = False
        self.model = model
        # `tokenizer` may be a processor (Qwen3.8-27B and other VL checkpoints).
        # Keep the original for any image-bearing path, but do all text work
        # through the unwrapped tokenizer.
        self.processor = tokenizer
        self.tokenizer = _unwrap_tokenizer(tokenizer)
        if self.tokenizer is not self.processor:
            print(f"[ChatBot] {type(self.processor).__name__} detected; "
                  f"using its inner {type(self.tokenizer).__name__} for all text operations.")
        self.__keep_memory = keep_memory
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.system_prompt = system_prompt
        self.disable_chat_template = _env_flag("NNGPT_DISABLE_CHAT_TEMPLATE")
        self.strip_think_output = _env_flag("NNGPT_STRIP_THINK_OUTPUT")
        self.budget_tokens = budget_tokens
        self.suppress_thinking = suppress_thinking

        # Tokenizers for models like ABrain/NNGPT-UniqueArch-Rag ship without a
        # chat_template, so apply_chat_template raises ValueError during generation
        # (swallowed as "no code generated"). Apply the same User/Assistant fallback
        # used for the training-data build so training and inference prompt formats match.
        # Tokenizers that already define chat_template (Qwen, DeepSeek, ...) are untouched.
        if not self.disable_chat_template and not getattr(self.tokenizer, "chat_template", None):
            inherited = getattr(self.processor, "chat_template", None)
            if inherited:
                # VL processors often hold the template while the inner tokenizer
                # does not. Reuse the real one rather than the generic fallback.
                print("[ChatBot] Taking chat_template from the processor.")
                self.tokenizer.chat_template = inherited
            else:
                print("[ChatBot] Tokenizer has no chat_template; applying default User/Assistant fallback.")
                self.tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

        # generate() errors out when pad_token_id is None, which is common on
        # checkpoints that only define eos.
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            eos = getattr(self.tokenizer, "eos_token", None)
            if eos is not None:
                self.tokenizer.pad_token = eos
                print(f"[ChatBot] pad_token was unset; using eos_token ({eos!r}).")

        # Check if model is ONNX (wrapped or direct ORTModel)
        self.is_onnx = (
                hasattr(model, 'ort_model') or  # Our OnnxCausalLMWrapper
                type(model).__name__ == 'ORTModelForCausalLM' or
                'ORTModel' in type(model).__name__
        )

        # Safety net: LLM.py already drops the ViT of VL checkpoints at load, but
        # a ChatBot can also be handed a model loaded elsewhere. drop_vision_tower
        # is idempotent, so a second call on an already-trimmed model is free.
        if not self.is_onnx:
            drop_vision_tower(self.model, log_prefix="[ChatBot]")

        if not self.is_onnx:
            generation_dtype = infer_generation_head_dtype(self.model)
            align_generation_head_dtype(
                self.model,
                generation_dtype,
                log_prefix="[ChatBot]",
            )

        # Announce the window actually in force. A silent mismatch between the
        # config window and the wrapper's max_seq_length is what made every
        # length guard in this file ineffective, so make it visible at startup.
        ctx = _context_limit(self.model, self.tokenizer)
        cfg_window = getattr(getattr(self.model, 'config', None), 'max_position_embeddings', None)
        if cfg_window and int(cfg_window) > ctx:
            print(f"[ChatBot] Usable context is {ctx} tokens, BELOW the config window "
                  f"({int(cfg_window)}). The model was loaded with a smaller "
                  f"max_seq_length; prompts are checked against {ctx}.")
        else:
            print(f"[ChatBot] Usable context: {ctx} tokens.")

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

        self._raise_vocab_bound()

    def _raise_vocab_bound(self):
        """Make tokenizer.vocab_size report the TRUE embedding bound.

        The batch/direct generation guards clamp any token id >= vocab_size to
        EOS. For models like deepseek-coder, tokenizer.vocab_size is the base BPE
        size (e.g. 100000) which EXCLUDES added special tokens, while the chat
        template prepends BOS id 100000 — so the BOS gets clamped to EOS and every
        prompt is corrupted, yielding zero generated NNs. The embedding table
        actually has config.vocab_size rows (e.g. 102400), so the id is valid.

        HF tokenizer.vocab_size is a read-only @property on the class, so a plain
        instance assignment does not stick; fall back to a per-instance subclass
        (class swap). Never blocks construction on failure."""
        try:
            tok = self.tokenizer
            old = getattr(tok, 'vocab_size', None)
            if not isinstance(old, int):
                print('[vocab_bound] tokenizer exposes no integer vocab_size; '
                      'skipping (id clamping will use len()/get_vocab()).', flush=True)
                return
            cfg = getattr(self.model, 'config', None)
            text_cfg = getattr(cfg, 'text_config', None)
            cfg_vocab = 0
            for obj in (cfg, text_cfg):
                n = getattr(obj, 'vocab_size', 0) or 0
                if isinstance(n, int):
                    cfg_vocab = max(cfg_vocab, n)
            try:
                tok_len = len(tok)
            except TypeError:
                tok_len = 0
            true_bound = max(tok_len, cfg_vocab)
            if true_bound <= old:
                return

            mechanism = None
            try:
                tok.vocab_size = true_bound
            except (AttributeError, TypeError):
                pass
            if tok.vocab_size == true_bound:
                mechanism = 'instance-attr'

            if mechanism is None:
                _base = type(tok)
                _Patched = type(
                    f'{_base.__name__}_VocabBound',
                    (_base,),
                    {'vocab_size': property(lambda _self: true_bound)},
                )
                tok.__class__ = _Patched
                if tok.vocab_size == true_bound:
                    mechanism = 'class-swap'

            if mechanism is None:
                print(f'[vocab_bound] WARNING: could not raise vocab bound; '
                      f'readback still {tok.vocab_size} (wanted {true_bound}).', flush=True)
                return
            print(f'[vocab_bound] vocab bound raised {old} -> {true_bound} via {mechanism}', flush=True)
        except Exception as e:  # never block construction
            print(f'[vocab_bound] WARNING: patch failed ({type(e).__name__}: {e}); '
                  f'run continues unchanged.', flush=True)

    def _clamp_token_ids(self, inputs, label=''):
        """Clamp out-of-range ids in place; no-op when the bound is unknown."""
        if 'input_ids' not in inputs:
            return
        vocab_size = _effective_vocab_size(self.tokenizer)
        if vocab_size is None:
            return
        input_ids = inputs['input_ids']
        max_token_id = input_ids.max().item()
        if max_token_id >= vocab_size:
            print(f"[WARN] Invalid token IDs detected{label}: "
                  f"max_id={max_token_id}, vocab_size={vocab_size}")
            inputs['input_ids'] = torch.clamp(input_ids, max=vocab_size - 1)
            print(f"[WARN] After clamping: max_id={inputs['input_ids'].max().item()}")

    def _eos_token_ids(self):
        """eos id(s) for generation, always including <|im_end|> when the tokenizer knows it."""
        eos = getattr(self.tokenizer, 'eos_token_id', None)
        if eos is None:
            ids = []
        elif isinstance(eos, (list, tuple)):
            ids = list(eos)
        else:
            ids = [eos]
        try:
            im_end = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        except (AttributeError, TypeError):
            im_end = None
        unk = getattr(self.tokenizer, 'unk_token_id', None)
        if im_end is not None and im_end != unk and im_end not in ids:
            ids.append(im_end)
        return ids or None

    def _build_messages(self, user_content: str) -> list:
        """Build a messages list with optional system role prepended."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages

    def _render_chat_template(self, messages) -> str:
        """apply_chat_template, asking hybrid-thinking models for non-thinking mode.

        Qwen3.8 is a hybrid thinking model: thinking is ON by default and the
        supported way to turn it off is the template flag, not a pre-closed
        <think> block. Qwen2.5 templates ignore the unknown variable, so the same
        call is safe for both; the TypeError guards cover templates that reject
        extra kwargs outright.
        """
        base = dict(tokenize=False, add_generation_prompt=True)
        if self.suppress_thinking:
            for extra in ({"enable_thinking": False},
                          {"chat_template_kwargs": {"enable_thinking": False}}):
                try:
                    return self.tokenizer.apply_chat_template(messages, **base, **extra)
                except TypeError:
                    continue
                except Exception:
                    break
        return self.tokenizer.apply_chat_template(messages, **base)

    def _apply_thinking_suppression(self, templated):
        """Close OlympicCoder-7B's auto-opened <think> block so the model skips
        chain-of-thought and writes code directly.

        This used to live inside _prepare_pipeline_input(), which is reached
        only from _direct_generate_batch() -- i.e. only when prompt_batch > 1.
        chat() handed raw messages straight to the pipeline and
        _direct_generate() built its own template, so on those two paths the
        suppress_thinking flag was accepted, stored and silently ignored. Both
        paths are also used as *fallbacks*, so suppression could switch itself
        off part-way through an epoch. It is applied in one place now and all
        three paths call it.

        Harmless when _render_chat_template already produced a closed block
        (Qwen3.8 non-thinking mode): the endswith check simply does not fire.
        """
        if not self.suppress_thinking or not templated:
            return templated
        if templated.rstrip().endswith('</think>'):
            return templated
        if templated.rstrip().endswith('<think>'):
            return templated.rstrip() + '</think>'
        # Strip only spaces/tabs, never the newline: the chat template emits
        # "<|im_start|>assistant\n" and a bare rstrip() deleted that newline,
        # producing "<|im_start|>assistant<think>" -- a header shape the model
        # has never seen in training.
        return templated.rstrip(' \t') + '<think></think>'

    def _templated_from_messages(self, messages, fallback_text=None):
        """Render a messages list to a prompt string, suppression included."""
        if not self.disable_chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
            templated = self._render_chat_template(messages)
        elif self.system_prompt:
            body = fallback_text if fallback_text is not None else '\n'.join(
                f"{m['role'].capitalize()}: {m['content']}" for m in messages)
            templated = f"System: {self.system_prompt}\nUser: {body}\nAssistant:"
        elif fallback_text is not None:
            templated = fallback_text
        else:
            templated = '\n'.join(
                f"{m['role'].capitalize()}: {m['content']}" for m in messages) + '\nAssistant:'
        return self._apply_thinking_suppression(templated)

    def _prepare_pipeline_input(self, prompt_text):
        """Build a pipeline-ready text prompt using chat template when available."""
        return self._templated_from_messages(
            self._build_messages(prompt_text), fallback_text=prompt_text)

    def _plan_generation(self, formatted_prompt, max_new_tokens, label=''):
        """(new_tokens, in_len, ctx) for one rendered prompt, or (None, in_len, ctx) to skip.

        Single place where a prompt is measured against the window that is
        really in force, so the batch path, the pipeline path and the direct
        path cannot disagree about what fits.
        """
        ctx = _context_limit(self.model, self.tokenizer)
        in_len = len(self.tokenizer(text=formatted_prompt, add_special_tokens=False)['input_ids'])
        new_tokens = _output_budget(ctx, in_len, max_new_tokens)
        if new_tokens is None:
            print(f"[gen] SKIPPED{label}: input_tokens={in_len} max_seq_length={ctx} "
                  f"leaves {ctx - in_len} for output, below the {N2_MIN_NEW_TOKENS}-token "
                  f"floor. Prompt not sent.", flush=True)
        return new_tokens, in_len, ctx

    def _input_budget(self, max_new_tokens, n_prompt_tokens=None, label=''):
        """How many prompt tokens may survive, with a loud alarm when they won't.

        The budget was tokenizer.model_max_length - max_new_tokens. That is
        unsafe because LoRA.train() assigns
            self.tokenizer.model_max_length = sft_max_length
        on the SHARED tokenizer this ChatBot holds. After the first fine-tune
        the window therefore collapses to sft_max_length, and with
        sft_max_length == max_new_tokens the budget becomes max(1, 0) == 1:
        the prompt is cut to a single token and the model, given nothing,
        free-associates from its pretraining prior.

        It then read max_position_embeddings directly and took the LARGER of the
        two, which broke the opposite way: under Unsloth the config advertises
        32768 while the loaded model only has buffers for 4096, so a 7462-token
        prompt was declared safe and generate() died on a tensor-shape mismatch.
        _context_limit() now settles this by taking the smallest real limit.
        """
        window = _context_limit(self.model, self.tokenizer)

        token_budget = max_new_tokens or 4096
        max_input_len = window - token_budget
        if max_input_len < self.MIN_PROMPT_TOKENS:
            print(f'[ALARM] Input budget is only {max_input_len} tokens '
                  f'(max_seq_length={window}, max_new_tokens={token_budget}). The prompt '
                  f'would be destroyed. Clamping to {self.MIN_PROMPT_TOKENS} and '
                  f'reducing generation instead -- lower max_new_tokens or raise '
                  f'max_seq_length at model load to silence this.')
            max_input_len = self.MIN_PROMPT_TOKENS

        if n_prompt_tokens is not None and n_prompt_tokens > max_input_len:
            print(f'[ALARM] PROMPT TRUNCATED{label}: {n_prompt_tokens} tokens -> '
                  f'{max_input_len} ({n_prompt_tokens - max_input_len} cut from the '
                  f'front), max_seq_length={window}. The task description is at the '
                  f'START of the prompt, so the model may not see what it is being '
                  f'asked to do.')
        return max_input_len

    def _resolve_device(self):
        if hasattr(self.model, 'device') and self.model.device is not None:
            return self.model.device
        if self.is_onnx:
            return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def _direct_generate_batch(self, prompts, max_new_tokens=None, max_len=None):
        """Run true batched generation via model.generate and strip prompt prefixes by token length."""
        if hasattr(self.model, "eval"):
            self.model.eval()

        formatted_prompts = [self._prepare_pipeline_input(p) for p in prompts]
        token_budget = max_new_tokens or 4096
        legacy = _legacy_truncation()
        ctx = _context_limit(self.model, self.tokenizer)
        results_by_pos = {}
        if legacy:
            max_input_len = max(1, ctx - token_budget)
            truncation_kwargs = {"truncation": True, "max_length": max_input_len}
            kept = list(range(len(formatted_prompts)))
        else:
            max_input_len = ctx - token_budget
            lengths = [len(ids) for ids in self.tokenizer(
                text=formatted_prompts, add_special_tokens=False)["input_ids"]]
            kept = []
            for pos, in_len in enumerate(lengths):
                if _output_budget(ctx, in_len, token_budget) is None:
                    print(f"[gen] SKIPPED prompt {pos}: input_tokens={in_len} "
                          f"max_seq_length={ctx} leaves {ctx - in_len} for output, "
                          f"below the {N2_MIN_NEW_TOKENS}-token floor.", flush=True)
                    results_by_pos[pos] = SKIPPED_OUTPUT
                else:
                    kept.append(pos)
            if not kept:
                print(f"[gen] every prompt in this batch exceeded max_seq_length={ctx}; "
                      f"nothing generated.", flush=True)
                return [results_by_pos[pos] for pos in range(len(formatted_prompts))]
            truncation_kwargs = {"truncation": False}

        original_padding_side = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = "left"
        try:
            inputs = self.tokenizer(
                text=[formatted_prompts[pos] for pos in kept],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,  # chat template already includes BOS; prevents EOS being appended
                **truncation_kwargs,
            )
        finally:
            self.tokenizer.padding_side = original_padding_side

        self._clamp_token_ids(inputs, label=' in batch')

        device = self._resolve_device()

        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        # With left-padding, all sequences in the batch share the same padded input length.
        # The generated tokens start at index padded_input_length in each output row.
        padded_input_length = inputs['input_ids'].shape[1]
        if legacy:
            new_tokens = max_new_tokens or 4096
        else:
            # Left-padding can push the batch past the window even when every
            # individual prompt fit; re-check against the padded length and drop
            # the whole batch rather than letting generate() raise.
            new_tokens = _output_budget(ctx, padded_input_length, token_budget)
            if new_tokens is None:
                print(f"[gen] SKIPPED batch: padded input_tokens={padded_input_length} "
                      f"max_seq_length={ctx} leaves no room for output.", flush=True)
                for pos in kept:
                    results_by_pos[pos] = SKIPPED_OUTPUT
                return [results_by_pos[pos] for pos in range(len(formatted_prompts))]
            print(f"[gen] input_tokens={padded_input_length} max_new_tokens={new_tokens} max_seq_length={ctx} batch={len(kept)}", flush=True)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=new_tokens,
                max_length=max_len,
                do_sample=True,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self._eos_token_ids(),
            )

        results = []
        for i in range(outputs.shape[0]):
            generated_ids = outputs[i][padded_input_length:]
            generated = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            if self.strip_think_output:
                generated = _strip_reasoning_output(generated)
            nn = extract_code(generated)
            results.append((nn, extract_hyperparam(generated), extract_transform(generated), generated))
        for pos, result in zip(kept, results):
            results_by_pos[pos] = result
        return [results_by_pos[pos] for pos in range(len(formatted_prompts))]

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
                # new_tokens was referenced here but never assigned, so this
                # branch raised NameError on every call and silently demoted the
                # run to _direct_generate(). Measure the prompt and skip it when
                # it cannot fit, exactly as the batch path does.
                templated = self._templated_from_messages(in_next)
                new_tokens, _in_len, _ctx = self._plan_generation(
                    templated, max_new_tokens, label=' (pipeline)')
                if new_tokens is None:
                    if _legacy_truncation():
                        new_tokens = max_new_tokens or 4096
                    else:
                        return SKIPPED_OUTPUT

                generation_kwargs = {
                    "max_new_tokens": new_tokens,
                    "do_sample": True,
                    "max_length": max_len,
                    "temperature": self.temperature,
                    "top_k": self.top_k,
                    "top_p": self.top_p,
                    "eos_token_id": self._eos_token_ids(),
                }
                # With suppression on, the pipeline must be given the rendered
                # string: handed a messages list it applies the chat template
                # itself and the pre-closed <think> block never reaches it.
                pipeline_input = templated if self.suppress_thinking else in_next
                try:
                    out_item = self.__pipeline(
                        pipeline_input,
                        return_full_text=False,
                        **generation_kwargs,
                    )[0]
                    out = _extract_generated_content(out_item)
                except TypeError:
                    out_item = self.__pipeline(pipeline_input, **generation_kwargs)[0]
                    out = _extract_generated_content(out_item)

                assert isinstance(out, str)
                if self.strip_think_output:
                    out = _strip_reasoning_output(out)

                if self.__keep_memory:
                    self.__messages.append({"role": "assistant", "content": out})

                return (*extract_all_to_train(out), out)

            except Exception as e:
                print(f"[ERROR] Pipeline generation failed: {e}")
                print("[INFO] Falling back to direct generation "
                      f"(suppress_thinking={self.suppress_thinking} is preserved)")

        # Direct generation (ONNX or PyTorch fallback)
        return self._direct_generate(in_next, max_new_tokens, max_len)

    def _format_messages(self, messages) -> str:
        """Prompt text exactly as _direct_generate builds it."""
        if not self.disable_chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
            return self._render_chat_template(messages)
        formatted_prompt = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])
        return f"{formatted_prompt}\nAssistant:"

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
                print("[WARN] Falling back to per-prompt generation. This path "
                      "is slower; candidates before and after this point were "
                      "generated differently, so compare them with care.")

        return [self.chat(p, max_len=max_len, max_new_tokens=max_new_tokens, engineer_prompt=engineer_prompt) for p in prompts]

    def _direct_generate(self, messages, max_new_tokens, max_len):
        """Direct model.generate() call without pipeline - works for ONNX and PyTorch"""
        try:
            # Apply chat template to format messages
            if not self.disable_chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
                formatted_prompt = self._render_chat_template(messages)
            else:
                # Fallback: concatenate messages
                formatted_prompt = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])
                formatted_prompt = f"{formatted_prompt}\nAssistant:"
            # This is a fallback path; without this line suppression turned
            # itself off whenever the pipeline raised.
            formatted_prompt = self._apply_thinking_suppression(formatted_prompt)

            legacy = _legacy_truncation()
            new_tokens, n_tok, ctx = self._plan_generation(
                formatted_prompt, max_new_tokens, label=' (direct)')

            if legacy:
                # Old behaviour: cut the prompt to fit and always generate.
                max_input_len = self._input_budget(max_new_tokens, n_tok, label=' (direct)')
                new_tokens = max_new_tokens or 4096
                tokenize_kwargs = {"truncation": True, "max_length": max_input_len}
            else:
                if new_tokens is None:
                    # Skip rather than truncate: the task description sits at the
                    # start of the prompt, so a cut prompt produces confident
                    # nonsense that is worse than no candidate at all.
                    return SKIPPED_OUTPUT
                tokenize_kwargs = {"truncation": False}

            # Tokenize input. The stray **truncation_kwargs that used to sit here
            # was never bound in this function and made every call to this
            # fallback die with NameError.
            inputs = self.tokenizer(
                text=formatted_prompt,
                return_tensors="pt",
                add_special_tokens=False,  # chat template already includes BOS; prevents EOS being appended
                **tokenize_kwargs,
            )

            # -- FIX 1: Validate token IDs before GPU move --
            self._clamp_token_ids(inputs)

            device = self._resolve_device()

            inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

            # FIX: Store input length before generation
            input_length = inputs['input_ids'].shape[-1]  # Use shape[-1] for sequence length
            print(f"[gen] input_tokens={input_length} max_new_tokens={new_tokens} "
                  f"max_seq_length={ctx} batch=1", flush=True)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=new_tokens,
                    max_length=max_len,
                    do_sample=True,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self._eos_token_ids(),
                )

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