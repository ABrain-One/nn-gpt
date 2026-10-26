from functools import partial

from datasets import Dataset
from pandas import DataFrame
from transformers import PreTrainedTokenizerBase

from ab.gpt.util.llm.LLMUtil import prepare_text_tokenizer, unwrap_tokenizer

# Fallback chat template for tokenizers that ship without one (e.g.
# ABrain/NNGPT-UniqueArch-Rag). Without it, apply_chat_template raises
# "tokenizer.chat_template is not set".
#
# We use the DeepSeek-Coder "User:/Assistant:" convention rather than ChatML: the
# template-less NNGPT models in this project are DeepSeek-Coder derivatives (vocab
# 102400, bos 100000, eos 100015) fine-tuned with exactly this format, so ChatML's
# <|im_start|>/<|im_end|> tokens are out-of-distribution and produce empty output.
# Plain "User:/Assistant:" text is also a safe generic default for any other
# template-less causal LM — it injects no model-specific special tokens.
# bos_token is intentionally omitted so it is added exactly once by the tokenizer
# (both the training-data build and generation tokenize with add_special_tokens=True).
# Tokenizers that already define chat_template (Qwen, DeepSeek, ...) never hit this.
DEFAULT_CHAT_TEMPLATE = (
    "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}{{ message['content'] + '\n\n' }}"
    "{% elif message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n\n' }}"
    "{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + eos_token }}"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
)


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch

    `text=` is not cosmetic. A multimodal processor's __call__ signature is
    (images, text, videos, audio), so a positional first argument is read as
    IMAGES and the run dies in decode_image with "Unsupported image file".
    The unwrap below normally removes the processor long before this point;
    the keyword keeps this function correct even when it does not, and is
    accepted unchanged by every plain tokenizer (`text` is its first parameter
    too).
    """
    tokenizer = unwrap_tokenizer(tokenizer)
    result = tokenizer(
        text=batch['text'],
        truncation=True,
        max_length=max_length,
    )
    # Also tokenize response to check its length
    if 'response' in batch:
        response_tokenized = tokenizer(
            text=batch['response'],
            truncation=False
        )
        # Add response_length field for filtering
        result['response'] = response_tokenized['input_ids']
    return result


class Prompt:
    def __init__(self, max_len: int, tokenizer, system_prompt: str = None):
        # `tokenizer` may be a processor (Qwen3VLProcessor) when the caller
        # built it from a VL checkpoint without going through LLM.__init__.
        # Unwrapping here keeps the training-data build and generation on the
        # same tokenizer and the same chat template.
        self.max_len = max_len
        self.tokenizer = prepare_text_tokenizer(tokenizer, log_prefix="[Prompt]")
        self.system_prompt = system_prompt

    def _build_messages(self, user_content: str, assistant_content: str = None, system_prompt: str = None) -> list:
        """Build a messages list for chat templates, supporting system/user/assistant roles.
        """
        sp = system_prompt if system_prompt is not None else self.system_prompt
        messages = []
        if sp:
            messages.append({"role": "system", "content": sp})
        messages.append({"role": "user", "content": user_content})
        if assistant_content is not None:
            messages.append(
                {"role": "assistant", "content": assistant_content})
        return messages

    def _apply_chat_template(self, messages, **kwargs) -> str:
        """Apply the tokenizer's chat template, falling back to a generic ChatML
        template for tokenizers that ship without one. Tokenizers that already
        define chat_template (Qwen, DeepSeek, ...) are left untouched.
        """
        if not getattr(self.tokenizer, "chat_template", None):
            print(
                "[Prompt] Tokenizer has no chat_template; applying default User/Assistant fallback.",
                flush=True)
            self.tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def get_raw_dataset(self, only_best_accuracy, n_training_prompts=None) -> DataFrame:
        """
            Implement this method such that it returns a pandas dataframe with the following columns:
            ["instruction", "context", "response", "category", "text"].
            It is recommended to keep the order but is not necessary.
            Only the field "text" is tokenized and used in the fine-tuning.
        """
        pass

    @staticmethod
    def _split_prompt_completion(text, response):
        """Split a rendered conversation into (prompt, completion).

        The completion is the assistant turn; the prompt is everything the
        template put before it, system turn and assistant header included.
        Splitting the rendered text rather than re-templating guarantees
        prompt + completion == text exactly, so nothing shifts by a token.
        """
        if not text or not response:
            return None, None
        idx = text.rfind(response)
        if idx <= 0:
            return None, None
        return text[:idx], text[idx:]

    def get_dataset(self, only_best_accuracy=False, seed=None, max_prompts=None,
                    max_new_tokens=4096, as_prompt_completion=True):
        frame = self.get_raw_dataset(only_best_accuracy, max_prompts)
        dataset = Dataset.from_pandas(frame)
        # Keep the rendered text and the target around; the length filter below
        # works on token ids, but the prompt/completion split needs the strings.
        raw_text = list(frame['text']) if 'text' in frame else []
        raw_response = list(frame['response']) if 'response' in frame else []
        print("Preprocessing dataset...")

        # Apply preprocessing to each batch of the dataset
        # Remove 'instruction', 'context', 'response', 'category' fields
        _preprocessing_function = partial(
            preprocess_batch, max_length=self.max_len, tokenizer=self.tokenizer)

        dataset = dataset.map(
            _preprocessing_function,
            batched=True,
            remove_columns=['instruction', 'context',
                            'response', 'text', 'category'],
        )
        # Filter out samples that have input_ids exceeding max_length
        # and response tokenized length exceeding max_new_tokens
        before_filter = len(dataset)
        keep_mask = [
            len(s['input_ids']) < self.max_len
            and len(s.get('response', [])) < max_new_tokens
            for s in dataset
        ]
        dataset = dataset.filter(
            lambda sample: len(sample['input_ids']) < self.max_len
            and len(sample.get('response', [])) < max_new_tokens
        )
        # Apply the same mask to the retained strings so the two stay aligned.
        raw_text = [t for t, k in zip(raw_text, keep_mask) if k]
        raw_response = [r for r, k in zip(raw_response, keep_mask) if k]
        kept = len(dataset)
        print(f"[SFT] {kept}/{before_filter} samples kept "
              f"(dropped {before_filter - kept}: longer than max_len={self.max_len} "
              f"or response longer than max_new_tokens={max_new_tokens})")
        if kept == 0:
            raise ValueError(
                f"All samples were filtered out at max_len={self.max_len}. Raise "
                f"context_length (and max_new_tokens) or shorten the prompts.")

        # Remove response_length field after filtering (it was only used for filtering)
        if 'response' in dataset.column_names:
            dataset = dataset.remove_columns(['response'])

        if as_prompt_completion:
            # Hand the trainer prompt/completion pairs instead of bare
            # input_ids. With only input_ids, LoRA.train() cannot find a 'text'
            # or 'prompt'/'completion' column, falls back to a plain Trainer
            # ("Dataset does not have 'text' field") and computes the loss over
            # the WHOLE sequence. The prompt holds the baseline network and is
            # several times longer than the target, so most of every update was
            # spent teaching the model to echo prompts. With these two columns
            # SFTTrainer masks the prompt and trains on the answer alone.
            keep = [i for i, (t, r) in enumerate(zip(raw_text, raw_response))
                    if self._split_prompt_completion(t, r)[0] is not None]
            dropped = len(raw_text) - len(keep)
            pairs = [self._split_prompt_completion(raw_text[i], raw_response[i]) for i in keep]
            if not pairs:
                raise ValueError(
                    "Could not split any sample into prompt/completion; pass "
                    "as_prompt_completion=False to train on full sequences.")
            if dropped:
                print(f"[SFT] {dropped} sample(s) could not be split into "
                      f"prompt/completion and were dropped")
            dataset = Dataset.from_dict({
                "prompt": [p for p, _ in pairs],
                "completion": [c for _, c in pairs],
            })
            print(f"[SFT] Training on completions only: {len(dataset)} prompt/completion pairs")

        # Shuffle dataset
        dataset = dataset.shuffle(seed=seed) if seed else dataset.shuffle()

        return dataset