from functools import partial

from datasets import Dataset
from pandas import DataFrame
from transformers import PreTrainedTokenizerBase


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """
    result = tokenizer(
        batch['text'],
        truncation=False
    )
    # Also tokenize response to check its length
    if 'response' in batch:
        response_tokenized = tokenizer(
            batch['response'],
            truncation=False
        )
        # Add response_length field for filtering
        result['response'] = response_tokenized['input_ids']
    return result


class Prompt:
    def __init__(self, max_len: int, tokenizer: PreTrainedTokenizerBase, system_prompt: str = None):
        self.max_len = max_len
        self.tokenizer = tokenizer
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

    def get_raw_dataset(self, only_best_accuracy, n_training_prompts=None) -> DataFrame:
        """
            Implement this method such that it returns a pandas dataframe with the following columns:
            ["instruction", "context", "response", "category", "text"].
            It is recommended to keep the order but is not necessary.
            Only the field "text" is tokenized and used in the fine-tuning.
        """
        pass

    def get_dataset(self, only_best_accuracy=False, seed=None, max_prompts=None, max_new_tokens=4096):
        dataset = Dataset.from_pandas(
            self.get_raw_dataset(only_best_accuracy, max_prompts))
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
        dataset = dataset.filter(
            lambda sample: len(sample['input_ids']) < self.max_len
            and len(sample.get('response', [])) < max_new_tokens
        )
        # Remove response_length field after filtering (it was only used for filtering)
        if 'response' in dataset.column_names:
            dataset = dataset.remove_columns(['response'])

        # Shuffle dataset
        dataset = dataset.shuffle(seed=seed) if seed else dataset.shuffle()

        return dataset
