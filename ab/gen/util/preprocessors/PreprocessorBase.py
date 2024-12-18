from functools import partial

from datasets import Dataset
from pandas import DataFrame
from transformers import PreTrainedTokenizerBase


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


class PreprocessorBase:
    def __init__(self, max_len: int, tokenizer: PreTrainedTokenizerBase):
        self.max_len = max_len
        self.tokenizer = tokenizer

    def get_raw_dataset(self) -> DataFrame:
        """
            Implement this method such that it returns a pandas dataframe with the following columns:
            ["instruction", "context", "response", "category", "text"].
            It is recommended to keep the order but is not necessary.
        """
        pass

    def get_dataset(self, seed=None):
        dataset = Dataset.from_pandas(self.get_raw_dataset())
        print("Preprocessing dataset...")

        # Apply preprocessing to each batch of the dataset
        # Remove 'instruction', 'context', 'response', 'category' fields
        _preprocessing_function = partial(preprocess_batch, max_length=self.max_len, tokenizer=self.tokenizer)
        dataset = dataset.map(
            _preprocessing_function,
            batched=True,
            remove_columns=["instruction", "context", "response", "text", "category"],
        )

        # Filter out samples that have input_ids exceeding max_length
        dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < self.max_len)

        # Shuffle dataset
        dataset = dataset.shuffle(seed=seed) if seed is not None else dataset.shuffle()

        return dataset
