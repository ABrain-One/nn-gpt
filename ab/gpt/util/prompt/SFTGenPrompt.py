from overrides import override
from pandas import DataFrame
from transformers import PreTrainedTokenizerBase
from datasets import Dataset

from ab.gpt.util.prompt.Prompt import Prompt
import ab.nn.api as lemur
import ab.gpt.util.SFTUtil as SFTUtil

def _normalize_nn_prefixes(nn_prefixes):
    if nn_prefixes is None:
        return ("rl-bb-test1",)
    if isinstance(nn_prefixes, str):
        prefixes = tuple(item.strip() for item in nn_prefixes.split(",") if item.strip())
    else:
        prefixes = tuple(str(item).strip() for item in nn_prefixes if str(item).strip())
    if not prefixes:
        raise ValueError("At least one SFT NN prefix is required")
    return prefixes


class SFTGenPrompt(Prompt):
    """
    Prompt processor for SFT mode.
    Does NOT pre-tokenize data, allowing SFTTrainer (in LoRA.py) to handle it.
    Uses SFTUtil for specialized parsing and formatting.
    """

    def __init__(self, max_len: int, tokenizer: PreTrainedTokenizerBase, nn_prefixes=None, dataset: str = None):
        # prompts_path is not needed for SFTGenPrompt as it uses SFTUtil templates
        super().__init__(max_len, tokenizer)
        self.nn_prefixes = _normalize_nn_prefixes(nn_prefixes)
        self.dataset = dataset

    @override
    def get_raw_dataset(self, only_best_accuracy, n_training_prompts=None) -> DataFrame:
        """
        Extracts data from Lemur and formats it using SFTUtil.
        Returns prompt/completion chat columns so TRL can build completion masks.
        """
        print(
            f"extracting data from Lemur for SFT with "
            f"nn_prefixes={self.nn_prefixes}, dataset={self.dataset}..."
        )
        data_kwargs = {
            "task": "img-classification",
            "nn_prefixes": self.nn_prefixes,
        }
        if self.dataset:
            data_kwargs["dataset"] = self.dataset
        df = lemur.data(**data_kwargs)
        print(f"extracted {len(df)} samples.")
        
        if n_training_prompts:
             df = df.sample(n=min(len(df), n_training_prompts))
        
        formatted_data = []
        for _, row in df.iterrows():
            full_code = row['nn_code']
            accuracy = row['accuracy']
            
            block_code, init_code, forward_code = SFTUtil.parse_nn_code(full_code)
            target_pattern = SFTUtil.extract_target_pattern_from_code(full_code)
            
            if block_code and init_code and forward_code and target_pattern:
                assistant_response = f"<block>\n{block_code}\n</block>\n<init>\n{init_code}\n</init>\n<forward>\n{forward_code}\n</forward>"
                user_prompt = SFTUtil.format_backbone_prompt(
                    accuracy=accuracy,
                    target_pattern=target_pattern,
                )
                formatted_data.append(
                    {
                        "prompt": [{"role": "user", "content": user_prompt}],
                        "completion": [{"role": "assistant", "content": assistant_response}],
                    }
                )
            else:
                 print(f"Skipping row {row.name} due to parsing failure or missing target_pattern")

        return DataFrame(formatted_data)

    @override
    def get_dataset(self, only_best_accuracy=False, seed=None, max_prompts=None, max_new_tokens=4096):
        """
        Override to return dataset WITHOUT tokenization/column stripping.
        LoRA.py's train() method will let SFTTrainer tokenize prompt/completion pairs.
        """
        raw_df = self.get_raw_dataset(only_best_accuracy, max_prompts)
        dataset = Dataset.from_pandas(raw_df)
        
        # Determine split from SFT.py logic (defaulting to random split here similar to others)
        # SFT.py does: dataset = dataset.train_test_split(test_size=0.1)
        # But here we just return the full dataset, LoRA.py handles splitting.
        
        if seed:
             dataset = dataset.shuffle(seed=seed)
        else:
             dataset = dataset.shuffle()
             
        return dataset
