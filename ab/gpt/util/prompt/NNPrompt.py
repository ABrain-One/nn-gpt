import ab.nn as lemur

from overrides import override
from pandas import DataFrame
from transformers import PreTrainedTokenizerBase

from ab.gpt.util.prompt.Prompt import Prompt


class NNPrompt(Prompt):
    """
    Assumes the existence of accuracies.json and folder-based dataset
    """
    def __init__(self, max_len: int, tokenizer: PreTrainedTokenizerBase, system_prompt: str = None):
        super().__init__(max_len, tokenizer, system_prompt=system_prompt)

    @override
    def get_raw_dataset(self, only_best_accuracy, n_training_prompts=None) -> DataFrame:
        """
        :return:
            pandas.Dataframe object with columns described in nn_api.data()
        """
        lemur_df = lemur.data(only_best_accuracy, max_rows=n_training_prompts)
        dataframe = DataFrame(columns=['instruction', 'context', 'response', 'category', 'text'])

        for idx, lemur_row in lemur_df.iterrows():
            inst = (
                f"Create a neural network structure in PyTorch for {lemur_row['task']} based on the {lemur_row['nn']} architecture,"
                f"that would achieve an accuracy of {lemur_row['accuracy']} after {lemur_row['epoch']} epochs of training"
                f"on the {lemur_row['dataset']} dataset. Name the main class 'Net'."
            )
            response = "```\n" + str(lemur_row['nn_code']) + "\n```"
            text = self.tokenizer.apply_chat_template(
                self._build_messages(inst, response),
                tokenize=False
            )

            dataframe.loc[len(dataframe)] = [inst, "", response, "", text]

        return dataframe