import json

import ab.nn as lemur
def override(func): return func
from pandas import DataFrame
from transformers import PreTrainedTokenizerBase

from ab.gpt.util.Const import conf_dir
from ab.gpt.util.prompt.Prompt import Prompt


def shuffle_data(df: DataFrame):
    return df.sample(frac=1).reset_index(drop=True)


models_with_the_same_prefix = False


class NNRLPrompt(Prompt):
    """
    Assumes the existence of accuracies.json and folder-based dataset
    """

    def __init__(self, max_len: int, tokenizer: PreTrainedTokenizerBase, prompts_path):
        super().__init__(max_len, tokenizer)
        self.prompts_path = prompts_path

    @override
    def get_raw_dataset(self, only_best_accuracy, n_training_prompts=None) -> DataFrame:
        """
        :return:
            pandas.Dataframe object with columns described in nn_api.data()
        """
        dataframe = DataFrame(columns=['instruction', 'context', 'response', 'category', 'text'])

        with open(self.prompts_path) as prompt_file:
            prompt_dict = json.load(prompt_file)
        assert isinstance(prompt_dict, dict)

        for key in prompt_dict.keys():
            prompt = '\n'.join(prompt_dict[key]['prompt'])
            # Get nn-dataset codes
            print('Preparing Data...')
            data = lemur.data(only_best_accuracy=only_best_accuracy, task=prompt_dict[key]['task'], max_rows=n_training_prompts)
            data = shuffle_data(data)
            print('Data acquisition complete')

            # Get addon nn-dataset codes
            addon_data = lemur.data(only_best_accuracy=only_best_accuracy, task=prompt_dict[key]['addon_task'], max_rows=n_training_prompts)
            print('Addon-Data acquisition complete')

            for _, row in data.iterrows():
                para_dict = dict()
                for it in prompt_dict[key]['input_list']:
                    para_dict[it['para']] = row[it['value']]
                if models_with_the_same_prefix and addon_data is not None:
                    ## Apply non-repeat filter if exists:
                    filter = f"nn.str.split('-',n=1).str[0]=='{row['nn'].split('-')[0]}'&nn!='{row['nn']}'"  # Default addon filter in this case should be models with same prefix
                    if 'no_repeat' in prompt_dict[key]:  # Compact test_prompt.json, items in prm is not supported
                        for filter_it in prompt_dict[key]['no_repeat']:
                            if isinstance(row[filter_it], str):
                                filter += f"&{filter_it}!='{row[filter_it]}'"
                            else:
                                filter += f"&{filter_it}!={row[filter_it]}"
                    if 'keep_same' in prompt_dict[key]:
                        for filter_it in prompt_dict[key]['keep_same']:
                            if isinstance(row[filter_it], str):
                                filter += f"&{filter_it}=='{row[filter_it]}'"
                            else:
                                filter += f"&{filter_it}=={row[filter_it]}"
                    addon_data = addon_data.query(filter)
                if len(addon_data) > 0:
                    addon_row = addon_data.sample(n=1).iloc[0]
                else:
                    continue  # The case no result matches requirement
                for it in prompt_dict[key]['addon_list']:
                    para_dict[it['para']] = addon_row[it['value']]
                inst = prompt.format(**para_dict)
                # Having the same name prefix before '-'
                # output = '\n'.join(prompt_dict[key]['output'])
                # response = output.format(**para_dict)
                response = para_dict  # Use the whole dictionary as response
                text = self.tokenizer.apply_chat_template(
                    [
                        {'role': 'user', 'content': inst},
                        {'role': 'assistant', 'content': json.dumps(para_dict, ensure_ascii=False, indent=2)}
                    ], tokenize=False
                )
                # print(f"Prompt: {inst}")
                # print(f"Output: {response}")
                dataframe.loc[len(dataframe)] = [inst, "", response, "", text]
            break  # Only one prompt is supported for now

        return dataframe
