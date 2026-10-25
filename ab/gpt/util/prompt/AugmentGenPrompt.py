import json

import pandas as pd
from pandas import DataFrame
from transformers import PreTrainedTokenizerBase
from overrides import override
from tqdm import tqdm

from ab.gpt.util.prompt.Prompt import Prompt


def shuffle_data(df: DataFrame):
    return df.sample(frac=1).reset_index(drop=True)


def load_schedule_pool(only_best_accuracy=True, fidelity=None) -> DataFrame:
    """
    Load evaluated augmentation schedules from the AugEval result pool.

    Returns columns: id_name (config_id), schedule_json (compact JSON string),
    accuracy (final accuracy).
    """
    from ab.gpt.brute.trans.augment.stat_utils import load_results

    df = load_results()
    if df.empty:
        print('Warning: No schedule results found. Returning empty DataFrame.')
        return pd.DataFrame(columns=['id_name', 'schedule_json', 'accuracy'])

    df = df[df['validity_class'] == 'ok'].copy()
    if fidelity is not None:
        df = df[df['fidelity'] == fidelity]
    elif df['fidelity'].nunique() > 1:
        print(f"[WARN] Schedule pool mixes fidelities {sorted(df['fidelity'].unique())}; ")
    df['accuracy'] = df['final_accuracy'].fillna(df['accuracy'])
    df = df.dropna(subset=['accuracy', 'config_id', 'augment'])
    df['schedule_json'] = df['augment'].apply(lambda s: json.dumps(s))
    df = df.rename(columns={'config_id': 'id_name'})

    if only_best_accuracy:
        df = df.sort_values('accuracy', ascending=False).drop_duplicates('id_name')

    df = df[['id_name', 'schedule_json', 'accuracy']].reset_index(drop=True)
    print(f'Loaded {len(df)} schedule data points from result pool.', flush=True)
    return df


class ScheduleGenPrompt(Prompt):
    """
    Builds instruction-tuning pairs from evaluated augmentation schedules:
    input = a baseline schedule with its accuracy, target = a different schedule with strictly higher accuracy
    """

    def __init__(self, max_len: int, tokenizer: PreTrainedTokenizerBase, prompts_path):
        super().__init__(max_len, tokenizer)
        self.prompts_path = prompts_path

    @override
    def get_raw_dataset(self, only_best_accuracy, n_training_prompts=None) -> DataFrame:
        with open(self.prompts_path) as prompt_file:
            prompt_dict = json.load(prompt_file)
            assert isinstance(prompt_dict, dict)

        print('Preparing schedule data...', flush=True)
        data = load_schedule_pool(only_best_accuracy)
        if data.empty:
            print('No data found to generate prompts.')
            return pd.DataFrame(columns=['instruction', 'context', 'response', 'category', 'text'])
        data = shuffle_data(data)

        prompt_lists = []
        for key in prompt_dict.keys():
            dataframe = DataFrame(columns=['instruction', 'context', 'response', 'category', 'text'])
            prompt_lists.append(dataframe)
            key_config = prompt_dict[key]
            prompt = '\n'.join(key_config['prompt'])
            output_template = '\n'.join(key_config.get('output', []))
            if not output_template:
                print(f"Warning: 'output' key missing in prompt config for {key}. Skipping key.")
                continue
            improve = key_config.get('improve', False)
            no_repeat = key_config.get('no_repeat', [])

            for _, row in tqdm(data.iterrows(), total=len(data)):
                if n_training_prompts and len(dataframe) >= n_training_prompts:
                    break

                addon = data[data.id_name != row['id_name']]
                if improve:
                    addon = addon[addon.accuracy > row['accuracy']]
                for col in no_repeat:
                    if col in data.columns:
                        addon = addon[addon[col] != row[col]]
                if addon.empty:
                    continue  
                addon_row = addon.sample(n=1).iloc[0]

                para_dict = {}
                for it in key_config['input_list']:
                    para_dict[it['para']] = row[it['value']]
                for it in key_config['addon_list']:
                    para_dict[it['para']] = addon_row[it['value']]

                try:
                    inst = prompt.format(**para_dict)
                    response = output_template.format(**para_dict)
                except KeyError as e:
                    print(f'Warning: Missing key {e} for formatting. Skipping row.')
                    continue

                text = self.tokenizer.apply_chat_template(
                    [
                        {'role': 'user', 'content': inst},
                        {'role': 'assistant', 'content': response}
                    ], tokenize=False
                )
                dataframe.loc[len(dataframe)] = [inst, '', response, '', text]

        print('Schedule prompts successfully generated', flush=True)
        return pd.concat(prompt_lists, ignore_index=True)
