import json
import os
import glob
import pandas as pd
from pandas import DataFrame
from transformers import PreTrainedTokenizerBase
def override(func): return func

from ab.gpt.util.prompt.Prompt import Prompt
from tqdm import tqdm
from ab.gpt.util.Const import trans_dir 


def shuffle_data(df: DataFrame):
    return df.sample(frac=1).reset_index(drop=True)


def load_data_from_folders(out_gen_dir: str, result_gen_dir: str, only_best_accuracy=True) -> DataFrame:
    """
    Loads transform code and results from the specified folders instead of lemur.
    """
    print("Loading data from folders...", flush=True)
    all_data = []
    json_files = glob.glob(os.path.join(result_gen_dir, "*.json"))
    
    for res_file in tqdm(json_files, desc="Reading data files"):
        base_name = os.path.basename(res_file).replace('.json', '')
        code_file = os.path.join(out_gen_dir, f"{base_name}.py") 
        
        if os.path.exists(code_file):
            try:
                with open(res_file, 'r') as f:
                    res_data = json.load(f)
                
                with open(code_file, 'r') as f:
                    code_content = f.read()
                
                # Combine data
                res_data['transform_code'] = code_content
                # For filtering duplicates
                res_data['id_name'] = base_name  
                
                # Add default prm if missing
                res_data.setdefault('prm', '{}') 
                
                all_data.append(res_data)
            except Exception as e:
                print(f"Warning: Could not load data for {base_name}. Error: {e}", flush=True)
        else:
            print(f"Warning: Missing code file {code_file} for result {res_file}", flush=True)

    if not all_data:
        # Return an empty DataFrame instead of raising ValueError
        print("Warning: No matching data found in folders. Returning empty DataFrame.")
        return pd.DataFrame()
        
    df = pd.DataFrame(all_data)
    
    # Taking the best entry per 'id_name'
   
    if only_best_accuracy and 'accuracy' in df.columns:
        df = df.sort_values('accuracy', ascending=False).drop_duplicates('id_name')
        
    print(f"Loaded {len(df)} data points from folders.", flush=True)
    return df


class TransformGenPrompt(Prompt):
    """
    Assumes the existence of results in result and code in out
    """

    def __init__(self, max_len: int, tokenizer: PreTrainedTokenizerBase, prompts_path, out_dir=None, res_dir=None):
        super().__init__(max_len, tokenizer)
        self.prompts_path = prompts_path
      
        self.out_gen_dir = out_dir
        self.result_gen_dir = res_dir

    @override
    def get_raw_dataset(self, only_best_accuracy, n_training_prompts=None) -> DataFrame:
        """
        Return pandas.Dataframe object with columns formatted for training.
        """
        prompt_lists = []

        with open(self.prompts_path) as prompt_file:
            prompt_dict = json.load(prompt_file)
            assert isinstance(prompt_dict, dict)
            
        # Load data from folders instead of lemur
        print('Preparing Data...', flush=True)
        # Convert Path objects to strings
        data = load_data_from_folders(str(self.out_gen_dir), str(self.result_gen_dir), only_best_accuracy)
        
        if data.empty:
            print("No data found to generate prompts.")
            return pd.DataFrame(columns=['instruction', 'context', 'response', 'category', 'text'])
            
        data = shuffle_data(data)
        print('Data acquisition complete', flush=True)

        for key in prompt_dict.keys():
            dataframe = DataFrame(columns=['instruction', 'context', 'response', 'category', 'text'])
            prompt_lists.append(dataframe)
            prompt = '\n'.join(prompt_dict[key]['prompt'])

            with_addons = 'addon_list' in prompt_dict[key] and prompt_dict[key]['addon_list']
            
            # Use the loaded data as the addon data
            addon_data = data 

            for _, row in tqdm(data.iterrows(), total=n_training_prompts or len(data)):
                if n_training_prompts and len(dataframe) >= n_training_prompts:
                    break
                
                row_dict = row.to_dict()
                para_dict = dict()
                
                for it in prompt_dict[key]['input_list']:
                    # Providing a default if a key is missing
                    para_dict[it['para']] = row_dict.get(it['value']) 

                if with_addons:
                    filter_q = f"id_name!='{row['id_name']}'"
                    
                    if 'same_pref' in prompt_dict[key] and prompt_dict[key]['same_pref']:
                         # Assuming id_name format is 'prefix-suffix'
                        prefix = row['id_name'].split('-')[0]
                        filter_q += f"&id_name.str.startswith('{prefix}-')"

                    if 'improve' in prompt_dict[key] and prompt_dict[key]['improve'] and 'accuracy' in row and row['accuracy'] is not None:
                        filter_q += f"&accuracy>{row['accuracy']}"

                    if 'no_repeat' in prompt_dict[key]:
                        for filter_it in prompt_dict[key]['no_repeat']:
                            if filter_it in row_dict:
                                val = row_dict[filter_it]
                                if isinstance(val, str):
                                    # Escape quotes/newlines for the query
                                    val_escaped = val.replace("'", "\\'").replace("\n", "\\n")
                                    filter_q += f"&{filter_it}!='{val_escaped}'"
                                else:
                                    filter_q += f"&{filter_it}!={val}"
                                    
                    try:
                        filtered_addon_data = addon_data.query(filter_q)
                    except Exception as e:
                        print(f"Warning: Query failed: {e}. Filter: {filter_q}")
                        continue


                    if len(filtered_addon_data) > 0:
                        shuffled = shuffle_data(filtered_addon_data)
                        addon_row = shuffled.sample(n=1).iloc[0].to_dict()
                    else:
                        # No result matches requirement
                        continue 
                    
                    for it in prompt_dict[key]['addon_list']:
                        para_dict[it['para']] = addon_row.get(it['value'])

               
                # Check for formatting errors in prompt
                try:
                    inst = prompt.format(**para_dict)
                except KeyError as e:
                    print(f"Warning: Missing key {e} for formatting prompt. Skipping row.")
                    continue
                
                # Get the output format from the prompt config
                if 'output' not in prompt_dict[key]:
                     print(f"Warning: 'output' key missing in prompt config for {key}. Skipping row.")
                     continue
                     
                output_template = '\n'.join(prompt_dict[key]['output'])
                
                try:
                    response = output_template.format(**para_dict)
                except KeyError as e:
                    print(f"Warning: Missing key {e} for formatting response. Skipping row.")
                    continue
              

                text = self.tokenizer.apply_chat_template(
                    [
                        {'role': 'user', 'content': inst},
                        {'role': 'assistant', 'content': response}
                    ], tokenize=False
                )

                dataframe.loc[len(dataframe)] = [inst, "", response, "", text]

        print('Prompts successfully generated', flush=True)
        del data, addon_data
        return pd.concat(prompt_lists, ignore_index=True)