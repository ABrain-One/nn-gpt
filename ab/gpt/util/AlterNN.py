import json
import os
import shutil

import ab.nn.api as nn_dataset
from ab.nn.util.Util import create_file
from tqdm import tqdm

from ab.gpt.util.Const import conf_test_dir, epoch_dir, new_nn_file, synth_dir, new_out_file
from ab.gpt.util.LLM import LLM
from ab.gpt.util.Util import extract_code, extract_delta


def format_prompt_with_supporting_models(prompt_template, para_dict, supporting_models):
    """
    Format prompt template with supporting models information.
    Handles complex formatting including supporting models display.
    """
    # Add number of supporting models to para_dict
    para_dict['n'] = len(supporting_models) if supporting_models else 0

    # Create a formatted string for supporting models
    if supporting_models:
        supporting_models_text = ""
        for i, model in enumerate(supporting_models, 1):
            supporting_models_text += f"\nSupporting Model {i}:\n"
            for key, value in model.items():
                supporting_models_text += f"  {key}: {value}\n"
        para_dict['supporting_models_prompt'] = supporting_models_text
    else:
        para_dict['supporting_models_prompt'] = "No supporting models available."

    # Format the prompt with all parameters
    try:
        formatted_prompt = prompt_template.format(**para_dict)
    except KeyError as e:
        # If there are missing parameters, use a fallback approach
        print(f"[WARNING] Missing parameter in prompt template: {e}")
        formatted_prompt = prompt_template
        # Replace available parameters
        for key, value in para_dict.items():
            formatted_prompt = formatted_prompt.replace(f"{{{key}}}", str(value))

    return formatted_prompt


def alter(epochs, test_conf, llm_name, gguf_file=None, n=1, temperature=0.6, top_k=50):
    # Load test prompts
    with open(conf_test_dir / test_conf) as f:
        prompt_dict = json.load(f)
    assert isinstance(prompt_dict, dict)

    model_loader = LLM(llm_name, gguf_file=gguf_file)
    model = model_loader.get_model()
    tokenizer = model_loader.get_tokenizer()
    print(f"Load Model Complete, Start Loop... (Will fetch {n} supporting models per prompt)")

    shutil.rmtree(epoch_dir(), ignore_errors=True)
    for epoch in range(epochs):
        out_path = epoch_dir(epoch)

        # Generate Prompts
        prompts = []
        for key in prompt_dict.keys():
            prompt = ""
            for pr in prompt_dict[key]['prompt']:
                prompt += pr + "\n"
            # Get nn-dataset codes
            data = nn_dataset.data(only_best_accuracy=True, task=prompt_dict[key]['task']).groupby(by="nn").sample(n=1)
            # Get addon nn-dataset codes
            addon_data = nn_dataset.data(only_best_accuracy=True, task=prompt_dict[key]['addon_task'])
            for _, row in data.iterrows():
                para_dict = dict()
                for it in prompt_dict[key]["input_list"]:
                    para_dict[it['para']] = row[it['value']]

                # Fetch n supporting models from database
                supporting_models = []
                if not (addon_data is None) and n > 0:
                    # Avoid sampling the same nn_code as the original
                    available_addon_data = addon_data.loc[addon_data.nn != row['nn']]

                    # Sample n supporting models (or as many as available if less than n)
                    n_samples = min(n, len(available_addon_data))
                    if n_samples > 0:
                        addon_rows = available_addon_data.sample(n=n_samples)

                        # Create a list to store supporting model information
                        for _, addon_row in addon_rows.iterrows():
                            model_info = {}
                            for it in prompt_dict[key]['addon_list']:
                                model_info[it['para']] = addon_row[it['value']]
                            supporting_models.append(model_info)

                    # Add the supporting models to the parameter dictionary
                    para_dict['supporting_models'] = supporting_models

                    # Also add individual parameters for backward compatibility
                    if supporting_models:  # Use the first model for backward compatibility
                        first_model = supporting_models[0]
                        for it in prompt_dict[key]['addon_list']:
                            para_dict[it['para']] = first_model[it['para']]
                # Format the prompt with supporting models
                formatted_prompt = format_prompt_with_supporting_models(prompt, para_dict, supporting_models)
                prompts.append((formatted_prompt, row))

        # produce new CV models
        B_index = 0
        for idx, prompt in tqdm(enumerate(prompts), desc="Generate Codes"):
            prompt, origdf = prompt
            model_dir = synth_dir(out_path) / f"B{B_index}"
            code_file = model_dir / new_nn_file
            df_file = model_dir / 'dataframe.df'
            inputs = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}, ], add_generation_prompt=True, return_tensors="pt").to(model.device)
            # tokenizer.eos_token_id is the id of <｜end▁of▁sentence｜>  token
            outputs = model.generate(inputs, max_new_tokens=64 * 1024, do_sample=True, temperature=temperature, top_k=top_k, top_p=0.95, num_return_sequences=1,
                                     eos_token_id=tokenizer.eos_token_id)
            out = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            print("Response Available!")
            nn_code = extract_code(out)
            if nn_code:
                print(f"[INFO]Saving code to: {code_file}")
                code_file.parent.mkdir(exist_ok=True, parents=True)  # Move here to avoid empty folder
                with open(code_file, 'w') as file:
                    file.write(nn_code)
                create_file(model_dir, new_out_file, out)
                if origdf is None:
                    if os.path.isfile(df_file):  # Clean up dataframe.df, if no additional information generated this time.
                        os.remove(df_file)
                else:
                    # Store DataFrame information, mainly for passing parameters to evaluator.
                    orig_code_file = model_dir / f"original_{origdf['nn']}.py"
                    with open(orig_code_file, 'w') as file:
                        file.write(origdf['nn_code'])

                    origdf.to_pickle(df_file)
                B_index += 1
            else:
                print("[INFO]Response Invalid!")
                continue


def alter_delta(epochs, test_conf, llm_name, gguf_file=None, n=1, temperature=0.6, top_k=50):
    """
    Generate improved neural network models using delta-based approach.
    
    Similar to alter() but:
    1. LLM generates code deltas (unified diffs) instead of full code
    2. Deltas are applied to baseline code to reconstruct improved code
    3. Requires delta-enabled config file (with use_delta: true)
    
    Args:
        epochs: Number of generation epochs
        test_conf: Config file name (should be delta-enabled, e.g., 'NN_gen_delta.json')
        llm_name: LLM model name/path
        gguf_file: Optional GGUF file path
        n: Number of supporting models
        temperature: Generation temperature
        top_k: Top-k sampling parameter
    """
    # Load test prompts
    with open(conf_test_dir / test_conf) as f:
        prompt_dict = json.load(f)
    assert isinstance(prompt_dict, dict)
    
    # Check if delta mode is enabled
    use_delta = False
    for key in prompt_dict.keys():
        key_config = prompt_dict[key]
        if isinstance(key_config, dict):
            use_delta = key_config.get('use_delta', False) or 'delta' in str(key).lower()
            if use_delta:
                break
    
    if not use_delta:
        print("[WARNING] Config file does not have delta mode enabled. Falling back to regular alter().")
        return alter(epochs, test_conf, llm_name, gguf_file, n, temperature, top_k)

    model_loader = LLM(llm_name, gguf_file=gguf_file)
    model = model_loader.get_model()
    tokenizer = model_loader.get_tokenizer()
    print(f"Load Model Complete, Start Loop... (Delta mode enabled)")

    shutil.rmtree(epoch_dir(), ignore_errors=True)
    for epoch in range(epochs):
        out_path = epoch_dir(epoch)

        # Generate Prompts
        prompts = []
        for key in prompt_dict.keys():
            prompt = ""
            for pr in prompt_dict[key]['prompt']:
                prompt += pr + "\n"
            # Get nn-dataset codes
            data = nn_dataset.data(only_best_accuracy=True, task=prompt_dict[key]['task']).groupby(by="nn").sample(n=1)
            # Get addon nn-dataset codes (if addon_task is specified)
            addon_data = None
            if prompt_dict[key].get('addon_task'):
                addon_data = nn_dataset.data(only_best_accuracy=True, task=prompt_dict[key]['addon_task'])
            
            for _, row in data.iterrows():
                para_dict = dict()
                for it in prompt_dict[key]["input_list"]:
                    para_dict[it['para']] = row[it['value']]
                
                # Handle addon_list (similar to original alter() function)
                if prompt_dict[key].get('addon_list') and addon_data is not None:
                    # Fetch n supporting models from database
                    supporting_models = []
                    if n > 0:
                        # Avoid sampling the same nn_code as the original
                        available_addon_data = addon_data.loc[addon_data.nn != row['nn']]
                        
                        # Sample n supporting models (or as many as available if less than n)
                        n_samples = min(n, len(available_addon_data))
                        if n_samples > 0:
                            addon_rows = available_addon_data.sample(n=n_samples)
                            
                            # Create a list to store supporting model information
                            for _, addon_row in addon_rows.iterrows():
                                model_info = {}
                                for it in prompt_dict[key]['addon_list']:
                                    model_info[it['para']] = addon_row[it['value']]
                                supporting_models.append(model_info)
                            
                            # Add the supporting models to the parameter dictionary
                            para_dict['supporting_models'] = supporting_models
                            
                            # Also add individual parameters for backward compatibility
                            if supporting_models:  # Use the first model for backward compatibility
                                first_model = supporting_models[0]
                                for it in prompt_dict[key]['addon_list']:
                                    para_dict[it['para']] = first_model[it['para']]
                    
                    # Format the prompt with supporting models
                    formatted_prompt = format_prompt_with_supporting_models(prompt, para_dict, supporting_models)
                    prompts.append((formatted_prompt, row))
                else:
                    # No addon_list, use simple prompt formatting
                    prompts.append((prompt.format(**para_dict), row))

        # produce new CV models
        B_index = 0
        for idx, prompt in tqdm(enumerate(prompts), desc="Generate Deltas"):
            prompt, origdf = prompt
            model_dir = synth_dir(out_path) / f"B{B_index}"
            code_file = model_dir / new_nn_file
            df_file = model_dir / 'dataframe.df'
            
            inputs = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}], 
                add_generation_prompt=True, 
                return_tensors="pt"
            ).to(model.device)
            
            outputs = model.generate(
                inputs, 
                max_new_tokens=64 * 1024, 
                do_sample=True, 
                temperature=temperature, 
                top_k=top_k, 
                top_p=0.95, 
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
            )
            out = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            print("Response Available!")
            
            # Extract delta from LLM output
            delta = extract_delta(out)
            
            if delta and origdf is not None:
                try:
                    from ab.gpt.util.DeltaUtil import apply_delta, validate_delta
                    baseline_code = origdf.get('nn_code', '')
                    
                    if baseline_code:
                        # Validate delta format before attempting to apply
                        if not validate_delta(delta):
                            print(f"[WARNING] Invalid delta format for model B{B_index}. Trying fallback to extract full code.")
                            delta = None  # Will trigger fallback below
                        else:
                            # Apply delta to baseline to get improved code
                            nn_code = apply_delta(baseline_code, delta)
                            
                            if nn_code:
                                print(f"[INFO] Successfully applied delta, saving code to: {code_file}")
                                code_file.parent.mkdir(exist_ok=True, parents=True)
                                with open(code_file, 'w') as file:
                                    file.write(nn_code)
                                create_file(model_dir, new_out_file, out)
                                
                                # Save original baseline code for reference
                                orig_code_file = model_dir / f"original_{origdf['nn']}.py"
                                with open(orig_code_file, 'w') as file:
                                    file.write(baseline_code)
                                
                                # Save delta for debugging
                                delta_file = model_dir / 'delta.diff'
                                with open(delta_file, 'w') as file:
                                    file.write(delta)
                                
                                origdf.to_pickle(df_file)
                                B_index += 1
                            else:
                                print(f"[WARNING] Delta application returned None for model B{B_index}. Trying fallback.")
                                delta = None  # Will trigger fallback below
                    else:
                        print(f"[WARNING] No baseline code found in origdf for model B{B_index}. Trying fallback.")
                        delta = None  # Will trigger fallback below
                except ImportError as e:
                    print(f"[ERROR] Failed to import delta utilities for model B{B_index}: {e}. Trying fallback.")
                    delta = None  # Will trigger fallback below
                except Exception as e:
                    print(f"[ERROR] Unexpected error applying delta for model B{B_index}: {e}. Trying fallback.")
                    delta = None  # Will trigger fallback below
            
            # Fallback: try to extract full code if delta extraction/application failed
            if not delta or origdf is None:
            else:
                # Fallback: try to extract full code if delta extraction failed
                nn_code = extract_code(out)
                if nn_code:
                    print(f"[INFO] Delta extraction failed, using extracted code as fallback: {code_file}")
                    code_file.parent.mkdir(exist_ok=True, parents=True)
                    with open(code_file, 'w') as file:
                        file.write(nn_code)
                    create_file(model_dir, new_out_file, out)
                    if origdf is not None:
                        origdf.to_pickle(df_file)
                    B_index += 1
                else:
                    print("[INFO] Response Invalid (no delta or code found)!")