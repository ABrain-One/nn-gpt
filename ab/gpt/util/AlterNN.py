import json
import os
import shutil
import torch
from tqdm import tqdm
import ab.nn.api as nn_dataset
from ab.nn.util.Util import create_file
from ab.gpt.util.Const import conf_test_dir, epoch_dir, new_nn_file, synth_dir, new_out_file
from ab.gpt.util.LLM import LLM
from ab.gpt.util.Util import extract_code

def format_prompt_with_supporting_models(prompt_template, para_dict, supporting_models):
    """Format prompt template with supporting models information."""
    para_dict['n'] = len(supporting_models) if supporting_models else 0
    if supporting_models:
        supporting_models_text = ""
        for i, model in enumerate(supporting_models, 1):
            supporting_models_text += f"\nSupporting Model {i}:\n"
            for key, value in model.items():
                supporting_models_text += f"  {key}: {value}\n"
        para_dict['supporting_models_prompt'] = supporting_models_text
    else:
        para_dict['supporting_models_prompt'] = "No supporting models available."

    try:
        formatted_prompt = prompt_template.format(**para_dict)
    except KeyError as e:
        print(f"[WARNING] Missing parameter in prompt template: {e}")
        formatted_prompt = prompt_template
        for key, value in para_dict.items():
            formatted_prompt = formatted_prompt.replace(f"{{{key}}}", str(value))
    return formatted_prompt

def alter(epochs, test_conf, llm_name, gguf_file=None, n=1, temperature=0.6, top_k=50, batch_size=8):
    # Load test prompts configuration
    with open(conf_test_dir / test_conf) as f:
        prompt_dict = json.load(f)
    
    # Load Model
    model_loader = LLM(llm_name, gguf_file=gguf_file)
    model = model_loader.get_model()
    tokenizer = model_loader.get_tokenizer()
    
    # --- Feature 1: Safety settings for Batch Inference ---
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    print(f"Load Model Complete. Batch Mode: {batch_size}. Fetching {n} supporting models per prompt.")

    shutil.rmtree(epoch_dir(), ignore_errors=True)
    
    

    for epoch in range(epochs):
        out_path = epoch_dir(epoch)
        prompts = []  # List of (raw_text_prompt, original_row_data)

        # --- Prepare All Prompts ---
        for key in prompt_dict.keys():
            prompt_base = ""
            for pr in prompt_dict[key]['prompt']:
                prompt_base += pr + "\n"
         


            # Fetch Data
            task_name = prompt_dict[key]['task']
            data = nn_dataset.data(only_best_accuracy=True, task=task_name).groupby(by="nn").sample(n=1)
            addon_data = nn_dataset.data(only_best_accuracy=True, task=prompt_dict[key]['addon_task'])
            
            for _, row in data.iterrows():
                para_dict = dict()
                for it in prompt_dict[key]["input_list"]:
                    para_dict[it['para']] = row[it['value']]

                # Fetch Supporting Models
                supporting_models = []
                if not (addon_data is None) and n > 0:
                    available_addon_data = addon_data.loc[addon_data.nn != row['nn']]
                    n_samples = min(n, len(available_addon_data))
                    if n_samples > 0:
                        addon_rows = available_addon_data.sample(n=n_samples)
                        for _, addon_row in addon_rows.iterrows():
                            model_info = {}
                            for it in prompt_dict[key]['addon_list']:
                                model_info[it['para']] = addon_row[it['value']]
                            supporting_models.append(model_info)
                    
                    para_dict['supporting_models'] = supporting_models
                    if supporting_models:
                        first_model = supporting_models[0]
                        for it in prompt_dict[key]['addon_list']:
                            para_dict[it['para']] = first_model[it['para']]

                formatted_prompt = format_prompt_with_supporting_models(prompt_base, para_dict, supporting_models)
                
                # Apply chat template
                chat_text = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': formatted_prompt}], 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                prompts.append((chat_text, row))

        # --- Run Batch Inference ---
        B_global_index = 0
        print(f"Total models to generate: {len(prompts)}")
        
        # Process in chunks of 'batch_size'
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Epoch {epoch} (Batched)"):
            batch = prompts[i : i + batch_size]
            batch_texts = [p[0] for p in batch]
            batch_rows = [p[1] for p in batch]

            model_inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=8192
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **model_inputs, 
                    max_new_tokens=4096,  # Safety limit for infinite loops
                    do_sample=True, 
                    temperature=temperature, 
                    top_k=top_k, 
                    top_p=0.95, 
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True
                )

            input_len = model_inputs.input_ids.shape[1]
            generated_tokens = outputs[:, input_len:]
            decoded_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            for j, out_text in enumerate(decoded_outputs):
                origdf = batch_rows[j]
                model_dir = synth_dir(out_path) / f"B{B_global_index}"
                code_file = model_dir / new_nn_file
                df_file = model_dir / 'dataframe.df'
                
                model_dir.mkdir(parents=True, exist_ok=True)
                create_file(model_dir, new_out_file, out_text)

                nn_code = extract_code(out_text)
                if nn_code:
                    with open(code_file, 'w') as file:
                        file.write(nn_code)
                    if origdf is None:
                        if os.path.isfile(df_file):
                            os.remove(df_file)
                    else:
                        orig_code_file = model_dir / f"original_{origdf['nn']}.py"
                        with open(orig_code_file, 'w') as file:
                            file.write(origdf['nn_code'])
                        origdf.to_pickle(df_file)
                
                B_global_index += 1
