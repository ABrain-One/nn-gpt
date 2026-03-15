import json
import os
import re
import shutil
import torch

import ab.nn.api as nn_dataset
from ab.nn.util.Util import create_file
from tqdm import tqdm

from ab.gpt.util.Const import conf_test_dir, epoch_dir, new_nn_file, synth_dir, new_out_file
from ab.gpt.util.LLM import LLM
from ab.gpt.util.AnalogicalPrompt import (
    get_nn_alter_analogical_prefill,
    get_nn_alter_analogical_prompt,
)
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


def repair_generated_nn_code(nn_code: str) -> str:
    if not nn_code:
        return nn_code

    fixed = nn_code

    needs_torch_import = "torch." in fixed or "torch.device" in fixed
    needs_nn_import = "nn." in fixed or "nn.Module" in fixed
    needs_optim_import = "optim." in fixed

    if needs_torch_import and "import torch" not in fixed:
        fixed = "import torch\n" + fixed
    if needs_nn_import and "import torch.nn as nn" not in fixed:
        prefix = "import torch\n" if fixed.startswith("import torch\n") else ""
        body = fixed[len(prefix):]
        fixed = f"{prefix}import torch.nn as nn\n{body}"
    if needs_optim_import and "import torch.optim as optim" not in fixed:
        header = ""
        body = fixed
        if body.startswith("import torch\n"):
            header += "import torch\n"
            body = body[len("import torch\n"):]
        if body.startswith("import torch.nn as nn\n"):
            header += "import torch.nn as nn\n"
            body = body[len("import torch.nn as nn\n"):]
        fixed = f"{header}import torch.optim as optim\n{body}"

    fixed = re.sub(r"\bc_in\s*=\s*in_shape\[0\]", "c_in = in_shape[1]", fixed)

    # Shape-safe fallback for common classifier failures under larger image transforms.
    # Convert the first fixed-width post-flatten Linear into LazyLinear so the classifier
    # binds to the real feature size at runtime instead of hallucinated spatial dims.
    if "nn.LazyLinear" not in fixed and ("flatten" in fixed.lower() or ".view(" in fixed):
        fixed = re.sub(
            r"nn\.Linear\(\s*([0-9][0-9\s\*\+\-\/]*|[A-Za-z_][A-Za-z0-9_]*\s*[\*\+\-\/][^,\n]+)\s*,\s*([A-Za-z_][A-Za-z0-9_]*|\d+)\s*\)",
            r"nn.LazyLinear(\2)",
            fixed,
            count=1,
        )

    return fixed


def alter(epochs, test_conf, llm_name, gguf_file=None, n=1, temperature=0.6, top_k=50, *args, **kwargs):
    inference_gpt_oss = kwargs.get('inference_gpt_oss', False)
    inference_gpt_oss_max_input_length = kwargs.get('inference_gpt_oss_max_input_length', None)
    analogical = kwargs.get("analogical", False)

    if inference_gpt_oss and inference_gpt_oss_max_input_length is None:
        inference_gpt_oss_max_input_length = 10**18
    batch_size = max(1, int(kwargs.get("batch_size", 1)))

    # Load test prompts
    with open(conf_test_dir / test_conf) as f:
        prompt_dict = json.load(f)
    assert isinstance(prompt_dict, dict)

    model_loader = LLM(llm_name, gguf_file=gguf_file)
    model = model_loader.get_model()
    tokenizer = model_loader.get_tokenizer()
    analogical_prompt = None
    analogical_prefill_ids = None
    analogical_prefill_text = None
    if analogical:
        analogical_prompt = get_nn_alter_analogical_prompt()
        analogical_prefill_text = get_nn_alter_analogical_prefill()
        analogical_prefill_ids = tokenizer(
            analogical_prefill_text,
            add_special_tokens=False,
        ).input_ids
    print(f"Load Model Complete, Start Loop... (Will fetch {n} supporting models per prompt)")
    if analogical:
        print("[INFO] Analogical prompting enabled for NNAlter.")

    shutil.rmtree(epoch_dir(), ignore_errors=True)
    for epoch in range(epochs):
        out_path = epoch_dir(epoch)

        # Generate Prompts
        prompts = []
        for key in prompt_dict.keys():
            if analogical_prompt is not None:
                prompt = analogical_prompt
            else:
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
        if batch_size > 1:
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            old_padding_side = getattr(tokenizer, "padding_side", "right")
            tokenizer.padding_side = "left"
            try:
                for i in tqdm(range(0, len(prompts), batch_size), desc=f"Generate Codes (Batch {batch_size})"):
                    batch_raw = prompts[i:i + batch_size]
                    batch_ids, batch_rows = [], []
                    for p_text, p_row in batch_raw:
                        try:
                            ids = tokenizer.apply_chat_template([{'role': 'user', 'content': p_text}], add_generation_prompt=True, tokenize=True)
                        except TypeError:
                            rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': p_text}], add_generation_prompt=True, tokenize=False)
                            ids = tokenizer(rendered, add_special_tokens=False).input_ids
                        if analogical_prefill_ids:
                            ids = ids + analogical_prefill_ids
                        if inference_gpt_oss and inference_gpt_oss_max_input_length is not None:
                            if len(ids) > inference_gpt_oss_max_input_length:
                                print(f"[INFO] Skipping prompt in batch: length {len(ids)} > {inference_gpt_oss_max_input_length}")
                                continue
                        batch_ids.append(ids)
                        batch_rows.append(p_row)
                    if not batch_ids: continue
                    try:
                        inputs = tokenizer.pad({'input_ids': batch_ids}, padding=True, return_tensors="pt").to(model.device)
                    except Exception:
                        inputs = tokenizer.pad([{'input_ids': x} for x in batch_ids], padding=True, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        model.eval()
                        outputs = model.generate(**inputs, max_new_tokens=8*1024, do_sample=True, temperature=temperature, top_k=top_k, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
                    input_len = inputs.input_ids.shape[1]
                    decoded_outputs = []
                    for k in range(outputs.size(0)):
                        if outputs.size(1) >= input_len and torch.equal(outputs[k, :input_len], inputs.input_ids[k]):
                            start = input_len
                        else:
                            start = int(inputs.attention_mask[k].sum().item()) if hasattr(inputs, "attention_mask") else input_len
                        decoded_outputs.append(tokenizer.decode(outputs[k, start:], skip_special_tokens=True))
                    for out, origdf in zip(decoded_outputs, batch_rows):
                        if analogical_prefill_text:
                            out = analogical_prefill_text + out
                        print("Response Available!")
                        nn_code = extract_code(out)
                        if nn_code:
                            nn_code = repair_generated_nn_code(nn_code)
                            model_dir = synth_dir(out_path) / f"B{B_index}"
                            code_file = model_dir / new_nn_file
                            df_file = model_dir / 'dataframe.df'
                            print(f"[INFO]Saving code to: {code_file}")
                            code_file.parent.mkdir(exist_ok=True, parents=True)
                            with open(code_file, 'w') as file: file.write(nn_code)
                            create_file(model_dir, new_out_file, out)
                            if origdf is None:
                                if os.path.isfile(df_file): os.remove(df_file)
                            else:
                                orig_code_file = model_dir / f"original_{origdf['nn']}.py"
                                with open(orig_code_file, 'w') as file: file.write(origdf['nn_code'])
                                origdf.to_pickle(df_file)
                            B_index += 1
                        else:
                            print("[INFO]Response Invalid!")
            finally:
                tokenizer.padding_side = old_padding_side
            continue

        for idx, prompt in tqdm(enumerate(prompts), desc="Generate Codes"):
            prompt, origdf = prompt
            model_dir = synth_dir(out_path) / f"B{B_index}"
            code_file = model_dir / new_nn_file
            df_file = model_dir / 'dataframe.df'
            inputs = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}, ], add_generation_prompt=True, return_tensors="pt")
            # Handle both tensor and BatchEncoding return types
            if hasattr(inputs, 'input_ids'):
                inputs = inputs.input_ids.to(model.device)
            else:
                inputs = inputs.to(model.device)
            if analogical_prefill_ids:
                prefill_tensor = torch.tensor([analogical_prefill_ids], device=model.device, dtype=inputs.dtype)
                inputs = torch.cat([inputs, prefill_tensor], dim=-1)

            if inference_gpt_oss:
                # Skip prompts that are too long to avoid O(n²) attention OOM
                if inputs.shape[-1] > inference_gpt_oss_max_input_length:
                    print(f"[INFO] Skipping prompt {idx}: input length {inputs.shape[-1]} > {inference_gpt_oss_max_input_length}")
                    continue
            # tokenizer.eos_token_id is the id of <｜end▁of▁sentence｜>  token
            with torch.no_grad():
                model.eval()
                outputs = model.generate(inputs, max_new_tokens=8*1024, do_sample=True, temperature=temperature, top_k=top_k, top_p=0.95, num_return_sequences=1,
                                     eos_token_id=tokenizer.eos_token_id)
            out = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            if analogical_prefill_text:
                out = analogical_prefill_text + out
            print("Response Available!")
            nn_code = extract_code(out)
            if nn_code:
                nn_code = repair_generated_nn_code(nn_code)
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
