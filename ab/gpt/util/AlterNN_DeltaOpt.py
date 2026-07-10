import json
import os
import re
import ast
import torch

import ab.nn.api as nn_dataset
from ab.nn.util.Util import create_file
from tqdm import tqdm

from ab.gpt.util.Const import conf_test_dir, epoch_dir, new_nn_file, synth_dir, new_out_file
from ab.gpt.util.LLM import LLM
from ab.gpt.util.Util import extract_code, extract_delta


def format_prompt_with_supporting_models(prompt_template, para_dict, supporting_models):
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

    safe_para_dict = {}
    for k, v in para_dict.items():
        v_str = str(v)
        if '{' in v_str or '}' in v_str:
            safe_para_dict[k] = v_str.replace('{', '{{').replace('}', '}}')
        else:
            safe_para_dict[k] = v_str

    class SafeDict(dict):
        def __missing__(self, key):
            return "{" + key + "}"

    try:
        formatted_prompt = prompt_template.format_map(SafeDict(safe_para_dict))
    except Exception as e:
        # Suppress the formatting warning as we handle it correctly below
        # print(f"[WARNING] Prompt formatting issue: {e}")
        formatted_prompt = prompt_template
        for key, value in safe_para_dict.items():
            formatted_prompt = formatted_prompt.replace(f"{{{key}}}", str(value))
    return formatted_prompt


def _check_python_syntax(code: str):
    """Returns (is_valid: bool, error_msg: str or None)."""
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"


def alter_delta(epochs, test_conf, llm_name, gguf_file=None, n=1, temperature=0.6, top_k=50, **kwargs):
    """
    Project-specific alter_delta with LangGraph-style reflection.
    Does NOT touch core AlterNN.py.

    On each failure, the exact error is appended to conversation history
    and LLM is asked to fix it. Falls back to clean baseline only after
    all retries are exhausted.
    """
    nn_prefixes = kwargs.get('nn_prefixes')
    max_retries = kwargs.get('max_retries', 3)

    with open(conf_test_dir / test_conf) as f:
        prompt_dict = json.load(f)
    assert isinstance(prompt_dict, dict)

    use_delta = False
    for key in prompt_dict.keys():
        key_config = prompt_dict[key]
        if isinstance(key_config, dict):
            use_delta = key_config.get('use_delta', False) or 'delta' in str(key).lower()
            if use_delta:
                break

    if not use_delta:
        print("[WARNING] Config file does not have delta mode enabled. Falling back to regular alter().")
        from ab.gpt.util.AlterNN import alter
        return alter(epochs, test_conf, llm_name, gguf_file, n, temperature, top_k)

    model_loader = LLM(llm_name, gguf_file=gguf_file, load_in_4bit=kwargs.get('load_in_4bit', False))
    model = model_loader.get_model()
    tokenizer = model_loader.get_tokenizer()
    print(f"Load Model Complete, Start Loop... (Delta + Reflection mode, max_retries={max_retries})")

    import glob
    from pathlib import Path
    base_epoch_dir = epoch_dir()
    base_epoch_dir.mkdir(parents=True, exist_ok=True)
    existing_epochs = glob.glob(str(base_epoch_dir / "Epoch_*"))
    max_epoch = -1
    for d in existing_epochs:
        try:
            idx = int(Path(d).name.split("_")[1])
            if idx > max_epoch:
                max_epoch = idx
        except Exception:
            pass
    start_epoch = max_epoch + 1
    print(f"[AUTO] Existing maximum epoch index found: Epoch_{max_epoch}. Next generations will write starting from Epoch_{start_epoch}.")

    for epoch in range(epochs):
        target_epoch = start_epoch + epoch
        out_path = epoch_dir(target_epoch)

        prompts = []
        for key in prompt_dict.keys():
            prompt = ""
            for pr in prompt_dict[key]['prompt']:
                prompt += pr + "\n"
            current_prefixes = nn_prefixes or prompt_dict[key].get('nn_prefixes')
            if isinstance(current_prefixes, list):
                current_prefixes = tuple(current_prefixes)
            data = nn_dataset.data(only_best_accuracy=True, task=prompt_dict[key]['task'], nn_prefixes=current_prefixes)
            if current_prefixes:
                data = data[data['nn'].isin(current_prefixes)]
            data = data.groupby(by="nn").sample(n=1)
            addon_data = None
            if prompt_dict[key].get('addon_task'):
                addon_data = nn_dataset.data(only_best_accuracy=True, task=prompt_dict[key]['addon_task'])

            for _, row in data.iterrows():
                para_dict = dict()
                for it in prompt_dict[key]["input_list"]:
                    para_dict[it['para']] = row[it['value']]

                supporting_models = []
                if prompt_dict[key].get('addon_list') and addon_data is not None and n > 0:
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

                formatted_prompt = format_prompt_with_supporting_models(prompt, para_dict, supporting_models)
                prompts.append((formatted_prompt, row))

        # ─── GENERATION LOOP WITH REFLECTION ───────────────────────────────
        B_index = 0
        for idx, prompt_data in tqdm(enumerate(prompts), desc="Generate Deltas"):
            prompt, origdf = prompt_data
            model_dir = synth_dir(out_path) / f"Blip2FastOpt-A{target_epoch}-B{B_index}"
            code_file = model_dir / new_nn_file
            df_file = model_dir / 'dataframe.df'

            success = False
            last_out = ""
            # Conversation history enables reflection: LLM sees its own mistake + feedback
            conversation_history = [{'role': 'user', 'content': prompt}]

            for attempt in range(max_retries):
                if attempt > 0:
                    print(f"\n[REFLECTION] Attempt {attempt + 1}/{max_retries} for Model_{B_index} — LLM sees its error feedback...")

                inputs = tokenizer.apply_chat_template(
                    conversation_history,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(model.device)

                with torch.no_grad():
                    model.eval()
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=6144,
                        do_sample=True,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=0.95,
                        num_return_sequences=1,
                        eos_token_id=tokenizer.eos_token_id
                    )
                last_out = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
                print("Response Available!")

                # Step 1: Extract nn_head tag
                from ab.gpt.util.Util import extract_str
                nn_head = extract_str(last_out, '<nn_head>', '</nn_head>')
                
                if not nn_head:
                    error_feedback = (
                        "Your previous response did not contain a valid CrossModalBridge class. "
                        "Please enclose your python class EXACTLY inside <nn_head> and </nn_head> tags."
                    )
                    conversation_history.append({'role': 'assistant', 'content': last_out})
                    conversation_history.append({'role': 'user', 'content': error_feedback})
                    print(f"[REFLECTION] No <nn_head> tag found. Sending correction feedback to LLM.")
                    continue

                # ── PERMANENT SANITIZATION (runs on every generated class before saving) ──
                import re as _re

                # 1. Fix: selfproj1 → self.proj1 (LLM drops the dot+self)
                nn_head = nn_head.replace('selfproj1', 'self.proj1')
                nn_head = nn_head.replace('selfproj2', 'self.proj2')
                nn_head = nn_head.replace('selfproj3', 'self.proj3')

                # 2. Cap hidden_dim / hidden_size literals to 1024 max to prevent OOM.
                # The OPT-2.7B model already uses ~13 GB; a huge bridge layer will OOM.
                def _cap_dim(m):
                    val = int(m.group(1))
                    return m.group(0).replace(m.group(1), str(min(val, 1024)))
                nn_head = _re.sub(r'(hidden_dim\s*=\s*)(\d+)', lambda m: m.group(1) + str(min(int(m.group(2)), 1024)), nn_head)
                nn_head = _re.sub(r'(hidden_size\s*=\s*)(\d+)', lambda m: m.group(1) + str(min(int(m.group(2)), 1024)), nn_head)
                # Cap the output dim of Linear layers (second arg) to 1024 when > 2048
                def _cap_linear(m):
                    out = int(m.group(2))
                    if out > 2048:
                        return m.group(0).replace(m.group(2), '1024', 1)
                    return m.group(0)
                nn_head = _re.sub(r'(nn\.Linear\s*\(\s*\d+\s*,\s*)(\d+)', _cap_linear, nn_head)
                # ── END SANITIZATION ────────────────────────────────────────────────────

                if origdf is None:
                    print(f"[WARNING] No baseline dataframe for Model_{B_index}. Skipping.")
                    break

                # Step 2: Inject class into baseline code
                try:
                    baseline_code = origdf.get('nn_code', '')
                    if not baseline_code:
                        print(f"[WARNING] No baseline code in origdf for Model_{B_index}. Skipping.")
                        break
                    
                    # 2. Inject class into baseline code using str.replace
                    injection_anchor = "class OPTCaptionDecoder(nn.Module):"
                    if injection_anchor not in baseline_code:
                        injection_anchor = "# ==============================================================================\n# OPT-2.7B decoder"
                        if injection_anchor not in baseline_code:
                            raise ValueError(f"Injection failed: anchor not found in baseline code.")
                    
                    parts = baseline_code.split(injection_anchor, 1)
                    assembled = parts[0] + nn_head + "\n\n" + injection_anchor + parts[1]
                    
                    old_linear = "        self.visual_projection = nn.Linear(QFORMER_HIDDEN, self.opt_embed_dim).to(device)"
                    new_bridge_init = """        try:
            self.visual_projection = CrossModalBridge(self.prm, self.opt_embed_dim)
        except TypeError:
            raise ValueError("CrossModalBridge must accept (prm, out_features) in __init__.")
        self.visual_projection = self.visual_projection.to(device)"""
                    
                    if old_linear not in assembled:
                        raise ValueError(f"Injection failed: visual_projection init line not found in baseline.")
                    nn_code = assembled.replace(old_linear, new_bridge_init, 1)

                    # Step 3: Python syntax check on the patched result
                    is_valid, syntax_error = _check_python_syntax(nn_code)
                    if not is_valid:
                        error_feedback = (
                            f"Your generated class produced Python code with a syntax error: {syntax_error}. "
                            f"Please fix the syntax issue and provide the corrected class inside <nn_head></nn_head> tags."
                        )
                        conversation_history.append({'role': 'assistant', 'content': last_out})
                        conversation_history.append({'role': 'user', 'content': error_feedback})
                        print(f"[REFLECTION] Syntax error in patched code ({syntax_error}). Sending correction feedback to LLM.")
                        continue

                    # Step 4: SUCCESS
                    print(f"[INFO] Successfully applied code injection on attempt {attempt + 1}, saving to: {code_file}")
                    code_file.parent.mkdir(exist_ok=True, parents=True)
                    with open(code_file, 'w') as file:
                        file.write(nn_code)
                    create_file(model_dir, new_out_file, last_out)

                    orig_code_file = model_dir / f"original_{origdf['nn']}.py"
                    with open(orig_code_file, 'w') as file:
                        file.write(baseline_code)

                    delta_file = model_dir / 'nn_head.py'
                    with open(delta_file, 'w') as file:
                        file.write(nn_head)

                    # --- OVERRIDE BASELINE PARAMS WITH LLM'S HP ---
                    hp_str = extract_str(last_out, '<hp>', '</hp>')
                    if hp_str:
                        try:
                            import json
                            # Clean escaped quotes that LLM sometimes outputs: {\"lr\": ...}
                            clean_hp = hp_str.strip().replace('\\"', '"')
                            if clean_hp.startswith('"') and clean_hp.endswith('"'):
                                clean_hp = clean_hp[1:-1]
                            hp_dict = json.loads(clean_hp)
                            # Enforce safe lr ceiling
                            if 'lr' in hp_dict and float(hp_dict['lr']) > 0.0005:
                                hp_dict['lr'] = 0.0001
                            if 'prm' not in origdf or not isinstance(origdf['prm'], dict):
                                origdf['prm'] = {}
                            origdf['prm'].update(hp_dict)
                        except Exception:
                            hp_dict = {}
                    else:
                        hp_dict = {}

                    # --- SMART OOM PREVENTION: scale down hp if heavy projection detected ---
                    import re as _re
                    # Find all hidden_dim / Linear sizes in the generated class
                    hidden_sizes = [int(x) for x in _re.findall(r'nn\.Linear\s*\(\s*\d+\s*,\s*(\d+)', nn_head)]
                    hidden_sizes += [int(x) for x in _re.findall(r'hidden_dim\s*=\s*(\d+)', nn_head)]
                    hidden_sizes += [int(x) for x in _re.findall(r'hidden_size\s*=\s*(\d+)', nn_head)]
                    max_hidden = max(hidden_sizes) if hidden_sizes else 0

                    if max_hidden > 2048:
                        # Very heavy: cut batch and lr aggressively
                        hp_dict['batch'] = 4
                        hp_dict['lr'] = 5e-5
                    elif max_hidden > 1024:
                        hp_dict['batch'] = 8
                        hp_dict['lr'] = hp_dict.get('lr', 0.0001)

                    if hp_dict:
                        if 'prm' not in origdf or not isinstance(origdf['prm'], dict):
                            origdf['prm'] = {}
                        origdf['prm'].update(hp_dict)
                        # Save hp.json so NNEval reads lr/batch directly
                        hp_json_file = model_dir / 'hp.json'
                        with open(hp_json_file, 'w') as f:
                            json.dump(hp_dict, f)

                    origdf.to_pickle(df_file)
                    B_index += 1
                    success = True
                    break


                except Exception as e:
                    error_feedback = (
                        f"Applying your code raised an unexpected error: {e}. "
                        f"Please review your code and regenerate it inside <nn_head></nn_head> tags."
                    )
                    conversation_history.append({'role': 'assistant', 'content': last_out})
                    conversation_history.append({'role': 'user', 'content': error_feedback})
                    print(f"[REFLECTION] Unexpected error: {e}. Sending correction feedback to LLM.")
                    continue

            # ─── SKIP: all retries exhausted — do NOT save baseline ────────
            if not success:
                print(f"[SKIP] All {max_retries} reflection attempts failed for Model_{B_index}. Skipping slot — no file saved.")
                B_index += 1  # advance to next model slot

