import json
import os
import shutil

import ab.nn.api as nn_dataset
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from ab.gpt.util.Const import conf_test_dir, epoch_dir, new_nn_file, synth_dir, new_out_file
from ab.gpt.util.Util import extract_code
from ab.nn.util.Util import create_file

def alter(epochs, test_conf, llm_name):
    # Load test prompts
    with open(conf_test_dir / test_conf) as f:
        prompt_dict = json.load(f)
    assert isinstance(prompt_dict, dict)

    print("Loading Tokenizer and Model...")
    tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
    print("Load Tokenizer Complete")
    model = AutoModelForCausalLM.from_pretrained(llm_name, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    print("Load Model Complete, Start Loop...")

    shutil.rmtree(epoch_dir(), ignore_errors=True)
    df = nn_dataset.data()
    df["prm_str"] = df["prm"].apply(lambda x: json.dumps(x, sort_keys=True) if isinstance(x, dict) else str(x))
    unique_cols = ['task', 'nn', 'transform_code', 'metric', 'metric_code']
    df_unique = df.drop_duplicates(subset=unique_cols).reset_index(drop=True)

    # Select best classification model (for segmentation addon)
    best_class_code = None
    if not df_unique[df_unique["task"] == "img-classification"].empty:
        class_subset = df_unique[df_unique["task"] == "img-classification"].copy()
        class_subset["accuracy"] = pd.to_numeric(class_subset["accuracy"], errors="coerce")
        best_row = class_subset.loc[class_subset["accuracy"].idxmax()]
        best_class_code = str(best_row["nn_code"])

    for epoch in range(epochs):
        out_path = epoch_dir(epoch)
        B_index = 0

        for key in prompt_dict.keys():
            template = prompt_dict[key]
            task = template["task"]
            addon_task = template.get("addon_task")
            data = nn_dataset.data(only_best_accuracy=True, task=task).drop_duplicates(subset=['nn'])
            addon_data = nn_dataset.data(only_best_accuracy=True, task=addon_task) if addon_task else None

            if data.empty:
                print(f"[WARN] No data found for task: {task}. Skipping...")
                continue

            for _, row in data.iterrows():
                prompt = ""
                original_code = row['nn_code']
                prm_str = json.dumps(row['prm'], sort_keys=True) if isinstance(row['prm'], dict) else str(row['prm'])

                para_dict = {it['para']: row[it['value']] for it in template['input_list'] if it['value'] in row}

                if addon_data is not None and not addon_data[addon_data.nn != row['nn']].empty:
                    addon_row = addon_data[addon_data.nn != row['nn']].sample(n=1).iloc[0]
                    for it in template['addon_list']:
                        para_dict[it['para']] = addon_row[it['value']]

                if task == "img-classification":
                    try:
                        acc_val = float(row.get("accuracy", 0.5))
                    except Exception:
                        acc_val = 0.5

                    base = acc_val if acc_val <= 1 else acc_val / 100.0
                    addon_accuracy_val = round(min(base + torch.rand(1).item() * 0.05 + 0.01, 0.9999), 4)
                    addon_accuracy_fmt = addon_accuracy_val if acc_val <= 1 else round(addon_accuracy_val * 100, 2)

                    para_dict["accuracy"] = row.get("accuracy", "0")
                    para_dict["addon_accuracy"] = addon_accuracy_fmt

                if task == "img-segmentation" and best_class_code:
                    para_dict["addon_nn_code"] = best_class_code

                if 'prompt' not in template:
                    print(f"[ERROR] Missing 'prompt' key in template: {key}")
                    continue

                for pr in template['prompt']:
                    prompt += pr.format(**para_dict) + "\n"

                print(f"[DEBUG] Prompt:\n{prompt}")

                model_dir = synth_dir(out_path) / f"B{B_index}"
                code_file = model_dir / new_nn_file
                df_file = model_dir / 'dataframe.df'
                inputs = tokenizer.apply_chat_template([
                    {'role': 'user', 'content': prompt},
                ], add_generation_prompt=True, return_tensors="pt").to(model.device)
                outputs = model.generate(inputs, max_new_tokens=64 * 1024, do_sample=True, temperature=0.6, top_k=50, top_p=0.95, num_return_sequences=1,
                                         eos_token_id=tokenizer.eos_token_id)
                out = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
                print("[DEBUG] Model Output:\n", out)
                nn_code = extract_code(out)
                if nn_code:
                    print(f"[INFO]Saving code to: {code_file}")
                    code_file.parent.mkdir(exist_ok=True, parents=True)
                    with open(code_file, 'w') as file:
                        file.write(nn_code)
                    create_file(model_dir, new_out_file, out)

                    if os.path.isfile(df_file):
                        os.remove(df_file)
                    row.to_pickle(df_file)
                    orig_code_file = model_dir / f"original_{row['nn']}.py"
                    with open(orig_code_file, 'w') as file:
                        file.write(row['nn_code'])
                    B_index += 1
                else:
                    print("[INFO]Response Invalid! No valid code extracted.")
                    continue