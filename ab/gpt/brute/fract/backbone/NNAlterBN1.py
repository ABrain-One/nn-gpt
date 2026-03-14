import json
import shutil
import itertools
import random
import torchvision
from pathlib import Path
import torch
import gc


from ab.gpt.util.Const import conf_test_dir, epoch_dir, new_nn_file, synth_dir, fract_dir
from ab.gpt.util.LLM import LLM

def filter_backbones_by_size(max_params_millions=2):
    print(f"Filtering backbones with < {max_params_millions}M parameters...")

    candidates = [name for name in dir(torchvision.models)
                  if not name.startswith("_")
                  and callable(getattr(torchvision.models, name))
                  and name[0].islower()
                  and "get_" not in name]

    safe_list = []

    for name in candidates:
        try:
            model = torchvision.models.get_model(name, weights=None)

            param_count = sum(p.numel() for p in model.parameters())
            param_count_m = param_count / 1e6

            if param_count_m < max_params_millions:
                safe_list.append(name)
            else:
                pass

            del model

        except Exception as e:
            continue

    gc.collect()
    print(f"Found {len(safe_list)} safe backbones.")
    return safe_list

def alter(epochs, test_conf, llm_name, gguf_file=None):
    print("Load Model Complete, Start Loop...")

    # Clean out old runs
    shutil.rmtree(epoch_dir(), ignore_errors=True)
    available_backbones = filter_backbones_by_size(max_params_millions=30)
    print(available_backbones)

    for epoch in range(epochs):
        out_path = epoch_dir(epoch)

        # --- Template ---
        TEMPLATE_PATH = fract_dir / 'backbone' / "FractalFusion_template.py"
        template = TEMPLATE_PATH.read_text()

        # --- Core elements for Custom CNN Stream ---
        element_list = [
            'nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias)',
            'nn.BatchNorm2d(out_channels)',
            'nn.ReLU(inplace=True)',
            'nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()',
            'nn.SiLU(inplace=True)',
            'nn.GELU()'
        ]

        # --- Generate Custom CNN Combos ---
        cnn_combinations = []

        for r in range(2, 3):
            for seq in itertools.product(element_list, repeat=r):

                if not any("Conv2d" in layer_str for layer_str in seq):
                    continue


                N = random.randint(2, 4)
                num_columns = random.randint(2, 3)

                cnn_combinations.append((seq, N, num_columns))

        print(f"Generated {len(cnn_combinations)} valid CNN structures (with Conv2d).")

        # --- Generate Variants ---
        max_variants = 100
        counter = 0

        random.shuffle(cnn_combinations)

        while counter < max_variants:
            if not cnn_combinations:
                break 

            bb_a, bb_b = random.sample(available_backbones, 2)

            cnn_config = cnn_combinations[counter % len(cnn_combinations)]
            perm, N, num_columns = cnn_config

            model_dir = synth_dir(out_path) / f"B{counter}"
            model_dir.mkdir(parents=True, exist_ok=True)

            # Build Custom Layer Sequence
            element_code = ",\n        ".join(perm)

            simple_perm_log = [p.split('(')[0].replace('nn.', '') for p in perm]
            element_list_str = str(simple_perm_log)

            # Fill template placeholders
            nn_code = (
                template
                .replace("$$", element_code)
                .replace("??", element_list_str)
                .replace("?1", str(N))
                .replace("?2", str(num_columns))
                .replace("?bb_a", f'"{bb_a}"')
                .replace("?bb_b", f'"{bb_b}"')
            )

            (model_dir / new_nn_file).write_text(nn_code)

            if counter % 50 == 0:
                print(f"Generated B{counter}: CNN={simple_perm_log}, BBs=[{bb_a}, {bb_b}]")
            counter += 1