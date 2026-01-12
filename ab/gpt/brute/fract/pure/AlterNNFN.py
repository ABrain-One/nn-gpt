import json
import shutil
import itertools
from pathlib import Path

from ab.gpt.util.Const import conf_test_dir, epoch_dir, new_nn_file, synth_dir, fract_dir
from ab.gpt.util.LLM import LLM


def alter(epochs, test_conf, llm_name, gguf_file=None):
    # Load test prompts (still required for compatibility, even if not used later)
    with open(conf_test_dir / test_conf) as f:
        prompt_dict = json.load(f)
    assert isinstance(prompt_dict, dict)

    model_loader = LLM(llm_name, gguf_file=gguf_file)
    model = model_loader.get_model()
    tokenizer = model_loader.get_tokenizer()
    print("Load Model Complete, Start Loop...")

    # Clean out old runs
    shutil.rmtree(epoch_dir(), ignore_errors=True)

    for epoch in range(epochs):
        out_path = epoch_dir(epoch)

        # --- Template ---
        TEMPLATE_PATH = fract_dir / 'pure' / "Fractal_template.py"
        template = TEMPLATE_PATH.read_text()

        # --- Core elements ---
        element_list = [
            'nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias)',
            'nn.MaxPool2d(kernel_size=3, stride=2)'
            'nn.BatchNorm2d(out_channels)',
            'nn.ReLU(inplace=True)',
            'nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()',
        ]

         # --- Generate all (perm, N, num_columns) combos ---
        all_combinations = []
        for r in range(2, 5 + 1):  # lengths 2..5
            for seq in itertools.product(element_list, repeat=r):
                for N in range(1, 6):              # 1..6
                    for num_columns in range(1, 8):  # 1..8
                        all_combinations.append((seq, N, num_columns))

        # --- Generate variants ---
        max_variants = 1200
        counter = 0

        for perm, N, num_columns in all_combinations:
            if counter >= max_variants:
                break

            model_dir = synth_dir(out_path) / f"B{counter}"
            model_dir.mkdir(parents=True, exist_ok=True)

            # Build variant’s layer sequence
            element_code = ",\n        ".join(perm)
            element_list_str = "['Conv2d', 'MaxPool2d', 'BatchNorm2d', 'ReLU', 'Dropout2d']"

                # Fill template placeholders
            nn_code = (
                    template
                    .replace("$$", element_code)
                    .replace("??", element_list_str)
                    .replace("?1", str(N))
                    .replace("?2", str(num_columns))
                )

                # Write new_nn.py only
            (model_dir / new_nn_file).write_text(nn_code)

            print(f"C{counter}: len={len(perm)}, N={N}, num_columns={num_columns}")
            counter += 1