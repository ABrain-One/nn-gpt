import json
import shutil
import itertools
from pathlib import Path

from ab.gpt.util.Const import conf_test_dir, epoch_dir, new_nn_file, synth_dir
from ab.gpt.util.LLM import LLM


def alter(epochs, test_conf, llm_name, gguf_file=None):
    # Load test prompts (kept for compatibility)
    with open(conf_test_dir / test_conf) as f:
        prompt_dict = json.load(f)
    assert isinstance(prompt_dict, dict)

    model_loader = LLM(llm_name, gguf_file=gguf_file)
    model = model_loader.get_model()
    tokenizer = model_loader.get_tokenizer()
    print("Load Model Complete, Start Loop...")

    # Clean old runs
    shutil.rmtree(epoch_dir(), ignore_errors=True)

    # -------- TEMPLATE --------
    PURE_DIR = Path(__file__).resolve().parent
    TEMPLATE_PATH = PURE_DIR / "Fractal_template.py"
    template = TEMPLATE_PATH.read_text()
  # -------- ELEMENT LIST ( $$ ) --------
    element_list = [
        'nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias)',
        'nn.MaxPool2d(kernel_size=3, stride=2)',
        'nn.BatchNorm2d(out_channels)',
        'nn.ReLU(inplace=True)',
        'nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()',
    ]

    element_list_str = "['Conv2d', 'MaxPool2d', 'BatchNorm2d', 'ReLU', 'Dropout2d']"
    
    # -------- ELEMENT LIST ( $ ) --------
    element_list_1 = [
        'nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias)',
        'nn.MaxPool2d(kernel_size=3, stride=2)',
    ]

    # -------- ELEMENT LIST ( SEC ) --------
    element_list_secondcolumn = [
        'nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias)',
        'nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()',
        'nn.MaxPool2d(kernel_size=3, stride=2)',
        'nn.MaxUnpool2d(kernel_size=3, stride=2, padding=padding)',
    ]

    # -------- GENERATION --------
    max_variants = 5
    counter = 0

    for epoch in range(epochs):
        out_path = epoch_dir(epoch)
        out_path.mkdir(parents=True, exist_ok=True)

        # ---- $$ LOOP ----
        for r in range(2, 6):
            for perm in itertools.product(element_list, repeat=r):
                element_code = ",\n        ".join(perm)

                for N in range(1, 6):
                    for num_columns in range(1, 8):

                        # ---- _SEC_ LOOP (1→4) ----
                        for k_at in range(1, 5):
                            for perm_at in itertools.product(element_list_secondcolumn, repeat=k_at):
                                element_at_code = ",\n        ".join(perm_at)

                                # ---- $ LOOP (0→2) ----
                                for k1 in range(0, 3):
                                    if k1 == 0:
                                        element1_perms = [()]
                                    else:
                                        element1_perms = itertools.product(element_list_1, repeat=k1)

                                    for perm1 in element1_perms:
                                        element1_code = ",\n        ".join(perm1) if k1 > 0 else ""

                                        if counter >= max_variants:
                                            return

                                        model_dir = synth_dir(out_path) / f"B{counter}"
                                        model_dir.mkdir(parents=True, exist_ok=True)

                                        nn_code = (
                                            template
                                            .replace("$$", element_code)
                                            .replace("$", element1_code)
                                            .replace("_SEC_", element_at_code)
                                            .replace("??", element_list_str)
                                            .replace("?1", str(N))
                                            .replace("?2", str(num_columns))
                                        )

                                        (model_dir / new_nn_file).write_text(nn_code)

                                        print(f"B{counter}: len$$={r}, len$={k1}, lenSEC={k_at}, N={N}, cols={num_columns}")

                                        counter += 1