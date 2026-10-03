"""Generate one runtime prompt through NNGenPrompt."""

import argparse
import json
import tempfile
from pathlib import Path

from ab.gpt.util.prompt.NNGenPrompt import NNGenPrompt


class TestTokenizer:
    chat_template = None

    def apply_chat_template(self, messages, tokenize=False):
        return "\n".join(
            f"{message['role']}: {message['content']}"
            for message in messages
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one PyTorch runtime prompt from the active database"
    )
    parser.add_argument("model_name")
    parser.add_argument("duration", type=int)
    args = parser.parse_args()

    source_config = Path(
        "/home/nayana/Desktop/nn-gpt/ab/gpt/conf/prompt/train/NN_gen_runtime.json"
    )
    config = json.loads(source_config.read_text())
    config_key = next(iter(config))
    config[config_key]["model_name"] = args.model_name
    config[config_key]["duration"] = args.duration

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as prompt_file:
        json.dump(config, prompt_file)
        prompt_file.flush()
        builder = NNGenPrompt(
            max_len=8192,
            tokenizer=TestTokenizer(),
            prompts_path=prompt_file.name,
        )
        frame = builder.get_raw_dataset(
            only_best_accuracy=False,
            n_training_prompts=1,
        )

    print(frame.iloc[0]["instruction"])


if __name__ == "__main__":
    main()
