import json
import tempfile
import unittest
from pathlib import Path

import ab.nn.api as lemur
from ab.gpt.util.prompt.NNGenPrompt import NNGenPrompt
from ab.nn.util.Const import ab_root_path


class TestNNGenRuntimePrompt(unittest.TestCase):
    def test_runtime_data_fills_prompt(self):
        records = lemur.run_data(type="pt", max_rows=1)
        self.assertFalse(records.empty)

        record = records.iloc[0]
        config_path = (
            ab_root_path
            / "ab"
            / "gpt"
            / "conf"
            / "prompt"
            / "train"
            / "NN_gen_runtime.json"
        )
        config = json.loads(config_path.read_text())
        config_key = next(iter(config))
        config[config_key]["model_name"] = record["model_name"]
        config[config_key]["duration"] = int(record["duration"])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as prompt_file:
            json.dump(config, prompt_file)
            prompt_file.flush()

            builder = NNGenPrompt(
                max_len=8192,
                tokenizer=_TestTokenizer(),
                prompts_path=prompt_file.name,
            )
            frame = builder.get_raw_dataset(
                only_best_accuracy=False,
                n_training_prompts=1,
            )

        self.assertEqual(len(frame), 1)
        prompt = frame.iloc[0]["instruction"]
        self.assertIn('"model_name"', prompt)
        self.assertIn('"type": "pt"', prompt)
        print(prompt)


class _TestTokenizer:
    chat_template = None

    def apply_chat_template(self, messages, tokenize=False):
        return "\n".join(
            f"{message['role']}: {message['content']}"
            for message in messages
        )


if __name__ == "__main__":
    unittest.main()
