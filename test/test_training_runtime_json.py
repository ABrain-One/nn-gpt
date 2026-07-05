import json
import unittest
from pathlib import Path

import ab.gpt.util.training_runtime as TrainingRuntime


class TrainingRuntimeJsonTests(unittest.TestCase):
    def test_json_safe_converts_nested_paths_to_strings(self):
        payload = {
            "resume": {
                "stage_checkpoint_dir": Path("/tmp/run/checkpoint"),
                "stage_adapter_dir": Path("/tmp/run/checkpoint/adapter"),
                "items": [Path("/tmp/one"), {"inner": Path("/tmp/two")}],
            },
            "active": True,
        }

        converted = TrainingRuntime.json_safe(payload)

        self.assertEqual(
            converted,
            {
                "resume": {
                    "stage_checkpoint_dir": "/tmp/run/checkpoint",
                    "stage_adapter_dir": "/tmp/run/checkpoint/adapter",
                    "items": ["/tmp/one", {"inner": "/tmp/two"}],
                },
                "active": True,
            },
        )
        json.dumps(converted)


if __name__ == "__main__":
    unittest.main()
