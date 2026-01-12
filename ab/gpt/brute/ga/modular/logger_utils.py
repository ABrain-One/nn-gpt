import json
import time
import os

class MutationLogger:
    def __init__(self, log_file=None):
        if log_file is None:
            log_file = os.path.join(os.path.dirname(__file__), "dataset/mutation_log.jsonl")
        self.log_file = log_file
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def log(self, event_type, prompt, code, reward, error=None):
        entry = {
            "timestamp": time.time(),
            "type": event_type,
            "prompt": prompt,
            "code": code,
            "reward": reward,
            "error": error
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
