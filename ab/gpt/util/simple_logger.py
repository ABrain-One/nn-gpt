#!/usr/bin/env python3
import json
import time
import os
from datetime import datetime
from typing import Any

class SimpleCodeLogger:
    
    def __init__(self, output_dir: str = "rl_output"):
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, "generation_log.json")
        self.training_log_file = os.path.join(output_dir, "training_progress.log")
        self.success_codes_file = os.path.join(output_dir, "success_codes.py")
        self.samples_file = os.path.join(output_dir, "generation_samples.jsonl")
        sft_append_logs = os.getenv("NNGPT_SFT_APPEND_LOGS", "").strip().lower() in {"1", "true", "yes", "on"}
        self.resume_mode = bool(
            os.getenv("NNGPT_RL_RESUME_CHECKPOINT_DIR", "").strip()
            or os.getenv("NNGPT_RL_RESUME_STAGE", "").strip()
            or os.getenv("NNGPT_SFT_RESUME_TRAINER_CHECKPOINT", "").strip()
            or os.getenv("NNGPT_SFT_RESUME_STAGE_CHECKPOINT", "").strip()
            or sft_append_logs
        )
        
        self.total_count = 0
        self.success_count = 0
        self.warmup_trainable_count = 0
        self.warmup_positive_count = 0
        self.timeout_count = 0
        self.improved_count = 0
        self.start_time = time.time()
        
        os.makedirs(self.output_dir, exist_ok=True)

        training_log_mode = 'a' if self.resume_mode and os.path.exists(self.training_log_file) else 'w'
        sample_log_mode = 'a' if self.resume_mode and os.path.exists(self.samples_file) else 'w'

        if self.resume_mode:
            if os.path.exists(self.samples_file):
                with open(self.samples_file, "r", encoding="utf-8") as f:
                    self.total_count = sum(1 for line in f if line.strip())
            if os.path.exists(self.log_file):
                try:
                    with open(self.log_file, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    summary = dict(payload.get("summary") or {})
                    self.success_count = int(summary.get("success_count", self.success_count) or self.success_count)
                    self.warmup_trainable_count = int(summary.get("warmup_trainable_count", self.warmup_trainable_count) or self.warmup_trainable_count)
                    self.warmup_positive_count = int(summary.get("warmup_positive_count", self.warmup_positive_count) or self.warmup_positive_count)
                    self.timeout_count = int(summary.get("timeout_count", self.timeout_count) or self.timeout_count)
                    self.improved_count = int(summary.get("improved_count", self.improved_count) or self.improved_count)
                    previous_start_time = summary.get("start_time")
                    if previous_start_time is not None:
                        self.start_time = float(previous_start_time)
                except Exception:
                    pass

        with open(self.training_log_file, training_log_mode, encoding='utf-8') as f:
            f.write(f"=== RL Training start {datetime.now().isoformat()} ===\n")
            f.write(f"Output directory: {os.path.abspath(self.output_dir)}\n")
        with open(self.samples_file, sample_log_mode, encoding='utf-8'):
            pass

    def log_to_file(self, message: str):
        with open(self.training_log_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime("%H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")
    
    def log_generation(self, prompt: str, completion: str, reward: float, api_result: Any = None):
        self.total_count += 1

        if isinstance(api_result, dict):
            trainable_ok = bool(
                api_result.get("built_ok")
                and api_result.get("forward_shape_ok")
                and api_result.get("backward_ok")
                and api_result.get("loss_drop_ok")
            )
            if api_result.get("timed_out"):
                self.timeout_count += 1
            if api_result.get("group_warmup") and trainable_ok:
                self.warmup_trainable_count += 1
                if float(reward) > 0.0:
                    self.warmup_positive_count += 1
            if (not api_result.get("group_warmup")) and trainable_ok and bool(api_result.get("group_train_acc_improved")):
                self.improved_count += 1
        
        self.log_to_file(f"Generation {self.total_count}: Reward={reward:.4f}")
        
        if (reward > 0) and (api_result is not None):
            self.success_count += 1
            self.log_to_file(f"Success case #{self.success_count}: Reward={reward:.4f}, code length={len(completion)}")
            self.log_to_file(f"   API result: {api_result}")
        else:
            self.log_to_file(f"Fail case: Reward={reward:.4f}")

        record = {
            "prompt": prompt,
            "completion": completion,
            "reward": reward,
            "api_result": api_result,
        }
        with open(self.samples_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        if self.total_count % 10 == 0:
            current_success_rate = (self.success_count / self.total_count * 100)
            self.log_to_file(
                f"progress: {self.total_count} generation，"
                f"{self.success_count} success，success rate {current_success_rate:.1f}% "
                f"warmup_trainable_count={self.warmup_trainable_count} "
                f"warmup_positive_count={self.warmup_positive_count} "
                f"timeout_count={self.timeout_count} "
                f"improved_count={self.improved_count}"
            )
     
    def save_log(self):
        log_data = {
            'summary': {
                'total_count': self.total_count,
                'success_count': self.success_count,
                'warmup_trainable_count': self.warmup_trainable_count,
                'warmup_positive_count': self.warmup_positive_count,
                'timeout_count': self.timeout_count,
                'improved_count': self.improved_count,
                'success_rate': (self.success_count / self.total_count) if self.total_count > 0 else 0,
                'start_time': self.start_time,
                'duration_minutes': (time.time() - self.start_time) / 60
            }
        }
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"log save to {self.log_file}")
