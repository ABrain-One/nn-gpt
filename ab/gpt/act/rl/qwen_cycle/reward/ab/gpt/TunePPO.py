#!/usr/bin/env python3
"""
TunePPO.py — PPO-based RL Training for nn-gpt

Fine-tunes the LLM to generate better CV model code using PPO.
Evaluates generated models on CIFAR-10 with real training.
Saves step-level results, plots, and correlation metrics.

Usage:
    python -m ab.gpt.TunePPO --simple_test
    python -m ab.gpt.TunePPO --num_prompts 500 --num_epochs 4
"""

from __future__ import annotations
import argparse, gc, json, os, random, re, sys, textwrap, time, inspect
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None

import ab.nn.api as api
from ab.gpt.rl_pipeline.trainer_runtime import (
    load_tokenizer, load_quantized_causal_lm,
    build_lora_config, attach_or_resume_lora,
    enable_non_reentrant_gradient_checkpointing,
)

# ── Configuration ───────────────────────────────────
DEFAULT_BASE_MODEL = "ABrain/NNGPT-Backbone-deepseek-coder-6.7b-instruct"
PPO_CLIP = 0.2
PPO_EPOCHS = 4
PPO_VALUE_COEF = 0.5
MAX_NEW_TOKENS = 768
MAX_LOG_PROB_TOKENS = 512
LEARNING_RATE = 5e-5
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
NN_TRAIN_EPOCHS = 10
OUT_DIR = "out/nngpt/ppo_best"


# ══════════════════════════════════════════════════════
# CIFAR-10 Evaluator
# ══════════════════════════════════════════════════════

class CIFARRewardEvaluator:
    def __init__(self, train_epochs=NN_TRAIN_EPOCHS):
        self.train_epochs = train_epochs
        print("[EVAL] Loading CIFAR-10...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
        print(f"[EVAL] CIFAR-10 ready, {train_epochs} epochs per model")

    def extract_code(self, raw_text):
        text = raw_text
        m = re.search(r'<nn>(.*?)</nn>', text, re.DOTALL)
        if m: text = m.group(1)
        m = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
        if m: text = m.group(1)
        elif re.search(r'```\s*(.*?)```', text, re.DOTALL):
            text = re.search(r'```\s*(.*?)```', text, re.DOTALL).group(1)
        for pat in [
            r'(class \w+\(nn\.Module\).*?)(?=\nclass |\n\ndef |\nif __name__|\Z)',
            r'(class \w+\(nn\.Module\).*)',
        ]:
            m = re.search(pat, text, re.DOTALL)
            if m:
                code = m.group(1).rstrip()
                code = textwrap.dedent(code)
                code = re.sub(r'class \w+\(nn\.Module\)', 'class Net(nn.Module)', code, count=1)
                def patch_init(match):
                    params = match.group(1)
                    defaults = {
                        "in_shape": "(1,3,32,32)", "out_shape": "10",
                        "in_channels": "3", "out_channels": "10",
                        "num_classes": "10", "n_classes": "10",
                        "prm": "None", "device": '"cuda"',
                        "dropout_prob": "0.3", "drop_prob": "0.3",
                        "loc_drop_prob": "0.3", "loc_dropout_prob": "0.3",
                        "dropout": "0.3", "num_columns": "4",
                        "hidden_dim": "128", "hidden_size": "128",
                        "num_layers": "4", "num_heads": "4",
                        "num_blocks": "4", "channels": "64",
                        "img_size": "32", "embed_dim": "128",
                        "model": "None", "backbone": "None",
                    }
                    new_params = []
                    for p in params.split(","):
                        p = p.strip()
                        if not p or p == "self": continue
                        name = p.split(":")[0].split("=")[0].strip()
                        if "=" in p: new_params.append(p.strip())
                        elif name in defaults: new_params.append(f"{name}={defaults[name]}")
                        else: new_params.append(f"{name}=None")
                    return "def __init__(self, " + ", ".join(new_params) + ", **kwargs)"
                code = re.sub(r'def __init__\(self,([^)]+)\)', patch_init, code)
                lines = code.split("\n")
                clean = [l for l in lines if not (
                    l.strip().startswith("self._") and "=" not in l.strip()
                    and "(" not in l.strip() and not l.strip().endswith(":"))]
                code = "\n".join(clean)
                header = ("import torch\nimport torch.nn as nn\n"
                          "import torch.nn.functional as F\n"
                          "import math\nfrom collections import OrderedDict\n\n")
                return header + code
        return None

    def try_instantiate(self, net_class):
        prm = {"lr": 0.001, "batch": 128, "momentum": 0.9, "dropout": 0.3}
        try:
            sig = inspect.signature(net_class.__init__)
            params = list(sig.parameters.keys())[1:]
            defaults = {
                "in_shape": (1, 3, 32, 32), "in_channels": 3,
                "out_shape": 10, "out_channels": 10, "num_classes": 10,
                "prm": prm, "device": "cuda",
                "dropout_prob": 0.3, "drop_prob": 0.3,
                "loc_drop_prob": 0.3, "loc_dropout_prob": 0.3,
                "dropout": 0.3, "num_columns": 4,
                "hidden_dim": 128, "hidden_size": 128,
                "num_layers": 4, "num_heads": 4,
                "channels": 64, "img_size": 32,
            }
            kwargs = {}
            for p in params:
                if p in defaults: kwargs[p] = defaults[p]
                elif "drop" in p.lower(): kwargs[p] = 0.3
                elif "num" in p.lower(): kwargs[p] = 4
                elif "dim" in p.lower() or "size" in p.lower(): kwargs[p] = 64
                else: kwargs[p] = None
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            try: return net_class(**kwargs).cuda()
            except Exception: pass
        except Exception: pass
        for args in [
            ((1, 3, 32, 32), 10, prm, "cuda"),
            ((1, 3, 32, 32), 10, prm),
            (),
        ]:
            try: return net_class(*args).cuda()
            except Exception: continue
        return None

    def _heuristic_score(self, code):
        if not code or len(code) < 50: return 0.0
        s, cl = 0.0, code.lower()
        if 'class' in cl and 'nn.module' in cl: s += 0.15
        if 'def forward' in cl: s += 0.15
        if 'nn.conv2d' in cl: s += 0.1
        if 'batchnorm' in cl: s += 0.1
        if 'dropout' in cl: s += 0.05
        if 'relu' in cl: s += 0.05
        return min(s, 0.6)

    def evaluate(self, generated_text):
        code = self.extract_code(generated_text)
        if code is None:
            return -0.5, "no_class_extracted", False
        # Stub classes for LEMUR helper functions
        class _StubBlock(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                in_c = args[0] if args else 64
                out_c = args[1] if len(args) > 1 else in_c
                self.conv = nn.Conv2d(in_c, out_c, 3, padding=1)
                self.bn = nn.BatchNorm2d(out_c)
            def forward(self, x):
                return F.relu(self.bn(self.conv(x)))
        namespace = {
            "torch": torch, "nn": nn, "F": F,
            "math": __import__("math"),
            "numpy": __import__("numpy"),
            "OrderedDict": __import__("collections").OrderedDict,
            "FractalBlock": _StubBlock, "FractalUnit": _StubBlock,
            "drop_conv3x3_block": _StubBlock, "DropBlock": _StubBlock,
            "make_scaler": lambda **kw: torch.amp.GradScaler("cuda", enabled=False),
            "autocast_ctx": lambda **kw: torch.amp.autocast("cuda", enabled=False),
            "use_amp": False,
        }
        try:
            exec(code, namespace)
        except Exception:
            return -0.4, "exec_failed", False

        if "Net" not in namespace:
            return -0.4, "no_Net_class", False

        net = self.try_instantiate(namespace["Net"])
        if net is None:
            score = self._heuristic_score(code)
            return score * 0.5 - 0.5, f"instantiation_failed({score:.2f})", False

        try:
            test_in = torch.randn(2, 3, 32, 32).cuda()
            test_out = net(test_in)
            if test_out.dim() > 2:
                pool = nn.AdaptiveAvgPool2d(1).cuda()
                orig_fwd = net.forward
                def fixed_fwd(x, _o=orig_fwd, _p=pool):
                    o = _o(x)
                    return _p(o).flatten(1) if o.dim() > 2 else o
                net.forward = fixed_fwd
                test_out = net(test_in)
            if test_out.shape[-1] != 10:
                del net; torch.cuda.empty_cache()
                return -0.1, f"wrong_shape_{test_out.shape}", False
        except Exception:
            del net; torch.cuda.empty_cache()
            return -0.2, "shape_test_failed", False

        try:
            opt = torch.optim.Adam(net.parameters(), lr=0.001)
            crit = nn.CrossEntropyLoss()
            for ep in range(self.train_epochs):
                net.train()
                for data, target in self.trainloader:
                    data, target = data.cuda(), target.cuda()
                    opt.zero_grad()
                    loss = crit(net(data), target)
                    loss.backward()
                    opt.step()
            net.eval()
            correct = total = 0
            with torch.no_grad():
                for data, target in self.testloader:
                    data, target = data.cuda(), target.cuda()
                    _, pred = net(data).max(1)
                    total += target.size(0)
                    correct += pred.eq(target).sum().item()
            accuracy = correct / total
            del net, opt, crit; torch.cuda.empty_cache()
            return accuracy, "OK", True
        except Exception:
            del net; torch.cuda.empty_cache()
            return -0.1, "training_failed", False


# ══════════════════════════════════════════════════════
# Value Head
# ══════════════════════════════════════════════════════

class ValueHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, hidden_states):
        return self.head(hidden_states[:, -1, :].float()).squeeze(-1)


# ══════════════════════════════════════════════════════
# PPO Trainer with Step-Level Tracking
# ══════════════════════════════════════════════════════

class PPOLLMTrainer:
    def __init__(self, model, tokenizer, value_head, evaluator, device, lr=LEARNING_RATE):
        self.model = model
        self.tokenizer = tokenizer
        self.value_head = value_head
        self.evaluator = evaluator
        self.device = device

        # Tracking
        self.history = []
        self.step_results = []
        self.all_accuracies = []
        self.all_real_accs = []
        self.first_accuracy = None
        self.best_accuracy = 0.0
        self.gen_count = 0
        self.step_count = 0

        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW([
            {"params": trainable, "lr": lr},
            {"params": value_head.parameters(), "lr": lr * 10},
        ], weight_decay=0.01)

    @torch.no_grad()
    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt",
                                truncation=True, max_length=512).to(self.device)
        out = self.model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True, temperature=0.8, top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id)
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        self.gen_count += 1
        return text, gen_ids

    def compute_log_probs_short(self, token_ids):
        ids = token_ids[:MAX_LOG_PROB_TOKENS].to(self.device)
        if len(ids) < 2:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        outputs = self.model(input_ids=ids.unsqueeze(0))
        logits = outputs.logits[0]
        log_probs = F.log_softmax(logits[:-1], dim=-1)
        token_lps = log_probs.gather(1, ids[1:].unsqueeze(1)).squeeze(1)
        return token_lps.sum()

    @torch.no_grad()
    def get_value(self, token_ids):
        ids = token_ids[:MAX_LOG_PROB_TOKENS].to(self.device)
        if len(ids) < 2: return 0.0
        out = self.model(input_ids=ids.unsqueeze(0), output_hidden_states=True)
        return self.value_head(out.hidden_states[-1]).item()

    def collect_rollout(self, prompts):
        self.model.eval()
        data = []
        for i, prompt in enumerate(prompts):
            print(f"  [PPO] Prompt {i+1}/{len(prompts)}:")
            print(f"    Generating model A...")
            text_a, ids_a = self.generate(prompt)
            print(f"    Generating model B...")
            text_b, ids_b = self.generate(prompt)

            print(f"    Training model A on CIFAR-10 ({self.evaluator.train_epochs} epochs)...")
            acc_a, status_a, real_a = self.evaluator.evaluate(text_a)
            print(f"    Training model B on CIFAR-10 ({self.evaluator.train_epochs} epochs)...")
            acc_b, status_b, real_b = self.evaluator.evaluate(text_b)

            # Track accuracies
            for acc, is_real in [(acc_a, real_a), (acc_b, real_b)]:
                if is_real and acc > 0:
                    self.all_accuracies.append(acc)
                    self.all_real_accs.append(acc)
                    if self.first_accuracy is None:
                        self.first_accuracy = acc
                    if acc > self.best_accuracy:
                        self.best_accuracy = acc

            # Continuous reward
            r_a, r_b = acc_a, acc_b

            print(f"    A: {acc_a*100:.1f}% ({status_a}) reward={r_a:.3f} | "
                  f"B: {acc_b*100:.1f}% ({status_b}) reward={r_b:.3f}")

            val_a = self.get_value(ids_a)
            val_b = self.get_value(ids_b)

            if len(ids_a) >= 2:
                data.append({"ids": ids_a, "reward": r_a, "value": val_a, "acc": acc_a, "real": real_a})
            if len(ids_b) >= 2:
                data.append({"ids": ids_b, "reward": r_b, "value": val_b, "acc": acc_b, "real": real_b})
            torch.cuda.empty_cache()
        return data

    def ppo_update(self, data):
        self.model.train()
        rewards = torch.tensor([d["reward"] for d in data], device=self.device)
        values = torch.tensor([d["value"] for d in data], device=self.device)
        advantages = rewards - values
        if len(advantages) >= 4 and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        old_lps = []
        for d in data:
            old_lps.append(self.compute_log_probs_short(d["ids"]).detach())
        old_lps = torch.stack(old_lps)

        total_ploss = total_vloss = 0
        for epoch in range(PPO_EPOCHS):
            new_lps, new_vals = [], []
            for d in data:
                new_lps.append(self.compute_log_probs_short(d["ids"]))
                ids = d["ids"][:MAX_LOG_PROB_TOKENS].to(self.device)
                with torch.no_grad():
                    out = self.model(input_ids=ids.unsqueeze(0), output_hidden_states=True)
                    new_vals.append(self.value_head(out.hidden_states[-1]))
            new_lps = torch.stack(new_lps)
            new_vals = torch.stack(new_vals)

            ratio = torch.exp(new_lps - old_lps)
            s1 = ratio * advantages
            s2 = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * advantages
            ploss = -torch.min(s1, s2).mean()
            vloss = F.mse_loss(new_vals.squeeze(), rewards)
            loss = ploss + PPO_VALUE_COEF * vloss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 1.0)
            torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), 1.0)
            self.optimizer.step()
            total_ploss += ploss.item()
            total_vloss += vloss.item()
            torch.cuda.empty_cache()

        return {"policy_loss": total_ploss / PPO_EPOCHS, "value_loss": total_vloss / PPO_EPOCHS}

    def _save_progress(self):
        """Auto-save results and plots during training."""
        os.makedirs(OUT_DIR, exist_ok=True)
        improvement = (self.best_accuracy - self.first_accuracy) if self.first_accuracy else 0
        success_rate = len(self.all_accuracies) / max(self.gen_count, 1)

        with open(f"{OUT_DIR}/step_results.json", "w") as f:
            json.dump({
                "step_results": self.step_results,
                "summary": {
                    "total_generations": self.gen_count,
                    "total_iterations": self.gen_count // 2,
                    "working_models": len(self.all_accuracies),
                    "success_rate": round(success_rate, 3),
                    "first_accuracy": round(self.first_accuracy, 4) if self.first_accuracy else None,
                    "best_accuracy": round(self.best_accuracy, 4),
                    "avg_accuracy": round(float(np.mean(self.all_accuracies)), 4) if self.all_accuracies else 0,
                    "improvement": round(improvement, 4),
                }
            }, f, indent=2)

        try:
            self._save_plots()
            print(f"  [PPO] Progress saved (step {self.step_count}, best={self.best_accuracy*100:.1f}%)")
        except Exception as e:
            print(f"  [PPO] Plot save failed: {e}")

    def _save_plots(self):
        """Generate 4-panel training progress plot."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if len(self.step_results) < 2:
            return

        steps = [s["step"] for s in self.step_results]
        avg_accs = [s["avg_cifar_accuracy"] * 100 for s in self.step_results]
        best_accs = [s["best_so_far"] * 100 for s in self.step_results]
        rewards = [s["avg_reward"] for s in self.step_results]
        p_losses = [s["policy_loss"] for s in self.step_results]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"PPO Training Progress (Step {steps[-1]}, "
                     f"Best: {self.best_accuracy*100:.1f}%, "
                     f"Gens: {self.gen_count})",
                     fontsize=14, fontweight="bold")

        # Smoothing window
        w = min(10, len(avg_accs) // 3) if len(avg_accs) > 3 else 1

        # Top-left: Average accuracy
        axes[0][0].plot(steps, avg_accs, "b-", linewidth=1, alpha=0.4)
        if w > 1:
            smooth = np.convolve(avg_accs, np.ones(w)/w, mode="valid")
            axes[0][0].plot(steps[w-1:], smooth, "b-", linewidth=2.5, label=f"Smoothed (w={w})")
            axes[0][0].legend()
        axes[0][0].set_title("Avg CIFAR-10 Accuracy per Step")
        axes[0][0].set_xlabel("Step"); axes[0][0].set_ylabel("Accuracy (%)")
        axes[0][0].grid(True, alpha=0.3)

        # Top-right: Best accuracy
        axes[0][1].plot(steps, best_accs, "g-", linewidth=2)
        axes[0][1].set_title("Best CIFAR-10 Accuracy Over Time")
        axes[0][1].set_xlabel("Step"); axes[0][1].set_ylabel("Best Accuracy (%)")
        axes[0][1].grid(True, alpha=0.3)

        # Bottom-left: Reward
        axes[1][0].plot(steps, rewards, "r-", linewidth=1, alpha=0.4)
        if w > 1:
            smooth_r = np.convolve(rewards, np.ones(w)/w, mode="valid")
            axes[1][0].plot(steps[w-1:], smooth_r, "r-", linewidth=2.5)
        axes[1][0].set_title("Avg Reward per Step")
        axes[1][0].set_xlabel("Step"); axes[1][0].set_ylabel("Reward")
        axes[1][0].grid(True, alpha=0.3)

        # Bottom-right: Policy loss
        axes[1][1].plot(steps, p_losses, "purple", linewidth=1.5, alpha=0.7)
        axes[1][1].set_title("Policy Loss")
        axes[1][1].set_xlabel("Step"); axes[1][1].set_ylabel("Loss")
        axes[1][1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("ppo_training_progress.png", dpi=150, bbox_inches="tight")
        plt.close()

    def train(self, prompts, num_epochs=2):
        print(f"\n{'='*60}")
        print("PPO TRAINING — Step-Level Tracking")
        print(f"{'='*60}")
        print(f"Prompts: {len(prompts)}, Epochs: {num_epochs}")
        print(f"PPO clip: {PPO_CLIP}, LR: {LEARNING_RATE}")
        print(f"Log prob tokens: {MAX_LOG_PROB_TOKENS}")
        print(f"CV training: {self.evaluator.train_epochs} epochs on CIFAR-10")
        print(f"Auto-save: every 10 steps")
        print(f"{'='*60}\n")

        prompts_per_step = int(os.environ.get('PROMPTS_PER_STEP', '4'))

        for epoch in range(num_epochs):
            random.shuffle(prompts)
            epoch_rewards = []
            epoch_accs = []
            epoch_losses = []
            steps = max(1, len(prompts) // prompts_per_step)

            for step in range(steps):
                start = step * prompts_per_step
                sp = prompts[start:start + prompts_per_step]
                if not sp: break

                print(f"\n[Epoch {epoch+1}/{num_epochs}] Step {step+1}/{steps}")

                data = self.collect_rollout(sp)
                if not data:
                    print("  [PPO] No valid data, skipping")
                    continue

                losses = self.ppo_update(data)
                avg_r = np.mean([d["reward"] for d in data])
                step_accs = [d["acc"] for d in data if d["real"]]
                avg_acc = np.mean(step_accs) if step_accs else 0

                epoch_rewards.append(avg_r)
                epoch_accs.extend(step_accs)
                epoch_losses.append(losses)

                # Step-level tracking
                self.step_count += 1
                self.step_results.append({
                    "step": self.step_count,
                    "epoch": epoch,
                    "avg_reward": float(avg_r),
                    "avg_cifar_accuracy": float(avg_acc),
                    "best_so_far": float(self.best_accuracy),
                    "first_accuracy": float(self.first_accuracy) if self.first_accuracy else 0,
                    "working_models": len(self.all_accuracies),
                    "total_generations": self.gen_count,
                    "success_rate": len(self.all_accuracies) / max(self.gen_count, 1),
                    "policy_loss": losses["policy_loss"],
                    "value_loss": losses["value_loss"],
                })

                n_real = sum(1 for d in data if d["real"])
                print(f"  [PPO] step={self.step_count} reward={avg_r:.3f} "
                      f"acc={avg_acc*100:.1f}% best={self.best_accuracy*100:.1f}% "
                      f"real={n_real}/{len(data)} p_loss={losses['policy_loss']:.4f}")

                # Auto-save every 10 steps
                if self.step_count % 10 == 0:
                    self._save_progress()

                del data
                torch.cuda.empty_cache()

            # Epoch summary
            if epoch_rewards:
                ea = np.mean(epoch_accs) if epoch_accs else 0
                self.history.append({
                    "epoch": epoch,
                    "avg_reward": float(np.mean(epoch_rewards)),
                    "avg_cifar_accuracy": float(ea),
                    "best_cifar_accuracy": float(self.best_accuracy),
                    "working_models": len(epoch_accs),
                    "generations": self.gen_count,
                })
                print(f"\n{'─'*40}")
                print(f"Epoch {epoch+1}: avg_acc={ea*100:.1f}% best={self.best_accuracy*100:.1f}% "
                      f"working={len(epoch_accs)} gens={self.gen_count}")
                print(f"{'─'*40}")

        # Final save
        self._save_progress()
        return self.history


# ── Dataset Loading ────────────────────────────────
def load_prompts(tokenizer, limit=200):
    data = api.data(only_best_accuracy=True, task='img-classification')
    print(f"[PPO] Loaded {len(data)} models from LEMUR")
    prompts = []
    for _, row in data.iterrows():
        if len(prompts) >= limit: break
        acc = float(row.get('accuracy', 0) or 0)
        ds = str(row.get('dataset', 'cifar-10'))
        ep = int(row.get('epoch', 1) or 1)
        text = (f"You are a machine learning model designer.\n"
                f"Generate a PyTorch neural network class Net(nn.Module) that achieves "
                f"at least {acc:.4f} accuracy at epoch {ep} on '{ds}' for img-classification.\n"
                f"Input: 3x32x32 images. Output: 10 classes.\n"
                f"Respond only with the Python class code inside <nn> tags.")
        try:
            fmt = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True, tokenize=False)
        except Exception:
            fmt = text
        prompts.append(fmt)
    random.shuffle(prompts)
    print(f"[PPO] Prepared {len(prompts)} prompts")
    return prompts


# ── Plotting ───────────────────────────────────────
def plot_results(history, path="ppo_cifar10_results.png"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    if not history: return
    epochs = [h["epoch"] for h in history]
    accs = [h["avg_cifar_accuracy"] for h in history]
    bests = [h.get("best_cifar_accuracy", 0) for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("PPO Training — CIFAR-10 Results", fontsize=14, fontweight="bold")
    ax1.plot(epochs, [a*100 for a in accs], "b-o", linewidth=2, label="Average")
    ax1.plot(epochs, [b*100 for b in bests], "g-s", linewidth=2, label="Best")
    ax1.set_title("CIFAR-10 Accuracy per Epoch"); ax1.set_xlabel("Epoch")
    ax1.set_ylabel("%"); ax1.legend(); ax1.grid(True, alpha=0.3)
    rewards = [h["avg_reward"] for h in history]
    ax2.plot(epochs, rewards, "r-o", linewidth=2)
    ax2.set_title("Avg Reward"); ax2.set_xlabel("Epoch"); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"[PPO] Epoch plot: {path}")


# ── Main ───────────────────────────────────────────

# ==== RUN CONFIG PATCH v4: strict reward, 2048 tok, G samples, KL+entropy, reward averaging, AST penalty, early stop, ALGO=none ====
import os as _os, json as _json, time as _time, re as _re
from ab.gpt import TuneKTO as _KTO
try:
    from analyze_collapse import exact_signature as _sig
except Exception:
    _sig = lambda code: None
OUT_DIR = _os.environ.get("RUN_DIR", OUT_DIR)
_E = lambda k, d: _os.environ.get(k, d)
_SEED = int(_E("SEED", "0"))
_ALGO = _E("ALGO", "ppo").lower()
_DIV = _E("DIV_PENALTY", "0") == "1"
_DIV_MODE = _E("DIV_MODE", "kto")
_DIV_ALPHA = float(_E("DIV_ALPHA", "0.1"))
_STRICT = _E("STRICT_REWARD", "0") == "1"
_G = int(_E("SAMPLES", "2"))
_KL = float(_E("KL_COEF", "0"))
_ENT = float(_E("ENT_COEF", "0"))
_RETRAIN = int(_E("RETRAIN", "1"))
_EARLY = _E("EARLY_STOP", "0") == "1"
MAX_NEW_TOKENS = int(_E("MAX_NEW", str(MAX_NEW_TOKENS)))
_GEN_LOG = f"{OUT_DIR}/generations.jsonl"
_STATE = {"gen": 0, "group": 0, "step": 0, "buffer": [], "sig_buffer": [], "cache": {},
          "inf_hist": [], "work_hist": [], "stopped": False}
if _ALGO not in ("ppo", "grpo", "none"):
    raise SystemExit(f"Unknown ALGO={_ALGO}")
_KTO_THR = getattr(_KTO, "SIM_THRESHOLD", None)
_KTO_W = getattr(_KTO, "SIM_PENALTY_WEIGHT", None)
_KTO_TYPE = getattr(_KTO, "SIM_PENALTY_TYPE", None)
_PRM = {"lr": 0.01, "momentum": 0.9, "dropout": 0.2}
_MAX_PARAMS = int(float(_E("MAX_PARAMS", "1e8")))


def _max_sim(code):
    if not code or not _STATE["buffer"]:
        return 0.0
    return float(max(_KTO.jaccard_ngram_similarity(code, ref) for ref in _STATE["buffer"]))


def _kto_penalty(code):
    if not code or not _STATE["buffer"]:
        return 0.0
    return float(_KTO.compute_similarity_penalty(code, _STATE["buffer"][-50:]))


def _strict_extract(text):
    code = text
    if "<nn>" in code:
        code = code.split("<nn>", 1)[1].split("</nn>", 1)[0]
    m = _re.search(r"```(?:python)?\s*\n(.*?)```", code, _re.S)
    if m:
        code = m.group(1)
    return code.replace("```python", "").replace("```", "")


def _strict_build(code):
    ns = {"torch": torch, "nn": nn, "F": F}
    try:
        exec(code, ns)
    except Exception as e:
        return None, f"exec:{type(e).__name__}", -0.4
    if "Net" not in ns:
        return None, "no_Net_class", -0.4
    try:
        with torch.device("meta"):
            _m = ns["Net"]((1, 3, 32, 32), (10,), _PRM, torch.device("meta"))
        _n = sum(p.numel() for p in _m.parameters()); del _m
        if _n > _MAX_PARAMS:
            return None, f"too_large:{_n/1e6:.0f}M", -0.1
    except Exception:
        pass
    try:
        with torch.device("cuda"):
            net = ns["Net"]((1, 3, 32, 32), (10,), _PRM, torch.device("cuda"))
        net = net.cuda()
    except Exception as e:
        torch.cuda.empty_cache()
        return None, f"init:{type(e).__name__}", -0.3
    try:
        with torch.no_grad():
            out = net(torch.randn(2, 3, 32, 32, device="cuda"))
        if not (isinstance(out, torch.Tensor) and tuple(out.shape) == (2, 10)):
            del net
            return None, "wrong_output_shape", -0.1
    except Exception as e:
        del net
        return None, f"forward:{type(e).__name__}", -0.2
    return (ns["Net"], net), "OK", None


def _train_eval(ev, net):
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    for _ in range(ev.train_epochs):
        net.train()
        for x, y in ev.trainloader:
            x, y = x.cuda(), y.cuda()
            opt.zero_grad()
            crit(net(x), y).backward()
            opt.step()
    net.eval()
    c = t = 0
    with torch.no_grad():
        for x, y in ev.testloader:
            x, y = x.cuda(), y.cuda()
            c += (net(x).argmax(1) == y).sum().item()
            t += y.size(0)
    return c / t



# ==== time limit for running generated code (hang guard) ====
import signal as _signal_mod, os as _os
def _gen_timeout(signum, frame):
    raise TimeoutError("generated code exceeded time limit")
_signal_mod.signal(_signal_mod.SIGALRM, _gen_timeout)
def _with_time_limit(fn):
    def wrapped(*a, **k):
        _signal_mod.alarm(int(_os.environ.get("BUILD_TIMEOUT", "120")))
        try:
            return fn(*a, **k)
        finally:
            _signal_mod.alarm(0)
    return wrapped
_strict_build = _with_time_limit(_strict_build)

def _strict_reward(self, text):
    code = _strict_extract(text)
    sig = _sig(code) if code and code.strip() else None
    if not code or not code.strip():
        return -0.5, "no_code", False, code, sig, [], False
    built, reason, pen = _strict_build(code)
    if built is None:
        torch.cuda.empty_cache()
        return pen, reason, False, code, sig, [], False
    NetCls, net = built
    if sig and sig in _STATE["cache"]:
        del net
        torch.cuda.empty_cache()
        accs = _STATE["cache"][sig]
        return float(np.mean(accs)), "OK", True, code, sig, accs, True
    accs = []
    try:
        for k in range(max(1, _RETRAIN)):
            if k > 0:
                with torch.device("cuda"):
                    net = NetCls((1, 3, 32, 32), (10,), _PRM, torch.device("cuda"))
                net = net.cuda()
            accs.append(_train_eval(self.evaluator, net))
            del net
            torch.cuda.empty_cache()
    except Exception as e:
        torch.cuda.empty_cache()
        return -0.1, f"training_failed:{type(e).__name__}", False, code, sig, accs, False
    if sig:
        _STATE["cache"][sig] = accs
    return float(np.mean(accs)), "OK", True, code, sig, accs, False


def _lenient_reward(self, text):
    acc, status, real = self.evaluator.evaluate(text)
    try:
        code = self.evaluator.extract_code(text)
    except Exception:
        code = None
    sig = _sig(code) if code else None
    return acc, str(status), bool(real), code, sig, ([acc] if real else []), False


_orig_load_prompts = load_prompts


def load_prompts(tokenizer, limit=200):
    random.seed(_SEED)
    torch.manual_seed(_SEED)
    prompts = _orig_load_prompts(tokenizer, limit)
    if _E("PROMPT_V2", "0") == "1":
        _OLD = "Respond only with the Python class code inside <nn> tags."
        _NEW = ("Write a complete, self-contained Python file inside <nn> tags: include all imports, "
                "define every helper function and class you use, and define class Net(nn.Module) with "
                "__init__(self, in_shape, out_shape, prm, device) and forward. "
                "Do not stop before the Net class is complete.")
        n_rep = sum(_OLD in q for q in prompts)
        prompts = [q.replace(_OLD, _NEW) for q in prompts]
        print(f"[RUN] PROMPT_V2: replaced instruction in {n_rep}/{len(prompts)} prompts")
    _os.makedirs(OUT_DIR, exist_ok=True)
    cfg = {"seed": _SEED, "algo": _ALGO, "div_penalty": _DIV, "div_mode": _DIV_MODE, "div_alpha": _DIV_ALPHA,
           "strict_reward": _STRICT, "samples_per_prompt": _G, "kl_coef": _KL, "ent_coef": _ENT,
           "retrain": _RETRAIN, "early_stop": _EARLY, "max_new_tokens": MAX_NEW_TOKENS,
           "prompts_per_step": int(_os.environ.get("PROMPTS_PER_STEP", "4")), "prompts": prompts}
    with open(f"{OUT_DIR}/run_prompts.json", "w") as f:
        _json.dump(cfg, f, indent=1)
    print("[RUN] " + " ".join(f"{k}={v}" for k, v in cfg.items() if k != "prompts")
          + f" n_prompts={len(prompts)} log={_GEN_LOG}")
    return prompts


def _collect_rollout(self, prompts):
    self.model.eval()
    data = []
    if _STATE["stopped"]:
        print("  [RUN] phase stopped early - skipping step")
        return data
    for i, prompt in enumerate(prompts):
        _STATE["group"] += 1
        print(f"  [{_ALGO.upper()}] Prompt {i+1}/{len(prompts)}:")
        gens = []
        for k in range(_G):
            print(f"    Generating sample {k+1}/{_G}...")
            try:
                text, ids = self.generate(prompt)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print("    [RUN] generation OOM - sample skipped")
                continue
            gens.append((k, text, ids))
        group_sigs, n_work = [], 0
        for k, text, ids in gens:
            print(f"    Evaluating sample {k+1}/{_G} ({len(ids)} tokens)...")
            if _STRICT:
                acc, status, real, code, sig, accs, cached = _strict_reward(self, text)
            else:
                acc, status, real, code, sig, accs, cached = _lenient_reward(self, text)
            if real and acc > 0:
                n_work += 1
                self.all_accuracies.append(acc)
                self.all_real_accs.append(acc)
                if self.first_accuracy is None:
                    self.first_accuracy = acc
                if acc > self.best_accuracy:
                    self.best_accuracy = acc
            sim = _max_sim(code)
            pen = 0.0
            if _DIV and real:
                if _DIV_MODE == "ast":
                    pen = _DIV_ALPHA if (sig and sig in _STATE["sig_buffer"]) else 0.0
                else:
                    pen = _kto_penalty(code)
            reward = float(acc) - pen
            if code:
                _STATE["buffer"] = (_STATE["buffer"] + [code])[-50:]
            if sig and real:
                _STATE["sig_buffer"] = (_STATE["sig_buffer"] + [sig])[-50:]
            group_sigs.append(sig if real else None)
            _STATE["gen"] += 1
            rec = {"gen": _STATE["gen"], "pid": _os.getpid(), "group": _STATE["group"], "time": _time.time(),
                   "algo": _ALGO, "strict_reward": _STRICT, "div_penalty": _DIV, "status": str(status),
                   "real": bool(real), "accuracy": float(acc) if real else None, "accs": accs,
                   "cached": cached, "sig": sig, "max_sim": sim, "penalty": pen, "reward": reward,
                   "n_tokens": int(len(ids)), "prompt": prompt, "code": code, "raw_output": text}
            with open(_GEN_LOG, "a") as f:
                f.write(_json.dumps(rec) + "\n")
            print(f"    {k+1}: {acc*100:.1f}% ({status}){' [cached]' if cached else ''} "
                  f"pen={pen:.3f} reward={reward:.3f}")
            val = self.get_value(ids) if _ALGO == "ppo" else 0.0
            if len(ids) >= 2:
                data.append({"ids": ids, "reward": reward, "value": val, "acc": acc,
                             "real": real, "group": _STATE["group"]})
        working = [s for s in group_sigs if s]
        _STATE["inf_hist"].append(1.0 if len(set(working)) >= 2 else 0.0)
        _STATE["work_hist"].append(n_work / _G)
        torch.cuda.empty_cache()
    _STATE["step"] += 1
    ri, rw = _STATE["inf_hist"][-12:], _STATE["work_hist"][-12:]
    share, work = sum(ri) / len(ri), sum(rw) / len(rw)
    print(f"  [RUN] informative share={share:.2f} working rate={work:.2f} (last {len(ri)} prompts)")
    if _EARLY and len(ri) >= 12 and share == 0.0 and work >= 0.5:
        _STATE["stopped"] = True
        print("  [RUN] EARLY STOP: working code but no informative prompts in the last 12")
    return data


def _seq_stats(self, ids):
    ids = ids[:MAX_LOG_PROB_TOKENS].to(self.device)
    out = self.model(input_ids=ids.unsqueeze(0))
    lp = F.log_softmax(out.logits[0, :-1].float(), dim=-1)
    tok = lp.gather(1, ids[1:].unsqueeze(1)).squeeze(1)
    ent = -(lp.exp() * lp).sum(-1).mean()
    return tok, ent


def _ppo_update(self, data):
    if _ALGO == "none":
        return {"policy_loss": 0.0, "value_loss": 0.0}
    data = [d for d in data if len(d["ids"]) >= 2]
    if not data:
        return {"policy_loss": 0.0, "value_loss": 0.0}
    self.model.train()
    if _ALGO == "ppo" and not getattr(self, "_vh_checked", False):
        in_opt = {id(p) for g in self.optimizer.param_groups for p in g["params"]}
        for p in self.value_head.parameters():
            p.requires_grad_(True)
        missing = [p for p in self.value_head.parameters() if id(p) not in in_opt]
        if missing:
            self.optimizer.add_param_group({"params": missing})
        print(f"[RUN] value head in optimizer (added {len(missing)})")
        self._vh_checked = True
    rewards = torch.tensor([float(d["reward"]) for d in data], device=self.device, dtype=torch.float32)
    if _ALGO == "grpo":
        advantages = torch.zeros_like(rewards)
        groups = {}
        for k, d in enumerate(data):
            groups.setdefault(d["group"], []).append(k)
        for idx in groups.values():
            if len(idx) > 1:
                r = rewards[idx]
                advantages[idx] = (r - r.mean()) / (r.std(unbiased=False) + 1e-4)
    else:
        values = torch.tensor([float(d["value"]) for d in data], device=self.device, dtype=torch.float32)
        advantages = rewards - values
        if len(advantages) >= 4 and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    with torch.no_grad():
        old_lps = torch.stack([_seq_stats(self, d["ids"])[0].sum() for d in data])
        ref_tok = None
        if _KL > 0:
            with self.model.disable_adapter():
                ref_tok = [_seq_stats(self, d["ids"])[0] for d in data]
    total_ploss = total_vloss = total_kl = total_ent = 0.0
    N = len(data)
    for _ in range(PPO_EPOCHS):
        # one sample at a time, gradients accumulated: same loss, lower peak memory
        self.optimizer.zero_grad()
        ep_p = ep_v = ep_k = ep_e = 0.0
        for i, d in enumerate(data):
            t, e = _seq_stats(self, d["ids"])
            ratio = torch.exp(t.sum() - old_lps[i])
            s1 = ratio * advantages[i]
            s2 = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * advantages[i]
            p_i = -torch.min(s1, s2)
            loss_i = p_i
            if _ENT > 0:
                loss_i = loss_i - _ENT * e
            if _KL > 0:
                r = ref_tok[i]
                k_i = (torch.exp(r - t) - (r - t) - 1).mean()
                loss_i = loss_i + _KL * k_i
                ep_k += k_i.item()
            if _ALGO == "ppo":
                ids = d["ids"][:MAX_LOG_PROB_TOKENS].to(self.device)
                with torch.no_grad():
                    out = self.model(input_ids=ids.unsqueeze(0), output_hidden_states=True)
                v_i = self.value_head(out.hidden_states[-1].detach()).reshape(-1)[-1].float()
                del out
                vl_i = (v_i - rewards[i]) ** 2
                loss_i = loss_i + PPO_VALUE_COEF * vl_i
                ep_v += vl_i.item()
            (loss_i / N).backward()
            ep_p += p_i.item(); ep_e += float(e.item())
            del t, e, loss_i, ratio, s1, s2, p_i
            torch.cuda.empty_cache()
        torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 1.0)
        torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), 1.0)
        self.optimizer.step()
        total_ploss += ep_p / N; total_vloss += ep_v / N; total_kl += ep_k / N; total_ent += ep_e / N
        torch.cuda.empty_cache()
    print(f"  [RUN] update: kl={total_kl/PPO_EPOCHS:.4f} entropy={total_ent/PPO_EPOCHS:.3f}")
    return {"policy_loss": total_ploss / PPO_EPOCHS, "value_loss": total_vloss / PPO_EPOCHS}


def _ppo_update_safe(self, data):
    try:
        return _ppo_update(self, data)
    except torch.cuda.OutOfMemoryError:
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        print("  [RUN] update OOM - step skipped")
        return {"policy_loss": 0.0, "value_loss": 0.0}


for _c in list(globals().values()):
    if isinstance(_c, type) and hasattr(_c, "collect_rollout") and hasattr(_c, "ppo_update"):
        _c.collect_rollout = _collect_rollout
        _c.ppo_update = _ppo_update_safe
        print(f"[RUN] patched {_c.__name__} (v4, algo={_ALGO})")
# ==== END PATCH v4 ====



def main():
    parser = argparse.ArgumentParser(description="PPO Training for nn-gpt")
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--num_prompts", type=int, default=40)
    parser.add_argument("--nn_train_epochs", type=int, default=NN_TRAIN_EPOCHS)
    parser.add_argument("--simple_test", action="store_true")
    args = parser.parse_args()

    if args.simple_test:
        args.num_epochs = 2
        args.num_prompts = 8

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16

    print(f"\n{'='*60}")
    print("TunePPO — PPO with Step-Level Tracking")
    print(f"{'='*60}")
    print(f"Model:   {args.base_model}")
    print(f"Epochs:  {args.num_epochs}")
    print(f"Prompts: {args.num_prompts}")
    print(f"CV epochs: {args.nn_train_epochs}")
    if torch.cuda.is_available():
        print(f"GPU:     {torch.cuda.get_device_name()}")
        print(f"Memory:  {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'='*60}\n")

    tokenizer = load_tokenizer(args.base_model)
    evaluator = CIFARRewardEvaluator(train_epochs=args.nn_train_epochs)
    prompts = load_prompts(tokenizer, limit=args.num_prompts)

    print(f"[PPO] Loading model: {args.base_model}...")
    precision = {"torch_dtype": dtype, "bf16": bf16, "fp16": not bf16,
                 "label": "bf16" if bf16 else "fp16"}
    model = load_quantized_causal_lm(
        model_source=args.base_model, precision=precision,
        train_device=device, use_deepspeed=False)
    try:
        from peft.utils.other import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    except Exception: pass

    peft_cfg = build_lora_config(r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)
    model = attach_or_resume_lora(model, peft_config=peft_cfg,
                                  stage_adapter_dir=None, log_prefix="[PPO]")
    model.print_trainable_parameters()

    value_head = ValueHead(model.config.hidden_size).to(device)
    enable_non_reentrant_gradient_checkpointing(model, log_prefix="[PPO]")

    trainer = PPOLLMTrainer(model, tokenizer, value_head, evaluator, device)
    history = trainer.train(prompts, num_epochs=args.num_epochs)

    # Save model
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(f"{OUT_DIR}/adapter")
    tokenizer.save_pretrained(f"{OUT_DIR}/tokenizer")
    torch.save(value_head.state_dict(), f"{OUT_DIR}/value_head.pt")
    with open(f"{OUT_DIR}/ppo_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Final summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total iterations:  {trainer.gen_count // 2}")
    print(f"Total generations: {trainer.gen_count}")
    if trainer.all_accuracies:
        print(f"Working models:    {len(trainer.all_accuracies)}")
        print(f"Success rate:      {len(trainer.all_accuracies)/trainer.gen_count*100:.1f}%")
        print(f"First accuracy:    {trainer.first_accuracy*100:.1f}%")
        print(f"Best accuracy:     {trainer.best_accuracy*100:.1f}%")
        print(f"Avg accuracy:      {np.mean(trainer.all_accuracies)*100:.1f}%")
        print(f"Improvement:       +{(trainer.best_accuracy - trainer.first_accuracy)*100:.1f}%")

    # Correlation metrics
    if scipy_stats and len(trainer.all_real_accs) > 5:
        iters = list(range(len(trainer.all_real_accs)))
        rho, rho_p = scipy_stats.spearmanr(iters, trainer.all_real_accs)
        tau, tau_p = scipy_stats.kendalltau(iters, trainer.all_real_accs)
        print(f"Spearman rho:      {rho:.3f} (p={rho_p:.4f})")
        print(f"Kendall tau:       {tau:.3f} (p={tau_p:.4f})")

    print(f"{'='*60}")
    print(f"Step results:      {OUT_DIR}/step_results.json")
    print(f"Training plot:     ppo_training_progress.png")
    print(f"{'='*60}")

    plot_results(history)


if __name__ == "__main__":
    main()
