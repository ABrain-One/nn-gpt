#!/usr/bin/env python3
"""
TunePPO.py — PPO with Real CIFAR-10 Evaluation

Generates CV model code with the LLM, actually trains each model
on CIFAR-10, and uses the real test accuracy as the PPO reward.

Usage:
    python -m ab.gpt.TunePPO --simple_test          # quick test (6 prompts, 2 epochs)
    python -m ab.gpt.TunePPO --num_prompts 20 --num_epochs 3   # real training
"""

from __future__ import annotations
import argparse, gc, json, os, random, re, sys, textwrap, time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import ab.nn.api as api
from ab.gpt.rl_pipeline.trainer_runtime import (
    load_tokenizer, load_quantized_causal_lm,
    build_lora_config, attach_or_resume_lora,
    enable_non_reentrant_gradient_checkpointing,
)

# ── Config ──────────────────────────────────────────
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
NN_TRAIN_EPOCHS = 5          # Epochs to train each generated CV model on CIFAR-10


# ══════════════════════════════════════════════════════
# Real CIFAR-10 Evaluator
# ══════════════════════════════════════════════════════

class CIFARRewardEvaluator:
    """
    Evaluates generated CV model code by actually training on CIFAR-10.
    Returns real test accuracy as the reward signal.
    """

    def __init__(self, train_epochs=NN_TRAIN_EPOCHS):
        self.train_epochs = train_epochs
        print(f"[EVAL] Loading CIFAR-10 dataset...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)
        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size=128, shuffle=False, num_workers=2)
        print(f"[EVAL] CIFAR-10 ready ({len(trainset)} train, {len(testset)} test)")
        print(f"[EVAL] Training each generated model for {train_epochs} epochs")

    def extract_code(self, raw_text):
        """Extract Net class from LLM output."""
        text = raw_text

        # Try <nn> tags
        m = re.search(r'<nn>(.*?)</nn>', text, re.DOTALL)
        if m:
            text = m.group(1)

        # Try markdown code blocks
        m = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
        if m:
            text = m.group(1)
        elif re.search(r'```\s*(.*?)```', text, re.DOTALL):
            text = re.search(r'```\s*(.*?)```', text, re.DOTALL).group(1)

        # Find class definition
        for pat in [
            r'(class \w+\(nn\.Module\).*?)(?=\nclass |\n\ndef |\nif __name__|\Z)',
            r'(class \w+\(nn\.Module\).*)',
        ]:
            m = re.search(pat, text, re.DOTALL)
            if m:
                code = m.group(1).rstrip()
                code = textwrap.dedent(code)
                code = re.sub(r'class \w+\(nn\.Module\)',
                              'class Net(nn.Module)', code, count=1)
                header = ("import torch\nimport torch.nn as nn\n"
                          "import torch.nn.functional as F\n"
                          "import math\nfrom collections import OrderedDict\n\n")
                # Patch ANY __init__ with args to add defaults + **kwargs
                def patch_init(match):
                    """Add default values to all constructor parameters."""
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
                        if not p or p == "self":
                            continue
                        # Remove type annotations
                        name = p.split(":")[0].split("=")[0].strip()
                        if "=" in p:
                            new_params.append(p.strip())
                        elif name in defaults:
                            new_params.append(f"{name}={defaults[name]}")
                        else:
                            new_params.append(f"{name}=None")
                    result = "def __init__(self, " + ", ".join(new_params) + ", **kwargs)"
                    return result
                code = re.sub(
                    r'def __init__\(self,([^)]+)\)',
                    patch_init,
                    code
                )
                # Remove broken attribute-only lines (e.g. self._something_undefined)
                lines = code.split("\n")
                clean_lines = []
                for line in lines:
                    stripped = line.strip()
                    # Skip lines that are just bare attribute access (no assignment, no call)
                    if (stripped.startswith("self._") and
                        "=" not in stripped and "(" not in stripped and
                        not stripped.endswith(":")):
                        continue
                    clean_lines.append(line)
                code = "\n".join(clean_lines)
                return header + code
        print(f'    [DEBUG] All instantiation attempts failed')
        return None

    def try_instantiate(self, net_class):
        """Try to instantiate by inspecting constructor and providing defaults."""
        import inspect
        try:
            sig = inspect.signature(net_class.__init__)
            params = list(sig.parameters.keys())[1:]  # skip 'self'
            defaults = {
                "in_shape": (1, 3, 32, 32), "in_channels": 3,
                "out_shape": 10, "out_channels": 10, "num_classes": 10,
                "prm": {"lr": 0.001, "batch": 128, "momentum": 0.9, "dropout": 0.3},
                "device": "cuda", "dropout_prob": 0.3, "drop_prob": 0.3,
                "loc_drop_prob": 0.3, "dropout": 0.3, "num_columns": 4,
                "hidden_dim": 128, "hidden_size": 128, "embed_dim": 128,
                "num_layers": 4, "num_heads": 4, "num_blocks": 4,
                "channels": 64, "n_classes": 10, "img_size": 32,
            }
            kwargs = {}
            for p in params:
                if p in defaults:
                    kwargs[p] = defaults[p]
                elif "drop" in p.lower():
                    kwargs[p] = 0.3
                elif "num" in p.lower() or "n_" in p.lower():
                    kwargs[p] = 4
                elif "dim" in p.lower() or "size" in p.lower() or "channel" in p.lower():
                    kwargs[p] = 64
                else:
                    kwargs[p] = None
            try:
                return net_class(**kwargs).cuda()
            except Exception:
                pass
            # Remove None values and retry
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            try:
                return net_class(**kwargs).cuda()
            except Exception:
                pass
        except Exception:
            pass
        """Try multiple LEMUR argument formats."""
        prm = {"lr": 0.001, "batch": 128, "momentum": 0.9, "dropout": 0.3}
        # Try all common argument patterns
        for args in [
            ((1, 3, 32, 32), 10, prm, "cuda"),      # out_shape as int + device
            ((1, 3, 32, 32), 10, prm),                # out_shape as int
            ((1, 3, 32, 32), (10,), prm, "cuda"),     # out_shape as tuple + device
            ((1, 3, 32, 32), (10,), prm),              # out_shape as tuple
            (3, 10),                                    # in_channels, num_classes
            (),                                         # no args
        ]:
            try:
                return net_class(*args).cuda()
            except Exception as e:
                print(f'    [DEBUG] args={[type(a).__name__ for a in args]} failed: {str(e)[:80]}')
                continue
        # Try keyword args
        for out in [10, (10,)]:
            try:
                return net_class(
                    in_shape=(1, 3, 32, 32), out_shape=out,
                    prm=prm, device="cuda").cuda()
            except Exception:
                pass
            try:
                return net_class(
                    in_shape=(1, 3, 32, 32), out_shape=out,
                    prm=prm).cuda()
            except Exception:
                pass
        try:
            return net_class(num_classes=10).cuda()
        except Exception:
            pass
        return None

    def _heuristic_score(self, code):
        """Heuristic code quality score when real evaluation fails."""
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
        """
        Full pipeline: extract code → parse → instantiate → shape test → 
        train on CIFAR-10 → return real test accuracy.
        
        Returns (accuracy, status_string)
        """
        # Extract code
        code = self.extract_code(generated_text)
        if code is None:
            return -0.5, "no_class_extracted"

        # Parse and exec
        namespace = {
            "torch": torch, "nn": nn, "F": F,
            "math": __import__("math"),
            "numpy": __import__("numpy"),
            "np": __import__("numpy"),
            "OrderedDict": __import__("collections").OrderedDict,
        }
        try:
            exec(code, namespace)
        except Exception as e:
            return -0.4, "exec_failed"

        if "Net" not in namespace:
            return -0.4, "no_Net_class"

        # Instantiate
        net = self.try_instantiate(namespace["Net"])
        if net is None:
            score = self._heuristic_score(code)
            return score * 0.5 - 0.5, f"instantiation_failed({score:.2f})"

        # Shape test — auto-fix models that forget to flatten/pool
        try:
            test_in = torch.randn(2, 3, 32, 32).cuda()
            test_out = net(test_in)
            if test_out.dim() > 2:
                # Model outputs (batch, classes, H, W) — wrap with global avg pool
                original_forward = net.forward
                pool = nn.AdaptiveAvgPool2d(1).cuda()
                def fixed_forward(x, _orig=original_forward, _pool=pool):
                    out = _orig(x)
                    if out.dim() > 2:
                        out = _pool(out).flatten(1)
                    return out
                net.forward = fixed_forward
                test_out = net(test_in)
            if test_out.dim() == 1:
                test_out = test_out.unsqueeze(0)
            if test_out.shape[-1] != 10:
                del net; torch.cuda.empty_cache()
                return -0.1, f"wrong_output_shape_{test_out.shape}"
        except Exception as e:
            del net; torch.cuda.empty_cache()
            return 0.0, f"shape_test_failed: {str(e)[:60]}"

        # Train on CIFAR-10
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

            # Test accuracy
            net.eval()
            correct = total = 0
            with torch.no_grad():
                for data, target in self.testloader:
                    data, target = data.cuda(), target.cuda()
                    _, pred = net(data).max(1)
                    total += target.size(0)
                    correct += pred.eq(target).sum().item()

            accuracy = correct / total
            del net, opt, crit
            torch.cuda.empty_cache()
            return accuracy, "OK"

        except Exception as e:
            del net
            torch.cuda.empty_cache()
            return 0.0, f"training_failed: {str(e)[:60]}"


# ══════════════════════════════════════════════════════
# Value Head (Critic)
# ══════════════════════════════════════════════════════

class ValueHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, hidden_states):
        return self.head(hidden_states[:, -1, :].float()).squeeze(-1)


# ══════════════════════════════════════════════════════
# PPO Trainer
# ══════════════════════════════════════════════════════

class PPOLLMTrainer:
    def __init__(self, model, tokenizer, value_head, evaluator, device, lr=LEARNING_RATE):
        self.model = model
        self.tokenizer = tokenizer
        self.value_head = value_head
        self.evaluator = evaluator
        self.device = device
        self.history = []
        self.all_accuracies = []
        self.gen_count = 0

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
        if len(ids) < 2:
            return 0.0
        out = self.model(input_ids=ids.unsqueeze(0), output_hidden_states=True)
        return self.value_head(out.hidden_states[-1]).item()

    def collect_rollout(self, prompts):
        """Generate pairs, evaluate on CIFAR-10, use continuous accuracy as reward."""
        self.model.eval()
        data = []

        for i, prompt in enumerate(prompts):
            print(f"  [PPO] Prompt {i+1}/{len(prompts)}:")

            # Generate two completions
            print(f"    Generating model A...")
            text_a, ids_a = self.generate(prompt)
            print(f"    Generating model B...")
            text_b, ids_b = self.generate(prompt)

            # REAL EVALUATION on CIFAR-10
            print(f"    Training model A on CIFAR-10 ({NN_TRAIN_EPOCHS} epochs)...")
            acc_a, status_a = self.evaluator.evaluate(text_a)
            print(f"    Training model B on CIFAR-10 ({NN_TRAIN_EPOCHS} epochs)...")
            acc_b, status_b = self.evaluator.evaluate(text_b)

            # Track all accuracies
            if acc_a > 0:
                self.all_accuracies.append(acc_a)
            if acc_b > 0:
                self.all_accuracies.append(acc_b)

            # Continuous reward: use raw accuracy directly
            # Continuous reward: raw accuracy (or heuristic score) as reward
            r_a = acc_a
            r_b = acc_b

            print(f"    A: {acc_a*100:.1f}% ({status_a}) | B: {acc_b*100:.1f}% ({status_b}) | reward: A={r_a:.3f} B={r_b:.3f}")

            val_a = self.get_value(ids_a)
            val_b = self.get_value(ids_b)

            if len(ids_a) >= 2:
                data.append({"ids": ids_a, "reward": r_a, "value": val_a, "acc": acc_a})
            if len(ids_b) >= 2:
                data.append({"ids": ids_b, "reward": r_b, "value": val_b, "acc": acc_b})

            torch.cuda.empty_cache()

        return data

    def ppo_update(self, data):
        """PPO clipped surrogate update."""
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
            new_lps = []
            new_vals = []
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

        return {"policy_loss": total_ploss / PPO_EPOCHS,
                "value_loss": total_vloss / PPO_EPOCHS}

    def train(self, prompts, num_epochs=2):
        print(f"\n{'='*60}")
        print("PPO TRAINING — Real CIFAR-10 Evaluation")
        print(f"{'='*60}")
        print(f"Prompts: {len(prompts)}, Epochs: {num_epochs}")
        print(f"PPO clip: {PPO_CLIP}, LR: {LEARNING_RATE}")
        print(f"CV model training: {NN_TRAIN_EPOCHS} epochs on CIFAR-10")
        print(f"{'='*60}\n")

        pairs_per_step = 4
        for epoch in range(num_epochs):
            random.shuffle(prompts)
            epoch_rewards = []
            epoch_accs = []
            epoch_losses = []
            steps = max(1, len(prompts) // pairs_per_step)

            for step in range(steps):
                start = step * pairs_per_step
                sp = prompts[start:start + pairs_per_step]
                if not sp:
                    break

                print(f"\n[Epoch {epoch+1}/{num_epochs}] Step {step+1}/{steps}")
                data = self.collect_rollout(sp)
                if not data:
                    continue

                losses = self.ppo_update(data)
                avg_r = np.mean([d["reward"] for d in data])
                step_accs = [d["acc"] for d in data if d["acc"] > 0]
                avg_acc = np.mean(step_accs) if step_accs else 0

                epoch_rewards.append(avg_r)
                epoch_accs.extend(step_accs)
                epoch_losses.append(losses)

                print(f"  [PPO] avg_reward={avg_r:.3f} avg_cifar_acc={avg_acc*100:.1f}% "
                      f"p_loss={losses['policy_loss']:.4f} v_loss={losses['value_loss']:.4f}")

                del data
                torch.cuda.empty_cache()

            if epoch_rewards:
                ea = np.mean(epoch_accs) if epoch_accs else 0
                self.history.append({
                    "epoch": epoch,
                    "avg_reward": float(np.mean(epoch_rewards)),
                    "avg_cifar_accuracy": float(ea),
                    "working_models": len(epoch_accs),
                    "generations": self.gen_count,
                })
                print(f"\n{'─'*40}")
                print(f"Epoch {epoch+1}: avg_cifar_acc={ea*100:.1f}% "
                      f"working_models={len(epoch_accs)} gens={self.gen_count}")
                print(f"{'─'*40}")

        return self.history


# ── Dataset ────────────────────────────────────────
def load_prompts(tokenizer, limit=200):
    data = api.data(only_best_accuracy=True, task='img-classification')
    print(f"[PPO] Loaded {len(data)} models from LEMUR")
    prompts = []
    for _, row in data.iterrows():
        if len(prompts) >= limit:
            break
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
        except:
            fmt = text
        prompts.append(fmt)
    random.shuffle(prompts)
    print(f"[PPO] Prepared {len(prompts)} prompts")
    return prompts


# ── Plotting ───────────────────────────────────────
def plot_results(history, path="ppo_cifar10_results.png"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    accs = [h["avg_cifar_accuracy"] for h in history]
    losses = [h.get("avg_reward", 0) for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("PPO Training — Real CIFAR-10 Accuracy", fontsize=14, fontweight="bold")
    ax1.plot(epochs, [a * 100 for a in accs], "g-o", linewidth=2)
    ax1.set_title("Avg CIFAR-10 Accuracy (%)"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("%"); ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, losses, "b-o", linewidth=2)
    ax2.set_title("Avg Reward"); ax2.set_xlabel("Epoch"); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); print(f"[PPO] Plot: {path}")


# ── Main ───────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
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
    print("TunePPO — PPO with Real CIFAR-10 Evaluation")
    print(f"{'='*60}")
    print(f"Model:   {args.base_model}")
    print(f"Epochs:  {args.num_epochs}")
    print(f"Prompts: {args.num_prompts}")
    print(f"CV train epochs: {args.nn_train_epochs}")
    if torch.cuda.is_available():
        print(f"GPU:     {torch.cuda.get_device_name()}")
        print(f"Memory:  {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'='*60}\n")

    # Load tokenizer
    tokenizer = load_tokenizer(args.base_model)

    # Load CIFAR-10 evaluator
    evaluator = CIFARRewardEvaluator(train_epochs=args.nn_train_epochs)

    # Load prompts
    prompts = load_prompts(tokenizer, limit=args.num_prompts)

    # Load model
    print(f"[PPO] Loading model: {args.base_model}...")
    precision = {"torch_dtype": dtype, "bf16": bf16, "fp16": not bf16,
                 "label": "bf16" if bf16 else "fp16"}
    model = load_quantized_causal_lm(
        model_source=args.base_model, precision=precision,
        train_device=device, use_deepspeed=False)
    try:
        from peft.utils.other import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    except: pass

    # LoRA
    peft_cfg = build_lora_config(r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)
    model = attach_or_resume_lora(model, peft_config=peft_cfg,
                                  stage_adapter_dir=None, log_prefix="[PPO]")
    model.print_trainable_parameters()

    # Value head
    value_head = ValueHead(model.config.hidden_size).to(device)
    enable_non_reentrant_gradient_checkpointing(model, log_prefix="[PPO]")

    # Train
    trainer = PPOLLMTrainer(model, tokenizer, value_head, evaluator, device)
    history = trainer.train(prompts, num_epochs=args.num_epochs)

    # Save
    out_dir = "out/nngpt/ppo_outputs"
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(f"{out_dir}/adapter")
    tokenizer.save_pretrained(f"{out_dir}/tokenizer")
    torch.save(value_head.state_dict(), f"{out_dir}/value_head.pt")
    with open(f"{out_dir}/ppo_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Final summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total generations: {trainer.gen_count}")
    if trainer.all_accuracies:
        print(f"Working models:    {len(trainer.all_accuracies)}")
        print(f"Best CIFAR-10 acc: {max(trainer.all_accuracies)*100:.1f}%")
        print(f"Avg CIFAR-10 acc:  {np.mean(trainer.all_accuracies)*100:.1f}%")
    print(f"Model saved to:    {out_dir}")
    print(f"{'='*60}")

    plot_results(history)


if __name__ == "__main__":
    main()
