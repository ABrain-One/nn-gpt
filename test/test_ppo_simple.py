#!/usr/bin/env python3
"""
test_ppo_simple.py — PPO Proof-of-Concept for nn-gpt

Demonstrates PPO (Proximal Policy Optimization) learning to generate
better neural network architectures by comparing pairs of generated
models and using binary preference as the reward signal.

This is a simplified version that:
1. Uses the LEMUR database to sample real CV model data
2. Implements PPO from scratch (actor-critic with clipping)
3. Trains a small policy network to predict which model is better
4. Shows learning curves proving PPO works

The full TunePPO.py will use this same PPO logic to fine-tune the LLM.

Usage:
    cd nn-gpt
    python test_ppo_simple.py

Requirements: torch, numpy, matplotlib, ab.nn (nn-dataset)
"""

import json
import math
import os
import random
import sys
from pathlib import Path
from collections import deque
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Try to import LEMUR dataset
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ab.nn.api as api
    HAS_LEMUR = True
    print("[PPO] LEMUR dataset available")
except ImportError:
    HAS_LEMUR = False
    print("[PPO] LEMUR not available, using synthetic data")


# ══════════════════════════════════════════════════════
# PPO Hyperparameters
# ══════════════════════════════════════════════════════
PPO_CLIP_EPSILON = 0.2       # Clipping parameter for PPO
PPO_EPOCHS = 4               # PPO update epochs per batch
PPO_LR = 3e-4                # Learning rate
PPO_GAMMA = 0.99             # Discount factor
PPO_GAE_LAMBDA = 0.95        # GAE lambda
PPO_VALUE_COEF = 0.5         # Value loss coefficient
PPO_ENTROPY_COEF = 0.01      # Entropy bonus coefficient
BATCH_SIZE = 32              # Samples per PPO update
TOTAL_EPISODES = 500         # Total training episodes
HIDDEN_DIM = 128             # Hidden layer size


# ══════════════════════════════════════════════════════
# Feature Extraction from CV Models
# ══════════════════════════════════════════════════════

def extract_model_features(code: str, accuracy: float, epoch: int,
                           dataset: str = "", task: str = "") -> np.ndarray:
    """
    Extract a fixed-size feature vector from a CV model's code and metrics.
    These features represent what the PPO agent observes about each model.
    """
    code_lower = code.lower() if code else ""

    features = [
        # Training metrics (normalized)
        min(accuracy, 1.0),
        min(epoch / 50.0, 1.0),

        # Architecture features
        float("residual" in code_lower or "+=" in code or "skip" in code_lower),
        float("batchnorm" in code_lower or "nn.BatchNorm" in code),
        float("layernorm" in code_lower or "nn.LayerNorm" in code),
        float("dropout" in code_lower),
        float("attention" in code_lower or "multihead" in code_lower),
        float("relu" in code_lower),
        float("gelu" in code_lower),

        # Layer counts (normalized)
        min(code_lower.count("nn.conv") / 10.0, 1.0),
        min(code_lower.count("nn.linear") / 10.0, 1.0),
        min(len(code) / 5000.0, 1.0),  # Code length

        # Dataset encoding
        float("cifar" in dataset.lower()) if dataset else 0.0,
        float("mnist" in dataset.lower()) if dataset else 0.0,
        float("svhn" in dataset.lower()) if dataset else 0.0,
        float("imagenet" in dataset.lower()) if dataset else 0.0,
    ]
    return np.array(features, dtype=np.float32)


# ══════════════════════════════════════════════════════
# Data Loading: LEMUR Database or Synthetic
# ══════════════════════════════════════════════════════

def load_model_pairs_from_lemur(n_pairs: int = 1000) -> List[Dict]:
    """Load real CV model data from LEMUR and create comparison pairs."""
    data = api.data(only_best_accuracy=True, task='img-classification')
    print(f"[PPO] Loaded {len(data)} models from LEMUR")

    pairs = []
    indices = list(range(len(data)))

    for _ in range(n_pairs):
        i, j = random.sample(indices, 2)
        row_a = data.iloc[i]
        row_b = data.iloc[j]

        acc_a = float(row_a.get('accuracy', 0) or 0)
        acc_b = float(row_b.get('accuracy', 0) or 0)

        code_a = str(row_a.get('nn', ''))
        code_b = str(row_b.get('nn', ''))
        dataset = str(row_a.get('dataset', 'cifar-10'))
        epoch_a = int(row_a.get('epoch', 1) or 1)
        epoch_b = int(row_b.get('epoch', 1) or 1)

        feat_a = extract_model_features(code_a, acc_a, epoch_a, dataset)
        feat_b = extract_model_features(code_b, acc_b, epoch_b, dataset)

        # Binary label: 1 if model A is better, 0 if model B is better
        label = 1.0 if acc_a > acc_b else 0.0

        pairs.append({
            "features_a": feat_a,
            "features_b": feat_b,
            "label": label,
            "acc_a": acc_a,
            "acc_b": acc_b,
        })

    return pairs


def generate_synthetic_pairs(n_pairs: int = 1000) -> List[Dict]:
    """Generate synthetic CV model comparison pairs for testing."""
    architectures = [
        {"name": "simple_conv", "base_acc": 0.65, "features": [0, 0, 0, 1, 0, 0]},
        {"name": "resnet_like", "base_acc": 0.85, "features": [1, 1, 0, 0, 0, 1]},
        {"name": "deep_bn", "base_acc": 0.80, "features": [0, 1, 0, 1, 0, 0]},
        {"name": "attention", "base_acc": 0.82, "features": [1, 1, 0, 0, 1, 0]},
        {"name": "tiny_fc", "base_acc": 0.35, "features": [0, 0, 0, 0, 0, 0]},
        {"name": "vgg_style", "base_acc": 0.78, "features": [0, 1, 0, 1, 0, 0]},
    ]

    pairs = []
    for _ in range(n_pairs):
        a, b = random.sample(architectures, 2)
        acc_a = a["base_acc"] + random.gauss(0, 0.05)
        acc_b = b["base_acc"] + random.gauss(0, 0.05)
        acc_a = max(0.0, min(1.0, acc_a))
        acc_b = max(0.0, min(1.0, acc_b))

        feat_a = np.array([acc_a, 0.5] + a["features"] +
                          [0.3, 0.2, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0],
                          dtype=np.float32)
        feat_b = np.array([acc_b, 0.5] + b["features"] +
                          [0.3, 0.2, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0],
                          dtype=np.float32)

        label = 1.0 if acc_a > acc_b else 0.0

        pairs.append({
            "features_a": feat_a,
            "features_b": feat_b,
            "label": label,
            "acc_a": acc_a,
            "acc_b": acc_b,
        })

    return pairs


# ══════════════════════════════════════════════════════
# PPO Actor-Critic Network
# ══════════════════════════════════════════════════════

class PPOActorCritic(nn.Module):
    """
    Actor-Critic network for binary comparison.

    Actor: Given features of two CV models, outputs probability
           that model A is better than model B.
    Critic: Estimates the expected reward (value function).

    This is the core of PPO — same architecture would be used
    inside TunePPO.py but applied to LLM-generated code.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Shared feature encoder
        self.shared = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # Concatenate both models
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Actor head: P(model A is better)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        # Critic head: V(state)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features_a: torch.Tensor, features_b: torch.Tensor):
        combined = torch.cat([features_a, features_b], dim=-1)
        shared_out = self.shared(combined)
        action_prob = self.actor(shared_out).squeeze(-1)
        value = self.critic(shared_out).squeeze(-1)
        return action_prob, value

    def get_action(self, features_a: torch.Tensor, features_b: torch.Tensor):
        """Sample an action (binary: A is better or not) and return log prob."""
        prob, value = self.forward(features_a, features_b)
        # Bernoulli distribution for binary action
        dist = torch.distributions.Bernoulli(prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, value, entropy


# ══════════════════════════════════════════════════════
# PPO Training Loop
# ══════════════════════════════════════════════════════

class PPOTrainer:
    """
    Proximal Policy Optimization trainer for binary model comparison.

    Key differences from GRPO (used in TuneRL.py):
    - PPO uses a learned value function (critic) for baseline
    - GRPO uses group-relative comparisons within a batch
    - PPO clips the policy ratio to prevent large updates
    - GRPO uses KL divergence penalty
    - PPO has explicit actor-critic architecture
    - GRPO operates on the language model directly

    Both use the same reward signal (accuracy of generated CV models).
    """

    def __init__(self, input_dim: int, device: str = "cpu"):
        self.device = device
        self.model = PPOActorCritic(input_dim, HIDDEN_DIM).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=PPO_LR)
        self.training_history = []

    def collect_rollout(self, pairs: List[Dict], batch_size: int = BATCH_SIZE):
        """
        Collect a batch of experiences (rollout).
        In the full pipeline, this would involve:
        1. LLM generates two CV models
        2. Both are trained for 3 epochs
        3. Accuracies compared → binary reward
        """
        batch = random.sample(pairs, min(batch_size, len(pairs)))

        states_a, states_b = [], []
        actions, log_probs, values, rewards, entropies = [], [], [], [], []

        self.model.eval()
        with torch.no_grad():
            for pair in batch:
                feat_a = torch.tensor(pair["features_a"]).to(self.device)
                feat_b = torch.tensor(pair["features_b"]).to(self.device)

                action, log_prob, value, entropy = self.model.get_action(
                    feat_a.unsqueeze(0), feat_b.unsqueeze(0)
                )

                # Reward: +1 if agent correctly identifies better model, -1 otherwise
                correct = (action.item() == pair["label"])
                reward = 1.0 if correct else -1.0

                states_a.append(feat_a)
                states_b.append(feat_b)
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                entropies.append(entropy)

        return {
            "states_a": torch.stack(states_a),
            "states_b": torch.stack(states_b),
            "actions": torch.stack(actions).squeeze(),
            "log_probs": torch.stack(log_probs).squeeze(),
            "values": torch.stack(values).squeeze(),
            "rewards": torch.tensor(rewards, device=self.device),
            "entropies": torch.stack(entropies).squeeze(),
        }

    def compute_gae(self, rewards, values):
        """
        Generalized Advantage Estimation (GAE).
        Computes advantage = how much better the action was vs expected.
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t + 1 < len(values) else 0
            delta = rewards[t] + PPO_GAMMA * next_value - values[t]
            gae = delta + PPO_GAMMA * PPO_GAE_LAMBDA * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def ppo_update(self, rollout):
        """
        PPO clipped surrogate update.

        This is the core PPO algorithm:
        1. Compute advantages using GAE
        2. For each PPO epoch:
           a. Compute new action probabilities
           b. Compute probability ratio (new/old)
           c. Clip the ratio to [1-ε, 1+ε]
           d. Take the minimum of clipped and unclipped objectives
           e. Update with combined actor + critic + entropy loss
        """
        advantages, returns = self.compute_gae(rollout["rewards"], rollout["values"])
        old_log_probs = rollout["log_probs"].detach()

        total_loss_sum = 0
        policy_loss_sum = 0
        value_loss_sum = 0

        self.model.train()

        for ppo_epoch in range(PPO_EPOCHS):
            # Forward pass
            prob, value = self.model(rollout["states_a"], rollout["states_b"])
            dist = torch.distributions.Bernoulli(prob)
            new_log_probs = dist.log_prob(rollout["actions"])
            entropy = dist.entropy().mean()

            # PPO clipped surrogate objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - PPO_CLIP_EPSILON, 1 + PPO_CLIP_EPSILON) * advantages

            # Losses
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value, returns)
            total_loss = (
                policy_loss
                + PPO_VALUE_COEF * value_loss
                - PPO_ENTROPY_COEF * entropy
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            total_loss_sum += total_loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()

        return {
            "total_loss": total_loss_sum / PPO_EPOCHS,
            "policy_loss": policy_loss_sum / PPO_EPOCHS,
            "value_loss": value_loss_sum / PPO_EPOCHS,
        }

    def train(self, pairs: List[Dict], n_episodes: int = TOTAL_EPISODES):
        """Run the full PPO training loop."""
        print(f"\n{'='*60}")
        print("PPO TRAINING — Binary Model Comparison")
        print(f"{'='*60}")
        print(f"Episodes: {n_episodes}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"PPO epochs: {PPO_EPOCHS}")
        print(f"Clip epsilon: {PPO_CLIP_EPSILON}")
        print(f"Learning rate: {PPO_LR}")
        print(f"Device: {self.device}")
        print(f"Training pairs: {len(pairs)}")
        print(f"{'='*60}\n")

        reward_history = deque(maxlen=100)

        for episode in range(n_episodes):
            # 1. Collect rollout
            rollout = self.collect_rollout(pairs)

            # 2. PPO update
            losses = self.ppo_update(rollout)

            # 3. Track metrics
            avg_reward = rollout["rewards"].mean().item()
            accuracy = (rollout["rewards"] > 0).float().mean().item()
            reward_history.append(avg_reward)

            self.training_history.append({
                "episode": episode,
                "avg_reward": avg_reward,
                "accuracy": accuracy,
                "policy_loss": losses["policy_loss"],
                "value_loss": losses["value_loss"],
                "total_loss": losses["total_loss"],
                "rolling_reward": np.mean(list(reward_history)),
            })

            if (episode + 1) % 50 == 0:
                rolling = np.mean(list(reward_history))
                print(
                    f"  Episode {episode+1:4d}/{n_episodes}: "
                    f"accuracy={accuracy:.3f}  "
                    f"avg_reward={avg_reward:.3f}  "
                    f"rolling_reward={rolling:.3f}  "
                    f"policy_loss={losses['policy_loss']:.4f}  "
                    f"value_loss={losses['value_loss']:.4f}"
                )

        return self.training_history


# ══════════════════════════════════════════════════════
# Evaluation & Visualization
# ══════════════════════════════════════════════════════

def evaluate(trainer: PPOTrainer, test_pairs: List[Dict]) -> float:
    """Evaluate PPO agent on held-out test pairs."""
    trainer.model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for pair in test_pairs:
            feat_a = torch.tensor(pair["features_a"]).unsqueeze(0).to(trainer.device)
            feat_b = torch.tensor(pair["features_b"]).unsqueeze(0).to(trainer.device)

            prob, _ = trainer.model(feat_a, feat_b)
            prediction = 1.0 if prob.item() > 0.5 else 0.0

            if prediction == pair["label"]:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy


def plot_results(history: List[Dict], test_accuracy: float):
    """Generate learning curve plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes = [h["episode"] for h in history]
    rewards = [h["avg_reward"] for h in history]
    accuracies = [h["accuracy"] for h in history]
    policy_losses = [h["policy_loss"] for h in history]
    value_losses = [h["value_loss"] for h in history]
    rolling = [h["rolling_reward"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"PPO Training — Binary Model Comparison (Test Accuracy: {test_accuracy:.1%})",
        fontsize=16, fontweight="bold",
    )

    # 1. Reward over time
    ax = axes[0, 0]
    ax.plot(episodes, rewards, alpha=0.2, color="green", label="Per-episode")
    ax.plot(episodes, rolling, color="green", linewidth=2, label="Rolling avg")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.set_title("Average Reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Classification accuracy
    ax = axes[0, 1]
    window = 20
    rolling_acc = np.convolve(accuracies, np.ones(window)/window, mode="valid")
    ax.plot(episodes, accuracies, alpha=0.2, color="blue", label="Per-episode")
    ax.plot(range(window-1, len(accuracies)), rolling_acc,
            color="blue", linewidth=2, label="Rolling avg")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Random baseline")
    ax.set_title("Comparison Accuracy (higher = better)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Policy loss
    ax = axes[1, 0]
    ax.plot(episodes, policy_losses, color="red", alpha=0.6, linewidth=1)
    ax.set_title("Policy Loss (PPO clipped)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # 4. Value loss
    ax = axes[1, 1]
    ax.plot(episodes, value_losses, color="orange", alpha=0.6, linewidth=1)
    ax.set_title("Value Loss (critic)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = "ppo_training_results.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {outpath}")


# ══════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════

def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[PPO] Using device: {device}")

    # Load data
    print("[PPO] Loading model comparison pairs...")
    if HAS_LEMUR:
        all_pairs = load_model_pairs_from_lemur(n_pairs=2000)
    else:
        all_pairs = generate_synthetic_pairs(n_pairs=2000)

    # Split: 80% train, 20% test
    split = int(len(all_pairs) * 0.8)
    train_pairs = all_pairs[:split]
    test_pairs = all_pairs[split:]
    print(f"[PPO] Train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")

    # Feature dimension
    input_dim = len(train_pairs[0]["features_a"])
    print(f"[PPO] Feature dimension: {input_dim}")

    # Random baseline
    random_acc = sum(1 for p in test_pairs
                     if (random.random() > 0.5) == (p["label"] == 1.0)) / len(test_pairs)
    print(f"[PPO] Random baseline accuracy: {random_acc:.3f}")

    # Train PPO
    trainer = PPOTrainer(input_dim=input_dim, device=device)
    history = trainer.train(train_pairs, n_episodes=TOTAL_EPISODES)

    # Evaluate
    test_accuracy = evaluate(trainer, test_pairs)
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Random baseline:  {random_acc:.1%}")
    print(f"PPO test accuracy: {test_accuracy:.1%}")
    print(f"Improvement:       {test_accuracy - random_acc:+.1%}")

    # Early vs late comparison
    early = history[:50]
    late = history[-50:]
    print(f"\nEarly avg reward:  {np.mean([h['avg_reward'] for h in early]):.3f}")
    print(f"Late avg reward:   {np.mean([h['avg_reward'] for h in late]):.3f}")
    print(f"Early accuracy:    {np.mean([h['accuracy'] for h in early]):.3f}")
    print(f"Late accuracy:     {np.mean([h['accuracy'] for h in late]):.3f}")
    print(f"{'='*60}")

    # Save model
    model_path = "out/nngpt/ppo_model.pt"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({
        "model_state_dict": trainer.model.state_dict(),
        "input_dim": input_dim,
        "history": history,
    }, model_path)
    print(f"\nModel saved to: {model_path}")

    # Plot
    plot_results(history, test_accuracy)


if __name__ == "__main__":
    main()
