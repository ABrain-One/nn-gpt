"""
test_ppo_simple.py — PPO Proof-of-Concept for nn-gpt

Tests that PPO can learn binary comparison of CV models
from the LEMUR database.

Run with pytest:   pytest test_ppo_simple.py -v
Run directly:      python test_ppo_simple.py
"""

import random
from collections import deque
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ab.nn.api as api
    HAS_LEMUR = True
except ImportError:
    HAS_LEMUR = False

# ── Config ──────────────────────────────────────────
PPO_CLIP = 0.2
PPO_EPOCHS = 4
PPO_LR = 3e-4
BATCH_SIZE = 32
HIDDEN_DIM = 128


# ── Feature Extraction ──────────────────────────────

def extract_model_features(code: str, accuracy: float, epoch: int,
                           dataset: str = "") -> np.ndarray:
    code_lower = code.lower() if code else ""
    return np.array([
        min(accuracy, 1.0),
        min(epoch / 50.0, 1.0),
        float("residual" in code_lower or "+=" in (code or "")),
        float("batchnorm" in code_lower or "nn.BatchNorm" in (code or "")),
        float("layernorm" in code_lower),
        float("dropout" in code_lower),
        float("attention" in code_lower),
        float("relu" in code_lower),
        float("gelu" in code_lower),
        min(code_lower.count("nn.conv") / 10.0, 1.0),
        min(code_lower.count("nn.linear") / 10.0, 1.0),
        min(len(code or "") / 5000.0, 1.0),
        float("cifar" in dataset.lower()) if dataset else 0.0,
        float("mnist" in dataset.lower()) if dataset else 0.0,
        float("svhn" in dataset.lower()) if dataset else 0.0,
        float("imagenet" in dataset.lower()) if dataset else 0.0,
    ], dtype=np.float32)


# ── Data Loading ────────────────────────────────────

def load_pairs_from_lemur(n_pairs=1000):
    data = api.data(only_best_accuracy=True, task='img-classification')
    pairs = []
    indices = list(range(len(data)))
    for _ in range(n_pairs):
        i, j = random.sample(indices, 2)
        row_a, row_b = data.iloc[i], data.iloc[j]
        acc_a = float(row_a.get('accuracy', 0) or 0)
        acc_b = float(row_b.get('accuracy', 0) or 0)
        feat_a = extract_model_features(
            str(row_a.get('nn', '')), acc_a,
            int(row_a.get('epoch', 1) or 1),
            str(row_a.get('dataset', 'cifar-10')))
        feat_b = extract_model_features(
            str(row_b.get('nn', '')), acc_b,
            int(row_b.get('epoch', 1) or 1),
            str(row_b.get('dataset', 'cifar-10')))
        pairs.append({
            "features_a": feat_a, "features_b": feat_b,
            "label": 1.0 if acc_a > acc_b else 0.0,
            "acc_a": acc_a, "acc_b": acc_b,
        })
    return pairs


def generate_synthetic_pairs(n_pairs=1000):
    archs = [
        {"base_acc": 0.65, "feat": [0, 0, 0, 1, 0, 0]},
        {"base_acc": 0.85, "feat": [1, 1, 0, 0, 0, 1]},
        {"base_acc": 0.80, "feat": [0, 1, 0, 1, 0, 0]},
        {"base_acc": 0.82, "feat": [1, 1, 0, 0, 1, 0]},
        {"base_acc": 0.35, "feat": [0, 0, 0, 0, 0, 0]},
        {"base_acc": 0.78, "feat": [0, 1, 0, 1, 0, 0]},
    ]
    pairs = []
    for _ in range(n_pairs):
        a, b = random.sample(archs, 2)
        acc_a = max(0, min(1, a["base_acc"] + random.gauss(0, 0.05)))
        acc_b = max(0, min(1, b["base_acc"] + random.gauss(0, 0.05)))
        feat_a = np.array([acc_a, 0.5] + a["feat"] + [0.3, 0.2, 0.5, 0, 1, 0, 0, 0], dtype=np.float32)
        feat_b = np.array([acc_b, 0.5] + b["feat"] + [0.3, 0.2, 0.5, 0, 1, 0, 0, 0], dtype=np.float32)
        pairs.append({
            "features_a": feat_a, "features_b": feat_b,
            "label": 1.0 if acc_a > acc_b else 0.0,
            "acc_a": acc_a, "acc_b": acc_b,
        })
    return pairs


# ── PPO Actor-Critic ────────────────────────────────

class PPOActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid())
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1))

    def forward(self, feat_a, feat_b):
        combined = torch.cat([feat_a, feat_b], dim=-1)
        shared = self.shared(combined)
        return self.actor(shared).squeeze(-1), self.critic(shared).squeeze(-1)

    def get_action(self, feat_a, feat_b):
        prob, value = self.forward(feat_a, feat_b)
        dist = torch.distributions.Bernoulli(prob)
        action = dist.sample()
        return action, dist.log_prob(action), value, dist.entropy()


# ── PPO Trainer ─────────────────────────────────────

class PPOTrainer:
    def __init__(self, input_dim, device="cpu"):
        self.device = device
        self.model = PPOActorCritic(input_dim, HIDDEN_DIM).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=PPO_LR)
        self.history = []

    def collect_rollout(self, pairs, batch_size=BATCH_SIZE):
        batch = random.sample(pairs, min(batch_size, len(pairs)))
        states_a, states_b, actions, log_probs, values, rewards = [], [], [], [], [], []
        self.model.eval()
        with torch.no_grad():
            for p in batch:
                fa = torch.tensor(p["features_a"]).to(self.device)
                fb = torch.tensor(p["features_b"]).to(self.device)
                a, lp, v, _ = self.model.get_action(fa.unsqueeze(0), fb.unsqueeze(0))
                r = 1.0 if (a.item() == p["label"]) else -1.0
                states_a.append(fa); states_b.append(fb)
                actions.append(a); log_probs.append(lp)
                values.append(v); rewards.append(r)
        return {
            "states_a": torch.stack(states_a),
            "states_b": torch.stack(states_b),
            "actions": torch.stack(actions).squeeze(),
            "log_probs": torch.stack(log_probs).squeeze(),
            "values": torch.stack(values).squeeze(),
            "rewards": torch.tensor(rewards, device=self.device),
        }

    def ppo_update(self, rollout):
        advantages = rollout["rewards"] - rollout["values"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = rollout["rewards"]
        old_lps = rollout["log_probs"].detach()
        self.model.train()
        total_loss = 0
        for _ in range(PPO_EPOCHS):
            prob, value = self.model(rollout["states_a"], rollout["states_b"])
            dist = torch.distributions.Bernoulli(prob)
            new_lps = dist.log_prob(rollout["actions"])
            ratio = torch.exp(new_lps - old_lps)
            s1 = ratio * advantages
            s2 = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * advantages
            loss = (-torch.min(s1, s2).mean() +
                    0.5 * F.mse_loss(value, returns) -
                    0.01 * dist.entropy().mean())
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / PPO_EPOCHS

    def train_loop(self, pairs, n_episodes=500):
        reward_hist = deque(maxlen=100)
        for ep in range(n_episodes):
            rollout = self.collect_rollout(pairs)
            self.ppo_update(rollout)
            avg_r = rollout["rewards"].mean().item()
            acc = (rollout["rewards"] > 0).float().mean().item()
            reward_hist.append(avg_r)
            self.history.append({
                "episode": ep, "avg_reward": avg_r,
                "accuracy": acc, "rolling_reward": np.mean(list(reward_hist)),
            })
        return self.history

    def evaluate(self, test_pairs):
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for p in test_pairs:
                fa = torch.tensor(p["features_a"]).unsqueeze(0).to(self.device)
                fb = torch.tensor(p["features_b"]).unsqueeze(0).to(self.device)
                prob, _ = self.model(fa, fb)
                pred = 1.0 if prob.item() > 0.5 else 0.0
                if pred == p["label"]:
                    correct += 1
        return correct / len(test_pairs) if test_pairs else 0


# ══════════════════════════════════════════════════════
# Pytest Tests (run with: pytest test_ppo_simple.py -v)
# ══════════════════════════════════════════════════════

def _run_ppo(n_episodes=500, seed=42):
    """Helper: run PPO training and return results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if HAS_LEMUR:
        all_pairs = load_pairs_from_lemur(n_pairs=2000)
    else:
        all_pairs = generate_synthetic_pairs(n_pairs=2000)

    split = int(len(all_pairs) * 0.8)
    train_pairs = all_pairs[:split]
    test_pairs = all_pairs[split:]
    input_dim = len(train_pairs[0]["features_a"])

    trainer = PPOTrainer(input_dim=input_dim, device=device)
    trainer.train_loop(train_pairs, n_episodes=n_episodes)
    test_acc = trainer.evaluate(test_pairs)

    return trainer, test_acc, train_pairs, test_pairs


def test_ppo_learns_above_random():
    """PPO should achieve significantly above random baseline (50%)."""
    _, test_acc, _, _ = _run_ppo(n_episodes=300)
    print(f"\n  PPO test accuracy: {test_acc:.1%}")
    assert test_acc > 0.60, (
        f"PPO accuracy {test_acc:.1%} should be above 60%"
    )


def test_ppo_reward_improves():
    """Average reward should increase from early to late training."""
    trainer, _, _, _ = _run_ppo(n_episodes=300)
    history = trainer.history
    early = np.mean([h["avg_reward"] for h in history[:50]])
    late = np.mean([h["avg_reward"] for h in history[-50:]])
    print(f"\n  Early avg reward: {early:.3f}")
    print(f"  Late avg reward:  {late:.3f}")
    assert late > early, (
        f"Late reward ({late:.3f}) should be higher than early ({early:.3f})"
    )


def test_ppo_model_saves():
    """PPO model should save and load correctly."""
    trainer, _, _, test_pairs = _run_ppo(n_episodes=100)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
        torch.save(trainer.model.state_dict(), path)
        input_dim = len(test_pairs[0]["features_a"])
        loaded = PPOActorCritic(input_dim, HIDDEN_DIM)
        loaded.load_state_dict(torch.load(path, weights_only=True))
        os.unlink(path)
    print("\n  Model save/load: OK")


# ══════════════════════════════════════════════════════
# Direct Execution (run with: python test_ppo_simple.py)
# ══════════════════════════════════════════════════════

def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[PPO] Device: {device}")

    if HAS_LEMUR:
        print("[PPO] Loading from LEMUR...")
        all_pairs = load_pairs_from_lemur(n_pairs=2000)
    else:
        print("[PPO] Using synthetic data")
        all_pairs = generate_synthetic_pairs(n_pairs=2000)

    split = int(len(all_pairs) * 0.8)
    train_pairs, test_pairs = all_pairs[:split], all_pairs[split:]
    input_dim = len(train_pairs[0]["features_a"])
    print(f"[PPO] Train: {len(train_pairs)}, Test: {len(test_pairs)}, Features: {input_dim}")

    trainer = PPOTrainer(input_dim=input_dim, device=device)
    trainer.train_loop(train_pairs, n_episodes=500)
    test_acc = trainer.evaluate(test_pairs)

    early = trainer.history[:50]
    late = trainer.history[-50:]
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"PPO test accuracy: {test_acc:.1%}")
    print(f"Early accuracy:    {np.mean([h['accuracy'] for h in early]):.3f}")
    print(f"Late accuracy:     {np.mean([h['accuracy'] for h in late]):.3f}")
    print(f"Early avg reward:  {np.mean([h['avg_reward'] for h in early]):.3f}")
    print(f"Late avg reward:   {np.mean([h['avg_reward'] for h in late]):.3f}")
    print(f"{'='*60}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        episodes = [h["episode"] for h in trainer.history]
        accs = [h["accuracy"] for h in trainer.history]
        rolling = [h["rolling_reward"] for h in trainer.history]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"PPO Binary Comparison (Test Acc: {test_acc:.1%})",
                     fontsize=14, fontweight="bold")
        w = 20
        ra = np.convolve(accs, np.ones(w)/w, mode="valid")
        ax1.plot(episodes, accs, alpha=0.2, color="blue")
        ax1.plot(range(w-1, len(accs)), ra, color="blue", linewidth=2)
        ax1.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Random")
        ax1.set_title("Comparison Accuracy"); ax1.legend(); ax1.grid(True, alpha=0.3)
        ax2.plot(episodes, rolling, color="green", linewidth=2)
        ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
        ax2.set_title("Rolling Avg Reward"); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("ppo_training_results.png", dpi=150)
        print(f"Plot: ppo_training_results.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
