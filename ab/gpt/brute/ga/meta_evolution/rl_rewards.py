import math

def calculate_meta_reward(current_score, best_ever_score, baseline_score, top3_mean, archive_novelty, valid_syntax):
    if not valid_syntax:
        print("   [RL] Penalty: Invalid Syntax/Crash.")
        return -1.0

    # [INNOVATION_ONLY] We only reward the LLM for producing novel architectures that map to new MAP-Elites cells.
    if archive_novelty > 0:
        # Exponential reward for high novelty, ensuring it is at least 10x the penalty
        reward = (archive_novelty * 2.0) + 10.0
        print(f"   [RL] INNOVATION (SUCCESS): Found {archive_novelty} novel architectures! Reward: {reward:.4f}")
    else:
        # Penalty for stagnation (no new architectures discovered)
        reward = -1.0
        print(f"   [RL] INNOVATION (STAGNATION): Zero novelty found. Reward: {reward:.4f}")

    # Secondary Reward: Quality Density (Top-3 Mean)
    # User requested this to be kept, while ensuring novelty remains the dominant reward.
    if top3_mean > 0:
        density_reward = (top3_mean / 100.0) * 1.5 
        reward += density_reward
        print(f"   [RL] SECONDARY: Top-3 Quality Density ({top3_mean:.2f}%). Bonus: {density_reward:.4f}")

    reward = min(reward, 25.0) 
    print(f"   [RL] Total Reward: {reward:.4f}")
    return reward