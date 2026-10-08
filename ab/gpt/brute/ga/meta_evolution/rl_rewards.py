import math

def calculate_meta_reward(current_score, best_ever_score, baseline_score, top3_mean, archive_novelty, valid_syntax):
    """Calculate meta-evolution reward.

    Primary signal: SOTA-relative fitness improvement (current_score - best_ever_score).
    Secondary signal: archive novelty (bonus, not dominant).
    Tertiary signal: top-3 quality density (small bonus).

    Dead parameters retained for backward compatibility:
    - baseline_score: available but not used in calculation (reserved for future use).
    """
    if not valid_syntax:
        print("   [RL] Penalty: Invalid Syntax/Crash.")
        return -1.0

    # PRIMARY: Fitness improvement over best-ever score.
    # This is the dominant reward signal — the GA should optimize for
    # improvement, not just novelty.
    fitness_improvement = max(0.0, current_score - best_ever_score)
    reward = fitness_improvement * 2.0  # Scale factor for meaningful reward values

    if fitness_improvement > 0:
        print(f"   [RL] PRIMARY: Fitness improvement +{fitness_improvement:.2f}% over best-ever ({best_ever_score:.2f}%). Reward: {reward:.4f}")

    # SECONDARY: Archive novelty (smaller weight — now a bonus, not dominant).
    if archive_novelty > 0:
        novelty_bonus = archive_novelty * 0.5
        reward += novelty_bonus
        print(f"   [RL] SECONDARY: Archive novelty ({archive_novelty} new cells). Bonus: {novelty_bonus:.4f}")

    # TERTIARY: Quality density (small bonus).
    if top3_mean > 0:
        density_reward = (top3_mean / 100.0) * 0.5
        reward += density_reward
        print(f"   [RL] TERTIARY: Top-3 Quality Density ({top3_mean:.2f}%). Bonus: {density_reward:.4f}")

    # Minimum penalty for stagnation (no improvement AND no novelty)
    if fitness_improvement == 0 and archive_novelty == 0:
        reward = max(reward, -2.0)
        print(f"   [RL] STAGNATION: No improvement, no novelty. Reward: {reward:.4f}")

    reward = min(reward, 25.0)
    print(f"   [RL] Total Reward: {reward:.4f}")
    return reward
