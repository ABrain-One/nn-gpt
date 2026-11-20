"""Entropy as an early warning: per-step token entropy and strict-valid samples of all training phases.
Input: data/train_steps.csv. Output: fig_qwen_stability.pdf/.png and the printed early-warning analysis."""
import collections, csv, statistics as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BL, OR, GY, INK, MU, GRID = "#2a78d6", "#eb6834", "#8a8984", "#1f1f1e", "#6f6e69", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MU, "xtick.color": MU, "ytick.color": MU,
                     "axes.labelcolor": INK, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})

P = collections.OrderedDict()
for r in csv.DictReader(open("data/train_steps.csv")):
    P.setdefault(r["phase"], []).append(r)
ENT = {ph: [float(r["entropy"]) if r["entropy"] else None for r in rows] for ph, rows in P.items()}
VAL = {ph: [int(r["valid"]) for r in rows] for ph, rows in P.items()}
COL = {"c3_B_grpo": "C3 GRPO (bonus)", "c4_A_ppo": "C4 PPO (bonus)"}
STAB = ["c1_A_ppo", "c1_B_grpo", "c2_A_ppo", "c2_B_grpo", "c3_A_ppo", "c3_B_grpo_fix", "c4fix_A_ppo", "c4fix_B_grpo"]


def onset(v):            # first of two consecutive steps with at most half of the phase's initial valid samples (mean of steps 1-3)
    b = st.mean(v[:3])
    for t in range(1, len(v)):
        if v[t - 1] <= b / 2 and v[t] <= b / 2:
            return t
    return None


def alarm(ph, kind, x):
    e = ENT[ph]
    for t, val in enumerate(e, 1):
        if val is not None and ((kind == "abs" and val > x) or (kind == "rel" and val > x * e[0])):
            return t
    return None


print("onset:", {ph: onset(VAL[ph]) for ph in COL}, "| stable phases with onset:", [ph for ph in STAB if onset(VAL[ph])])
print("max entropy of stable phases:", {ph: max(e for e in ENT[ph] if e) for ph in STAB})
for kind, x in [("abs", 0.8), ("abs", 0.9), ("abs", 1.2), ("rel", 2.0), ("rel", 3.0)]:
    out = [f"{ph}: alarm {alarm(ph, kind, x)} lead {onset(VAL[ph]) - alarm(ph, kind, x)}" for ph in COL]
    print(kind, x, " | ".join(out), "| false alarms:", [ph for ph in STAB if alarm(ph, kind, x)])

S = list(range(1, 17))
fig, ax = plt.subplots(2, 1, figsize=(3.4, 4.3), sharex=True)
for i, ph in enumerate(STAB):
    lab = "stable phases (8)" if i == 0 else None
    c = BL if ph in ("c3_B_grpo_fix", "c4fix_A_ppo", "c4fix_B_grpo") else GY
    e = [(t, v) for t, v in zip(S, ENT[ph]) if v is not None]
    ax[0].plot([t for t, _ in e], [v for _, v in e], color=c, lw=0.9, alpha=0.75, zorder=2)
    ax[1].plot(S, [100 * v / 16 for v in VAL[ph]], color=c, lw=0.9, alpha=0.55, zorder=2)
for ph, name in COL.items():
    ls = "-" if ph == "c3_B_grpo" else "--"
    ax[0].plot(S, ENT[ph], color=OR, lw=2, ls=ls, marker="o", ms=2.8, zorder=4, label=name)
    ax[1].plot(S, [100 * v / 16 for v in VAL[ph]], color=OR, lw=2, ls=ls, marker="o", ms=2.8, zorder=4)
    o, a = onset(VAL[ph]), alarm(ph, "abs", 0.8)
    ax[1].plot([o], [100 * VAL[ph][o - 1] / 16], marker="v", ms=7, color=OR, mec=INK, mew=0.6, zorder=5)
    ax[0].plot([a], [ENT[ph][a - 1]], marker="^", ms=7, color=OR, mec=INK, mew=0.6, zorder=5)
ax[0].plot([], [], color=GY, lw=0.9, label="stable, with bonus (5)")
ax[0].plot([], [], color=BL, lw=0.9, label="stable, fixed settings (3)")
ax[0].plot([], [], marker="^", ls="", color=OR, mec=INK, mew=0.6, label="alarm: entropy > 0.8")
ax[1].plot([], [], marker="v", ls="", color=OR, mec=INK, mew=0.6, label="collapse onset")
for y, txt in [(0.8, "0.8"), (1.2, "1.2 (used)")]:
    ax[0].axhline(y, color=MU, lw=0.8, ls=":", zorder=1)
    ax[0].text(16.4, y, txt, ha="right", va="bottom", fontsize=6, color=MU)
ax[0].set_ylim(0, 2.45); ax[0].set_ylabel("token entropy")
ax[0].set_title("(a) token entropy per update step", loc="left", fontsize=8, color=INK)
ax[0].legend(frameon=False, fontsize=5.8, loc="upper left")
ax[1].set_ylim(0, 105); ax[1].set_ylabel("strict-valid samples (%)")
ax[1].set_title("(b) strict-valid samples per step (of 16)", loc="left", fontsize=8, color=INK)
ax[1].legend(frameon=False, fontsize=5.8, loc="lower left")
ax[1].set_xlabel("update step within the phase"); ax[1].set_xticks([1, 4, 8, 12, 16])
for a in ax:
    a.grid(axis="y", color=GRID, zorder=0)
fig.tight_layout()
fig.savefig("fig_qwen_stability.pdf"); fig.savefig("fig_qwen_stability.png", dpi=200)
