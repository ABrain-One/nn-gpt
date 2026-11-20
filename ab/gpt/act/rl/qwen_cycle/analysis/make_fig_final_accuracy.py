"""Final method, seed 0: accuracy of the generated networks on the prompts that the base model and all eight phases
solve (paired by prompt). Same layout as fig_qwen_accuracy (development run).
Input: data/frozen_eval_seed0.csv, data/train_steps_seed0.csv. Output: fig_final_accuracy.pdf/.png and printed numbers."""
import collections, csv, math, statistics as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats

BL, OR, GY, GR, PU, INK, MU, GRID = "#2a78d6", "#eb6834", "#8a8984", "#2e9e5b", "#7b4bc4", "#1f1f1e", "#6f6e69", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MU, "xtick.color": MU, "ytick.color": MU,
                     "axes.labelcolor": INK, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})

M = ["base", "c1_ppo", "c1_grpo", "c2_ppo", "c2_grpo", "c3_ppo", "c3_grpo", "c4_ppo", "c4_grpo"]
LAB = ["base", "C1\nP", "C1\nG", "C2\nP", "C2\nG", "C3\nP", "C3\nG", "C4\nP", "C4\nG"]
_h = collections.defaultdict(float)
for r in csv.DictReader(open("data/train_steps_seed0.csv")):
    _h[r["phase"]] += (float(r["t_gen"]) + float(r["t_reward"]) + float(r["t_update"])) / 3600
HOURS = [""] + [f"{_h[m]:.1f} h" for m in M[1:]]                     # RL training time of the phase

acc = collections.defaultdict(lambda: collections.defaultdict(list))
for r in csv.DictReader(open("data/frozen_eval_seed0.csv")):
    if r["model"] in M and r["accuracy"]:
        acc[r["model"]][r["prompt_hash"]].append(100 * float(r["accuracy"]))
common = sorted(set.intersection(*[set(acc[m]) for m in M]))
per = {m: [st.mean(acc[m][p]) for p in common] for m in M}            # per-prompt mean accuracy (over the 3 sampling runs)
best = {m: max(x for q in acc[m] for x in acc[m][q]) for m in M}   # best single network, all prompts


def tci(x):
    m = st.mean(x); h = stats.t.ppf(0.975, len(x) - 1) * st.stdev(x) / math.sqrt(len(x)); return m, m - h, m + h


rows = []
for m in M:
    mm = tci(per[m]); d = [a - b for a, b in zip(per[m], per["base"])]
    dd = tci(d) if m != "base" else (0.0, 0.0, 0.0)
    p = stats.ttest_rel(per[m], per["base"]).pvalue if m != "base" else float("nan")
    allm = st.mean([x for q in acc[m] for x in acc[m][q]])
    rows.append((m, mm, dd, p, best[m], allm, len(acc[m])))
print(f"common prompts (base + 8 phases): {len(common)} of 97 distinct")
for m, mm, dd, p, b, allm, n in rows:
    print(f"{m:7s} solved {n:3d} | common mean {mm[0]:.1f} [{mm[1]:.1f}, {mm[2]:.1f}] | paired vs base {dd[0]:+.1f} "
          f"[{dd[1]:+.1f}, {dd[2]:+.1f}] p={p:.1e} | best {b:.1f} | all-prompt mean {allm:.1f}")

X = list(range(len(M)))
fig, ax = plt.subplots(2, 1, figsize=(3.4, 4.7), sharex=True, gridspec_kw={"height_ratios": [1.25, 1]})
a = ax[0]
b0 = rows[0][1]
a.axhspan(b0[1], b0[2], color=GY, alpha=0.18, lw=0, zorder=1)
a.axhline(b0[0], color=GY, ls="--", lw=1.1, zorder=2, label="base model, same prompts")
mm = [r[1] for r in rows]
a.fill_between(X, [v[1] for v in mm], [v[2] for v in mm], color=GR, alpha=0.2, lw=0, zorder=2)
a.plot(X, [v[0] for v in mm], color=GR, lw=2, marker="o", ms=5, zorder=4, label="mean, 95% CI")
a.plot(X, [r[5] for r in rows], color=GR, lw=1, ls=":", marker="o", ms=3, mfc="white", zorder=3,
       label="mean, all solved prompts")
for i, v in enumerate(mm):
    a.text(i, v[1] - 1.0, f"{v[0]:.1f}", ha="center", va="top", fontsize=6.3, color=INK)
bb = [r[4] for r in rows]; bsf = [max(bb[:i + 1]) for i in range(len(bb))]
a.plot(X, bb, color=OR, lw=1.3, ls="--", marker="s", ms=4, zorder=4, label="best network")
a.plot(X, bsf, color=OR, lw=0.9, ls=":", zorder=3, label="best so far")
for i, v in enumerate(bb):
    a.text(i, v + 0.7, f"{v:.1f}", ha="center", va="bottom", fontsize=6.3, color=INK)
a.set_ylim(66, 100); a.set_ylabel("CIFAR-10 accuracy (%)")
a.set_title(f"(a) accuracy, {len(common)} prompts solved by all", loc="left", fontsize=8, color=INK)
a.legend(frameon=False, fontsize=5.6, loc="upper left", ncol=2, columnspacing=0.8, handlelength=1.8)

a = ax[1]
dd = [r[2] for r in rows]
a.axhline(0, color=MU, lw=0.8)
a.fill_between(X, [v[1] for v in dd], [v[2] for v in dd], color=GR, alpha=0.2, lw=0)
a.plot(X, [v[0] for v in dd], color=GR, lw=2, marker="o", ms=5, label="paired mean Δ, 95% CI")
for i, v in enumerate(dd[1:], 1):
    a.text(i, v[1] - 0.5, f"{v[0]:+.1f}", ha="center", va="top", fontsize=6.3, color=INK)
db = [x - bb[0] for x in bb]
# best-of-3 per prompt (best of the three seeds), paired with the base model's best-of-3 on the same prompt
b3 = {m: [max(acc[m][p]) for p in common] for m in M}
d3 = [(0.0, 0.0, 0.0)] + [tci([x - y for x, y in zip(b3[m], b3["base"])]) for m in M[1:]]
for m, v in zip(M, d3):
    pv = stats.ttest_rel(b3[m], b3["base"]).pvalue if m != "base" else float("nan")
    print(f"best-of-3 {m:7s} mean {st.mean(b3[m]):.1f} | paired vs base {v[0]:+.1f} [{v[1]:+.1f}, {v[2]:+.1f}] p={pv:.1e}")
a.fill_between(X, [v[1] for v in d3], [v[2] for v in d3], color=PU, alpha=0.13, lw=0)
a.plot(X, [v[0] for v in d3], color=PU, lw=1.4, ls="-.", marker="D", ms=3.5, label="best-of-3 Δ per prompt, 95% CI")
a.plot(X, db, color=OR, lw=1.3, ls="--", marker="s", ms=4, label="Δ of the best network")
a.set_ylim(-6, 16); a.set_ylabel("Δ accuracy vs. base (pp)")
a.set_title("(b) difference to base, paired by prompt", loc="left", fontsize=8, color=INK)
a.legend(frameon=False, fontsize=5.8, loc="upper left")
a.set_xticks(X); a.set_xticklabels([f"{l}\n{h}" for l, h in zip(LAB, HOURS)], fontsize=5.6)
for a in ax:
    a.grid(axis="y", color=GRID, zorder=0); a.set_xlim(-0.4, 8.4)
fig.tight_layout()
fig.savefig("fig_final_accuracy.pdf"); fig.savefig("fig_final_accuracy.png", dpi=200)
