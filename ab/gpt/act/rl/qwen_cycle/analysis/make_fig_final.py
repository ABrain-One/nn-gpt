"""Final method (seed 0) vs development run: strict-valid rate, and compute and layouts of the generated networks.
Input: data/frozen_eval_seed0.csv (base + 8 phases of seed 0, with architecture descriptors), data/frozen_eval.csv (first run).
Output: fig_final.pdf/.png and the numbers printed for the paper."""
import collections, csv, math, statistics as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats

BL, GY, PU, INK, MU, GRID = "#2a78d6", "#8a8984", "#7b4bc4", "#1f1f1e", "#6f6e69", "#e4e3df"
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MU, "xtick.color": MU, "ytick.color": MU, "axes.labelcolor": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
S0 = ["base", "c1_ppo", "c1_grpo", "c2_ppo", "c2_grpo", "c3_ppo", "c3_grpo", "c4_ppo", "c4_grpo"]
LAB = ["B", "1P", "1G", "2P", "2G", "3P", "3G", "4P", "4G"]
OLD = {"base": 0, "c1": 2, "c2": 4, "c3A": 5, "c3fix": 6, "c4fixA": 7, "c4fix": 8}   # first run, main path -> x position

def read(path, keep):
    R = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["model"] in keep: R[r["model"]].append(r)
    return R
new, old = read("data/frozen_eval_seed0.csv", S0), read("data/frozen_eval.csv", OLD)

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)
def tci(x):
    m = st.mean(x); h = stats.t.ppf(0.975, len(x) - 1) * st.stdev(x) / math.sqrt(len(x)); return m, m - h, m + h
def per_prompt(rows):
    d = collections.defaultdict(list)
    for r in rows:
        if r["accuracy"]: d[r["prompt_hash"]].append(100 * float(r["accuracy"]))
    return d

P = {("new", m): per_prompt(new[m]) for m in S0}
P.update({("old", m): per_prompt(old[m]) for m in OLD if m != "base"})
common = sorted(set.intersection(*[set(v) for v in P.values()]))
base = P[("new", "base")]
def paired(key, f):
    d = [f(P[key][p]) - f(base[p]) for p in common]
    return tci(d), stats.ttest_rel([f(P[key][p]) for p in common], [f(base[p]) for p in common]).pvalue
print(f"common prompts (base, 8 seed-0 phases, 6 first-run models): {len(common)}")

fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.0), gridspec_kw={"wspace": 0.32})
X = range(len(S0))
# (a) strict-valid
a = ax[0]
vn = [wilson(sum(r["strict"] == "OK" for r in new[m]), len(new[m])) for m in S0]
vo = [(OLD[m], wilson(sum(r["strict"] == "OK" for r in old[m]), len(old[m]))) for m in OLD]
a.plot([x for x, _ in vo], [v[0] for _, v in vo], color=GY, lw=1.2, marker="o", ms=3.5, mfc="white", label="development run", zorder=3)
a.fill_between(X, [v[1] for v in vn], [v[2] for v in vn], color=BL, alpha=0.15, lw=0)
a.plot(X, [v[0] for v in vn], color=BL, lw=2, marker="o", ms=4, label="final method, seed 0", zorder=4)
a.set_ylim(30, 102); a.set_ylabel("% of 300 outputs")
a.set_title("(a) strict-valid outputs", loc="left", fontsize=7.5, color=INK)
a.legend(frameon=False, fontsize=5.8, loc="lower right")
# (c) compute of the generated networks
c = ax[1]
def arch(m, f):
    return [f(r) for r in new[m] if r["strict"] == "OK" and r["macs"]]
mac = [sorted(arch(m, lambda r: float(r["macs"]) / 1e6)) for m in S0]
q = lambda x, f: x[int(f * (len(x) - 1))]
c.fill_between(X, [q(v, .25) for v in mac], [q(v, .75) for v in mac], color=BL, alpha=0.15, lw=0)
c.plot(X, [q(v, .5) for v in mac], color=BL, lw=2, marker="o", ms=4)
c.set_yscale("log"); c.set_ylabel("MMACs per image")
c.set_title("(b) compute (median, IQR)", loc="left", fontsize=7.5, color=INK)
# (d) diversity: share of the most common layer layout
d = ax[2]
share, nlay = [], []
for m in S0:
    lay = collections.Counter(arch(m, lambda r: r["layout"])); share.append(100 * lay.most_common(1)[0][1] / sum(lay.values())); nlay.append(len(lay))
d.bar(X, share, color=BL, width=0.62)
for i, (s, n) in enumerate(zip(share, nlay)):
    d.text(i, s + 2, str(n), ha="center", fontsize=6, color=INK)
d.set_ylim(0, 108); d.set_ylabel("% of valid networks")
d.set_title("(c) most common layout", loc="left", fontsize=7.5, color=INK)
for a_ in ax:
    a_.set_xticks(list(X)); a_.set_xticklabels(LAB, fontsize=6); a_.grid(axis="y", color=GRID, lw=0.6, zorder=0); a_.set_axisbelow(True)
fig.savefig("fig_final.pdf", bbox_inches="tight"); fig.savefig("fig_final.png", dpi=200, bbox_inches="tight")
for m, v in zip(S0, vn):
    print(f"{m:8s} valid {v[0]:.1f}")
for m in OLD:
    if m != "base": print(f"development {m:7s} paired mean vs base on the {len(common)} shared prompts {paired(('old', m), st.mean)[0][0]:+.1f}")
print(f"seed-0 final on the same prompts {paired(('new', 'c4_grpo'), st.mean)[0][0]:+.1f}")
print("median MACs", [round(q(v, .5)) for v in mac], "share", [round(s) for s in share], "layouts", nlay)
