"""Numbers of Tables 2, 3 and 4 (final run, seed 0) from the released data.
Input: data/frozen_eval_seed0.csv, data/train_steps_seed0.csv. Usage: python make_tables.py"""
import collections, csv, math, statistics as st
from scipy import stats
from scipy.stats import fisher_exact

M = ["base", "c1_ppo", "c1_grpo", "c2_ppo", "c2_grpo", "c3_ppo", "c3_grpo", "c4_ppo", "c4_grpo"]
R = collections.defaultdict(list)
for r in csv.DictReader(open("data/frozen_eval_seed0.csv")):
    R[r["model"]].append(r)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


def tci(x):
    m = st.mean(x); h = stats.t.ppf(0.975, len(x) - 1) * st.stdev(x) / math.sqrt(len(x)); return m, m - h, m + h


S = {}
for m in M:
    rows = R[m]; ok = sum(r["strict"] == "OK" for r in rows)
    acc = [100 * float(r["accuracy"]) for r in rows if r["accuracy"]]
    per = collections.defaultdict(list)
    for r in rows:
        if r["accuracy"]: per[r["prompt_hash"]].append(100 * float(r["accuracy"]))
    S[m] = dict(ok=ok, n=len(rows), acc=acc, per=per)
common = sorted(set.intersection(*[set(S[m]["per"]) for m in M]))
mean_c = {m: [st.mean(S[m]["per"][p]) for p in common] for m in M}
best_c = {m: [max(S[m]["per"][p]) for p in common] for m in M}

print(f"Table 2 (paired on the {len(common)} prompts solved by the base model and all 8 phases)")
for m in M:
    d = S[m]; v = wilson(d["ok"], d["n"]); a = tci(d["acc"]); exp = len(d["acc"]) / d["n"] * st.mean(d["acc"])
    line = f"  {m:8s} {d['ok']}/{d['n']} {v[0]:.1f} [{v[1]:.1f}, {v[2]:.1f}] | acc {a[0]:.1f} [{a[1]:.1f}, {a[2]:.1f}] | best {max(d['acc']):.1f} | exp {exp:.1f}"
    if m != "base":
        dm = tci([x - y for x, y in zip(mean_c[m], mean_c["base"])]); db = st.mean(best_c[m]) - st.mean(best_c["base"])
        pm = stats.ttest_rel(mean_c[m], mean_c["base"]).pvalue; pb = stats.ttest_rel(best_c[m], best_c["base"]).pvalue
        pf = fisher_exact([[d["ok"], d["n"] - d["ok"]], [S["base"]["ok"], S["base"]["n"] - S["base"]["ok"]]], alternative="greater").pvalue
        better = sum(x > y for x, y in zip(mean_c[m], mean_c["base"]))
        line += f" | delta {dm[0]:+.1f} [{dm[1]:+.1f}, {dm[2]:+.1f}] p={pm:.1g} | best-of-3 {db:+.1f} p={pb:.1g} | better {better}/{len(common)} | Fisher p={pf:.1g}"
    print(line)

print("\nTable 3 (effect of each phase relative to the previous model)")
steps = collections.defaultdict(list)
for r in csv.DictReader(open("data/train_steps_seed0.csv")):
    steps[r["phase"]].append(r)
for prev, m in zip(M, M[1:]):
    a, b = S[prev], S[m]
    pv = fisher_exact([[b["ok"], b["n"] - b["ok"]], [a["ok"], a["n"] - a["ok"]]]).pvalue
    pa = stats.ttest_rel(mean_c[m], mean_c[prev]).pvalue
    exp = lambda d: len(d["acc"]) / d["n"] * st.mean(d["acc"])
    e = [float(s["entropy"]) for s in steps[m] if s["entropy"] not in ("", "None")]
    h = sum(float(s["t_gen"]) + float(s["t_reward"]) + float(s["t_update"]) for s in steps[m]) / 3600
    upd = sum(s["n_used"] not in ("0", "") for s in steps[m])
    print(f"  {m:8s} dvalid {100 * (b['ok'] / b['n'] - a['ok'] / a['n']):+.1f} p={pv:.1g} | dacc {st.mean(mean_c[m]) - st.mean(mean_c[prev]):+.1f} p={pa:.1g}"
          f" | dexp {exp(b) - exp(a):+.1f} | entropy {e[0]:.2f}->{e[-1]:.2f} | {h:.1f} h | steps with update {upd}/{len(steps[m])}")

print("\nTable 4 (architectures of the valid networks)")
q = lambda x, f: x[int(f * (len(x) - 1))]
for m in M:
    V = [r for r in R[m] if r["strict"] == "OK" and r["macs"]]
    lay = collections.Counter(r["layout"] for r in V)
    P = sorted(float(r["params"]) for r in V); C = sorted(float(r["macs"]) for r in V)
    X = [(math.log10(float(r["macs"]) + 1), float(r["accuracy"])) for r in V if r["accuracy"]]
    rho = stats.spearmanr([x for x, _ in X], [y for _, y in X])
    print(f"  {m:8s} layouts {len(lay):3d} | top {100 * lay.most_common(1)[0][1] / len(V):.0f}% | params {q(P, .5) / 1e6:.2f}M | MACs {q(C, .5) / 1e6:.0f}M"
          f" | conv {st.median(int(r['n_conv']) for r in V):.0f} | rho {rho.statistic:+.2f} (p={rho.pvalue:.1g}) | rebuilt {len(V)}/{sum(r['strict'] == 'OK' for r in R[m])}")
