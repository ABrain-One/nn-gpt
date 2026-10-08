"""Offline graph-canonicalization audit for the KTO/DivPO novelty filter.

Read-only post-hoc analysis. Takes a completed run's output_subdir (the folder
holding cycle_*/nneval/<model>/), canonicalizes every generated architecture into
an isomorphism-invariant graph hash (torch.fx make_fx aten trace + a Weisfeiler-
Lehman label-refinement hash), and measures how many architectures the lexical
MinHash-Jaccard filter counted as "novel/unique" are in fact graph-isomorphic
duplicates. It answers one question for the paper: does lexical novelty over-count
structural diversity, and by how much?

Jaccard-novel is read straight from the pipeline's own verdict on disk: a model
kept as new_nn.py was admitted as novel; one moved to new_nn.notnovel.py was
flagged a duplicate. This never writes inside cycle_*/ and never renames anything,
so it is safe to run alongside or after training and cannot affect resume.

  # inside the container (CPU only, no GPU needed):
  python -m ab.gpt.act.kto.graph_novelty_audit \
      --run_dir out_<TAG>/nngpt/divpo_anchors --scope all --plot
  python -m ab.gpt.act.kto.graph_novelty_audit --selftest
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── canonical graph hash ──────────────────────────────────────────────────────

class _DefaultPrm(dict):
    """Permissive hyperparameter dict: any missing key reads back a float."""
    def __missing__(self, key):  # noqa: D401
        return 0.5


def _h(s: str) -> str:
    return hashlib.blake2b(s.encode("utf-8"), digest_size=10).hexdigest()


def _wl_hash(labels: List[str], adj_in: List[List[int]],
             adj_out: List[List[int]], rounds: int) -> str:
    """Directed Weisfeiler-Lehman hash: refine node labels by their in/out
    neighbourhoods, then hash the sorted multiset of final labels."""
    cur = [_h(l) for l in labels]
    for _ in range(rounds):
        nxt = []
        for i in range(len(cur)):
            ins = ",".join(sorted(cur[j] for j in adj_in[i]))
            outs = ",".join(sorted(cur[j] for j in adj_out[i]))
            nxt.append(_h(f"{cur[i]}<{ins}>{outs}"))
        cur = nxt
    n_edges = sum(len(a) for a in adj_out)
    return _h(f"{len(labels)}|{n_edges}|" + ",".join(sorted(cur)))


def _label_aten(node, granularity: str) -> str:
    op = node.op
    if op == "placeholder":
        return "in"
    if op == "output":
        return "out"
    if op == "get_attr":
        return "param"
    if op == "call_function":
        base = str(node.target)
    elif op == "call_method":
        base = "m:" + str(node.target)
    elif op == "call_module":
        base = "mod:" + str(node.target)
    else:
        base = op
    if granularity == "typed":
        consts = []
        for a in node.args:
            if isinstance(a, (bool, int, float)):
                consts.append(repr(a))
            elif isinstance(a, (list, tuple)) and a and all(
                    isinstance(z, (bool, int, float)) for z in a):
                consts.append(repr(tuple(a)))
        if consts:
            base += "|" + ",".join(consts)
    return base


def _graph_from_gm(gm, tracer: str, granularity: str):
    nodes = list(gm.graph.nodes)
    idx = {id(n): i for i, n in enumerate(nodes)}
    adj_in: List[List[int]] = [[] for _ in nodes]
    adj_out: List[List[int]] = [[] for _ in nodes]
    if tracer == "symbolic":
        submods = dict(gm.named_modules())
        labels = []
        for n in nodes:
            if n.op == "call_module":
                labels.append("mod:" + type(submods.get(n.target)).__name__)
            elif n.op == "call_function":
                labels.append(getattr(n.target, "__name__", str(n.target)))
            elif n.op == "call_method":
                labels.append("m:" + str(n.target))
            elif n.op in ("placeholder", "output"):
                labels.append("in" if n.op == "placeholder" else "out")
            else:
                labels.append(n.op)
    else:
        labels = [_label_aten(n, granularity) for n in nodes]
    for i, n in enumerate(nodes):
        for p in n.all_input_nodes:
            j = idx[id(p)]
            adj_out[j].append(i)
            adj_in[i].append(j)
    return labels, adj_in, adj_out


def _load_net(path: Path):
    name = "audit_mod_" + _h(str(path))[:12]
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, "Net", None), name


def graph_hash_for_file(path: Path, sizes: List[int], in_ch: int, n_cls: int,
                        tracer: str, granularity: str, wl_rounds: int
                        ) -> Tuple[Optional[str], str]:
    """Return (hash, status). status is 'ok' or a short error tag."""
    import torch  # local import: torch lives inside the container

    name = None
    try:
        Net, name = _load_net(path)
        if Net is None:
            return None, "no_Net_class"
    except Exception as e:  # noqa: BLE001
        if name:
            sys.modules.pop(name, None)
        return None, f"exec_error:{type(e).__name__}"

    last = "unknown"
    try:
        for s in sizes:
            try:
                model = Net((1, in_ch, s, s), (n_cls,), _DefaultPrm(), torch.device("cpu"))
                model.eval()
                x = torch.randn(1, in_ch, s, s)
                with torch.no_grad():
                    if tracer == "symbolic":
                        gm = torch.fx.symbolic_trace(model)
                    else:
                        from torch.fx.experimental.proxy_tensor import make_fx
                        gm = make_fx(model, tracing_mode="real")(x)
                labels, ai, ao = _graph_from_gm(gm, tracer, granularity)
                return _wl_hash(labels, ai, ao, wl_rounds), "ok"
            except Exception as e:  # noqa: BLE001 — try the next input size
                last = f"{type(e).__name__}"
        return None, f"trace_error:{last}"
    finally:
        sys.modules.pop(name, None)


# ── corpus discovery ──────────────────────────────────────────────────────────

def _cycle_num(p: Path) -> int:
    m = re.search(r"cycle_(\d+)", p.as_posix())
    return int(m.group(1)) if m else 0


def discover_models(run_dir: Path) -> List[Dict]:
    """Every generated architecture, tagged with its Jaccard verdict from disk."""
    out: List[Dict] = []
    for cyc in sorted(run_dir.glob("cycle_*"), key=_cycle_num):
        nneval = cyc / "nneval"
        if not nneval.is_dir():
            continue
        for md in sorted(nneval.iterdir(), key=lambda d: d.name):
            if not md.is_dir():
                continue
            novel_f = md / "new_nn.py"
            dup_f = md / "new_nn.notnovel.py"
            if novel_f.exists():
                out.append({"cycle": _cycle_num(cyc), "model_id": md.name,
                            "path": novel_f, "jaccard_novel": True})
            elif dup_f.exists():
                out.append({"cycle": _cycle_num(cyc), "model_id": md.name,
                            "path": dup_f, "jaccard_novel": False})
    return out


# ── audit ─────────────────────────────────────────────────────────────────────

def run_audit(run_dir: Path, out_dir: Path, scope: str, sizes: List[int],
              in_ch: int, n_cls: int, tracer: str, granularity: str,
              wl_rounds: int, limit: int, make_plot: bool) -> Dict:
    models = discover_models(run_dir)
    if not models:
        raise SystemExit(f"No generated architectures found under {run_dir} "
                         "(expected cycle_*/nneval/<model>/new_nn.py).")

    novel = [m for m in models if m["jaccard_novel"]]
    nonnovel = [m for m in models if not m["jaccard_novel"]]
    if limit > 0:
        novel = novel[:limit]

    errors: Counter = Counter()
    seen: Dict[str, Dict] = {}          # graph hash -> first novel model that had it
    clusters: Dict[str, List[str]] = defaultdict(list)
    per_cycle: Dict[int, Dict[str, int]] = defaultdict(
        lambda: {"novel": 0, "traced": 0, "distinct": 0, "dup": 0})

    traced = 0
    for i, m in enumerate(novel, 1):
        pc = per_cycle[m["cycle"]]
        pc["novel"] += 1
        gh, status = graph_hash_for_file(m["path"], sizes, in_ch, n_cls,
                                         tracer, granularity, wl_rounds)
        if status != "ok":
            errors[status] += 1
        else:
            traced += 1
            pc["traced"] += 1
            tag = f"c{m['cycle']}/{m['model_id']}"
            if gh in seen:
                pc["dup"] += 1
            else:
                seen[gh] = m
                pc["distinct"] += 1
            clusters[gh].append(tag)
        if i % 25 == 0 or i == len(novel):
            print(f"  novel traced {i}/{len(novel)} "
                  f"(ok={traced}, distinct={len(seen)})", flush=True)

    distinct = len(seen)
    dups = traced - distinct
    overcount = (dups / traced) if traced else 0.0

    result: Dict = {
        "run_dir": str(run_dir),
        "tracer": tracer,
        "granularity": granularity,
        "wl_rounds": wl_rounds,
        "sizes_tried": sizes,
        "scope": scope,
        "jaccard_novel_total": len(novel),
        "jaccard_nonnovel_total": len(nonnovel),
        "novel_traced": traced,
        "novel_trace_failures": len(novel) - traced,
        "trace_coverage": round(traced / len(novel), 4) if novel else 0.0,
        "graph_distinct": distinct,
        "graph_duplicates_in_novel": dups,
        "lexical_overcount_rate": round(overcount, 4),
        "trace_error_breakdown": dict(errors.most_common()),
        "top_collision_clusters": [
            {"members": v, "size": len(v)}
            for _, v in sorted(clusters.items(), key=lambda kv: -len(kv[1]))
            if len(v) > 1
        ][:10],
        "per_cycle": {str(c): per_cycle[c] for c in sorted(per_cycle)},
    }

    # Precision / asymmetric-cost check: are Jaccard's *discarded* models really
    # structural duplicates, or did the lexical filter throw away distinct archs?
    if scope == "all" and nonnovel:
        nn_traced = nn_confirmed = nn_false_pos = 0
        nn_errors: Counter = Counter()
        nn_seen = set(seen.keys())
        for j, m in enumerate(nonnovel, 1):
            gh, status = graph_hash_for_file(m["path"], sizes, in_ch, n_cls,
                                             tracer, granularity, wl_rounds)
            if status != "ok":
                nn_errors[status] += 1
                continue
            nn_traced += 1
            if gh in nn_seen:
                nn_confirmed += 1
            else:
                nn_false_pos += 1
                nn_seen.add(gh)
            if j % 25 == 0 or j == len(nonnovel):
                print(f"  nonnovel traced {j}/{len(nonnovel)}", flush=True)
        result["nonnovel_traced"] = nn_traced
        result["nonnovel_confirmed_graph_dup"] = nn_confirmed
        result["nonnovel_graph_distinct_false_positive"] = nn_false_pos
        result["nonnovel_false_positive_rate"] = (
            round(nn_false_pos / nn_traced, 4) if nn_traced else 0.0)
        result["nonnovel_trace_error_breakdown"] = dict(nn_errors.most_common())

    _write_reports(result, out_dir, make_plot)
    return result


def _headline(r: Dict) -> str:
    line = (f"Jaccard-novel: {r['jaccard_novel_total']} · "
            f"traced: {r['novel_traced']} ({r['trace_coverage'] * 100:.1f}%) · "
            f"graph-distinct: {r['graph_distinct']} · "
            f"graph-duplicates: {r['graph_duplicates_in_novel']} · "
            f"lexical over-count: {r['lexical_overcount_rate'] * 100:.1f}%  "
            f"[tracer={r['tracer']} granularity={r['granularity']}]")
    if "nonnovel_false_positive_rate" in r:
        line += (f"\nJaccard-discarded: {r['jaccard_nonnovel_total']} · "
                 f"traced: {r['nonnovel_traced']} · "
                 f"confirmed graph-dup: {r['nonnovel_confirmed_graph_dup']} · "
                 f"graph-distinct (lexical false-positive): "
                 f"{r['nonnovel_graph_distinct_false_positive']} "
                 f"({r['nonnovel_false_positive_rate'] * 100:.1f}%)")
    return line


def _write_reports(r: Dict, out_dir: Path, make_plot: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "graph_audit_summary.json").write_text(
        json.dumps(r, indent=2), encoding="utf-8")
    lines = [_headline(r), "",
             "cycle  jaccard_novel  traced  graph_distinct  graph_dup"]
    for c, pc in r["per_cycle"].items():
        lines.append(f"{c:>5}  {pc['novel']:>13}  {pc['traced']:>6}  "
                     f"{pc['distinct']:>14}  {pc['dup']:>9}")
    (out_dir / "graph_audit_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + _headline(r))
    print(f"\nWrote {out_dir / 'graph_audit_summary.txt'}")
    print(f"Wrote {out_dir / 'graph_audit_summary.json'}")
    if make_plot:
        _plot(r, out_dir)


def _plot(r: Dict, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plot] skipped (matplotlib unavailable: {e})")
        return
    cycles = [int(c) for c in r["per_cycle"]]
    novelv = [r["per_cycle"][str(c)]["novel"] for c in cycles]
    distinctv = [r["per_cycle"][str(c)]["distinct"] for c in cycles]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([c - 0.2 for c in cycles], novelv, width=0.4,
           label="Jaccard-novel", color="#4C72B0")
    ax.bar([c + 0.2 for c in cycles], distinctv, width=0.4,
           label="graph-distinct (this audit)", color="#C44E52")
    ax.set_xlabel("cycle")
    ax.set_ylabel("architectures")
    ax.set_title(f"Lexical vs structural novelty — over-count "
                 f"{r['lexical_overcount_rate'] * 100:.1f}%")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "graph_audit.png", dpi=140)
    print(f"Wrote {out_dir / 'graph_audit.png'}")


# ── self-test (run in-container to validate the machinery) ────────────────────

def _selftest() -> int:
    import tempfile
    import textwrap
    try:
        import torch  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"selftest needs torch (run inside the container): {e}")
        return 2

    same_a = textwrap.dedent('''
        import torch, torch.nn as nn
        def supported_hyperparameters(): return {'lr'}
        class Net(nn.Module):
            def __init__(self, in_shape, out_shape, prm, device):
                super().__init__()
                self.stem = nn.Conv2d(in_shape[1], 8, 3, padding=1)
                self.norm = nn.BatchNorm2d(8)
                self.act = nn.ReLU()
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.head = nn.Linear(8, out_shape[0])
            def forward(self, x):
                x = self.act(self.norm(self.stem(x)))
                x = torch.flatten(self.pool(x), 1)
                return self.head(x)
    ''')
    # Same computation graph, different identifiers/comments/structure order.
    same_b = textwrap.dedent('''
        import torch, torch.nn as nn
        def supported_hyperparameters(): return {'lr'}
        class Net(nn.Module):
            def __init__(self, in_shape, out_shape, prm, device):
                super().__init__()  # renamed everything vs variant A
                self.c = nn.Conv2d(in_shape[1], 8, 3, padding=1)
                self.b = nn.BatchNorm2d(8)
                self.r = nn.ReLU()
                self.gap = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(8, out_shape[0])
            def forward(self, z):
                y = self.r(self.b(self.c(z)))
                y = torch.flatten(self.gap(y), 1)
                return self.fc(y)
    ''')
    # Structurally different: two conv stages, no BN.
    diff_c = textwrap.dedent('''
        import torch, torch.nn as nn
        def supported_hyperparameters(): return {'lr'}
        class Net(nn.Module):
            def __init__(self, in_shape, out_shape, prm, device):
                super().__init__()
                self.c1 = nn.Conv2d(in_shape[1], 8, 3, padding=1)
                self.c2 = nn.Conv2d(8, 16, 3, padding=1)
                self.r = nn.ReLU()
                self.gap = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(16, out_shape[0])
            def forward(self, x):
                x = self.r(self.c1(x))
                x = self.r(self.c2(x))
                x = torch.flatten(self.gap(x), 1)
                return self.fc(x)
    ''')

    tmp = Path(tempfile.mkdtemp(prefix="graph_audit_selftest_"))
    (tmp / "a.py").write_text(same_a, encoding="utf-8")
    (tmp / "b.py").write_text(same_b, encoding="utf-8")
    (tmp / "c.py").write_text(diff_c, encoding="utf-8")
    sizes = [32]
    ha, sa = graph_hash_for_file(tmp / "a.py", sizes, 3, 10, "aten", "topology", 3)
    hb, sb = graph_hash_for_file(tmp / "b.py", sizes, 3, 10, "aten", "topology", 3)
    hc, sc = graph_hash_for_file(tmp / "c.py", sizes, 3, 10, "aten", "topology", 3)
    print(f"A: {sa} {ha}\nB: {sb} {hb}\nC: {sc} {hc}")
    ok = (sa == sb == sc == "ok") and ha == hb and ha != hc
    print("SELFTEST", "PASS" if ok else "FAIL",
          "— A and B (renamed twins) collide, C (extra conv, no BN) is distinct")
    return 0 if ok else 1


# ── cli ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", type=str,
                   help="run output_subdir holding cycle_*/nneval/<model>/")
    p.add_argument("--out", type=str, default=None,
                   help="where to write graph_audit_* (default: --run_dir)")
    p.add_argument("--scope", choices=["novel", "all"], default="novel",
                   help="'novel' audits Jaccard-novel only; 'all' also checks "
                        "whether Jaccard-discarded models were truly duplicates")
    p.add_argument("--tracer", choices=["aten", "symbolic"], default="aten")
    p.add_argument("--granularity", choices=["topology", "typed"], default="topology")
    p.add_argument("--sizes", type=str, default="32,64,96,224,256")
    p.add_argument("--in_channels", type=int, default=3)
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--wl_rounds", type=int, default=3)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        sys.exit(_selftest())
    if not args.run_dir:
        p.error("--run_dir is required (or use --selftest)")

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out) if args.out else run_dir
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    run_audit(run_dir, out_dir, args.scope, sizes, args.in_channels,
              args.num_classes, args.tracer, args.granularity, args.wl_rounds,
              args.limit, args.plot)


if __name__ == "__main__":
    main()
