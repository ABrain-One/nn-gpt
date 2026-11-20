#!/usr/bin/env python3
"""Step 1 - Architecture collapse analysis (no GPU needed)."""
import argparse, ast, csv, glob, hashlib, json, os, re, sys

CODE_KEYS = ("code", "nn_code", "generated_code", "nn", "model_code", "text", "completion")
ACC_KEYS = ("accuracy", "acc", "reward", "test_accuracy")

def looks_like_code(s):
    return isinstance(s, str) and "class" in s and ("nn.Module" in s or "def forward" in s)

def extract_code(s):
    m = re.search(r"<nn>(.*?)</nn>", s, re.S)
    if m: return m.group(1)
    m = re.search(r"```(?:python)?\n(.*?)```", s, re.S)
    if m: return m.group(1)
    return s

def walk(obj, out):
    if isinstance(obj, dict):
        code = next((obj[k] for k in CODE_KEYS if k in obj and looks_like_code(obj[k])), None)
        if code is None:
            code = next((v for v in obj.values() if looks_like_code(v)), None)
        if code is not None:
            acc = next((obj[k] for k in ACC_KEYS if isinstance(obj.get(k), (int, float))), None)
            out.append((extract_code(code), acc)); return
        for v in obj.values(): walk(v, out)
    elif isinstance(obj, list):
        for v in obj: walk(v, out)

def load(path):
    recs = []
    if os.path.isdir(path):
        for f in sorted(glob.glob(os.path.join(path, "**", "*.py"), recursive=True)):
            recs.append((open(f, encoding="utf-8", errors="ignore").read(), None))
    elif path.endswith(".jsonl"):
        for line in open(path, encoding="utf-8"):
            if line.strip(): walk(json.loads(line), recs)
    else:
        walk(json.load(open(path, encoding="utf-8")), recs)
    return recs

class Normalize(ast.NodeTransformer):
    KEEP = {"nn","torch","F","self","super","range","len","int","float","max","min",
            "list","tuple","dict","getattr","setattr","isinstance"}
    def __init__(self): self.map = {}
    def _name(self, n):
        if n in self.KEEP: return n
        return self.map.setdefault(n, f"v{len(self.map)}")
    def visit_Name(self, node): node.id = self._name(node.id); return node
    def visit_arg(self, node): node.arg = self._name(node.arg); node.annotation = None; return node
    def visit_FunctionDef(self, node):
        if node.name not in ("__init__", "forward"): node.name = self._name(node.name)
        node.returns = None; self._strip_doc(node); self.generic_visit(node); return node
    def visit_ClassDef(self, node):
        node.name = self._name(node.name); self._strip_doc(node); self.generic_visit(node); return node
    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id in ("nn","torch","F"): return node
        node.attr = self._name(node.attr); return node
    def visit_Import(self, node): return None
    def visit_ImportFrom(self, node): return None
    @staticmethod
    def _strip_doc(node):
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:] or [ast.Pass()]

def exact_signature(code):
    try: tree = ast.parse(code)
    except SyntaxError: return None
    tree = Normalize().visit(tree)
    return hashlib.md5(ast.dump(tree, annotate_fields=False).encode()).hexdigest()

def structural_signature(code):
    try: tree = ast.parse(code)
    except SyntaxError: return None
    layers = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) \
           and node.func.value.id == "nn" and node.func.attr not in ("Module","ModuleList","Sequential","ModuleDict"):
            nums = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, (int, float))]
            nums += [k.value.value for k in node.keywords if isinstance(k.value, ast.Constant) and isinstance(k.value.value, (int, float))]
            layers.append((node.lineno, node.col_offset, f"{node.func.attr}{tuple(nums)}"))
    layers.sort()
    sig = "|".join(l[2] for l in layers)
    return hashlib.md5(sig.encode()).hexdigest() if sig else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path"); ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--label", default="PPO"); ap.add_argument("--out", default="collapse")
    a = ap.parse_args()
    recs = load(a.path)
    if not recs: sys.exit("No generated code found. Send the ls -la output and the first lines of the file.")
    print(f"Loaded {len(recs)} generations from {a.path}")
    rows, seen = [], set()
    for start in range(0, len(recs), a.window):
        win = recs[start:start + a.window]
        ex = [s for s in (exact_signature(c) for c, _ in win) if s]
        st = [s for s in (structural_signature(c) for c, _ in win) if s]
        accs = [x for _, x in win if isinstance(x, (int, float)) and x > 0]
        new = len(set(ex) - seen); seen |= set(ex)
        rows.append({"window": f"{start+1}-{start+len(win)}", "end_iter": start+len(win), "n": len(win),
                     "parsable": len(ex), "unique_exact": len(set(ex)), "unique_structural": len(set(st)),
                     "new_vs_all_previous": new, "cumulative_unique_exact": len(seen),
                     "mean_acc": round(sum(accs)/len(accs), 4) if accs else "",
                     "best_acc": round(max(accs), 4) if accs else ""})
    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\n{'window':>11} {'parse':>5} {'uniq':>5} {'struct':>6} {'new':>4} {'cum':>5} {'mean':>6} {'best':>6}")
    for r in rows:
        print(f"{r['window']:>11} {r['parsable']:>5} {r['unique_exact']:>5} {r['unique_structural']:>6} "
              f"{r['new_vs_all_previous']:>4} {r['cumulative_unique_exact']:>5} {str(r['mean_acc']):>6} {str(r['best_acc']):>6}")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        x = [r["end_iter"] for r in rows]
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(x, [r["unique_exact"] for r in rows], "o-", label="unique (normalized AST)")
        ax.plot(x, [r["unique_structural"] for r in rows], "s--", label="unique (layer structure)")
        ax.plot(x, [r["new_vs_all_previous"] for r in rows], "^:", label="never seen before")
        ax.set_xlabel("iteration"); ax.set_ylabel(f"architectures per {a.window}-iteration window")
        ax.set_title(f"{a.label}: architectural diversity over training"); ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(a.out + ".png", dpi=200)
        print(f"\nSaved {a.out}.csv and {a.out}.png")
    except ImportError:
        print(f"\nSaved {a.out}.csv (install matplotlib for the plot)")
    first, last = rows[0]["unique_exact"], rows[-1]["unique_exact"]
    print(f"Unique per window: first={first}, last={last} ({'DROP - collapse visible' if last < 0.7*first else 'no clear drop'})")

if __name__ == "__main__":
    main()
