"""Does the chained-receiver fix change identity, and by how much?

THE MINIMAL PROOF FIRST. Six hand-written pairs that differ ONLY in something the unfixed
graph cannot see: a loss constructed then moved to a device, a chained tensor op, an indexed
receiver. If the fix works, each pair collides before and separates after. If some pair
collides after too, the fix is incomplete and the pair says where.

THEN THE CORPUS, because a fix that separates toy cases and moves nothing real is not worth
shipping. Reported as distinct-count before/after and, more usefully, the number of GROUPS
the unfixed version merged that the fixed one splits.

  python verify_fix.py --db DB --task img-denoising [--limit N]
"""
import argparse
import collections
import os
import sqlite3
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.dirname(HERE), "/a/mm/incoming"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

PAIRS = [
    ("loss_to_device",
     "import torch.nn as nn\nclass Net(nn.Module):\n"
     "    def train_setup(self, d, p):\n        self.crit = nn.MSELoss().to(d)\n",
     "import torch.nn as nn\nclass Net(nn.Module):\n"
     "    def train_setup(self, d, p):\n        self.crit = nn.L1Loss().to(d)\n"),
    ("loss_bare",                       # control: already distinguished, must stay so
     "import torch.nn as nn\nclass Net(nn.Module):\n"
     "    def train_setup(self, d, p):\n        self.crit = nn.MSELoss()\n",
     "import torch.nn as nn\nclass Net(nn.Module):\n"
     "    def train_setup(self, d, p):\n        self.crit = nn.L1Loss()\n"),
    ("chained_forward",
     "import torch.nn as nn\nclass Net(nn.Module):\n"
     "    def forward(self, x):\n        return self.conv(x).relu()\n",
     "import torch.nn as nn\nclass Net(nn.Module):\n"
     "    def forward(self, x):\n        return self.pool(x).relu()\n"),
    ("chained_reduce",
     "import torch\nclass Net(torch.nn.Module):\n"
     "    def forward(self, x):\n        return self.f(x).mean()\n",
     "import torch\nclass Net(torch.nn.Module):\n"
     "    def forward(self, x):\n        return self.f(x).sum()\n"),
    ("optimizer_chained",
     "import torch\nclass Net(torch.nn.Module):\n"
     "    def train_setup(self, d, p):\n"
     "        self.o = torch.optim.Adam(self.parameters()).__class__\n",
     "import torch\nclass Net(torch.nn.Module):\n"
     "    def train_setup(self, d, p):\n"
     "        self.o = torch.optim.SGD(self.parameters()).__class__\n"),
    ("import_alias",                    # control: aliasing must NOT change identity
     "import torch.nn as nn\nclass Net(nn.Module):\n"
     "    def forward(self, x):\n        return nn.functional.relu(x)\n",
     "import torch.nn as N\nclass Net(N.Module):\n"
     "    def forward(self, x):\n        return N.functional.relu(x)\n"),
]


def hasher(module_name):
    """-> f(tmpdir, name, code) giving the exact hash under one implementation."""
    import importlib
    P = importlib.import_module(module_name)
    import uid_full

    def h(tmp, name, code):
        # uid_full binds P at import time, so point its reference at whichever
        # implementation this hasher is testing.
        old, uid_full.P = uid_full.P, P
        try:
            K = uid_full.load_class(tmp, name, code)
            g = uid_full.source_graph_ext(K)
            if not g.keys:
                raise ValueError("empty graph")
            return P.wl_hash(g)
        finally:
            uid_full.P = old
            sys.modules.pop("m_" + "".join(
                c if c.isalnum() else "_" for c in name)[:60], None)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/a/mm/nn-dataset/db/ab.nn.db")
    ap.add_argument("--task", default="img-denoising")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    old_h = hasher("ast_model_graph")
    new_h = hasher("ast_model_graph_fixed")
    tmp = tempfile.mkdtemp(prefix="verify_")

    print("=== MINIMAL PAIRS ===")
    print(f"  {'case':<20}{'unfixed':>12}{'fixed':>12}")
    ok = True
    for i, (nm, A, B) in enumerate(PAIRS):
        row = []
        for h in (old_h, new_h):
            try:
                same = h(tmp, f"{nm}_a{i}", A) == h(tmp, f"{nm}_b{i}", B)
                row.append("COLLIDE" if same else "distinct")
            except Exception as e:
                row.append(f"ERR {type(e).__name__}")
        print(f"  {nm:<20}{row[0]:>12}{row[1]:>12}")
        # import_alias is the one case that MUST collide in both: aliasing is style.
        want = "COLLIDE" if nm == "import_alias" else "distinct"
        if row[1] != want:
            ok = False
            print(f"      ^^ expected {want} after the fix")
    print(f"\n  minimal pairs: {'PASS' if ok else 'FAIL'}")

    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    rows = con.execute("SELECT n.name, n.code FROM nn n WHERE n.name IN "
                       "(SELECT DISTINCT nn FROM stat WHERE task=?) ORDER BY n.name",
                       (a.task,)).fetchall()
    con.close()
    if a.limit:
        rows = rows[:a.limit]
    print(f"\n=== CORPUS: {len(rows)} {a.task} models ===")

    O, N, fail = {}, {}, 0
    t0 = time.time()
    for i, (nm, code) in enumerate(rows, 1):
        for store, h in ((O, old_h), (N, new_h)):
            try:
                store[nm] = h(tmp, nm, code)
            except Exception:
                pass
        if nm not in O or nm not in N:
            fail += 1
        if i % 200 == 0 or i == len(rows):
            print(f"  [{i}/{len(rows)}] {time.time()-t0:.0f}s", flush=True)

    both = [n for n in O if n in N]
    print(f"\n  hashed by both      {len(both)}   (failed {fail})")

    # HARNESS SANITY CHECK, not a result. If the two implementations produce byte-identical
    # hashes for every model, the fixed module was never actually loaded and every number
    # below is vacuous -- a silent no-op is indistinguishable from a fix that does not fire.
    changed = sum(1 for n in both if O[n] != N[n])
    print(f"  hash VALUE changed  {changed}/{len(both)}  ({changed/max(len(both),1):.1%})")
    if changed == 0:
        print("  !! the fixed implementation produced identical hashes for EVERY model.")
        print("  !! that means it was not loaded -- treat the comparison below as invalid.")
    print(f"  DISTINCT unfixed    {len(set(O[n] for n in both))}")
    print(f"  DISTINCT fixed      {len(set(N[n] for n in both))}")

    # The number that matters: groups the unfixed version merged that the fix splits.
    gb = collections.defaultdict(set)
    for n in both:
        gb[O[n]].add(N[n])
    split = {k: v for k, v in gb.items() if len(v) > 1}
    print(f"\n  groups the fix SPLITS   {len(split)} of {len(gb)}"
          f"  ({len(split)/max(len(gb),1):.1%})")
    print(f"  models in those groups  "
          f"{sum(1 for n in both if len(gb[O[n]]) > 1)}")

    # Reverse direction is a CORRECTNESS CHECK, not a statistic. The fix only adds
    # edges, so it can never merge two graphs that were distinct; anything here is a bug.
    rb = collections.defaultdict(set)
    for n in both:
        rb[N[n]].add(O[n])
    merged = sum(1 for v in rb.values() if len(v) > 1)
    print(f"  groups the fix MERGES   {merged}   <- must be 0; the fix only adds edges")

    ex = [n for n in sorted(both) if len(gb[O[n]]) > 1][:8]
    if ex:
        print(f"\n  examples now separated: {', '.join(ex)}")
    return 0 if ok and merged == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
