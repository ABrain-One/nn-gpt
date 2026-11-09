"""Vet candidate seed families before they enter the pool: legal, runnable, and NEW.

A seed that fails any of these is worse than no seed. An over-cap seed is rejected silently by
run_seed_ga's _seed_is_edge_legal and the variant quietly runs with fewer parents than
intended; a seed that duplicates an existing architecture spends a whole GA run re-exploring a
basin the corpus already covers.

Three gates, in the order that fails cheapest first:
  1 PREFLIGHT  imports, builds, forwards, shape-preserves, inside the MAC cap. Uses the same
               check_isolated the live pipeline uses, so a seed that passes here cannot fail
               later for a reason preflight would have caught.
  2 DISTINCT   AST exact hash and fx graded cosine against the WHOLE corpus. A candidate whose
               nearest neighbour sits above the gate threshold would be rejected the moment
               the GA produced anything close to it.
  3 TRAINABLE  one optimiser step on random data -- catches a learn() that diverges or whose
               auxiliary losses reference tensors forward() never set.
"""
import argparse
import glob
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, "/a/mm/incoming"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--db", default="/a/mm/nn-dataset/db/ab.nn.db")
    ap.add_argument("--cos-max", type=float, default=0.95)
    ap.add_argument("--mac-cap", type=float, default=9053.0)
    a = ap.parse_args()

    import torch
    import preflight as PF
    from diversity_gate import DiversityGate

    print("=== seeding the reference set from the corpus ===", flush=True)
    gate = DiversityGate(store_path=None, graded=True, cos_max=a.cos_max,
                         ast_exact=True, log=lambda *_: None)
    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    corpus = con.execute("SELECT name, code FROM nn WHERE name IN "
                         "(SELECT DISTINCT nn FROM stat WHERE task='img-denoising')"
                         ).fetchall()
    con.close()
    for nm, code in corpus:
        try:
            _, fp = gate.check(code)
            gate.add(nm, fp)
        except Exception:
            pass
    print(f"  {len(gate.accepted)} architectures, {len(gate.embeds)} vectors\n", flush=True)

    results = []
    for f in a.files:
        name = os.path.basename(f)[:-3]
        src = open(f).read()
        print(f"=== {name} ===", flush=True)

        ok, stage, detail, arch = PF.check_isolated(src, True, 360, a.mac_cap, cpu_limit=120)
        print(f"  [1] preflight   {'PASS' if ok else 'FAIL'}  stage={stage}"
              + (f"  {str(detail)[:90]}" if not ok else ""))
        if not ok:
            results.append((name, False, stage, None, None))
            continue

        accept, fp = gate.check(src)
        m = gate.last_match or {}
        cos = m.get("cos")
        print(f"  [2] distinct    {'PASS' if accept else 'FAIL'}"
              f"  nearest={m.get('key')}  cos={cos if cos is None else round(cos,4)}")

        # 3: one real optimiser step. A model can build and forward and still have a learn()
        # that touches a tensor forward() only sets on some paths.
        trainable, terr = True, ""
        try:
            ns = {}
            exec(compile(src, f"<{name}>", "exec"), ns)
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            net = ns["Net"]((2, 3, 64, 64), (3, 64, 64), {"lr": 1e-4}, dev)
            batch = [(torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64))]
            net.learn(batch)
            for p in net.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    trainable, terr = False, "non-finite gradient"
                    break
        except Exception as e:
            trainable, terr = False, f"{type(e).__name__}: {e}"[:110]
        print(f"  [3] trainable   {'PASS' if trainable else 'FAIL'}  {terr}")

        results.append((name, ok and accept and trainable, stage, cos, terr))

    print(f"\n{'='*64}\n{'seed':24}{'verdict':>10}{'nearest cos':>14}")
    for name, good, stage, cos, terr in results:
        print(f"{name:24}{'USABLE' if good else 'REJECT':>10}"
              f"{('-' if cos is None else f'{cos:.4f}'):>14}")
    n = sum(1 for r in results if r[1])
    print(f"\n  {n} of {len(results)} usable as new GA seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
