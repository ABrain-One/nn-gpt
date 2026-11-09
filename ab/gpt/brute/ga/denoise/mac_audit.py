"""Measure MACs for the denoise seed corpus against the 9053 MMac edge budget.

WHY THIS IS STANDALONE. The cap has been asserted in three places and verified in none of
them from this tree:
  * run_seed_ga.py's DEFAULT_SEEDS comment records DenoiseUnet2F/7F/10F at
    14005/30152/29054 MMac -- a measurement made somewhere else, at some earlier time.
  * _seed_is_edge_legal() FAILS OPEN: any exception in measure_cost() returns True. It is
    the right call inside the GA (a thop hiccup must not empty the seed pool) and it means
    a broken backend looks exactly like a clean pass.
  * measure_cost() lazily imports mutator.lemur_io, and the `mutator` package lives in
    ab/gpt/brute/ast -- a DIFFERENT tree from edge_cost.py. Any caller that adds only the
    denoise directory to sys.path gets ModuleNotFoundError for every model, fails open,
    and reports every seed legal. That is what happened on the first two attempts here.

So: import both paths explicitly, and treat an unmeasurable model as UNKNOWN rather than
as a pass. Run in-container -- torch and thop are not on the submit node.

Usage:  python mac_audit.py [--cap 9053] [names...]
"""
import argparse
import os
import sys

CAP_DEFAULT = 9053.0
HERE = os.path.dirname(os.path.abspath(__file__))
AST_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "ast"))

DEFAULT_NAMES = [
    "DenoiseUnet1F", "DenoiseUnet2F", "DenoiseUnet3F", "DenoiseUnet4F", "DenoiseUnet5F",
    "DenoiseUnet6F", "DenoiseUnet7F", "DenoiseUnet8F", "DenoiseUnet9F", "DenoiseUnet10F",
    "LiteDenoiseNet",
]


def nn_dir():
    for d in ("/a/mm/nn-dataset/ab/nn/nn",
              "/shared/ssd/home/b-b-ahmed/thesis/nn-dataset/ab/nn/nn",
              os.path.join(HERE, "..", "..", "..", "..", "..", "..",
                           "nn-dataset", "ab", "nn", "nn")):
        if os.path.isdir(d):
            return d
    raise SystemExit("cannot locate ab/nn/nn")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=CAP_DEFAULT)
    ap.add_argument("names", nargs="*", default=None)
    a = ap.parse_args()

    for p in (HERE, AST_DIR, "/a/mm/nn-gpt/ab/gpt/brute/ast"):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    try:
        import edge_cost as EC
    except Exception as e:
        raise SystemExit(f"edge_cost unimportable: {type(e).__name__}: {e}")

    nd = nn_dir()
    names = a.names or DEFAULT_NAMES
    print(f"cap = {a.cap:.0f} MMac   source = {nd}\n")
    print(f"{'model':<20}{'MMac':>10}  verdict")
    print("-" * 46)
    over, unknown, ok = [], [], []
    for n in names:
        p = os.path.join(nd, f"{n}.py")
        if not os.path.exists(p):
            print(f"{n:<20}{'-':>10}  NO SOURCE"); unknown.append(n); continue
        try:
            c = EC.measure_cost(open(p).read())
        except Exception as e:
            print(f"{n:<20}{'ERR':>10}  {type(e).__name__}: {e}")
            unknown.append(n); continue
        if not c.get("ok"):
            print(f"{n:<20}{'?':>10}  UNMEASURABLE: {c.get('err')}")
            unknown.append(n); continue
        m = c["macs"] / 1e6
        verdict = "OK" if m <= a.cap else f"OVER CAP ({m / a.cap:.1f}x)"
        print(f"{n:<20}{m:>10.0f}  {verdict}")
        (ok if m <= a.cap else over).append((n, m))

    print(f"\n{len(ok)} within cap, {len(over)} over, {len(unknown)} unknown")
    if over:
        print("  OVER: " + ", ".join(f"{n} ({m:.0f})" for n, m in over))
    if unknown:
        # An unknown is NOT a pass. Saying so is the whole point of this script.
        print("  UNKNOWN (must not be treated as legal): " + ", ".join(unknown))
    return 1 if unknown else 0


if __name__ == "__main__":
    raise SystemExit(main())
