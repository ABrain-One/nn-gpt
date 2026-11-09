"""Is the graded half of DiversityGate actually functional, or a silent no-op?

arch_embed.both_source() wraps its whole body in `except Exception: return None, None`,
and diversity_gate.py only logs "untraceable" when an exception ESCAPES both_source -- which
it never can. So a total failure of the graded check is indistinguishable from success in
every log we have. This probe calls it directly under the two PYTHONPATHs of interest.
"""
import sys, traceback
D = "/a/mm/nn-gpt/ab/gpt/brute/ga/denoise"
AST = "/a/mm/nn-gpt/ab/gpt/brute/ast"

for label, paths in (("preflight's path (no .../brute/ast)", [D, "/a/mm/nn-gpt"]),
                     ("with .../brute/ast added",            [D, "/a/mm/nn-gpt", AST])):
    for m in list(sys.modules):
        if m.split(".")[0] in ("arch_embed", "mutator"):
            del sys.modules[m]
    sys.path = [p for p in sys.path if p not in (D, AST, "/a/mm/nn-gpt")]
    for p in reversed(paths):
        sys.path.insert(0, p)
    print(f"\n=== {label} ===")
    try:
        import arch_embed
        import sqlite3
        con = sqlite3.connect("/a/mm/nn-dataset/db/ab.nn.db")
        code = con.execute("SELECT code FROM nn ORDER BY id LIMIT 1").fetchone()[0]
        h, e = arch_embed.both_source(code)
        print(f"  both_source -> hash={'OK' if h else 'None'} embed={'OK' if e is not None else 'None'}")
        if e is None:
            # reproduce the swallowed failure
            try:
                arch_embed._load(code)
            except Exception:
                print("  underlying _load error:")
                traceback.print_exc()
    except Exception:
        traceback.print_exc()
