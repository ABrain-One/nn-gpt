"""Pre-load the diversity gate's reference set with the WHOLE corpus, not just one run's output.

THE GAP THIS CLOSES, measured in generation 5's preflight:

    arch-dedupe: 1199 architectures already in the DB     <- EXACT check: whole corpus
    gate reference set: 143 graphs, 143 embeds            <- COSINE check: 143 models

The two halves of the gate cover wildly different populations. `--arch-dedupe-db` compares a
candidate's exact WL hash against every architecture in the database. The GRADED half compares
its embedding only against `gate.embeds`, which is built by add() from candidates THIS RUN
accepted -- 143 LLM models. It has never seen the 1,122 GA models. A candidate sitting at
cosine 0.99 to a GA architecture passes, because the gate does not know that architecture
exists.

So "cosine < 0.95 to everything" was only ever true within one run. This makes it true against
the dataset.

USE both_source(), NOT arch_uid. This is the one thing that has to be right. diversity_gate's
_fingerprint calls arch_embed.both_source(code), which instantiates through mutator.lemur_io;
arch_uid.build() uses a different constructor signature, and arch_uid.py's own docstring
records what happens when the two are mixed -- "the same architecture gets different hashes...
the cross-check then reported '0 architectures already in the dataset' for candidates whose
architectures were provably already there". A store written with the wrong constructor is not
a weaker gate, it is a gate that silently compares incomparable numbers and reports success.

MERGES, never overwrites: an existing store holds architectures the gate accepted during a
live run, and those are exactly the ones that must keep being rejected.

  python seed_diversity_store.py --out /a/mm/sweep_out/llm_evolve_par3/diversity_store_full.pt
"""
import argparse
import os
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, "/a/mm/nn-gpt/ab/gpt/brute/ast"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

TASK = "img-denoising"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/a/mm/nn-dataset/db/ab.nn.db")
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--merge", default=None,
                    help="existing store to fold in (its entries are kept)")
    ap.add_argument("--min-db", type=float, default=0.0,
                    help="only index models above this dB. 0 = everything, which is the "
                         "right default: a bad model is still an architecture we have "
                         "already tried and should not be handed back.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from functional_similarity import _disable_db_prm_lookup
    _disable_db_prm_lookup()
    from arch_embed import both_source

    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT n.name, n.code FROM nn n WHERE n.name IN "
        "(SELECT DISTINCT nn FROM stat WHERE task=? AND accuracy*48 >= ?)",
        (a.task, a.min_db)).fetchall()
    con.close()
    print(f"=== pre-seeding the diversity store | {len(rows)} {a.task} models ===",
          flush=True)

    hashes, embeds, keys = set(), [], []
    if a.merge and os.path.exists(a.merge):
        try:
            old = torch.load(a.merge, weights_only=False)
            if isinstance(old, dict):
                hashes = set(old.get("hashes", []))
                embeds = list(old.get("embeds", []))
                keys = list(old.get("embed_keys", []))
                if len(keys) < len(embeds):
                    keys += [None] * (len(embeds) - len(keys))
            else:
                hashes = set(old)
            print(f"  merged {len(hashes)} hashes / {len(embeds)} embeds from {a.merge}",
                  flush=True)
        except Exception as e:
            print(f"  WARNING could not read {a.merge}: {e}", flush=True)

    before_h, before_e = len(hashes), len(embeds)
    ok, failed, t0 = 0, 0, time.time()
    for i, (name, code) in enumerate(rows, 1):
        try:
            h, e = both_source(code)          # returns (None, None) on trace failure
        except Exception:
            h, e = None, None
        if h is None:
            failed += 1
        else:
            hashes.add(h)
            if e is not None:
                embeds.append(e)
                keys.append(name)
            ok += 1
        if i % 100 == 0 or i == len(rows):
            print(f"  [{i}/{len(rows)}] indexed={ok} untraceable={failed} "
                  f"{time.time()-t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tmp = a.out + ".tmp"
    torch.save({"hashes": list(hashes), "embeds": embeds, "embed_keys": keys}, tmp)
    os.replace(tmp, a.out)                    # atomic, matching how the gate persists

    print(f"\n=== store written ===")
    print(f"  exact hashes  {before_h:>6} -> {len(hashes):>6}")
    print(f"  embeddings    {before_e:>6} -> {len(embeds):>6}")
    print(f"  untraceable   {failed:>6}  (no architectural id; the gate cannot judge these)")
    print(f"  {a.out}")
    print(f"\n  The graded gate now compares every candidate against {len(embeds)} "
          f"architectures\n  instead of {before_e}. 'cosine < 0.95 to everything' finally "
          f"means the dataset,\n  not one run's own output.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
