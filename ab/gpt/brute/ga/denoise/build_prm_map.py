"""Extract each model's REAL hyperparameters into one lookup file, once.

WHY. embed_corpus.py handed every model an invented prm dict
{lr, momentum, batch, dropout, epoch_max}, and 285 models read a key that was not in it --
'stochastic_depth_prob', 'patch_size', 'attention_dropout', 'norm_eps', 'dropout_aux' --
and died with KeyError. Inventing more defaults would be the wrong repair: 'patch_size'
CHANGES THE GRAPH, so guessing it produces a confident hash for an architecture that was
never trained, which is worse than a failure because nothing downstream can tell.

The real values are in the DB. stat.prm is a uid into a prm(uid, name, value) table of
7.3M rows, so every model's actual training hyperparameters are recoverable exactly.

Scanned once here rather than per shard: prm has no usable index on uid, so a per-shard
scan would repeat a 7.3M-row pass 32 times.
"""
import json, sqlite3, sys, time

db = sys.argv[1] if len(sys.argv) > 1 else "/shared/ssd/home/b-b-ahmed/thesis/nn-gpt/db/ab.nn.db"
out = sys.argv[2] if len(sys.argv) > 2 else "/shared/ssd/home/b-b-ahmed/thesis/sweep_out/manifest/prm_map.json"
task = sys.argv[3] if len(sys.argv) > 3 else "img-classification"

c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
t0 = time.time()
# One prm uid per model. MIN() is deterministic; models with several runs differ in lr/batch,
# neither of which changes the graph.
want = dict(c.execute("SELECT nn, MIN(prm) FROM stat WHERE task=? GROUP BY nn", (task,)))
need = set(want.values())
print(f"{len(want)} models, {len(need)} distinct prm uids ({time.time()-t0:.0f}s)", flush=True)

by_uid = {}
n = 0
for uid, name, value in c.execute("SELECT uid, name, value FROM prm"):
    n += 1
    if uid in need:
        by_uid.setdefault(uid, {})[name] = value
    if n % 2000000 == 0:
        print(f"  scanned {n} prm rows ({time.time()-t0:.0f}s)", flush=True)
c.close()

m = {nn: by_uid.get(uid, {}) for nn, uid in want.items()}
have = sum(1 for v in m.values() if v)
json.dump(m, open(out, "w"))
keys = {}
for v in m.values():
    for k in v:
        keys[k] = keys.get(k, 0) + 1
print(f"\n{have}/{len(m)} models have real hyperparameters ({time.time()-t0:.0f}s)")
print("keys present:")
for k, v in sorted(keys.items(), key=lambda kv: -kv[1])[:20]:
    print(f"   {k:26s} {v}")
print(f"wrote {out}")
