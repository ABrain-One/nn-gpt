import json, os, sys, torch
import ab.gpt.TunePPO as T                     # same build + training code as the RL reward
ev = T.CIFARRewardEvaluator(train_epochs=10)
for f in sys.argv[1:]:
    R = [json.loads(l) for l in open(f)]; out = f.replace("_strict.jsonl", "_acc.jsonl")
    with open(out, "w") as g:
        for r in R:
            r["accuracy"] = None
            if r.get("strict") == "OK":
                built, status, pen = T._strict_build(T._strict_extract(r["text"]))
                if built is not None:
                    try:
                        r["accuracy"] = T._train_eval(ev, built[1])
                    except Exception as e:
                        r["train_error"] = f"{type(e).__name__}: {str(e)[:120]}"
                        print(f"  training_failed: {r['train_error']}", flush=True)
                    del built; torch.cuda.empty_cache()
                print(f"{os.path.basename(f)} idx {r['idx']}: acc {r['accuracy']}", flush=True)
            g.write(json.dumps(r) + "\n")
    print("DONE", out, flush=True)
