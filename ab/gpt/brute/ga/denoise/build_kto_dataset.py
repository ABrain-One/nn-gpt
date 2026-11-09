"""Turn generation episodes into the KTO JSONL that TuneNNGenKTO consumes, plus a frozen
evaluation set.

Every decision below is forced by something measured earlier today. They are stated because
each one, done the obvious way instead, fails SILENTLY -- the run completes and the dataset is
wrong.

EPISODE KEY IS run/gen/id, NEVER id.
    Candidate ids are `g<gen>_<idx>_<strategy>` and every run restarts generation numbering,
    so `g1_0000_seeded_constraint_based` exists in par, par2, par3, par4 and par5. Measured:
    6,012 records collapse to 2,545 distinct ids. A dict keyed on id loses 57.7% and raises
    nothing.

COMPLETION IS response_raw.txt, NEVER candidate.py.
    Compared on 1,500 episodes: they differ in 1,500 of 1,500. Raw output opens with a `<nn>`
    wrapper that candidate.py has stripped, and `<nn>...</nn>` is the repo-wide output contract
    (see conf NN_gen.json). Training on the stripped form teaches the model to emit source our
    own extractor cannot parse.

LABEL IS RECOMPUTED, NEVER READ FROM preflight.json.
    `ok=True` means "passed every stage including the gate", but the gate ADMITS untraceable
    candidates by contract. Measured: 44.6% of stored positives are not actually novel -- 59%
    of the never-checked ones were duplicates, and 123 gate-checked ones went stale because the
    corpus grew after they were generated. episode_distinct.py recomputes this from the AST
    hash, which covers 99.4% of episodes against torch.fx's ~57%.

PROMPT IS TRIMMED BY DROPPING WHOLE REFERENCE BLOCKS, NEVER BY TRUNCATION.
    KTOTrainer truncates prompts FROM THE LEFT, and the left of ours is the feedback and the
    task instructions -- truncation keeps the boilerplate and discards the brief. Budget is
    6,144 tokens, measured as the point where GPU 0 (which holds lm_head and therefore the
    logits) sits at 19.2 of 23.6 GiB under device_map across 4 GPUs.

SPLIT IS BY ARCHITECTURE HASH, NEVER RANDOM.
    The corpus holds 1,053 cross-shard duplicate hashes. A random split puts the same
    architecture on both sides and the dev metric measures memorisation.
"""
import argparse
import collections
import glob
import json
import os
import random
import re
import sys

# THE HEADER IS "SUPPORTING REFERENCE n", NOT "REFERENCE n", AND THE FENCE IS ```python.
# The first version of this regex matched neither, so nothing was ever trimmed: 4,342 of
# 5,509 episodes were discarded as "unfittable" while the trimmer silently found no blocks
# to remove. Verified against the real prompt text before reuse.
REF_RE = re.compile(
    r"\nSUPPORTING REFERENCE \d+ - measured [^\n]*\n```python\n.*?\n```\n", re.S)


def trim_prompt(user_text, tok, budget, system_text=""):
    """Drop trailing supporting REFERENCE blocks until the rendered prompt fits.

    The PRIMARY reference (the seed being altered, labelled "REFERENCE - measured ...") is
    never dropped: it is what the candidate is an alteration OF. Only the numbered supporting
    references go, last first, because arm-balancing showed the mix of references matters more
    than their count -- two drawn from different arms carry the signal that took yield from 3
    to 17.5 per generation.
    """
    def n_tok(u):
        msgs = ([{"role": "system", "content": system_text}] if system_text else []) + \
               [{"role": "user", "content": u}]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return len(tok(s, add_special_tokens=False)["input_ids"])

    cur = user_text
    dropped = 0
    while n_tok(cur) > budget:
        blocks = list(REF_RE.finditer(cur))
        if not blocks:
            return None, dropped, n_tok(cur)      # cannot fit without cutting the brief
        last = blocks[-1]
        cur = cur[:last.start()] + "\n" + cur[last.end():]
        dropped += 1
    return cur, dropped, n_tok(cur)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/a/mm/sweep_out")
    ap.add_argument("--distinct", default="/a/mm/sweep_out/manifest/episode_distinct")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--budget", type=int, default=6144)
    ap.add_argument("--dev-frac", type=float, default=0.05)
    ap.add_argument("--eval-und-n", type=int, default=64,
                    help="ALSO hold out this many undesirable-only groups for eval. They "
                         "must be reserved BEFORE the split -- carving eval prompts out of "
                         "kto_train.jsonl afterwards means evaluating on prompts the model "
                         "was trained on, which reads as a large improvement and is a leak.")
    ap.add_argument("--eval-n", type=int, default=64,
                    help="frozen prompts held out ENTIRELY, for before/after yield")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--undesirable-ratio", type=float, default=3.0,
                    help="target |undesirable|/|desirable| in TRAIN, applied by the repo's "
                         "KTODataManager.balance. The repo default of 1.0 would discard 79% "
                         "of the failures, and failures are why KTO was chosen over SFT.")
    ap.add_argument("--out-dir", default="/a/mm/sweep_out/kto_data")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)

    dist = {}
    for p in sorted(glob.glob(os.path.join(a.distinct, "shard*.json"))):
        dist.update(json.load(open(p)))
    print(f"=== recomputed distinctness for {len(dist)} episodes ===")

    eps = []
    for p in sorted(glob.glob(os.path.join(a.root, "llm_evolve*/gen*/pre/preflight.json"))):
        root = p.rsplit("/pre/", 1)[0]
        rel = os.path.relpath(root, a.root).replace(os.sep, "/")
        try:
            recs = json.load(open(p))
        except Exception:
            continue
        for x in recs:
            key = f"{rel}/{x.get('id')}"
            d = os.path.join(root, "run", "cand", x.get("id", ""))
            if key in dist and os.path.exists(os.path.join(d, "response_raw.txt")):
                eps.append((key, d, dist[key]))
    print(f"  episodes with source + label: {len(eps)}")

    # GLOBAL DEDUP. Shards were greedy only within themselves, so the same architecture can
    # be marked novel in two shards. First occurrence in sorted order keeps the label.
    eps.sort(key=lambda e: e[0])
    seen_hash = set()
    rows, skipped_fit, dup_global = [], 0, 0
    counts = collections.Counter()
    for key, d, rec in eps:
        h = rec.get("ast_hash")
        novel = bool(rec.get("novel_exact"))
        if novel and h is not None:
            if h in seen_hash:
                novel = False
                dup_global += 1
            else:
                seen_hash.add(h)
        label = bool(rec.get("ok")) and novel

        sysf = os.path.join(d, "prompt_system.txt")
        system_text = open(sysf).read() if os.path.exists(sysf) else ""
        user_text = open(os.path.join(d, "prompt_user.txt")).read()
        completion = open(os.path.join(d, "response_raw.txt")).read()
        trimmed, ndrop, ntok = trim_prompt(user_text, tok, a.budget, system_text)
        if trimmed is None:
            skipped_fit += 1
            continue
        msgs = ([{"role": "system", "content": system_text}] if system_text else []) + \
               [{"role": "user", "content": trimmed}]
        try:
            mtime = os.path.getmtime(os.path.join(d, "response_raw.txt"))
        except OSError:
            mtime = 0.0
        rows.append({"prompt_messages": msgs, "completion": completion, "label": label,
                     "_meta": {"key": key, "ast_hash": h, "stage": rec.get("stage"),
                               "refs_dropped": ndrop, "prompt_tokens": ntok,
                               "mtime": mtime}})
        counts["desirable" if label else "undesirable"] += 1
        counts[f"drop{ndrop}"] += 1

    print(f"\n  built {len(rows)} records")
    print(f"    desirable   {counts['desirable']}")
    print(f"    undesirable {counts['undesirable']}")
    print(f"    ratio 1:{counts['undesirable']/max(counts['desirable'],1):.1f}"
          f"   -> desirable_weight "
          f"{counts['undesirable']/max(counts['desirable'],1):.1f}")
    print(f"    demoted by global dedup {dup_global}")
    print(f"    unfittable (dropped)    {skipped_fit}")
    for k in sorted(x for x in counts if x.startswith("drop")):
        print(f"    references dropped {k[4:]}: {counts[k]}")

    # SPLIT BY ARCHITECTURE HASH so a hash never spans train and dev.
    rng = random.Random(a.seed)
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["_meta"]["ast_hash"] or f"_none{i}"].append(i)
    gk = sorted(groups)
    rng.shuffle(gk)

    # The frozen eval set is carved from DESIRABLE-bearing groups first, so the before/after
    # comparison is run on prompts that at least once produced something acceptable --
    # otherwise a zero-yield eval set cannot show improvement in either direction.
    # CAP THE EVAL CARVE-OUT AT A FRACTION OF THE DESIRABLE GROUPS. Taking every
    # desirable-bearing group until eval_n is reached consumed ALL of them the first time
    # this ran -- kto_train.jsonl came out with 1,056 records and ZERO positives, which KTO
    # cannot learn from and which no error would have reported.
    # STRATIFY BOTH HOLD-OUTS. Desirable groups are 8% of the total, so filling dev from a
    # shuffled list gave it ZERO positives on the first run -- TRL still trains, still reports
    # an eval loss, and that loss says nothing about the class the whole objective is about.
    # Eval and dev each take their share of desirable and undesirable groups explicitly.
    des_g = [k for k in gk if any(rows[i]["label"] for i in groups[k])]
    und_g = [k for k in gk if k not in set(des_g)]
    max_eval = max(1, min(a.eval_n, int(0.25 * len(des_g))))
    n_dev_des = max(1, int(a.dev_frac * len(des_g)))
    n_dev_und = max(1, int(a.dev_frac * len(und_g)))
    # dev takes und_g[:n_dev_und]; eval takes the slice immediately after, so the two
    # never overlap and neither reaches train.
    print(f"  groups: {len(des_g)} desirable-bearing, {len(und_g)} undesirable-only")
    print(f"  eval reserves {max_eval} desirable groups; "
          f"dev takes {n_dev_des} desirable + {n_dev_und} undesirable")

    eval_idx, dev_idx = [], []
    for k in des_g[:max_eval]:
        eval_idx += groups[k]
    # Undesirable-only eval groups: 4,099 available, so reserving 64 costs 1.6% of the
    # negatives and nothing at all from the 357 positives. It also makes the eval prompt mix
    # representative of what the model actually faces, instead of only prompts that
    # historically produced a success.
    n_eval_und = min(a.eval_und_n, max(0, len(und_g) - n_dev_und - 1))
    for k in und_g[n_dev_und:n_dev_und + n_eval_und]:
        eval_idx += groups[k]
    print(f"  eval also reserves {n_eval_und} undesirable-only groups")
    for k in des_g[max_eval:max_eval + n_dev_des]:
        dev_idx += groups[k]
    for k in und_g[:n_dev_und]:
        dev_idx += groups[k]
    ev, dv = set(eval_idx), set(dev_idx)
    train = [r for i, r in enumerate(rows) if i not in ev and i not in dv]
    dev = [rows[i] for i in dev_idx]

    # BALANCE WITH THE REPO'S OWN METHOD, not a reimplementation. balance() reads only
    # undesirable_ratio and _balance_seed off self, so a shim runs the real code path without
    # TrainingDataManager's train.jsonl scaffolding.
    #
    # IT KEEPS THE TAIL: `undesirables[-max_U:]`, on the documented reasoning that later
    # cycles' negatives are sharper and more on-distribution. Our rows are sorted by KEY, so
    # the tail would be an alphabetical accident (llm_evolve_smoke sorts last). Sorting by the
    # response file's mtime first makes "most recent" mean what the method intends -- which
    # here is par5, the run whose prompt format and arm-balanced references match what the
    # tuned model will actually be given.
    sys.path.insert(0, "/a/mm/nn-gpt")
    # POST-REFACTOR PATH FIRST. The 2026-08 NNGPT refactoring moved the LLM fine-tuning and
    # network-modification scripts into ab.gpt.act, so ab.gpt.kto_pipeline became
    # ab.gpt.act.kto. Both are tried because this tree has not pulled the refactor yet, and the
    # import must keep working before and after it lands -- this is the ONLY line in the whole
    # denoise package the refactoring touches (everything else we use, ab.gpt.util.{LLM,LLMUtil,
    # KTO,Const,acc_client} and ab.gpt.brute, is unchanged upstream).
    try:
        from ab.gpt.act.kto.kto_data_manager import KTODataManager
    except ImportError:
        from ab.gpt.kto_pipeline.kto_data_manager import KTODataManager

    class _Shim:
        undesirable_ratio = a.undesirable_ratio
        _balance_seed = a.seed

    des = [r for r in train if r["label"]]
    und = sorted((r for r in train if not r["label"]),
                 key=lambda r: r["_meta"]["mtime"])
    train, bstats = KTODataManager.balance(_Shim(), des, und, a.undesirable_ratio)
    print(f"\n=== BALANCE (repo KTODataManager.balance, ratio {a.undesirable_ratio}) ===")
    for k, v in bstats.items():
        print(f"  {k:24} {v}")
    kept = collections.Counter(r["_meta"]["key"].split("/")[0]
                               for r in train if not r["label"])
    print(f"  undesirables kept by run: {dict(kept.most_common())}")

    def dump(recs, name):
        p = os.path.join(a.out_dir, name)
        with open(p, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        d = sum(1 for r in recs if r["label"])
        print(f"  wrote {p}  {len(recs)} records ({d} desirable)")
        return p

    print(f"\n=== SPLIT (by architecture hash, {len(gk)} groups) ===")
    dump(train, "kto_train.jsonl")
    dump(dev, "kto_dev.jsonl")

    # The frozen eval set keeps PROMPTS ONLY: yield is measured by generating fresh
    # completions before and after, so shipping the old completion would invite scoring the
    # dataset instead of the model.
    seen_prompt = set()
    ev_rows = []
    for i in eval_idx:
        r = rows[i]
        h = hash(json.dumps(r["prompt_messages"]))
        if h in seen_prompt:
            continue
        seen_prompt.add(h)
        ev_rows.append({"prompt_messages": r["prompt_messages"], "_meta": r["_meta"]})
    p = os.path.join(a.out_dir, "frozen_eval_prompts.jsonl")
    with open(p, "w") as f:
        for r in ev_rows:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {p}  {len(ev_rows)} distinct held-out prompts")

    json.dump({"records": len(rows), "desirable": counts["desirable"],
               "undesirable": counts["undesirable"],
               "desirable_weight": round(
                   counts["undesirable"] / max(counts["desirable"], 1), 2),
               "train": len(train), "dev": len(dev), "eval_prompts": len(ev_rows),
               "budget": a.budget, "global_dedup": dup_global,
               "unfittable": skipped_fit},
              open(os.path.join(a.out_dir, "summary.json"), "w"), indent=2)
    print(f"\n  wrote {a.out_dir}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
