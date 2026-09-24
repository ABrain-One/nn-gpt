"""
OOD generalization scale-up: evaluate any fine-tuned adapter on N held-out targets.

Question: does the fine-tuned LLR generator produce edits that IMPROVE strong,
never-seen models from OTHER architecture families (not the llr-/llr2-/llr3-
brute-force set it was trained on)? Scales an earlier n=3 pilot to ~25 targets spanning many families.

CAVEAT: the reported delta is against the target's STORED database accuracy, which was
not trained under this protocol and can be far below a matched control. Judge gains
against a fresh llr_uniform control (see ood_epoch_recheck_gen.py).

Parameterised so the SAME target set can be run against different adapters —
e.g. the old frozen-loop cycle-3 adapter (before-fix baseline) and the new
8-cycle run's best checkpoint (after-fix) — for a clean before/after contrast.

This script only GENERATES candidate model dirs (reads the `nn` table, writes
dirs). It does NOT write to the `stat` table, so it is safe to run while another
job holds the shared SQLite DB. Evaluate in a second NNEval step with
--no-save_to_db (also DB-read-only); then run ood_analyze.py.

Usage:
    python -m ab.gpt.brute.llr.ood_scale_up \
        --adapter-path /a/mm/backups/llrgen_20cycle_run_20260723/epoch/A3/Qwen/Qwen2.5-Coder-7B-Instruct \
        --n-targets 25 --out-epoch-dir 950 --prefix llrood --tag cyc3old
"""
import argparse
import json
import re
import sqlite3
from pathlib import Path

import torch
from peft import PeftModel

from ab.gpt.util.Const import epoch_dir, synth_dir, new_nn_file, conf_test_dir
from ab.gpt.util.llm.LLM import LLM
from ab.gpt.util.llm.LLMUtil import quantization_config_4bit
from ab.gpt.util.llm.Chatbot import ChatBot
from ab.gpt.util.nn.DeltaUtil import (
    shrink_nn_code_for_prompt,
    apply_delta,
    apply_delta_to_optimizer_assignment,
)
from ab.gpt.util.Util import extract_delta, create_file
from ab.nn.util.db.Write import init_population

DB_PATH = "/a/mm/db/ab.nn.db"
# Families the generator was TRAINED on — never use these as "held-out" targets.
TRAIN_PREFIXES = ("llr-", "llr2-", "llr3-", "llrgen-", "llrgen2-", "llrep", "llreptest")

# A diverse strategy per target, cycled — spread across group counts / types so
# each model gets a reasonable distinct shot rather than a lucky/unlucky draw.
STRATEGY_PICKS = [
    "llr_2grp_01x", "llr2_2grp_inv_01x", "llr3_2grp_05x", "llr_3grp_lin",
    "llr3_6grp_cos_inv", "llr2_4grp_cos", "llr3_4grp_geo_splits", "llr_2grp_75pct",
    "llr3_2grp_inv_05x", "llr2_cyclic_01x", "llr_4grp_lin", "llr3_3grp_step_head",
]


def _family(nn_name: str) -> str:
    m = re.match(r'^(.*?)-[0-9a-f]{8,}$', nn_name)
    return m.group(1) if m else nn_name


def select_targets(cur, dataset: str, n_targets: int, per_family: int):
    """Top-accuracy held-out models on `dataset` (epoch 1), spread across families."""
    rows = cur.execute(
        "SELECT nn, accuracy, transform FROM stat "
        "WHERE dataset=? AND epoch=1 AND metric='acc' ORDER BY accuracy DESC LIMIT 1500",
        (dataset,),
    ).fetchall()
    picked, fam_count = [], {}
    for nn_name, acc, transform in rows:
        if nn_name.startswith(TRAIN_PREFIXES):
            continue
        if '-' not in nn_name:            # skip bare vanilla arch names (GoogLeNet, ...)
            continue
        fam = _family(nn_name)
        if fam_count.get(fam, 0) >= per_family:
            continue
        picked.append((nn_name, float(acc), transform, fam))
        fam_count[fam] = fam_count.get(fam, 0) + 1
        if len(picked) >= n_targets:
            break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adapter-path', required=True,
                    help="Path to the LoRA adapter dir (…/Qwen/Qwen2.5-Coder-7B-Instruct).")
    ap.add_argument('--n-targets', type=int, default=25)
    ap.add_argument('--per-family', type=int, default=3,
                    help="Max targets per architecture family (spread).")
    ap.add_argument('--out-epoch-dir', type=int, default=950,
                    help="A{N}/synth_nn slot for candidate dirs (default A950; avoids A0..A20).")
    ap.add_argument('--dataset', default='cifar-100')
    ap.add_argument('--task', default='img-classification')
    ap.add_argument('--metric', default='acc')
    ap.add_argument('--prefix', default='llrood', help="DB name prefix at eval time.")
    ap.add_argument('--tag', default='run', help="Short label for the manifest (e.g. cyc3old / newbest).")
    ap.add_argument('--temperature', type=float, default=0.35)
    args = ap.parse_args()

    db_path = Path(DB_PATH)
    if not db_path.exists() or db_path.stat().st_size < 500_000:
        db_path.unlink(missing_ok=True)
        init_population()
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    targets = select_targets(cur, args.dataset, args.n_targets, args.per_family)
    print(f"[TARGETS] {len(targets)} held-out {args.dataset} models across "
          f"{len({t[3] for t in targets})} families:")
    for nn_name, acc, tr, fam in targets:
        print(f"    {acc:.4f}  {fam:20s} {nn_name[:44]}")

    # Strategy specs
    from ab.gpt.brute.llr.layerwise_lr import LAYERWISE_STRATEGIES as S1
    from ab.gpt.brute.llr.layerwise_lr2 import LAYERWISE_STRATEGIES as S2
    from ab.gpt.brute.llr.layerwise_lr3 import LAYERWISE_STRATEGIES as S3
    ALL = {s["name"]: s for s in S1 + S2 + S3}

    print("[GPU] before load:", torch.cuda.memory_allocated() / 2**30, "GiB")
    llm = LLM(model_path="Qwen/Qwen2.5-Coder-7B-Instruct",
              bnb_config=quantization_config_4bit, context_length=6144)
    print(f"[ADAPTER] attaching {args.adapter_path}")
    model = PeftModel.from_pretrained(llm.model, args.adapter_path, is_trainable=False)
    model.eval()
    chat_bot = ChatBot(model, llm.tokenizer, temperature=args.temperature, top_k=50, top_p=0.9)

    conf = json.load(open(conf_test_dir / "NN_gen_layerwise_lr.json"))
    key = conf["apply_layerwise_lr"]
    system_text = "\n".join(key["system"])
    template = "\n".join(key["prompt"])
    chat_bot.system_prompt = system_text

    out_root = synth_dir(epoch_dir(args.out_epoch_dir))
    out_root.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    manifest = []
    for idx, (nn_name, acc, transform, fam) in enumerate(targets):
        row = cur.execute("SELECT code FROM nn WHERE name=?", (nn_name,)).fetchone()
        if not row:
            print(f"[SKIP] {nn_name} not in nn table")
            continue
        full_code = row[0]
        strat_name = STRATEGY_PICKS[idx % len(STRATEGY_PICKS)]
        spec = ALL[strat_name]
        para = {
            "nn_code": shrink_nn_code_for_prompt(full_code), "accuracy": acc, "epoch": 1,
            "dataset": args.dataset, "task": args.task, "metric": args.metric,
            "addon_strategy": spec["name"], "addon_strategy_type": spec["type"],
            "addon_n_groups": spec["n_groups"], "addon_multipliers": spec["multipliers"],
            "addon_split_ratios": spec["split_ratios"], "addon_description": spec["description"],
        }
        print(f"\n=== B{idx}: {fam}/{nn_name[:30]} base={acc:.4f} strat={strat_name} ===")
        _, hp, tr, full_out = chat_bot.chat(template.format(**para),
                                            engineer_prompt=False, max_new_tokens=1024)
        delta = extract_delta(full_out)
        # Prefer the robust AST anchor-patcher (the 0/10 -> 9/10 fix); fall back
        # to plain apply_delta if the anchor path returns None.
        code = None
        if delta:
            code = apply_delta_to_optimizer_assignment(full_code, delta) or apply_delta(full_code, delta)
        applied = code is not None

        mdir = out_root / f"B{idx}"
        mdir.mkdir(parents=True, exist_ok=True)
        create_file(mdir, new_nn_file, code if applied else "")
        create_file(mdir, "full_output.txt", full_out)
        create_file(mdir, f"original_{nn_name}.py", full_code)
        pd.Series({
            "nn": nn_name, "nn_code": full_code, "accuracy": acc, "epoch": 1,
            "dataset": args.dataset, "task": args.task, "metric": args.metric,
            "prm": {"transform": transform},
        }).to_pickle(mdir / "dataframe.df")

        manifest.append({
            "B": idx, "adapter_tag": args.tag, "nn": nn_name, "family": fam,
            "baseline_acc": round(acc, 4), "strategy": strat_name,
            "transform": transform, "delta_applied": applied,
        })
        print(f"  delta_applied={applied}")
        del full_out
        torch.cuda.empty_cache()

    import csv
    project_root = Path(__file__).resolve().parents[4]
    man_path = project_root / 'out' / 'nngpt' / f'ood_manifest_{args.tag}_A{args.out_epoch_dir}.csv'
    with open(man_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys())); w.writeheader(); w.writerows(manifest)

    applied_n = sum(m["delta_applied"] for m in manifest)
    print(f"\n{'=' * 60}")
    print(f"Generated {len(manifest)} candidates ({applied_n} delta-applied) into {out_root}")
    print(f"Manifest: {man_path}")
    print(f"Next (DB-read-only eval): python -m ab.gpt.act.eval.Eval --only_epoch {args.out_epoch_dir} "
          f"--nn_train_epochs 1 --dataset {args.dataset} --task {args.task} --metric {args.metric} "
          f"--no-save_to_db --nn_name_prefix {args.prefix}")
    print(f"Then: python -m ab.gpt.brute.llr.ood_analyze --tag {args.tag} --out-epoch-dir {args.out_epoch_dir}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
