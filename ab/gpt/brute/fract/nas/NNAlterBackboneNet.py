import argparse
import json
from pathlib import Path
from datetime import datetime

from ab.gpt.brute.fract.nas.NNAlterBN import alter
from ab.gpt.NNEval import main as eval_main
from ab.gpt.util.Const import nngpt_dir, synth_dir, epoch_dir


def _best_accuracy(run_summary):
    best = 0.0
    for epoch in run_summary.get("epochs", []):
        for r in epoch.get("model_results", []):
            if r.get("success"):
                acc = r.get("accuracy") or 0.0
                if acc > best:
                    best = acc
    return best


def _archive_eval_artifacts(epochs, dataset_name):
    safe = dataset_name.replace('/', '_').replace(' ', '_')
    for i in range(epochs):
        s = Path(synth_dir(epoch_dir(i)))
        if not s.exists():
            continue
        for model_dir in s.iterdir():
            if model_dir.is_dir():
                for artifact in ('eval_info.json', 'eval_summary.json'):
                    p = model_dir / artifact
                    if p.exists():
                        stem, ext = artifact.rsplit('.', 1)
                        p.rename(model_dir / f"{stem}_{safe}.{ext}")


def _collect_models(run_summary):
    return [
        {"model_id": r.get("model_id"), "accuracy": r.get("accuracy"), "success": r.get("success")}
        for epoch in run_summary.get("epochs", [])
        for r in epoch.get("model_results", [])
    ]


def _compute_model_avg(nb_result, datasets):
    """Per-model accuracy across all datasets + average."""
    model_accs = {}
    for ds in datasets:
        for m in nb_result["datasets"].get(ds, {}).get("models", []):
            mid = m.get("model_id")
            if mid and m.get("success"):
                model_accs.setdefault(mid, {})[ds] = m.get("accuracy") or 0.0
    summary = {}
    for mid, ds_accs in model_accs.items():
        vals = list(ds_accs.values())
        summary[mid] = {**ds_accs, "avg": round(sum(vals) / len(vals), 6)}
    return summary


def _save_model_summaries(gen_epochs, model_avg, num_backbones, run_tag):
    """Write model_summary.json into each Bx dir AND a permanent archive."""
    archive_dir = Path(nngpt_dir) / "backbone_model_summaries" / run_tag / f"{num_backbones}_backbones"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for i in range(gen_epochs):
        s = Path(synth_dir(epoch_dir(i)))
        if not s.exists():
            continue
        for model_dir in s.iterdir():
            if model_dir.is_dir():
                mid = model_dir.name
                entry = model_avg.get(mid)
                if entry:
                    data = {"num_backbones": num_backbones, "model_id": mid, **entry}
                    payload = json.dumps(data, indent=2)
                    (model_dir / "model_summary.json").write_text(payload)
                    (archive_dir / f"{mid}.json").write_text(payload)
    print(f"  Summaries archived -> {archive_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=1, help="Generation epochs.")
    parser.add_argument('-m', '--max_variants', type=int, default=50, help="Models per epoch.")
    parser.add_argument('-nb', '--num_backbones', type=int, nargs='+', default=None,
                        help="Backbone count(s). Pass one to fix (e.g. -nb 2), "
                             "or multiple to compare (e.g. -nb 1 2 3 4).")
    parser.add_argument('-te', '--train_epochs', type=int, default=1, help="Training epochs per model.")
    parser.add_argument('--nn_name_prefix', type=str, default=None,
                        help="Prefix for model naming or database logs.")
    parser.add_argument('--datasets', type=str, nargs='+', default=['imagenette'],
                        help="Datasets to evaluate on (e.g. --datasets imagenette cifar-10 cifar-100 mnist).")
    parser.add_argument('--prm_json', type=str, default=None,
                        help='Eval params as JSON string, e.g. \'{"max_steps": 100, "lr": 0.01}\'')
    args = parser.parse_args()

    prm_json = json.loads(args.prm_json) if args.prm_json else None
    backbone_counts = args.num_backbones if args.num_backbones else [None]
    datasets = args.datasets

    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H-%M-%S')
    run_tag = f"{date_str}_{time_str}"
    run_dir = f"{date_str}/{time_str}"
    
    # Override NNGPT_EPOCH_DIR_OVERRIDE env var to isolate models into date/time subdirectory
    import os
    os.environ["NNGPT_EPOCH_DIR_OVERRIDE"] = str(Path(nngpt_dir) / 'llm' / date_str / time_str)
    
    all_results = {}
    first_iteration = True
    for nb in backbone_counts:
        label = f"{nb}_backbones"
        print(f"\n{'='*60}\n  Generating with {nb} backbone(s)\n{'='*60}")
        # model_prefix = f"{nb}b_"
        model_prefix = ""
        alter(args.epochs, 'NN_alter.json', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
              max_variants=args.max_variants, num_backbones=nb, clean=first_iteration, model_prefix=model_prefix)
        first_iteration = False

        all_results[label] = {"num_backbones": nb, "datasets": {}}

        for ds in datasets:
            print(f"\n  -- Evaluating {label} on dataset: {ds} --")
            current_prefix = f"{args.nn_name_prefix}-{nb}b" if args.nn_name_prefix else None
            run_summary = eval_main(
                nn_name_prefix=current_prefix,
                nn_train_epochs=args.train_epochs,
                dataset=ds,
                save_to_db=False,
                prm_json=prm_json,
                # eval_model_prefix=f"{nb}b_",
            )
            
            _archive_eval_artifacts(args.epochs, ds)

            # Custom saving to database
            from ab.nn.util.Util import uuid4
            from ab.nn.util.Const import db_file
            from ab.gpt.util.Const import new_lemur_nn_dir
            import sqlite3

            for epoch_info in run_summary.get("epochs", []):
                for r in epoch_info.get("model_results", []):
                    if r.get("success") and r.get("code_file"):
                        code_file_path = Path(r["code_file"])
                        if code_file_path.exists():
                            code = code_file_path.read_text(encoding="utf-8")
                            checksum = uuid4(code)
                            
                            # Find the actual name that was copied to lemur
                            nn_name = checksum
                            if new_lemur_nn_dir.exists():
                                matches = list(new_lemur_nn_dir.glob(f"*{checksum}.py"))
                                if matches:
                                    nn_name = matches[0].stem
                                    
                            desc_file = code_file_path.parent / "model_description.json"
                            desc_json = desc_file.read_text(encoding="utf-8") if desc_file.exists() else None
                            
                            print(f"  [DB] Custom saving model {nn_name} to database...")
                            conn = sqlite3.connect(db_file)
                            try:
                                cursor = conn.cursor()
                                cursor.execute("SELECT name FROM nn WHERE name = ?", (nn_name,))
                                if cursor.fetchone():
                                    cursor.execute(
                                        "UPDATE nn SET code = ?, id = ?, model_description_json = ? WHERE name = ?",
                                        (code, checksum, desc_json, nn_name)
                                    )
                                else:
                                    cursor.execute(
                                        "INSERT INTO nn (name, code, id, model_description_json) VALUES (?, ?, ?, ?)",
                                        (nn_name, code, checksum, desc_json)
                                    )
                                conn.commit()
                                print(f"  [DB] Successfully saved {nn_name}")
                            except Exception as db_err:
                                print(f"  [DB ERROR] Failed to save {nn_name}: {db_err}")
                            finally:
                                conn.close()

            best = _best_accuracy(run_summary)
            all_results[label]["datasets"][ds] = {
                "best_accuracy": best,
                "models": _collect_models(run_summary),
            }
            print(f"  >> {label} / {ds}: best_accuracy={best:.4f}")

        model_avg = _compute_model_avg(all_results[label], datasets)
        all_results[label]["model_avg"] = model_avg
        _save_model_summaries(args.epochs, model_avg, nb, run_dir)
        print(f"  Per-model avg saved to each Bx/model_summary.json")

    out_path = Path(nngpt_dir) / 'llm' / date_str / time_str / f"backbone_comparison_{run_tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n{'='*60}\n  Comparison saved -> {out_path}\n{'='*60}")

    col_w = 14
    header = f"  {'Backbones':<16}" + "".join(f"{ds:>{col_w}}" for ds in datasets) + f"{'average':>{col_w}}"
    print(header)
    print(f"  {'-' * (16 + col_w * (len(datasets) + 1))}")
    for label, v in all_results.items():
        row = f"  {label:<16}"
        accs = []
        for ds in datasets:
            best = v["datasets"].get(ds, {}).get("best_accuracy", 0.0)
            row += f"{best:>{col_w}.4f}"
            accs.append(best)
        avg = sum(accs) / len(accs) if accs else 0.0
        row += f"{avg:>{col_w}.4f}"
        print(row)


if __name__ == "__main__":
    main()
