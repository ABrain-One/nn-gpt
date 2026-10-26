#!/usr/bin/env python3
"""Pipeline-v2 models in NN Dataset (LEMUR) format, for contribution to nn-dataset.

Every model that trained in the first MAX_CYCLE generations of a complete
out_<run_prefix>_<config>_s<seed> run is a candidate (cycle_*/nneval/gen_*/eval_info.json).
Models are identified as nn-dataset does (ab.nn.util.Util.uuid4: the ArchUID architecture
certificate); each architecture is exported once, with the trial of its earliest
generation (ties: run name, then model id).

  <out>/nn/<prefix>-<id>.py
  <out>/train/<task>_<dataset>_<metric>_<prefix>-<id>/<epoch>.json    one trial record
  <out>/manifest.csv          every trained model: run, generation, accuracy, WL hash, ...
  <out>/export_summary.json   settings, ArchUID source, per-run counts

--previous keeps the files of architectures an earlier export already holds, so a later
batch never changes a submitted record.

  python -m ab.gpt.act.kto.export_nn_dataset --runs_root . --out nn_dataset_export \\
      --arch_uid_module .cache/nn_dataset/<commit>/ArchUID.py
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import re
import shutil
import warnings
from pathlib import Path

TRIAL_KEYS = ("accuracy", "batch", "dropout", "duration", "epoch", "lr", "metric_acc",
              "momentum", "transform", "uid")
TRAIN_STAT_KEYS = ("train_loss", "test_loss", "train_accuracy", "gradient_norm",
                   "samples_per_second", "best_accuracy", "best_epoch", "cpu_count", "cpu_type",
                   "cpu_usage_percent", "total_ram_kb", "occupied_ram_kb", "ram_usage_percent",
                   "gpu_type", "gpu_memory_kb", "gpu_total_memory_kb", "occupied_gpu_memory_kb",
                   "gpu_memory_usage_percent")
REQUIRED = ("accuracy", "duration", "transform", "uid")
MANIFEST_COLS = ("name", "arch_id", "exported", "run", "base_model", "rule", "selection",
                 "struct_archive", "anchors", "generation", "model_id", "accuracy", "wl_hash",
                 "params", "code_md5", "git_commit")


def load_arch_uid(module_path: str | None):
    if module_path:
        spec = importlib.util.spec_from_file_location("nn_dataset_ArchUID", module_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        src = Path(module_path)
    else:
        from ab.nn.util import ArchUID as mod
        src = Path(mod.__file__)
    return mod, {"path": str(src), "sha256": hashlib.sha256(src.read_bytes()).hexdigest()}


def code_md5(src: str) -> str:
    return hashlib.md5(re.sub(r"\s", "", src).encode()).hexdigest()


def arch_id(mod, src: str) -> str:
    """ab.nn.util.Util.uuid4 for model code: the certificate, else the whitespace-free md5."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if mod.is_model_code(src):
            try:
                g = mod.source_graph(src)
                if len(g.keys):
                    return mod.wl_hash(g)
            except Exception:  # noqa: BLE001
                pass
    return code_md5(src)


def _jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _complete(run_dir: Path, max_cycle: int) -> bool:
    try:
        json.loads((run_dir / f"cycle_{max_cycle}" / "metrics.json").read_text(encoding="utf-8"))
        return True
    except Exception:  # noqa: BLE001
        return False


def discover_runs(root: Path, prefixes, seed: int, subdir: str):
    runs = []
    for p in prefixes:
        pat = re.compile(rf"^out_{re.escape(p)}_(.+)_s{seed}$")
        for d in sorted(root.glob(f"out_{p}_*_s{seed}")):
            run_dir = d / "nngpt" / subdir
            if pat.match(d.name) and (run_dir / "v2_config.json").exists():
                cfg = json.loads((run_dir / "v2_config.json").read_text(encoding="utf-8"))
                runs.append((d.name, run_dir, cfg))
    return runs


def trained_models(run: str, run_dir: Path, cfg: dict, max_cycle: int):
    for gen in range(1, max_cycle + 1):
        cdir = run_dir / f"cycle_{gen}"
        cands = {r.get("model_id"): r for r in _jsonl(cdir / "candidates.jsonl")}
        for mdir in sorted((cdir / "nneval").glob("gen_*")):
            info_f, code_f = mdir / "eval_info.json", mdir / "new_nn.py"
            if not (info_f.exists() and code_f.exists()):
                continue
            info = json.loads(info_f.read_text(encoding="utf-8"))
            args, res = info.get("eval_args", {}), info.get("eval_results", {})
            if res.get("accuracy") is None:
                continue
            cand = cands.get(mdir.name, {})
            yield {
                "run": run, "generation": gen, "model_id": mdir.name, "code_file": code_f,
                "task": args["task"], "dataset": args["dataset"], "metric": args["metric"],
                "prm": args.get("prm", {}), "accuracy": res["accuracy"],
                "wl_hash": cand.get("wl_hash", ""), "params": (cand.get("desc") or {}).get("params", ""),
                "base_model": cfg.get("base_model", ""), "rule": cfg.get("rule", ""),
                "selection": cfg.get("selection", ""), "struct_archive": cfg.get("struct_archive", ""),
                "anchors": cfg.get("anchors_per_generation", ""), "git_commit": cfg.get("git_commit", ""),
            }


def trial_record(prm: dict) -> dict:
    rec = {k: prm[k] for k in TRIAL_KEYS if k in prm}
    stat = {k: prm[k] for k in TRAIN_STAT_KEYS if k in prm}
    if stat:
        rec["train_stat"] = stat
    return rec


def check_export(out: Path, prefix: str) -> list:
    """Format check against the nn-dataset layout: names, model API, trial records."""
    errors = []
    name_re = re.compile(rf"^{re.escape(prefix)}-[0-9a-f]{{32}}$")
    names = {p.stem for p in (out / "nn").glob("*.py")}
    stat_names = set()
    for d in (out / "train").iterdir():
        parts = d.name.split("_", 3)
        if len(parts) != 4:
            errors.append(f"bad stat folder name: {d.name}")
            continue
        stat_names.add(parts[3])
        files = list(d.glob("*.json"))
        if not files or any(not f.stem.isdigit() for f in files):
            errors.append(f"{d.name}: expected <epoch>.json files only")
        for f in files:
            rows = json.loads(f.read_text(encoding="utf-8"))
            if not isinstance(rows, list) or not rows:
                errors.append(f"{f}: not a non-empty list of trials")
                continue
            for r in rows:
                miss = [k for k in REQUIRED if k not in r]
                if miss or not isinstance(r.get("transform"), str) or \
                        not isinstance(r.get("accuracy"), (int, float)) or not isinstance(r.get("duration"), int):
                    errors.append(f"{f}: trial missing/ill-typed {miss or REQUIRED}")
    for n in names:
        if not name_re.match(n):
            errors.append(f"bad model name: {n}")
        tree = ast.parse((out / "nn" / f"{n}.py").read_text(encoding="utf-8"))
        top = {x.name for x in tree.body if isinstance(x, (ast.FunctionDef, ast.ClassDef))}
        if not {"Net", "supported_hyperparameters"} <= top:
            errors.append(f"{n}.py: missing top-level Net or supported_hyperparameters")
    if names != stat_names:
        errors.append(f"models without stats: {sorted(names - stat_names)[:5]}, "
                      f"stats without model: {sorted(stat_names - names)[:5]}")
    return errors


def main() -> None:
    ap = argparse.ArgumentParser(description="Export pipeline-v2 models in NN Dataset format")
    ap.add_argument("--runs_root", default=".")
    ap.add_argument("--run_prefixes", nargs="+", default=["v2", "v2qwen"])
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--subdir", default="pref_v2")
    ap.add_argument("--max_cycle", type=int, default=21)
    ap.add_argument("--prefix", default="divpo", help="NN Dataset name prefix")
    ap.add_argument("--out", default="nn_dataset_export")
    ap.add_argument("--previous", default=None, help="Earlier export whose architectures keep their files")
    ap.add_argument("--arch_uid_module", default=None,
                    help="nn-dataset ab/nn/util/ArchUID.py to identify models with "
                         "(default: the installed ab.nn.util.ArchUID)")
    ap.add_argument("--include_incomplete", action="store_true",
                    help="Also export runs that have not reached --max_cycle")
    a = ap.parse_args()

    mod, mod_info = load_arch_uid(a.arch_uid_module)
    out = Path(a.out)
    runs_meta, occ = {}, []
    for run, run_dir, cfg in discover_runs(Path(a.runs_root), a.run_prefixes, a.seed, a.subdir):
        done = _complete(run_dir, a.max_cycle)
        runs_meta[run] = {"complete": done, "base_model": cfg.get("base_model", ""), "included": False}
        if done or a.include_incomplete:
            rows = list(trained_models(run, run_dir, cfg, a.max_cycle))
            runs_meta[run].update(included=True, trained=len(rows))
            occ += rows
    if not occ:
        raise SystemExit("No trained models found in the selected runs.")

    ids = {}
    for o in occ:
        src = o["code_file"].read_text(encoding="utf-8")
        o["code_md5"] = code_md5(src)
        if o["code_md5"] not in ids:
            ids[o["code_md5"]] = arch_id(mod, src)
        o["arch_id"] = ids[o["code_md5"]]
        o["name"] = f"{a.prefix}-{o['arch_id']}"

    prev = Path(a.previous) if a.previous else None
    prev_names = {p.stem for p in (prev / "nn").glob("*.py")} if prev and (prev / "nn").is_dir() else set()
    prev_manifest = list(csv.DictReader(open(prev / "manifest.csv", encoding="utf-8"))) \
        if prev and (prev / "manifest.csv").exists() else []
    prev_rows = {(r["run"], r["model_id"], r["generation"]) for r in prev_manifest if r["exported"] == "1"}

    groups = {}
    for o in sorted(occ, key=lambda o: (o["generation"], o["run"], o["model_id"])):
        groups.setdefault(o["name"], []).append(o)

    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "nn").mkdir(parents=True)
    (tmp / "train").mkdir()

    def keep_previous(name: str) -> None:
        shutil.copyfile(prev / "nn" / f"{name}.py", tmp / "nn" / f"{name}.py")
        for d in (prev / "train").glob(f"*_{name}"):
            shutil.copytree(d, tmp / "train" / d.name)

    kept = 0
    for name, members in groups.items():
        if name in prev_names:
            kept += 1
            keep_previous(name)
            for o in members:
                o["exported"] = int((o["run"], o["model_id"], str(o["generation"])) in prev_rows)
            continue
        first = members[0]
        for o in members:
            o["exported"] = int(o is first)
        shutil.copyfile(first["code_file"], tmp / "nn" / f"{name}.py")
        sdir = tmp / "train" / f"{first['task']}_{first['dataset']}_{first['metric']}_{name}"
        sdir.mkdir()
        epoch = int(first["prm"].get("epoch", 1))
        (sdir / f"{epoch}.json").write_text(json.dumps([trial_record(first["prm"])], indent=4),
                                            encoding="utf-8")
    # Architectures the earlier export holds but this one no longer covers stay as they were.
    carried = sorted(prev_names - groups.keys())
    for name in carried:
        keep_previous(name)

    errors = check_export(tmp, a.prefix)
    if errors:
        print("\n".join(errors[:50]))
        raise SystemExit(f"Format check failed ({len(errors)} problems); {tmp} left for inspection.")

    with open(tmp / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(occ, key=lambda o: (o["run"], o["generation"], o["model_id"])))
        # Rows of earlier exports that the selected runs no longer contain, so every
        # exported record keeps the trial it came from.
        orphan = {n for n in prev_names & groups.keys() if not any(o["exported"] for o in groups[n])}
        w.writerows(r for r in prev_manifest if r["name"] in set(carried)
                    or (r["name"] in orphan and r["exported"] == "1"))
    for run, meta in runs_meta.items():
        if meta["included"]:
            meta["exported_records"] = sum(o["exported"] for o in occ if o["run"] == run)
    summary = {
        "prefix": a.prefix, "max_cycle": a.max_cycle, "run_prefixes": a.run_prefixes, "seed": a.seed,
        "identity": "ab.nn.util.Util.uuid4 (ArchUID certificate)", "arch_uid_module": mod_info,
        "previous": str(prev) if prev else None,
        "trained_models": len(occ), "architectures": len(groups),
        "same_architecture_merged": len(occ) - len(groups), "kept_from_previous": kept,
        "carried_from_previous": len(carried), "exported_models": len(groups) + len(carried),
        "runs": runs_meta,
    }
    (tmp / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    old = out.with_name(out.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if out.exists():
        out.rename(old)
    tmp.rename(out)
    shutil.rmtree(old, ignore_errors=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2))
    for run, meta in runs_meta.items():
        print(f"  {run:45s} {meta}")
    print(f"-> {out}/ ({len(groups) + len(carried)} models)")


if __name__ == "__main__":
    main()
