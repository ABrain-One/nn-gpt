#!/usr/bin/env python3
"""Stage-2 sampling for the predictor-validation experiment (TEVC revision).

Selects, from the finished Stage-1 runs, a stratified set of accepted architectures
for predictor validation, following the pre-registered design:

  * Four key configurations (default: DivPO-anchors+gate, DivPO-core,
    KTO-nonnovel-undesirable, KTO-noveltydb-discard).
  * From each, 30 accepted architectures = 10 per tercile of single-epoch accuracy,
    always including the top-10 by single-epoch accuracy (120 total).

It reads per-model single-epoch accuracy from cycle_<n>/nneval/gen_*/eval_info.json
(the exact files the plotter reads) and copies each selected new_nn.py into an
NNEval-compatible directory  <out>/<config>/gen_XXXX/new_nn.py  plus a manifest.

Read-only w.r.t. the source runs (only copies out). Run in-container on the cluster,
where the out_* run directories live. No GPU needed.

Example:
  python -m ab.gpt.act.kto.exp_stage2_sample \
      --out stage2_pool \
      --config DivPO-anchors+gate=out_3184144/nngpt/divpo_anchors_graphgate \
      --config DivPO-core=out_XXXX/nngpt/divpo_core \
      --config KTO-nonnovel=out_XXXX/nngpt/kto_nonnovel_undesirable \
      --config KTO-discard=out_XXXX/nngpt/kto_noveltydb_discard
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _acc_from_eval_info(payload: Dict[str, Any]) -> Optional[float]:
    """Single-epoch accuracy from a per-model eval_info.json (mirrors plot_kto_cycles.py)."""
    res = payload.get("eval_results")
    a = None
    if isinstance(res, (list, tuple)) and len(res) >= 2:
        a = res[1]
    elif isinstance(res, dict):
        a = res.get("accuracy", res.get("acc"))
        if a is None:
            eps = res.get("epochs", [])
            if eps and isinstance(eps[0], dict):
                a = eps[0].get("accuracy", eps[0].get("acc"))
    try:
        return float(a) if a is not None else None
    except (TypeError, ValueError):
        return None


def collect_accepted(config_dir: Path, threshold: float) -> List[Tuple[Path, float]]:
    """Every accepted (evaluated, >= threshold) model in a config's run dir, as
    (new_nn.py path, single-epoch accuracy). 'Accepted' mirrors the pipeline:
    the generation carries a live new_nn.py (novel) and an eval_info.json above the bar."""
    out: List[Tuple[Path, float]] = []
    for cyc in sorted(config_dir.glob("cycle_*")):
        nneval = cyc / "nneval"
        if not nneval.is_dir():
            continue
        for gen in sorted(nneval.iterdir()):
            nn = gen / "new_nn.py"
            info = gen / "eval_info.json"
            if not (nn.exists() and info.exists()):
                continue
            try:
                acc = _acc_from_eval_info(json.loads(info.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                acc = None
            if acc is not None and acc >= threshold:
                out.append((nn, acc))
    return out


def stratified_pick(pool: List[Tuple[Path, float]], per_tercile: int) -> List[Tuple[Path, float, int]]:
    """10 per tercile of single-epoch accuracy, with the global top-10 guaranteed in.
    Returns (path, acc, tercile) with tercile in {0,1,2} (2 = highest)."""
    ranked = sorted(pool, key=lambda t: t[1])  # ascending
    n = len(ranked)
    if n == 0:
        return []
    b1, b2 = n // 3, 2 * n // 3
    terciles = {0: ranked[:b1], 1: ranked[b1:b2], 2: ranked[b2:]}
    top10 = set(id(t) for t in sorted(pool, key=lambda t: -t[1])[:per_tercile])
    picked: List[Tuple[Path, float, int]] = []
    for ti in (2, 1, 0):
        group = terciles[ti]
        # highest-accuracy first inside the tercile; guarantee the global top-10
        group_sorted = sorted(group, key=lambda t: -t[1])
        forced = [t for t in group_sorted if id(t) in top10]
        rest = [t for t in group_sorted if id(t) not in top10]
        chosen = (forced + rest)[:per_tercile]
        picked.extend((p, a, ti) for (p, a) in chosen)
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-2 stratified architecture sampler")
    ap.add_argument("--out", required=True, help="Output pool dir (NNEval-compatible layout)")
    ap.add_argument("--config", action="append", default=[], metavar="LABEL=RUN_DIR",
                    help="Repeatable: paper-label=path/to/out_TAG/nngpt/<config>")
    ap.add_argument("--threshold", type=float, default=0.40, help="Acceptance accuracy bar")
    ap.add_argument("--per_tercile", type=int, default=10)
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.jsonl"
    manifest = []
    gi = 0
    for spec in args.config:
        if "=" not in spec:
            raise SystemExit(f"--config expects LABEL=RUN_DIR, got: {spec}")
        label, run_dir = spec.split("=", 1)
        cdir = Path(run_dir)
        pool = collect_accepted(cdir, args.threshold)
        picked = stratified_pick(pool, args.per_tercile)
        dst_cfg = out_root / label.replace("/", "_")
        dst_cfg.mkdir(parents=True, exist_ok=True)
        print(f"[{label}] accepted pool={len(pool)}  selected={len(picked)}")
        for src, acc, tercile in picked:
            gen_name = f"gen_{gi:04d}"
            dst = dst_cfg / gen_name
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst / "new_nn.py")
            rec = {"gen": gen_name, "config": label, "tercile": tercile,
                   "acc_1epoch": acc, "src": str(src)}
            (dst / "source.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
            manifest.append(rec)
            gi += 1
    manifest_path.write_text("\n".join(json.dumps(m) for m in manifest) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest)} architectures to {out_root} (manifest: {manifest_path})")


if __name__ == "__main__":
    main()
