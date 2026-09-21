#!/usr/bin/env python3
"""
Self-contained DivPO (Diverse Preference Optimization) pipeline.

A NEW algorithm on top of the existing self-contained pipeline: it SUBCLASSES
SelfContainedKTOPipeline and reuses its generation → eval → novelty(DB+prior) →
cycle/output machinery UNCHANGED, swapping only two stages:

  * bucketing (desirable/undesirable)  →  DivPO pair construction
  * KTO training                       →  DPO training (ab.gpt.act.tune.DPO)

DivPO (Lanchantin et al., 2025, arXiv:2501.18101) fixes preference-optimization
mode collapse by selecting *diverse* preference pairs:

  * chosen   = novel (Jaccard < sim_threshold vs LEMUR DB + all prior gens) AND
               accuracy ≥ threshold        → rare-and-good
  * rejected = non-novel (near-duplicate)  OR accuracy < threshold / failed
               → common-or-bad
  * (optional) diverse LEMUR-DB architectures injected as extra CHOSEN anchors
    to keep the positive set diverse (the recursive-self-training / model-
    collapse mitigation).

Every non-algorithm parameter (LLM, temperature, eval protocol, novelty-from-DB,
cycles, output isolation, LoRA drift-control) is inherited from the base so runs
are directly comparable to the KTO / noveltydb baselines. This file MODIFIES NO
existing file — it only imports the base class and helpers.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ab.gpt.act.kto.kto_selfcontained_finetune import (
    SelfContainedKTOPipeline,
    build_prompt_messages,
    save_cycle_results,
    _read_jsonl,
    _write_jsonl,
    _fenced,
    _unfenced,
    logger,
)


class SelfContainedDivPOPipeline(SelfContainedKTOPipeline):
    """DivPO variant of the self-contained pipeline (diverse pair selection + DPO)."""

    def __init__(
        self,
        *,
        divpo_db_anchors: int = 0,
        dpo_beta: float = 0.1,
        divpo_neg_per_pos: int = 3,
        divpo_max_pairs: int = 1000,
        graph_gate: bool = False,
        graph_gate_granularity: str = "topology",
        graph_gate_sizes: str = "32,64,96,224,256",
        graph_gate_wl_rounds: int = 3,
        graph_gate_num_classes: int = 10,
        graph_gate_in_channels: int = 3,
        **base_kwargs,
    ):
        # DivPO needs the DB-similarity novelty pre-filter (skip eval of near-dupes,
        # Jaccard vs DB + all prior gens). Force it on regardless of caller flags.
        base_kwargs["novelty_db"] = True
        super().__init__(**base_kwargs)

        self.divpo_db_anchors = int(divpo_db_anchors)
        self.dpo_beta = float(dpo_beta)
        self.divpo_neg_per_pos = max(1, int(divpo_neg_per_pos))
        self.divpo_max_pairs = int(divpo_max_pairs)

        # Graph-canonicalization novelty gate (opt-in; default off keeps every prior
        # run byte-identical and resume-safe — _prefilter_novelty falls straight
        # through to super()). When on, an architecture is novel only if it is
        # Jaccard-novel AND its canonical graph (torch.fx make_fx + Weisfeiler-Lehman
        # hash) has not been generated before, closing the lexical over-count.
        self.graph_gate = bool(graph_gate)
        self.graph_gate_granularity = graph_gate_granularity
        self.graph_gate_sizes = [int(s) for s in str(graph_gate_sizes).split(",") if s.strip()]
        self.graph_gate_wl_rounds = int(graph_gate_wl_rounds)
        self.graph_gate_num_classes = int(graph_gate_num_classes)
        self.graph_gate_in_channels = int(graph_gate_in_channels)
        self._graph_seen_records: List[Dict[str, Any]] = []
        if self.graph_gate:
            self._graph_seen_file = self.output_dir / "divpo_graph_seen.jsonl"
            self._graph_seen_records = _read_jsonl(self._graph_seen_file)

        # Preference-pair cache — accumulates across cycles like the KTO caches.
        self.divpo_pairs_cache_file = self.output_dir / "divpo_pairs_cache.jsonl"
        self.divpo_pairs: List[Dict[str, Any]] = _read_jsonl(self.divpo_pairs_cache_file)

        # On resume, re-seed the similarity index with prior generations recovered
        # from the pair cache (super() seeded it from the empty KTO caches + DB).
        if self.sim_index is not None and self.divpo_pairs:
            seed = []
            for p in self.divpo_pairs:
                seed.append(_unfenced(p.get("chosen", "")))
                seed.append(_unfenced(p.get("rejected", "")))
            n = self.sim_index.add_codes([c for c in seed if c])
            logger.info(f"[divpo] re-seeded similarity index with {n} codes "
                        f"from {len(self.divpo_pairs)} prior pairs")

        # Diverse real-data anchors (model-collapse mitigation).
        self.db_codes: List[str] = []
        if self.divpo_db_anchors > 0:
            self._load_db_codes()

        logger.info("=" * 80)
        logger.info("SELF-CONTAINED DivPO PIPELINE (diverse preference optimization)")
        logger.info(f"  DB anchors / cycle : {self.divpo_db_anchors} "
                    f"({len(self.db_codes)} DB architectures loaded)")
        logger.info(f"  DPO beta           : {self.dpo_beta}")
        logger.info(f"  negatives / positive: {self.divpo_neg_per_pos}")
        logger.info(f"  max pairs (train)  : {self.divpo_max_pairs}")
        logger.info(f"  novelty threshold  : Jaccard >= {self.sim_threshold} = non-novel")
        logger.info(f"  quality floor      : accuracy >= {self.accuracy_threshold}")
        logger.info(f"  starting pairs     : {len(self.divpo_pairs)}")
        if self.graph_gate:
            logger.info(f"  graph gate         : ON (granularity={self.graph_gate_granularity}, "
                        f"sizes={self.graph_gate_sizes}, {len(self._graph_seen_records)} hashes loaded)")
        logger.info("=" * 80)

    def _load_db_codes(self) -> None:
        """Load LEMUR DB architecture source for diverse chosen-anchors."""
        try:
            from ab.nn.api import data
            df = data(task=self.sim_db_task, dataset=self.sim_db_dataset, unique_nn=True)
            if "nn_code" in getattr(df, "columns", []):
                self.db_codes = [c for c in df["nn_code"].tolist()
                                 if isinstance(c, str) and c.strip()]
            logger.info(f"[divpo] loaded {len(self.db_codes)} LEMUR DB architectures for anchoring")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[divpo] could not load DB codes for anchors ({e}); anchors disabled")
            self.db_codes = []

    # ── stage 3': DivPO pair construction (replaces bucketing) ──────────────────

    def build_divpo_dataset(
        self,
        cycle: int,
        nneval_dir: Path,
        generation_records: List[Dict[str, Any]],
        eval_by_id: Dict[str, Dict[str, Any]],
    ) -> Tuple[Optional[Path], Dict[str, Any], Dict[str, Any]]:
        """Sort generations into chosen (rare-good) / rejected (common-bad), inject
        DB anchors, form augmented pairs, accumulate, and write divpo_train.jsonl."""
        rng = random.Random(cycle * 7919 + 13)
        prompt_messages = build_prompt_messages(
            dataset=self.dataset, params_limit=self.params_limit, diversity_hint=None
        )
        eff_threshold = self.accuracy_threshold  # DivPO quality floor (fixed, comparable)

        # Idempotency across resumes: drop this cycle's previously-added pairs.
        self.divpo_pairs = [p for p in self.divpo_pairs
                            if p.get("_meta", {}).get("cycle") != cycle]

        chosen_pool: List[Tuple[str, str, Optional[float]]] = []   # (code, id, acc)
        rejected_pool: List[Tuple[str, str, str]] = []            # (code, id, reason)
        accuracies: List[float] = []
        n_nonnovel = 0

        for rec in generation_records:
            model_id = rec.get("model_id")
            model_dir = nneval_dir / (model_id or "")

            # Near-duplicate (pre-filtered, eval skipped) → common → natural negative.
            if rec.get("not_novel"):
                n_nonnovel += 1
                aside = model_dir / "new_nn.notnovel.py"
                if aside.exists():
                    code = aside.read_text(encoding="utf-8", errors="replace")
                    rejected_pool.append((code, model_id, "non_novel"))
                continue

            # Unparseable → salvage raw text as a negative when substantial.
            if not rec.get("ok"):
                raw = ""
                raw_file = rec.get("raw_file")
                if raw_file and Path(raw_file).exists():
                    raw = Path(raw_file).read_text(encoding="utf-8", errors="replace")
                if raw.strip() and len(raw.strip()) >= 40:
                    rejected_pool.append((raw.strip(), model_id, "unparseable"))
                continue

            code_file = model_dir / "new_nn.py"
            if not code_file.exists():
                continue
            code = code_file.read_text(encoding="utf-8", errors="replace")

            ev = eval_by_id.get(model_id)
            if ev is None or not ev.get("success"):
                rejected_pool.append((code, model_id, "failed"))
                continue
            try:
                acc = float(ev.get("accuracy")) if ev.get("accuracy") is not None else 0.0
            except (TypeError, ValueError):
                acc = 0.0
            accuracies.append(acc)
            if acc >= eff_threshold:
                chosen_pool.append((code, model_id, acc))          # rare-and-good
            else:
                rejected_pool.append((code, model_id, "low_accuracy"))  # novel-but-bad

        n_chosen_gen = len(chosen_pool)

        # Diverse real-data anchors: inject DB architectures as extra CHOSEN.
        n_anchor = 0
        if self.divpo_db_anchors > 0 and self.db_codes:
            k = min(self.divpo_db_anchors, len(self.db_codes))
            for db_code in rng.sample(self.db_codes, k):
                chosen_pool.append((db_code, "db_anchor", None))
                n_anchor += 1

        # Form pairs: each chosen × N sampled rejected (augment + accumulate).
        new_pairs: List[Dict[str, Any]] = []
        if chosen_pool and rejected_pool:
            for (ccode, cid, cacc) in chosen_pool:
                cfenced = _fenced(ccode)
                k = min(self.divpo_neg_per_pos, len(rejected_pool))
                for (rcode, rid, reason) in rng.sample(rejected_pool, k):
                    rfenced = _fenced(rcode)
                    if cfenced == rfenced:
                        continue
                    new_pairs.append({
                        "prompt_messages": prompt_messages,
                        "chosen": cfenced,
                        "rejected": rfenced,
                        "_meta": {
                            "cycle": cycle, "chosen_id": cid, "rejected_id": rid,
                            "chosen_acc": cacc, "rejected_reason": reason,
                            "chosen_source": "db_anchor" if cid == "db_anchor" else "gen",
                        },
                    })

        self.divpo_pairs.extend(new_pairs)
        _write_jsonl(self.divpo_pairs, self.divpo_pairs_cache_file)

        # Cap the training set to the most-recent max_pairs (bounds training time).
        if len(self.divpo_pairs) > self.divpo_max_pairs:
            pairs_used = self.divpo_pairs[-self.divpo_max_pairs:]
        else:
            pairs_used = list(self.divpo_pairs)
        divpo_file = self._cycle_dir(cycle) / "divpo_train.jsonl"
        _write_jsonl(pairs_used, divpo_file)

        best_acc = max(accuracies) if accuracies else 0.0
        passed = [a for a in accuracies if a >= eff_threshold]
        avg_acc = (sum(passed) / len(passed)) if passed else 0.0
        avg_all = (sum(accuracies) / len(accuracies)) if accuracies else 0.0

        # bucketing-compatible stats so the existing plotter (novel vs not-novel,
        # accuracy, card) works unchanged on DivPO runs.
        pair_stats = {
            "new_desirable": n_chosen_gen,           # novel-good generations → plot "novel"
            "new_undesirable": len(rejected_pool),   # common/bad this cycle
            "not_novel_skipped": n_nonnovel,         # near-duplicates → plot "not novel"
            "undesirable_breakdown": {"non_novel": n_nonnovel},
            "db_anchors": n_anchor,
            "chosen_total": len(chosen_pool),
            "rejected_total": len(rejected_pool),
            "evaluated_accuracies": len(accuracies),
            "best_accuracy": best_acc,
            "avg_accuracy": avg_acc,
            "avg_accuracy_all": avg_all,
            "effective_threshold": eff_threshold,
            "divpo_pairs_new": len(new_pairs),
            "divpo_pairs_total": len(self.divpo_pairs),
            "desirable_total": len(self.divpo_pairs),  # "data accumulated" proxy
            "undesirable_total": 0,
        }
        ds_stats = {
            "pairs_used": len(pairs_used),
            "pairs_total": len(self.divpo_pairs),
            "divpo_file": str(divpo_file),
        }
        logger.info("")
        logger.info(f"[cycle {cycle}] DivPO PAIRS:")
        logger.info(f"  chosen  : {len(chosen_pool)} (gen {n_chosen_gen} + DB anchors {n_anchor})")
        logger.info(f"  rejected: {len(rejected_pool)} (non-novel {n_nonnovel})")
        logger.info(f"  pairs   : +{len(new_pairs)} new → {len(pairs_used)} used / "
                    f"{len(self.divpo_pairs)} total")
        logger.info(f"  best / avg(>=thr) / avg(all): {best_acc*100:.2f}% / "
                    f"{avg_acc*100:.2f}% / {avg_all*100:.2f}%")

        if len(pairs_used) < self.min_train_examples:
            logger.warning(f"[cycle {cycle}] only {len(pairs_used)} pairs "
                           f"(< {self.min_train_examples}); skipping DPO training this cycle.")
            return None, pair_stats, ds_stats
        return divpo_file, pair_stats, ds_stats

    # ── stage 5': DPO fine-tune (replaces KTO training) ─────────────────────────

    def run_divpo_training(self, cycle: int, divpo_file: Path) -> Dict[str, Any]:
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"CYCLE {cycle}: DPO (DivPO) FINE-TUNING")
        logger.info("=" * 80)

        checkpoint_dir = self._checkpoint_dir(cycle)
        if checkpoint_dir.exists() and (checkpoint_dir / "adapter_config.json").exists():
            logger.info(f"[cycle {cycle}] checkpoint already exists, skipping: {checkpoint_dir}")
            return {"success": True, "checkpoint_dir": str(checkpoint_dir), "skipped": True}

        prev_adapter = self._prev_adapter(cycle)
        cmd = [
            sys.executable, "-u", "-m", "ab.gpt.act.tune.DPO",
            "--llm_conf", self.llm_conf,
            "--dpo_data_file", str(divpo_file),
            "--dpo_checkpoint_dir", str(checkpoint_dir),
            "--dpo_beta", str(self.dpo_beta),
            "--num_train_epochs", str(self.num_train_epochs),
            "--max_prompt_length", str(self.kto_max_prompt_length),
            "--max_completion_length", str(self.kto_max_completion_length),
            # Same LoRA drift-control as KTO (comparability).
            "--learning_rate", str(self.kto_learning_rate),
            "--r", str(self.kto_lora_r),
            "--lora_alpha", str(self.kto_lora_alpha),
            "--max_grad_norm", str(self.kto_max_grad_norm),
        ]
        if prev_adapter is not None:
            logger.info(f"[cycle {cycle}] warm-starting from previous adapter: {prev_adapter}")
            cmd.extend(["--peft", str(prev_adapter)])
        else:
            logger.info(f"[cycle {cycle}] training fresh LoRA on base {self.base_model}")

        start = time.time()
        try:
            self._run_subprocess(cmd, f"DPO training cycle {cycle}")
        except subprocess.CalledProcessError as e:
            return {"success": False, "error": f"dpo_training_failed: exit_{e.returncode}",
                    "training_time_minutes": (time.time() - start) / 60}

        minutes = (time.time() - start) / 60
        if not (checkpoint_dir / "adapter_config.json").exists():
            logger.error(f"[cycle {cycle}] training finished but no adapter at {checkpoint_dir}")
            return {"success": False, "error": "checkpoint_missing",
                    "training_time_minutes": minutes}

        logger.info(f"[cycle {cycle}] DPO training complete in {minutes:.1f} min → {checkpoint_dir}")
        return {"success": True, "checkpoint_dir": str(checkpoint_dir),
                "training_time_minutes": minutes, "dpo_beta": self.dpo_beta}

    def _prev_adapter(self, cycle: int) -> Optional[Path]:
        """Warm-start from the most recent VALID checkpoint (one that actually has
        adapter_config.json), walking back past any cycle whose DPO training failed
        and left an empty checkpoint dir. Without this, one failed cycle cascades:
        the next cycle's generation is handed a non-existent adapter and dies too."""
        for c in range(cycle - 1, 0, -1):
            ckpt = self._checkpoint_dir(c)
            if (ckpt / "adapter_config.json").exists():
                return ckpt
        return None

    # ── stage 2b': novelty pre-filter + optional graph-canonicalization gate ────

    def _prefilter_novelty(self, cycle: int, nneval_dir: Path,
                           generation_records: List[Dict[str, Any]]) -> int:
        """With the graph gate off this is the inherited Jaccard-only pre-filter
        (byte-identical → existing runs resume untouched). With it on, an
        architecture is additionally flagged non-novel when its canonical graph
        (make_fx aten trace + WL hash) duplicates one generated in an earlier cycle
        — catching the structural duplicates Jaccard misses. Trace failures fail
        open (Jaccard-only). Seen hashes persist per-cycle so a mid-run resume never
        re-traces the whole history nor sees a future cycle's structures."""
        if not self.graph_gate or self.sim_index is None:
            return super()._prefilter_novelty(cycle, nneval_dir, generation_records)

        from ab.gpt.act.kto.graph_novelty_audit import graph_hash_for_file

        def _hash(path: Path) -> Optional[str]:
            h, _ = graph_hash_for_file(
                path, self.graph_gate_sizes, self.graph_gate_in_channels,
                self.graph_gate_num_classes, "aten",
                self.graph_gate_granularity, self.graph_gate_wl_rounds)
            return h

        # Idempotency across resumes (mirrors build_divpo_dataset): forget this
        # cycle's and any later cycle's persisted hashes; they are recomputed here.
        self._graph_seen_records = [r for r in self._graph_seen_records
                                    if int(r.get("cycle", -1)) < cycle]
        seen = {r["h"] for r in self._graph_seen_records}

        def _remember(h: Optional[str]) -> bool:
            if h is None:
                return False
            was = h in seen
            if not was:
                seen.add(h)
                self._graph_seen_records.append({"cycle": cycle, "h": h})
            return was

        n = n_graph_only = n_traced = n_tracefail = 0
        for rec in generation_records:
            if not rec.get("ok"):
                continue
            model_dir = nneval_dir / rec.get("model_id", "")
            nn_file = model_dir / "new_nn.py"
            aside = model_dir / "new_nn.notnovel.py"

            # Already flagged on a prior run: keep the flag but re-register its graph
            # so later cycles still dedup against this structure.
            if aside.exists() and not nn_file.exists():
                rec["not_novel"] = True
                _remember(_hash(aside))
                n += 1
                continue
            if not nn_file.exists():
                continue
            try:
                code = nn_file.read_text(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                continue

            jaccard_dup = self.sim_index.nearest_jaccard(code) >= self.sim_threshold
            self.sim_index.add(code)  # every generation joins the reference set

            h = _hash(nn_file)
            if h is None:
                n_tracefail += 1
            else:
                n_traced += 1
            graph_dup = _remember(h)  # True iff this graph was seen earlier

            if jaccard_dup or graph_dup:
                rec["not_novel"] = True
                if graph_dup and not jaccard_dup:
                    n_graph_only += 1
                try:
                    nn_file.rename(aside)
                except Exception:  # noqa: BLE001
                    pass
                n += 1

        _write_jsonl(self._graph_seen_records, self._graph_seen_file)
        logger.info(f"[cycle {cycle}] graph-gate pre-filter: {n} non-novel "
                    f"({n_graph_only} caught by graph beyond Jaccard) · "
                    f"traced {n_traced} ok / {n_tracefail} fail → eval skipped")
        return n

    # ── orchestration (mirrors base run_cycle; swaps the two DivPO stages) ──────

    def run_cycle(self, cycle: int) -> Dict[str, Any]:
        t0 = time.time()
        logger.info("")
        logger.info("#" * 80)
        logger.info(f"# CYCLE {cycle} / {self.cycles}  (DivPO)")
        logger.info("#" * 80)

        nneval_dir, gen_records = self.generate_models(cycle)
        self._prefilter_novelty(cycle, nneval_dir, gen_records)  # skips eval of near-dupes
        eval_by_id = self.evaluate_models(cycle, nneval_dir)
        self._free_gpu()
        divpo_file, pair_stats, ds_stats = self.build_divpo_dataset(
            cycle, nneval_dir, gen_records, eval_by_id
        )

        if divpo_file is not None:
            train_stats = self.run_divpo_training(cycle, divpo_file)
        else:
            train_stats = {"success": False, "skipped": True, "reason": "insufficient_pairs"}

        result = {
            "cycle": cycle,
            "timestamp": datetime.now().isoformat(),
            "generated": len(gen_records),
            "extractable": sum(1 for r in gen_records if r.get("ok")),
            "bucketing": pair_stats,
            "dataset": ds_stats,
            "training": train_stats,
            "cycle_time_minutes": (time.time() - t0) / 60,
        }
        save_cycle_results(result, self._cycle_dir(cycle) / "metrics.json")
        return result


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Self-contained DivPO (diverse preference optimization) pipeline")
    # Baseline params (kept identical to the noveltydb runs for comparability).
    p.add_argument("--llm_conf", type=str, default="nngpt_unique_arch_rag.json")
    p.add_argument("--cycles", type=int, default=31)
    p.add_argument("--models_per_cycle", type=int, default=100)
    p.add_argument("--accuracy_threshold", type=float, default=0.40)
    p.add_argument("--num_train_epochs", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--gen_max_rejections", type=int, default=3)
    p.add_argument("--sim_threshold", type=float, default=0.85)
    p.add_argument("--sim_db_task", type=str, default="img-classification")
    p.add_argument("--sim_db_dataset", type=str, default="cifar-10")
    p.add_argument("--eval_transform", type=str, default="norm_256_flip")
    p.add_argument("--eval_lr", type=float, default=0.01)
    p.add_argument("--eval_momentum", type=float, default=0.9)
    p.add_argument("--eval_batch", type=int, default=10)
    p.add_argument("--eval_multi_gpu", action="store_true",
                   help="Distribute evaluation across all visible GPUs (NNEval worker pool); "
                        "pins generation/training to one GPU. Needs >1 GPU allocated.")
    p.add_argument("--output_subdir", type=str, default="divpo_selfcontained")
    p.add_argument("--resume_from_cycle", type=int, default=None)

    # DivPO algorithm params.
    p.add_argument("--dpo_beta", type=float, default=0.1,
                   help="DPO KL strength; higher keeps the policy nearer the diverse base model")
    p.add_argument("--divpo_db_anchors", type=int, default=0,
                   help="Inject this many diverse LEMUR-DB architectures as chosen anchors per cycle")
    p.add_argument("--divpo_neg_per_pos", type=int, default=3,
                   help="Rejected examples sampled per chosen (augment sparse positives)")
    p.add_argument("--divpo_max_pairs", type=int, default=1000,
                   help="Cap on accumulated pairs used per DPO training (most recent kept)")

    # Graph-canonicalization novelty gate (opt-in; default off = Jaccard-only).
    p.add_argument("--graph_gate", action="store_true",
                   help="Also flag graph-isomorphic duplicates as non-novel "
                        "(make_fx aten trace + Weisfeiler-Lehman hash vs prior cycles)")
    p.add_argument("--graph_gate_granularity", choices=["topology", "typed"], default="topology")
    p.add_argument("--graph_gate_sizes", type=str, default="32,64,96,224,256")
    p.add_argument("--graph_gate_wl_rounds", type=int, default=3)
    p.add_argument("--graph_gate_num_classes", type=int, default=10)
    p.add_argument("--graph_gate_in_channels", type=int, default=3)
    args = p.parse_args()

    pipeline = SelfContainedDivPOPipeline(
        divpo_db_anchors=args.divpo_db_anchors,
        dpo_beta=args.dpo_beta,
        divpo_neg_per_pos=args.divpo_neg_per_pos,
        divpo_max_pairs=args.divpo_max_pairs,
        graph_gate=args.graph_gate,
        graph_gate_granularity=args.graph_gate_granularity,
        graph_gate_sizes=args.graph_gate_sizes,
        graph_gate_wl_rounds=args.graph_gate_wl_rounds,
        graph_gate_num_classes=args.graph_gate_num_classes,
        graph_gate_in_channels=args.graph_gate_in_channels,
        # inherited baseline params
        llm_conf=args.llm_conf,
        cycles=args.cycles,
        models_per_cycle=args.models_per_cycle,
        accuracy_threshold=args.accuracy_threshold,
        num_train_epochs=args.num_train_epochs,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        gen_max_rejections=args.gen_max_rejections,
        sim_threshold=args.sim_threshold,
        sim_db_task=args.sim_db_task,
        sim_db_dataset=args.sim_db_dataset,
        eval_transform=args.eval_transform,
        eval_lr=args.eval_lr,
        eval_momentum=args.eval_momentum,
        eval_batch=args.eval_batch,
        eval_multi_gpu=args.eval_multi_gpu,
        output_subdir=args.output_subdir,
        resume_from_cycle=args.resume_from_cycle,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
