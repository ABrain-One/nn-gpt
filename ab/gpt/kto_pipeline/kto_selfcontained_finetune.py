#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ab.gpt.kto_pipeline.kto_generator import build_prompt_messages
from ab.gpt.iterative_pipeline.novelty_checker import NoveltyChecker
from ab.gpt.util.Const import conf_llm_dir, nngpt_dir
from ab.gpt.util.CycleResults import save_cycle_results

logger = logging.getLogger("kto_selfcontained")


# ── small helpers ─────────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _fenced(code: str) -> str:
    """Wrap raw model code in a python fence — the canonical good completion."""
    return f"```python\n{code.strip()}\n```"


def _unfenced(text: str) -> str:
    """Recover raw code from a ```python ... ``` fence (best-effort)."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if "```" in t:
            t = t[: t.rfind("```")]
    return t.strip()


# ── pipeline ──────────────────────────────────────────────────────────────────

class SelfContainedKTOPipeline:
    """Iterative KTO pipeline that bootstraps its own preference data."""

    def __init__(
        self,
        llm_conf: str,
        cycles: int = 10,
        models_per_cycle: int = 30,
        accuracy_threshold: float = 0.50,
        novelty_check: bool = True,
        eval_skip: bool = True,
        undesirable_ratio: float = 1.0,
        max_undesirable_total: int = 1000,
        min_train_examples: int = 10,
        # curriculum threshold (#1): "fixed" = flat bar; "quantile" = max(floor,
        # q-percentile of the cycle's accuracies); "linear" = floor + slope*cycle.
        threshold_mode: str = "fixed",
        threshold_quantile: float = 0.6,
        threshold_ceil: float = 0.62,
        threshold_slope: float = 0.01,
        # dynamic class weights (#2): "auto" sets undesirable_weight so that
        # desirable_weight*n_D ~= class_weight_target * undesirable_weight*n_U.
        class_weight_mode: str = "fixed",
        class_weight_target: float = 1.0,
        # similarity penalty (#3): subtract alpha*shaped_jaccard from the KTO chosen
        # reward for near-duplicate generations (vs LEMUR DB + our own prior gens).
        # When on, non-novel passers are NOT dropped — they enter as desirable with
        # a penalty; novelty stays a reported metric.
        sim_penalty: bool = False,
        sim_alpha: float = 0.4,
        sim_threshold: float = 0.85,
        sim_shape: str = "exponential",
        sim_db_task: str = "img-classification",
        sim_db_dataset: str = "cifar-10",
        # KTO hyperparameters
        kto_beta: float = 0.1,
        kto_desirable_weight: float = 1.0,
        kto_undesirable_weight: float = 1.0,
        num_train_epochs: int = 3,
        # KTO drift control — conservative defaults.  A single KTO fine-tune at
        # the SFT-default lr=1e-5 / r=32 was observed (in the older KTO pipeline)
        # to collapse the generator into a degenerate template, so we halve the
        # learning rate and LoRA rank and tighten gradient clipping.
        kto_learning_rate: float = 5e-6,
        kto_lora_r: int = 16,
        kto_lora_alpha: int = 16,
        kto_max_grad_norm: float = 0.3,
        max_prompt_length: int = 1536,
        # generation knobs
        temperature: float = 0.4,
        top_k: int = 50,
        top_p: float = 0.95,
        max_new_tokens: int = 2048,
        gen_max_rejections: int = 5,
        # evaluation knobs (CIFAR-10, 1-epoch protocol matching the model card)
        dataset: str = "cifar-10",
        # Defaults match the ABrain model-card / ab.nn protocol exactly
        # (lr 0.01, momentum 0.9, dropout 0.2, batch 10, norm_256_flip, 1 epoch)
        # so accuracies are directly comparable to the card's 63.98% best / 50.99% avg.
        eval_transform: str = "norm_256_flip",
        eval_lr: float = 0.01,
        eval_momentum: float = 0.9,
        eval_batch: int = 10,
        eval_dropout: float = 0.2,
        eval_train_epochs: int = 1,
        params_limit: int = 500_000,
        save_to_db: bool = False,
        # plumbing
        output_subdir: str = "kto_selfcontained",
        resume_from_cycle: Optional[int] = None,
        max_retries: int = 2,
        seed: int = 43,
    ):
        self.llm_conf = self._resolve_llm_conf(llm_conf)
        with open(self.llm_conf, "r", encoding="utf-8") as f:
            self.llm_conf_data = json.load(f)
        self.base_model = self.llm_conf_data["base_model_name"]
        self.context_length = int(self.llm_conf_data.get("context_length", 4096))

        self.cycles = cycles
        self.models_per_cycle = models_per_cycle
        self.accuracy_threshold = accuracy_threshold
        self.novelty_check = novelty_check
        self.eval_skip = eval_skip
        self.undesirable_ratio = undesirable_ratio
        self.max_undesirable_total = max_undesirable_total
        self.min_train_examples = min_train_examples

        self.threshold_mode = threshold_mode
        self.threshold_quantile = threshold_quantile
        self.threshold_ceil = threshold_ceil
        self.threshold_slope = threshold_slope
        self.class_weight_mode = class_weight_mode
        self.class_weight_target = class_weight_target

        self.sim_penalty = sim_penalty
        self.sim_alpha = sim_alpha
        self.sim_threshold = sim_threshold
        self.sim_shape = sim_shape
        self.sim_db_task = sim_db_task
        self.sim_db_dataset = sim_db_dataset

        self.kto_beta = kto_beta
        self.kto_desirable_weight = kto_desirable_weight
        self.kto_undesirable_weight = kto_undesirable_weight
        self.num_train_epochs = num_train_epochs
        self.kto_learning_rate = kto_learning_rate
        self.kto_lora_r = kto_lora_r
        self.kto_lora_alpha = kto_lora_alpha
        self.kto_max_grad_norm = kto_max_grad_norm

        # KTO concatenates prompt + completion; their token budgets must sum to
        # ≤ the model context length (else the sequence overflows position
        # embeddings).  Give the prompt a fixed slice and the rest to the code.
        self.kto_max_prompt_length = min(max_prompt_length, max(512, self.context_length - 1024))
        self.kto_max_completion_length = max(1024, self.context_length - self.kto_max_prompt_length)

        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.gen_max_rejections = gen_max_rejections

        self.dataset = dataset
        self.eval_transform = eval_transform
        self.eval_lr = eval_lr
        self.eval_momentum = eval_momentum
        self.eval_batch = eval_batch
        self.eval_dropout = eval_dropout
        self.eval_train_epochs = eval_train_epochs
        self.params_limit = params_limit
        self.save_to_db = save_to_db

        self.resume_from_cycle = resume_from_cycle
        self.max_retries = max_retries
        self.seed = seed

        # Output tree — MUST be under nngpt_dir (NNEval.relative_to(nngpt_dir)).
        self.output_dir = nngpt_dir / output_subdir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Persistent preference-data caches (the "dataset" the pipeline grows).
        self.desirable_cache_file = self.output_dir / "kto_desirable_cache.jsonl"
        self.undesirable_cache_file = self.output_dir / "kto_undesirable_cache.jsonl"
        self.desirable: List[Dict[str, Any]] = _read_jsonl(self.desirable_cache_file)
        self.undesirable: List[Dict[str, Any]] = _read_jsonl(self.undesirable_cache_file)

        # Novelty checker — starts EMPTY (no dataset); grows as we accept models.
        self.novelty_checker = NoveltyChecker(self.output_dir / "seen_models.json")

        # Similarity-penalty index (#3): LEMUR DB code + our own prior generations.
        # Built once; new generations are added as they're accepted (cumulative).
        self.sim_index = None
        if self.sim_penalty:
            from ab.gpt.kto_pipeline.similarity_penalty import SimilarityIndex
            self.sim_index = SimilarityIndex(threshold=self.sim_threshold)
            n_db = self.sim_index.add_db(self.sim_db_task, self.sim_db_dataset)
            n_prev = self.sim_index.add_codes(
                [_unfenced(r.get("completion", "")) for r in self.desirable])
            logger.info(f"[sim] index built: {n_db} DB models + {n_prev} prior generations "
                        f"(alpha={self.sim_alpha}, threshold={self.sim_threshold}, "
                        f"shape={self.sim_shape})")

        self._setup_logging()
        self.cycle_results: List[Dict[str, Any]] = []

        logger.info("=" * 80)
        logger.info("SELF-CONTAINED ITERATIVE KTO PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Base model           : {self.base_model}")
        logger.info(f"Output dir           : {self.output_dir}")
        logger.info(f"Cycles               : {self.cycles}")
        logger.info(f"Models per cycle     : {self.models_per_cycle}")
        logger.info(f"Accuracy threshold   : {self.accuracy_threshold}")
        logger.info(f"Novelty check        : {self.novelty_check}")
        logger.info(f"KTO beta             : {self.kto_beta}")
        logger.info(f"Desirable weight     : {self.kto_desirable_weight}")
        logger.info(f"Undesirable weight   : {self.kto_undesirable_weight}")
        logger.info(f"Undesirable ratio    : {self.undesirable_ratio}")
        logger.info(f"Threshold mode       : {self.threshold_mode}")
        logger.info(f"Class weight mode    : {self.class_weight_mode}")
        logger.info(f"Starting desirable   : {len(self.desirable)}")
        logger.info(f"Starting undesirable : {len(self.undesirable)}")
        logger.info("=" * 80)

        # Guard: ABrain's first-epoch CIFAR-10 ceiling is ~0.68 (model card best
        # 0.6398).  A threshold at/above that makes "desirable" unreachable, so
        # every cycle collects only negatives and KTO never trains.  Warn loudly.
        if self.accuracy_threshold >= 0.66:
            logger.warning(
                f"accuracy_threshold={self.accuracy_threshold:.2f} is at/above the "
                "model's realistic first-epoch ceiling (~0.68). Desirable models "
                "will be near-impossible and KTO will be skipped every cycle. "
                "Use ~0.45-0.55."
            )

    # ── setup helpers ─────────────────────────────────────────────────────────

    def _resolve_llm_conf(self, llm_conf: str) -> str:
        p = Path(llm_conf)
        if p.exists():
            return str(p)
        candidate = conf_llm_dir / p.name
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(f"LLM config not found: {llm_conf}")

    def _setup_logging(self) -> None:
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh = logging.FileHandler(self.output_dir / "kto_pipeline.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    @staticmethod
    def _free_gpu() -> None:
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — cleanup is best-effort
            pass

    @staticmethod
    def _percentile(values: List[float], q: float) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        if len(s) == 1:
            return s[0]
        pos = max(0.0, min(1.0, q)) * (len(s) - 1)
        lo, hi = math.floor(pos), math.ceil(pos)
        if lo == hi:
            return s[int(lo)]
        return s[int(lo)] + (s[int(hi)] - s[int(lo)]) * (pos - lo)

    def _effective_threshold(self, cycle: int, cycle_accs: List[float]) -> float:
        """Curriculum bar: floor for "fixed"; rising for "quantile"/"linear"."""
        floor = self.accuracy_threshold
        if self.threshold_mode == "quantile" and cycle_accs:
            return min(max(floor, self._percentile(cycle_accs, self.threshold_quantile)),
                       self.threshold_ceil)
        if self.threshold_mode == "linear":
            return min(floor + self.threshold_slope * max(0, cycle - 1), self.threshold_ceil)
        return floor

    def _class_weights(self, ds_stats: Dict[str, Any]) -> Tuple[float, float]:
        """KTO imbalance rule: balance desirable_weight*n_D vs undesirable_weight*n_U."""
        dw, uw = self.kto_desirable_weight, self.kto_undesirable_weight
        if self.class_weight_mode == "auto":
            n_d = max(1, int(ds_stats.get("desirable_used", 0)))
            n_u = max(1, int(ds_stats.get("undesirable_used", 0)))
            uw = (dw * n_d) / (max(1e-6, self.class_weight_target) * n_u)
            uw = min(10.0, max(0.1, uw))
        return dw, uw

    def _cycle_dir(self, cycle: int) -> Path:
        d = self.output_dir / f"cycle_{cycle}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _checkpoint_dir(self, cycle: int) -> Path:
        return self.output_dir / f"cycle_{cycle}" / "checkpoint"

    def _prev_adapter(self, cycle: int) -> Optional[Path]:
        """Adapter to warm-start from: previous cycle's checkpoint, else None."""
        if cycle <= 1:
            return None
        prev = self._checkpoint_dir(cycle - 1)
        return prev if prev.exists() else None

    # ── stage 1: generation ────────────────────────────────────────────────────

    def generate_models(self, cycle: int) -> Tuple[Path, List[Dict[str, Any]]]:
        cycle_dir = self._cycle_dir(cycle)
        nneval_dir = cycle_dir / "nneval"
        records_file = cycle_dir / "generation_records.jsonl"

        if records_file.exists() and nneval_dir.exists():
            logger.info(f"[cycle {cycle}] Reusing existing generations: {records_file}")
            return nneval_dir, _read_jsonl(records_file)

        adapter = self._prev_adapter(cycle)
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"CYCLE {cycle}: GENERATION ({self.models_per_cycle} models)")
        logger.info(f"  base={self.base_model}  adapter={adapter}")
        logger.info("=" * 80)

        cmd = [
            sys.executable, "-u", "-m", "ab.gpt.kto_pipeline.kto_generator",
            "--base_model", self.base_model,
            "--out_dir", str(nneval_dir),
            "--records_file", str(records_file),
            "--num_models", str(self.models_per_cycle),
            "--start_index", "0",
            "--temperature", str(self.temperature),
            "--top_k", str(self.top_k),
            "--top_p", str(self.top_p),
            "--max_new_tokens", str(self.max_new_tokens),
            "--max_rejections", str(self.gen_max_rejections),
            "--dataset", self.dataset,
            "--params_limit", str(self.params_limit),
            "--context_length", str(self.context_length),
            "--seed", str(self.seed + cycle * 1000),
        ]
        if adapter is not None:
            cmd.extend(["--adapter", str(adapter)])

        self._run_subprocess(cmd, f"generation cycle {cycle}")

        records = _read_jsonl(records_file)
        n_ok = sum(1 for r in records if r.get("ok"))
        logger.info(f"[cycle {cycle}] generation done: {n_ok}/{len(records)} extractable")
        return nneval_dir, records

    # ── stage 2: evaluation ────────────────────────────────────────────────────

    def evaluate_models(self, cycle: int, nneval_dir: Path) -> Dict[str, Dict[str, Any]]:
        """Train+evaluate every gen_XXXX/new_nn.py on CIFAR-10 via NNEval."""
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"CYCLE {cycle}: EVALUATION")
        logger.info("=" * 80)

        has_models = any(
            (nneval_dir / d.name / "new_nn.py").exists()
            for d in nneval_dir.iterdir() if d.is_dir()
        ) if nneval_dir.exists() else False
        if not has_models:
            logger.warning(f"[cycle {cycle}] no extractable models to evaluate")
            return {}

        # Import here so generation/training subprocess dispatch doesn't pay the
        # heavy ab.nn import cost, and so a missing optional dep fails loudly only
        # at eval time.
        from ab.gpt import NNEval

        eval_by_id: Dict[str, Dict[str, Any]] = {}
        try:
            summary = NNEval.main(
                nn_name_prefix=None,
                nn_train_epochs=self.eval_train_epochs,
                only_epoch=0,
                save_to_db=self.save_to_db,
                nn_alter_epochs=1,
                task="img-classification",
                dataset=self.dataset,
                metric="acc",
                lr=self.eval_lr,
                batch=self.eval_batch,
                dropout=self.eval_dropout,
                momentum=self.eval_momentum,
                transform=self.eval_transform,
                custom_synth_dir=str(nneval_dir),
                cycle=cycle,
                use_sequential=True,
                use_all_visible_gpus=False,
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"[cycle {cycle}] NNEval raised: {type(e).__name__}: {e}")
            summary = {"epochs": []}

        epochs = summary.get("epochs", []) or []
        model_results = epochs[0].get("model_results", []) if epochs else []
        for item in model_results:
            mid = item.get("model_id")
            if mid:
                eval_by_id[mid] = item

        n_succ = sum(1 for v in eval_by_id.values() if v.get("success"))
        logger.info(f"[cycle {cycle}] evaluation done: {n_succ}/{len(eval_by_id)} trained successfully")
        return eval_by_id

    # ── stage 2b: skip eval of duplicates (novelty is structural, no GPU needed) ─

    def _prefilter_novelty(self, cycle: int, nneval_dir: Path,
                           generation_records: List[Dict[str, Any]]) -> int:
        """Flag generations that duplicate an already-accepted design and exclude
        them from evaluation. Novelty is decided from the code alone, so there's no
        point spending GPU on duplicates that won't enter training. Flagged records
        get new_nn.py moved aside (so NNEval ignores them) and are counted as
        not_novel_skipped in bucketing. Idempotent across resumes via the moved file.
        Disabled by eval_skip=False (e.g. benchmark runs that must evaluate ALL),
        and always disabled under the similarity penalty (non-novel passers must be
        evaluated so they can enter training as desirable).
        """
        if not self.novelty_check or not self.eval_skip or self.sim_index is not None:
            return 0
        n = 0
        for rec in generation_records:
            if not rec.get("ok"):
                continue
            model_dir = nneval_dir / rec.get("model_id", "")
            nn_file = model_dir / "new_nn.py"
            aside = model_dir / "new_nn.notnovel.py"
            if aside.exists() and not nn_file.exists():
                rec["not_novel"] = True  # already filtered on a prior run
                n += 1
                continue
            if not nn_file.exists():
                continue
            try:
                code = nn_file.read_text(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                continue
            if not self.novelty_checker.is_novel(code, rec.get("model_id")):
                rec["not_novel"] = True
                try:
                    nn_file.rename(aside)
                except Exception:  # noqa: BLE001
                    pass
                n += 1
        if n:
            logger.info(f"[cycle {cycle}] novelty pre-filter: skipping eval of {n} "
                        "duplicate architecture(s) already accepted in earlier cycles")
        return n

    # ── stage 3: bucketing into desirable / undesirable ─────────────────────────

    def bucket_models(
        self,
        cycle: int,
        nneval_dir: Path,
        generation_records: List[Dict[str, Any]],
        eval_by_id: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Sort every generation into a KTO preference class and append to the
        accumulated caches.  Returns this cycle's bucketing stats.
        """
        prompt_messages = build_prompt_messages(
            dataset=self.dataset, params_limit=self.params_limit, diversity_hint=None
        )

        # Idempotency: if this cycle was already (partially) bucketed (e.g. on a
        # resume), drop its prior cache entries so we don't double-count.
        self.desirable = [r for r in self.desirable if r.get("_meta", {}).get("cycle") != cycle]
        self.undesirable = [r for r in self.undesirable if r.get("_meta", {}).get("cycle") != cycle]

        n_desirable = 0
        n_und_compile = 0
        n_und_runtime = 0
        n_und_lowacc = 0
        n_not_novel = 0
        n_und_unparseable = 0
        n_skipped = 0
        accuracies: List[float] = []
        cycle_penalties: List[float] = []  # similarity penalties of this cycle's desirables

        def add_desirable(code: str, model_id: str, accuracy: float) -> None:
            pen = 0.0
            if self.sim_index is not None:
                pen = self.sim_index.penalty_for(code, self.sim_shape)  # vs DB + prior gens
                self.sim_index.add(code)                                # becomes a "prior" too
                cycle_penalties.append(pen)
            self.desirable.append({
                "prompt_messages": prompt_messages,
                "completion": _fenced(code),
                "label": True,
                "sim_penalty": pen,
                "_meta": {
                    "model_id": model_id, "cycle": cycle,
                    "accuracy": accuracy, "reason": "passed",
                },
            })

        def add_undesirable(completion: str, model_id: str, reason: str,
                            accuracy: Optional[float]) -> None:
            self.undesirable.append({
                "prompt_messages": prompt_messages,
                "completion": completion,
                "label": False,
                "_meta": {
                    "model_id": model_id, "cycle": cycle,
                    "accuracy": accuracy, "reason": reason,
                },
            })

        cycle_accs: List[float] = []
        for ev in eval_by_id.values():
            if ev.get("success") and ev.get("accuracy") is not None:
                try:
                    cycle_accs.append(float(ev.get("accuracy")))
                except (TypeError, ValueError):
                    pass
        eff_threshold = self._effective_threshold(cycle, cycle_accs)

        for rec in generation_records:
            model_id = rec.get("model_id")
            model_dir = nneval_dir / model_id

            # Duplicate of an already-accepted design — pre-filtered before eval.
            if rec.get("not_novel"):
                n_not_novel += 1
                continue

            # ── unparseable generation: salvage raw text as a hard negative ──
            if not rec.get("ok"):
                raw = ""
                raw_file = rec.get("raw_file")
                if raw_file and Path(raw_file).exists():
                    raw = Path(raw_file).read_text(encoding="utf-8", errors="replace")
                if not raw.strip() or len(raw.strip()) < 40:
                    n_skipped += 1
                    continue
                add_undesirable(raw.strip(), model_id,
                                rec.get("error") or "code_extraction_failed", None)
                n_und_unparseable += 1
                continue

            code_file = model_dir / "new_nn.py"
            if not code_file.exists():
                n_skipped += 1
                continue
            code = code_file.read_text(encoding="utf-8", errors="replace")

            ev = eval_by_id.get(model_id)
            if ev is None:
                # Generated + extractable but evaluator never returned a verdict
                # (e.g. evaluator died before reaching it).  Treat as undesirable
                # — the model could not be shown to clear the bar.
                add_undesirable(_fenced(code), model_id, "not_evaluated", None)
                n_und_runtime += 1
                continue

            if not ev.get("success"):
                add_undesirable(_fenced(code), model_id,
                                str(ev.get("error", "evaluation_failed"))[:300], 0.0)
                # Distinguish compile-ish vs runtime by error text (best-effort).
                err = str(ev.get("error", "")).lower()
                if any(k in err for k in ("verification", "compile", "syntax", "import", "missing")):
                    n_und_compile += 1
                else:
                    n_und_runtime += 1
                continue

            accuracy = ev.get("accuracy")
            try:
                accuracy = float(accuracy) if accuracy is not None else 0.0
            except (TypeError, ValueError):
                accuracy = 0.0
            accuracies.append(accuracy)

            if accuracy < eff_threshold:
                add_undesirable(_fenced(code), model_id, "low_accuracy", accuracy)
                n_und_lowacc += 1
                continue

            # passed the accuracy bar. Without the similarity penalty, exact
            # duplicates of already-accepted designs are skipped (legacy dedup).
            # WITH the similarity penalty (sim_index set), non-novel passers are
            # NOT skipped — they enter as desirable carrying a penalty; novelty
            # stays a reported metric only.
            is_novel = self.novelty_checker.is_novel(code, model_id) if self.novelty_check else True
            if is_novel:
                n_desirable += 1
            else:
                n_not_novel += 1
                if self.sim_index is None:
                    continue  # legacy: drop the duplicate
            add_desirable(code, model_id, accuracy)
            self.novelty_checker.mark_as_seen(code, model_id, source=f"cycle_{cycle}")

        # Persist caches + novelty.
        _write_jsonl(self.desirable, self.desirable_cache_file)
        _write_jsonl(self.undesirable, self.undesirable_cache_file)
        self.novelty_checker.save_cache()

        n_und_total = (n_und_compile + n_und_runtime + n_und_lowacc
                       + n_und_unparseable)
        best_acc = max(accuracies) if accuracies else 0.0
        # Card-style avg: mean over models that cleared the threshold. Keep the
        # all-valid mean as a secondary field.
        passed_accs = [a for a in accuracies if a >= eff_threshold]
        avg_acc = (sum(passed_accs) / len(passed_accs)) if passed_accs else 0.0
        avg_acc_all = (sum(accuracies) / len(accuracies)) if accuracies else 0.0

        stats = {
            "new_desirable": n_desirable,
            "new_undesirable": n_und_total,
            "undesirable_breakdown": {
                "non_compiling": n_und_compile,
                "runtime_error": n_und_runtime,
                "low_accuracy": n_und_lowacc,
                "unparseable": n_und_unparseable,
            },
            "not_novel_skipped": n_not_novel,
            "sim_penalty_mean": (sum(cycle_penalties) / len(cycle_penalties)) if cycle_penalties else 0.0,
            "sim_penalty_nonzero": sum(1 for p in cycle_penalties if p > 0.0),
            "skipped_no_signal": n_skipped,
            "evaluated_accuracies": len(accuracies),
            "best_accuracy": best_acc,
            "avg_accuracy": avg_acc,
            "avg_accuracy_all": avg_acc_all,
            "effective_threshold": eff_threshold,
            "desirable_total": len(self.desirable),
            "undesirable_total": len(self.undesirable),
        }
        logger.info("")
        logger.info(f"[cycle {cycle}] BUCKETING:")
        logger.info(f"  + desirable (passed)     : {n_desirable}")
        logger.info(f"  - undesirable (total)    : {n_und_total}")
        logger.info(f"      non-compiling        : {n_und_compile}")
        logger.info(f"      runtime error        : {n_und_runtime}")
        logger.info(f"      low accuracy         : {n_und_lowacc}")
        logger.info(f"      unparseable          : {n_und_unparseable}")
        logger.info(f"  ~ not novel (skipped)    : {n_not_novel}")
        logger.info(f"  skipped (no signal)      : {n_skipped}")
        logger.info(f"  best / avg(>=thr) / avg(all): {best_acc*100:.2f}% / "
                    f"{avg_acc*100:.2f}% / {avg_acc_all*100:.2f}%")
        logger.info(f"  effective threshold      : {eff_threshold*100:.2f}% ({self.threshold_mode})")
        logger.info(f"  cumulative desirable     : {len(self.desirable)}")
        logger.info(f"  cumulative undesirable   : {len(self.undesirable)}")
        return stats

    # ── stage 4: build the balanced KTO dataset ─────────────────────────────────

    def build_kto_dataset(self, cycle: int) -> Tuple[Optional[Path], Dict[str, Any]]:
        cycle_dir = self._cycle_dir(cycle)
        kto_file = cycle_dir / "kto_train.jsonl"

        desirables = list(self.desirable)
        undesirables = list(self.undesirable)
        D = len(desirables)
        Ua = len(undesirables)

        # Cap undesirables to keep KTO balanced; keep the most-recent (sharper)
        # negatives.  Also respect the absolute total cap.
        max_u = math.floor(D * self.undesirable_ratio) if D > 0 else Ua
        max_u = min(max_u, self.max_undesirable_total)
        if Ua > max_u and max_u >= 0:
            selected_u = undesirables[-max_u:] if max_u > 0 else []
        else:
            selected_u = undesirables

        combined = [
            {"prompt_messages": r["prompt_messages"], "completion": r["completion"],
             "label": True, "sim_penalty": float(r.get("sim_penalty", 0.0)),
             "_meta": r.get("_meta", {})}
            for r in desirables
        ] + [
            {"prompt_messages": r["prompt_messages"], "completion": r["completion"],
             "label": False, "sim_penalty": 0.0, "_meta": r.get("_meta", {})}
            for r in selected_u
        ]
        _write_jsonl(combined, kto_file)

        stats = {
            "desirable_used": D,
            "undesirable_available": Ua,
            "undesirable_used": len(selected_u),
            "total": len(combined),
            "kto_file": str(kto_file),
        }
        logger.info("")
        logger.info(f"[cycle {cycle}] KTO dataset: {D} desirable + "
                    f"{len(selected_u)}/{Ua} undesirable = {len(combined)} examples → {kto_file}")

        if D == 0 or len(selected_u) == 0 or len(combined) < self.min_train_examples:
            logger.warning(
                f"[cycle {cycle}] insufficient/one-sided data "
                f"(desirable={D}, undesirable={len(selected_u)}, total={len(combined)}); "
                f"KTO needs both classes and ≥{self.min_train_examples} examples — "
                "skipping training this cycle, will accumulate more next cycle."
            )
            return None, stats
        return kto_file, stats

    # ── stage 5: KTO fine-tune ──────────────────────────────────────────────────

    def run_kto_training(self, cycle: int, kto_file: Path,
                         desirable_weight: Optional[float] = None,
                         undesirable_weight: Optional[float] = None) -> Dict[str, Any]:
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"CYCLE {cycle}: KTO FINE-TUNING")
        logger.info("=" * 80)

        dw = desirable_weight if desirable_weight is not None else self.kto_desirable_weight
        uw = undesirable_weight if undesirable_weight is not None else self.kto_undesirable_weight
        logger.info(f"[cycle {cycle}] class weights: desirable={dw:.3f}, undesirable={uw:.3f} "
                    f"({self.class_weight_mode})")

        checkpoint_dir = self._checkpoint_dir(cycle)
        if checkpoint_dir.exists() and (checkpoint_dir / "adapter_config.json").exists():
            logger.info(f"[cycle {cycle}] checkpoint already exists, skipping: {checkpoint_dir}")
            return {"success": True, "checkpoint_dir": str(checkpoint_dir), "skipped": True}

        prev_adapter = self._prev_adapter(cycle)

        cmd = [
            sys.executable, "-u", "-m", "ab.gpt.TuneNNGenKTO",
            "--llm_conf", self.llm_conf,
            "--kto_data_file", str(kto_file),
            "--kto_checkpoint_dir", str(checkpoint_dir),
            "--kto_beta", str(self.kto_beta),
            "--kto_desirable_weight", str(dw),
            "--kto_undesirable_weight", str(uw),
            "--num_train_epochs", str(self.num_train_epochs),
            "--max_prompt_length", str(self.kto_max_prompt_length),
            "--max_completion_length", str(self.kto_max_completion_length),
            # KTO drift control (see __init__).
            "--learning_rate", str(self.kto_learning_rate),
            "--r", str(self.kto_lora_r),
            "--lora_alpha", str(self.kto_lora_alpha),
            "--max_grad_norm", str(self.kto_max_grad_norm),
        ]
        if self.sim_penalty:
            cmd.extend(["--sim_alpha", str(self.sim_alpha)])
        if prev_adapter is not None:
            logger.info(f"[cycle {cycle}] warm-starting from previous adapter: {prev_adapter}")
            cmd.extend(["--peft", str(prev_adapter)])
        else:
            logger.info(f"[cycle {cycle}] training fresh LoRA on base {self.base_model}")

        start = time.time()
        try:
            self._run_subprocess(cmd, f"KTO training cycle {cycle}")
        except subprocess.CalledProcessError as e:
            return {"success": False, "error": f"kto_training_failed: exit_{e.returncode}",
                    "training_time_minutes": (time.time() - start) / 60}

        minutes = (time.time() - start) / 60
        if not (checkpoint_dir / "adapter_config.json").exists():
            logger.error(f"[cycle {cycle}] training finished but no adapter at {checkpoint_dir}")
            return {"success": False, "error": "checkpoint_missing",
                    "training_time_minutes": minutes}

        logger.info(f"[cycle {cycle}] KTO training complete in {minutes:.1f} min → {checkpoint_dir}")
        return {"success": True, "checkpoint_dir": str(checkpoint_dir),
                "training_time_minutes": minutes,
                "desirable_weight": dw, "undesirable_weight": uw}

    # ── orchestration ───────────────────────────────────────────────────────────

    def run_cycle(self, cycle: int) -> Dict[str, Any]:
        t0 = time.time()
        logger.info("")
        logger.info("#" * 80)
        logger.info(f"# CYCLE {cycle} / {self.cycles}")
        logger.info("#" * 80)

        nneval_dir, gen_records = self.generate_models(cycle)
        self._prefilter_novelty(cycle, nneval_dir, gen_records)
        eval_by_id = self.evaluate_models(cycle, nneval_dir)
        # Release any CUDA memory the in-process evaluator held before the KTO
        # training subprocess loads the 7B model.
        self._free_gpu()
        bucket_stats = self.bucket_models(cycle, nneval_dir, gen_records, eval_by_id)
        kto_file, ds_stats = self.build_kto_dataset(cycle)

        if kto_file is not None:
            dw, uw = self._class_weights(ds_stats)
            train_stats = self.run_kto_training(cycle, kto_file, dw, uw)
        else:
            train_stats = {"success": False, "skipped": True, "reason": "insufficient_data"}

        result = {
            "cycle": cycle,
            "timestamp": datetime.now().isoformat(),
            "generated": len(gen_records),
            "extractable": sum(1 for r in gen_records if r.get("ok")),
            "bucketing": bucket_stats,
            "dataset": ds_stats,
            "training": train_stats,
            "cycle_time_minutes": (time.time() - t0) / 60,
        }
        save_cycle_results(result, self._cycle_dir(cycle) / "metrics.json")
        return result

    def run(self) -> Dict[str, Any]:
        start_cycle = self.resume_from_cycle or 1
        for cycle in range(start_cycle, self.cycles + 1):
            try:
                result = self.run_cycle(cycle)
                self.cycle_results.append(result)
            except Exception as e:  # noqa: BLE001 — keep the experiment alive
                import traceback
                logger.error(f"[cycle {cycle}] FAILED: {type(e).__name__}: {e}")
                logger.error(traceback.format_exc())
                self.cycle_results.append({"cycle": cycle, "error": str(e)})
            self._save_aggregate()

        logger.info("")
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 80)
        self._save_aggregate()
        return {"cycles": self.cycle_results}

    def _save_aggregate(self) -> None:
        agg = {
            "base_model": self.base_model,
            "accuracy_threshold": self.accuracy_threshold,
            "kto_beta": self.kto_beta,
            "desirable_total": len(self.desirable),
            "undesirable_total": len(self.undesirable),
            "cycles": self.cycle_results,
        }
        save_cycle_results(agg, self.output_dir / "all_cycles_results.json")

    # ── subprocess runner with retry ────────────────────────────────────────────

    def _run_subprocess(self, cmd: List[str], name: str) -> None:
        logger.info(f"Running {name}: {' '.join(cmd)}")
        # Force MKL to use GNU OpenMP.  The KTO trainer subprocess's import order
        # (torch + datasets + trl + peft) otherwise trips:
        #   "MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1"
        # and exits before training starts.  Harmless for the generator subprocess.
        env = os.environ.copy()
        env["MKL_THREADING_LAYER"] = "GNU"
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            result = subprocess.run(cmd, capture_output=False, text=True, check=False, env=env)
            if result.returncode == 0:
                return
            last_exc = subprocess.CalledProcessError(result.returncode, cmd)
            logger.error(f"{name} failed (attempt {attempt}/{self.max_retries}, "
                         f"exit {result.returncode})")
            if attempt < self.max_retries:
                time.sleep(15.0 * attempt)
        assert last_exc is not None
        raise last_exc


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-contained iterative KTO fine-tuning (no initial dataset)"
    )
    parser.add_argument("--llm_conf", type=str, default="nngpt_unique_arch_rag.json",
                        help="LLM config JSON (default: nngpt_unique_arch_rag.json → ABrain model)")
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--models_per_cycle", type=int, default=30)
    parser.add_argument("--accuracy_threshold", type=float, default=0.50,
                        help="First-epoch CIFAR-10 accuracy a model must clear to be desirable "
                             "(ABrain's first-epoch ceiling is ~0.68; keep this well below it)")
    parser.add_argument("--novelty_check", action="store_true", default=True)
    parser.add_argument("--no_novelty_check", dest="novelty_check", action="store_false")
    parser.add_argument("--eval_skip", dest="eval_skip", action="store_true", default=True,
                        help="Skip eval of duplicate (non-novel) generations to save GPU")
    parser.add_argument("--no_eval_skip", dest="eval_skip", action="store_false",
                        help="Evaluate ALL generations; novelty stays a reported metric only")
    parser.add_argument("--undesirable_ratio", type=float, default=1.0)
    parser.add_argument("--max_undesirable_total", type=int, default=1000)
    parser.add_argument("--min_train_examples", type=int, default=10)

    # curriculum threshold (#1)
    parser.add_argument("--threshold_mode", type=str, default="fixed",
                        choices=["fixed", "quantile", "linear"])
    parser.add_argument("--threshold_quantile", type=float, default=0.6)
    parser.add_argument("--threshold_ceil", type=float, default=0.62)
    parser.add_argument("--threshold_slope", type=float, default=0.01)
    # dynamic class weights (#2)
    parser.add_argument("--class_weight_mode", type=str, default="fixed",
                        choices=["fixed", "auto"])
    parser.add_argument("--class_weight_target", type=float, default=1.0)

    # similarity penalty (#3)
    parser.add_argument("--sim_penalty", action="store_true", default=False,
                        help="Penalize near-duplicate generations in the KTO reward "
                             "(vs LEMUR DB + own prior gens); non-novel passers kept as desirable")
    parser.add_argument("--sim_alpha", type=float, default=0.4)
    parser.add_argument("--sim_threshold", type=float, default=0.85)
    parser.add_argument("--sim_shape", type=str, default="exponential",
                        choices=["exponential", "linear"])
    parser.add_argument("--sim_db_task", type=str, default="img-classification")
    parser.add_argument("--sim_db_dataset", type=str, default="cifar-10")

    parser.add_argument("--kto_beta", type=float, default=0.1)
    parser.add_argument("--kto_desirable_weight", type=float, default=1.0)
    parser.add_argument("--kto_undesirable_weight", type=float, default=1.0)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--kto_learning_rate", type=float, default=5e-6)
    parser.add_argument("--kto_lora_r", type=int, default=16)
    parser.add_argument("--kto_lora_alpha", type=int, default=16)
    parser.add_argument("--kto_max_grad_norm", type=float, default=0.3)
    parser.add_argument("--max_prompt_length", type=int, default=1536,
                        help="Prompt token budget; completion gets context_length - this")

    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--gen_max_rejections", type=int, default=5,
                        help="Regenerate a model up to N times if it fails the "
                             "structural pre-check (instantiate + forward on 3x32x32)")

    parser.add_argument("--dataset", type=str, default="cifar-10")
    parser.add_argument("--eval_transform", type=str, default="norm_256_flip",
                        help="Card/ab.nn protocol = norm_256_flip (256x256)")
    parser.add_argument("--eval_lr", type=float, default=0.01)
    parser.add_argument("--eval_momentum", type=float, default=0.9)
    parser.add_argument("--eval_batch", type=int, default=10,
                        help="Card/ab.nn protocol = 10")
    parser.add_argument("--eval_dropout", type=float, default=0.2)
    parser.add_argument("--eval_train_epochs", type=int, default=1)
    parser.add_argument("--params_limit", type=int, default=500_000)
    parser.add_argument("--save_to_db", action="store_true", default=False)

    parser.add_argument("--output_subdir", type=str, default="kto_selfcontained")
    parser.add_argument("--resume_from_cycle", type=int, default=None)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=43)

    args = parser.parse_args()

    pipeline = SelfContainedKTOPipeline(
        llm_conf=args.llm_conf,
        cycles=args.cycles,
        models_per_cycle=args.models_per_cycle,
        accuracy_threshold=args.accuracy_threshold,
        novelty_check=args.novelty_check,
        eval_skip=args.eval_skip,
        undesirable_ratio=args.undesirable_ratio,
        max_undesirable_total=args.max_undesirable_total,
        min_train_examples=args.min_train_examples,
        threshold_mode=args.threshold_mode,
        threshold_quantile=args.threshold_quantile,
        threshold_ceil=args.threshold_ceil,
        threshold_slope=args.threshold_slope,
        class_weight_mode=args.class_weight_mode,
        class_weight_target=args.class_weight_target,
        sim_penalty=args.sim_penalty,
        sim_alpha=args.sim_alpha,
        sim_threshold=args.sim_threshold,
        sim_shape=args.sim_shape,
        sim_db_task=args.sim_db_task,
        sim_db_dataset=args.sim_db_dataset,
        kto_beta=args.kto_beta,
        kto_desirable_weight=args.kto_desirable_weight,
        kto_undesirable_weight=args.kto_undesirable_weight,
        num_train_epochs=args.num_train_epochs,
        kto_learning_rate=args.kto_learning_rate,
        kto_lora_r=args.kto_lora_r,
        kto_lora_alpha=args.kto_lora_alpha,
        kto_max_grad_norm=args.kto_max_grad_norm,
        max_prompt_length=args.max_prompt_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        gen_max_rejections=args.gen_max_rejections,
        dataset=args.dataset,
        eval_transform=args.eval_transform,
        eval_lr=args.eval_lr,
        eval_momentum=args.eval_momentum,
        eval_batch=args.eval_batch,
        eval_dropout=args.eval_dropout,
        eval_train_epochs=args.eval_train_epochs,
        params_limit=args.params_limit,
        save_to_db=args.save_to_db,
        output_subdir=args.output_subdir,
        resume_from_cycle=args.resume_from_cycle,
        max_retries=args.max_retries,
        seed=args.seed,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
