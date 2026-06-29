#!/usr/bin/env python


import json
import logging
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from ab.gpt.iterative_finetune import IterativeFinetuner, logger
from ab.gpt.iterative_pipeline.novelty_checker import NoveltyChecker
from ab.gpt.iterative_pipeline.pipeline_validation import (
    RetryHandler,
    StageValidator,
)
from ab.gpt.iterative_pipeline.gpu_memory_manager import (
    check_gpu_memory,
    clear_gpu_cache,
    ensure_gpu_memory,
    get_gpu_memory_info,
    kill_gpu_processes,
)
from ab.gpt.kto_pipeline.kto_data_manager import KTODataManager
from ab.gpt.TuneNNGen import get_pipeline_defaults
from ab.gpt.util.Const import nngpt_dir
from ab.nn.util.Const import out_dir


class KTOIterativeFinetuner(IterativeFinetuner):
    """KTO version of the iterative fine-tuning pipeline."""

    def __init__(
        self,
        llm_conf: str,
        cycles: int = 21,
        models_per_cycle: int = 150,
        samples_per_prompt: int = 1,
        accuracy_threshold: float = 0.40,
        min_selected_k: int = 15,
        fallback_threshold: float = 0.35,
        adaptive_threshold: bool = False,
        novelty_check: bool = True,
        resume_from_cycle: Optional[int] = None,
        max_retries: int = 3,
        use_optimized_training: bool = True,
        num_train_epochs: int = 5,
        # ── KTO-specific knobs ────────────────────────────────────────────
        undesirable_floor_accuracy: float = 0.10,
        undesirable_ratio: float = 1.0,
        max_undesirable_per_cycle: int = 500,
        kto_beta: float = 0.1,
        kto_desirable_weight: float = 1.0,
        kto_undesirable_weight: float = 1.0,
        kto_output_subdir: str = "kto_pipeline",
    ):
        # 1. Parent init sets up curation data, base SFT data_manager, logging,
        #    novelty checker — all using out_dir/curation_output as the working
        #    directory.  We accept those side-effects (they're idempotent) and
        #    then redirect output to the KTO-specific tree.
        super().__init__(
            llm_conf=llm_conf,
            cycles=cycles,
            models_per_cycle=models_per_cycle,
            samples_per_prompt=samples_per_prompt,
            accuracy_threshold=accuracy_threshold,
            min_selected_k=min_selected_k,
            fallback_threshold=fallback_threshold,
            adaptive_threshold=adaptive_threshold,
            novelty_check=novelty_check,
            resume_from_cycle=resume_from_cycle,
            max_retries=max_retries,
            use_optimized_training=use_optimized_training,
            num_train_epochs=num_train_epochs,
        )

        # 2. Override output_dir → nngpt_dir/<kto_output_subdir>
        # Must be under nngpt_dir so NNEval.py's relative_to(nngpt_dir) call
        # in _collect_epoch_requests() does not raise ValueError.
        kto_output = nngpt_dir / kto_output_subdir
        kto_output.mkdir(parents=True, exist_ok=True)
        self.output_dir = kto_output

        
        shared_log_marker = str(out_dir / "curation_output" / "iterative_pipeline.log")
        for h in list(logger.handlers):
            base = getattr(h, "baseFilename", None)
            if base and Path(base) == Path(shared_log_marker):
                logger.removeHandler(h)
                try:
                    h.close()
                except Exception:
                    pass

        kto_log_file = self.output_dir / "iterative_pipeline.log"
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        # Avoid duplicate handlers if __init__ is called twice
        existing_files = {
            getattr(h, "baseFilename", None) for h in logger.handlers
        }
        if str(kto_log_file) not in existing_files:
            fh = logging.FileHandler(kto_log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        logger.info(f"[KTO] Output directory: {self.output_dir}")

        # 4. Re-init novelty checker with a KTO-specific cache so we don't
        #    cross-contaminate the SFT run's seen-models set.
        self.novelty_checker = NoveltyChecker(self.output_dir / "seen_models.json")
        if self.novelty_check_enabled:
            self._initialize_novelty_checker()

        # 5. Replace data_manager with KTO-aware version.  Same interface
        #    (augment_training_data signature is identical), so run_cycle in
        #    the parent class works unchanged.
        self.data_manager = KTODataManager(
            base_data_dir=str(self.base_data_dir),
            undesirable_cache_file=self.output_dir / "kto_undesirable_cache.jsonl",
            undesirable_ratio=undesirable_ratio,
            max_undesirable_per_cycle=max_undesirable_per_cycle,
        )

        # 6. KTO hyperparameters
        self.undesirable_floor_accuracy = undesirable_floor_accuracy
        self.undesirable_ratio = undesirable_ratio
        self.max_undesirable_per_cycle = max_undesirable_per_cycle
        self.kto_beta = kto_beta
        self.kto_desirable_weight = kto_desirable_weight
        self.kto_undesirable_weight = kto_undesirable_weight
        self.kto_output_subdir = kto_output_subdir

       
        import os
        seed_env = os.environ.get("KTO_SEED_CHECKPOINT", "").strip()
        self.kto_seed_checkpoint = (
            Path(seed_env) if seed_env else (out_dir / "qlora-sft" / "final")
        )

        # 7. Tracks which cycle is currently in flight (set by run_cycle).
        #    Needed by filter_successful_novel to tag undesirables.
        self._current_cycle: int = 0

        logger.info(
            f"[KTO] beta={self.kto_beta}, "
            f"desirable_weight={self.kto_desirable_weight}, "
            f"undesirable_weight={self.kto_undesirable_weight}, "
            f"undesirable_floor={self.undesirable_floor_accuracy:.2f}, "
            f"undesirable_ratio={self.undesirable_ratio}"
        )

    # ── lightweight overrides ────────────────────────────────────────────

    def run_cycle(self, cycle: int) -> Dict[str, Any]:
        """Track current cycle so filter_successful_novel can tag undesirables."""
        self._current_cycle = cycle
        return super().run_cycle(cycle)

    def filter_successful_novel(
        self,
        generation_results: Dict[str, Any],
        evaluation_results: Dict[str, Any],
        starting_checksum: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Inherit the parent's selection logic for desirables, then ALSO
        scan the same evaluation result set for undesirables and record
        them in the KTODataManager's accumulated cache.
        """
        selected = super().filter_successful_novel(
            generation_results, evaluation_results, starting_checksum
        )
        selected_ids = {m["model_id"] for m in selected}

        # Build a model_id → eval_info map (covers BOTH success=True and False)
        eval_models = evaluation_results.get("models", [])
        eval_by_id: Dict[str, Dict[str, Any]] = {
            m["model_id"]: m for m in eval_models if "model_id" in m
        }

        # Locate this cycle's nneval directory so we can read code for
        # models whose eval entry doesn't carry a code_file field.
        cycle_nneval_dir = self.output_dir / f"cycle_{self._current_cycle}" / "nneval"

        results_list = generation_results.get("results", [])
        n_recorded = 0
        n_recorded_extraction_fail = 0
        n_skipped_no_code = 0
        n_skipped_no_eval = 0

        def _read_first_existing(*candidates: Optional[str]) -> Optional[str]:
            for c in candidates:
                if not c:
                    continue
                p = Path(c)
                if p.exists():
                    try:
                        return p.read_text(encoding="utf-8", errors="replace")
                    except Exception:
                        continue
            return None

        def _is_error_stub(text: str) -> bool:
            """
            The generator writes a tiny stub like "# No code generated\n# Error: ..."
            when the model itself crashed before producing any tokens (typically a
            CUDA device-mismatch, OOM, or NaN).  These contain zero architectural
            signal — recording them as KTO undesirables teaches the policy nothing
            useful and pollutes the cache.  Reject by both prefix and total bulk.
            """
            stripped = text.strip()
            if stripped.startswith("# No code generated"):
                return True
            # Heuristic: a real (failed) generation always contains at least one
            # python keyword (class / def / import).  Pure error-stub text won't.
            if len(stripped) < 100 and not any(
                kw in stripped for kw in ("class ", "def ", "import ", "nn.Module")
            ):
                return True
            return False

        def _extract_acc_from_eval_info(info: Dict[str, Any]) -> Optional[float]:
            """Pull accuracy out of a per-model eval_info.json payload."""
            results = info.get("eval_results") if isinstance(info, dict) else None
            if not isinstance(results, dict):
                return None
            acc = results.get("accuracy", results.get("acc"))
            try:
                return float(acc) if acc is not None else None
            except (TypeError, ValueError):
                return None

        for i, result in enumerate(results_list):
            model_id = f"gen_{starting_checksum + i:04d}"
            if model_id in selected_ids:
                continue  # already a desirable

            
            if not result.get("ok", False):
                # Prefer raw_output (actual model text, even if unparseable)
                # over fail_code (which can be a "# No code generated" stub
                # when the model crashed before emitting any tokens).
                bad_text = _read_first_existing(
                    result.get("raw_output"),
                    result.get("fail_code"),
                )
                if bad_text is None or not bad_text.strip():
                    n_skipped_no_code += 1
                    continue

                if _is_error_stub(bad_text):
                    # Crash-before-generation: nothing the policy can learn from.
                    n_skipped_no_code += 1
                    continue

                metadata = {
                    "accuracy": 0.0,
                    "params": 0,
                    "cycle": self._current_cycle,
                    "checkpoint": "current",
                }
                failure_reason = result.get("error") or "code_extraction_failed"
                if self.data_manager.record_undesirable(
                    model_id, failure_reason, bad_text, metadata
                ):
                    n_recorded += 1
                    n_recorded_extraction_fail += 1
                else:
                    n_skipped_no_code += 1
                continue

            # ── Path B: ok=True but not selected → eval-based undesirable ────
            eval_info = eval_by_id.get(model_id)
            if eval_info is None:
                
                model_eval_dir = cycle_nneval_dir / model_id
                err_file = model_eval_dir / "error.txt"
                info_file = model_eval_dir / "eval_info.json"
                if err_file.exists():
                    err_text = _read_first_existing(str(err_file)) or "evaluation_failed"
                    eval_info = {"success": False, "error": err_text.strip()[:500]}
                elif info_file.exists():
                    try:
                        info = json.loads(info_file.read_text())
                        acc = _extract_acc_from_eval_info(info)
                        eval_info = {"success": True, "accuracy": acc}
                    except Exception:
                        eval_info = None
                if eval_info is None:
                    # Genuinely no verdict for this model (never reached by the
                    # evaluator before it died) — can't classify it.
                    n_skipped_no_eval += 1
                    continue

            success = eval_info.get("success", False)
            accuracy = eval_info.get("accuracy")

            
            code_file_str = eval_info.get("code_file") or ""
            code_path = Path(code_file_str) if code_file_str else None
            if code_path is None or not code_path.exists():
                code_path = cycle_nneval_dir / model_id / "new_nn.py"
            if not code_path.exists():
                gen_file = result.get("file")
                if gen_file:
                    code_path = Path(gen_file)

            if not code_path.exists():
                n_skipped_no_code += 1
                continue

            try:
                code = code_path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                logger.warning(f"[KTO] Failed to read code for {model_id}: {e}")
                n_skipped_no_code += 1
                continue

            # Decide failure reason for the undesirable label
            if not success:
                failure_reason = eval_info.get("error") or "evaluation_failed"
            elif accuracy is not None and accuracy < self.undesirable_floor_accuracy:
                failure_reason = "low_accuracy"
            else:
                # success=True, accuracy >= floor, but not selected → likely
                # rejected for novelty.  Treat as a soft-negative example
                # (still a real generation we want to weight low).
                failure_reason = "not_novel_or_below_threshold"

            metadata = {
                "accuracy": accuracy if accuracy is not None else 0.0,
                "params": result.get("params", 0),
                "cycle": self._current_cycle,
                "checkpoint": "current",
            }

            if self.data_manager.record_undesirable(
                model_id, failure_reason, code, metadata
            ):
                n_recorded += 1

        self.data_manager.save_undesirable_cache()

        logger.info(
            f"[KTO][cycle {self._current_cycle}] "
            f"undesirables recorded this cycle: {n_recorded} "
            f"(of which extraction-failures={n_recorded_extraction_fail}, "
            f"skipped_no_code={n_skipped_no_code}, skipped_no_eval={n_skipped_no_eval}); "
            f"cumulative cache size: {len(self.data_manager._accumulated_undesirable)}"
        )

        return selected

    

    def run_finetuning(self, cycle: int, data_dir: Path) -> Dict[str, Any]:
        """
        Run KTO fine-tuning for one cycle by spawning
        `python -m ab.gpt.TuneNNGenKTO --kto_data_file <path/kto_train.jsonl> ...`
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"CYCLE {cycle}: FINE-TUNING (KTO)")
        logger.info("=" * 80)
        logger.info(f"Training data: {data_dir}")

        import os
        if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            logger.info("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

        logger.info("Clearing GPU cache before fine-tuning...")
        clear_gpu_cache()
        current_pid = str(os.getpid())
        kill_gpu_processes(exclude_pids=[current_pid])
        clear_gpu_cache()

        has_memory, msg = check_gpu_memory(min_free_gb=8.0)
        if not has_memory:
            logger.warning(f"GPU memory warning: {msg}")
            if not ensure_gpu_memory(min_free_gb=8.0, aggressive=True):
                logger.error("Insufficient GPU memory for fine-tuning.")
                return {
                    "success": False,
                    "error": f"insufficient_gpu_memory: {msg}",
                    "training_time_minutes": 0,
                }

        # Skip if this cycle's checkpoint already exists & validates
        isolated_checkpoint = self.output_dir / f"cycle_{cycle}" / "checkpoint"
        if isolated_checkpoint.exists():
            is_valid, msg = StageValidator.validate_finetuning_output(isolated_checkpoint)
            if is_valid:
                logger.info(f"✓ Existing checkpoint validated: {isolated_checkpoint}")
                return {
                    "success": True,
                    "checkpoint_dir": str(isolated_checkpoint),
                    "training_time_minutes": 0.0,
                    "skipped": True,
                }
            logger.warning(f"Existing checkpoint failed validation: {msg} — re-training")

        total, used, free = get_gpu_memory_info()
        logger.info(f"GPU memory before fine-tuning: {free:.2f}GB free / {total:.2f}GB total")

        ft_output_dir = self.output_dir / f"cycle_{cycle}" / "finetuning_output"
        ft_output_dir.mkdir(parents=True, exist_ok=True)

        # The KTO file lives inside the data_dir as kto_train.jsonl
        # (written by KTODataManager.augment_training_data — or by
        # KTODataManager._build_initial_kto_file for cycle 1).
        kto_train_file = Path(data_dir) / "kto_train.jsonl"
        if not kto_train_file.exists():
            logger.error(f"KTO training file not found: {kto_train_file}")
            return {
                "success": False,
                "error": f"kto_train_file_missing: {kto_train_file}",
                "training_time_minutes": 0,
            }

        # Per-run checkpoint dir avoids collision when 1.3B and 7B run concurrently.
        # kto_output_subdir is e.g. "kto_pipeline_1p3b" or "kto_pipeline_7b".
        kto_ckpt_dir = out_dir / "qlora-kto" / self.kto_output_subdir / "final"

        # KTO  ←── changed: spawn TuneNNGenKTO, not TuneNNGen
        cmd = [
            sys.executable, "-m", "ab.gpt.TuneNNGenKTO",
            "--llm_conf", self.llm_conf,
            "--kto_data_file", str(kto_train_file),
            "--kto_checkpoint_dir", str(kto_ckpt_dir),
            "--kto_beta", str(self.kto_beta),
            "--kto_desirable_weight", str(self.kto_desirable_weight),
            "--kto_undesirable_weight", str(self.kto_undesirable_weight),
        ]

        # Choose the warm-start adapter (--peft) to merge before attaching the
        # fresh KTO LoRA:
        #   cycle 1  → the SFT seed checkpoint (format-teacher; see __init__).
        #   cycle 2+ → the previous cycle's KTO checkpoint (continual learning).
        if cycle == 1:
            if self.kto_seed_checkpoint and Path(self.kto_seed_checkpoint).exists():
                logger.info(
                    f"[KTO] Cycle 1 warm-start from SFT seed checkpoint: "
                    f"{self.kto_seed_checkpoint}"
                )
                cmd.extend(["--peft", str(self.kto_seed_checkpoint)])
            else:
                logger.warning(
                    "[KTO] No SFT seed checkpoint found at "
                    f"{self.kto_seed_checkpoint}. Cycle 1 will KTO-train from RAW "
                    "BASE, which usually cannot learn the LEMUR Net(in_shape,...) "
                    "format — expect generated models to crash at evaluation "
                    "(e.g. Conv2d(1,...) channel mismatches). Run the SFT pipeline "
                    "first or set KTO_SEED_CHECKPOINT to a format-trained adapter."
                )
        else:
            prev_checkpoint = self.output_dir / f"cycle_{cycle - 1}" / "checkpoint"
            if prev_checkpoint.exists():
                logger.info(f"Loading previous cycle checkpoint: {prev_checkpoint}")
                cmd.extend(["--peft", str(prev_checkpoint)])
            else:
                logger.warning(f"Previous cycle checkpoint not found: {prev_checkpoint}")

        if self.use_optimized_training:
            d = get_pipeline_defaults()
           
            kto_learning_rate = 5e-6
            kto_lora_r = 16
            kto_lora_alpha = 16
            kto_max_grad_norm = 0.3
            cmd.extend([
                "--per_device_train_batch_size", "2",  # KTO requires actual batch > 1 for KL estimation
                "--learning_rate", str(kto_learning_rate),
                "--r", str(kto_lora_r),
                "--lora_alpha", str(kto_lora_alpha),
                "--weight_decay", str(d["weight_decay"]),
                "--warmup_steps", str(d["warmup_steps"]),
                "--num_train_epochs", str(self.num_train_epochs),
                "--logging_steps", str(d["logging_steps"]),
                "--max_grad_norm", str(kto_max_grad_norm),
                "--target_modules", d["target_modules"],
                "--max_new_tokens", str(d["max_new_tokens"]),
                "--temperature", str(d["temperature"]),
                "--top_k", str(d["top_k"]),
                "--evaluation_strategy", d["evaluation_strategy"],
                "--eval_steps", str(d["eval_steps"]),
                "--per_device_eval_batch_size", str(d["per_device_eval_batch_size"]),
                "--save_strategy", d["save_strategy"],
                "--save_steps", str(d["save_steps"]),
                "--save_total_limit", str(d["save_total_limit"]),
                "--metric_for_best_model", d["metric_for_best_model"],
            ])
            if d["load_best_model_at_end"]:
                cmd.append("--load_best_model_at_end")
            logger.info("Using optimized training configuration")
        else:
            logger.info("Using default training configuration")

        logger.info(f"Running fine-tuning: {' '.join(cmd)}")
        start_time = time.time()

        def _run_ft_cmd():
            clear_gpu_cache()
            result = subprocess.run(cmd, capture_output=False, text=True, check=False)
            if result.returncode != 0:
                logger.error(f"KTO fine-tuning failed with exit code {result.returncode}")
                clear_gpu_cache()
                raise subprocess.CalledProcessError(result.returncode, cmd)
            return result

        try:
            RetryHandler.retry_with_backoff(
                _run_ft_cmd,
                max_retries=self.max_retries,
                initial_delay=30.0,
                backoff_factor=2.0,
                exceptions=(subprocess.CalledProcessError, OSError),
                operation_name=f"KTO fine-tuning cycle {cycle}",
            )
        except subprocess.CalledProcessError as e:
            training_time = time.time() - start_time
            logger.error(f"KTO fine-tuning failed after {self.max_retries} retries (exit code {e.returncode})")
            return {
                "success": False,
                "error": f"kto_fine_tuning_failed: exit_code_{e.returncode}",
                "training_time_minutes": training_time / 60,
            }
        except Exception as e:
            training_time = time.time() - start_time
            logger.error(f"KTO fine-tuning failed with unexpected error: {e}")
            return {
                "success": False,
                "error": f"kto_fine_tuning_failed: {str(e)}",
                "training_time_minutes": training_time / 60,
            }

        training_time = time.time() - start_time
        logger.info(f"KTO fine-tuning completed in {training_time / 60:.1f} minutes")

        # KTO  ←── changed: TuneNNGenKTO saves to per-run qlora-kto/<subdir>/final
        source_checkpoint = kto_ckpt_dir
        if not source_checkpoint.exists():
            logger.error(f"Checkpoint directory not found: {source_checkpoint}")
            return {
                "success": False,
                "error": "checkpoint_not_found",
                "training_time_minutes": training_time / 60,
            }

        isolated_checkpoint = self.output_dir / f"cycle_{cycle}" / "checkpoint"
        isolated_checkpoint.mkdir(parents=True, exist_ok=True)

        def _copy_checkpoint():
            logger.info(f"Copying KTO checkpoint to: {isolated_checkpoint}")
            if isolated_checkpoint.exists():
                shutil.rmtree(isolated_checkpoint)
            shutil.copytree(source_checkpoint, isolated_checkpoint)
            return True

        try:
            RetryHandler.retry_with_backoff(
                _copy_checkpoint,
                max_retries=3,
                initial_delay=2.0,
                backoff_factor=1.5,
                exceptions=(OSError, shutil.Error),
                operation_name=f"Copy KTO checkpoint cycle {cycle}",
            )
        except Exception as e:
            logger.error(f"Failed to copy KTO checkpoint: {e}")
            return {
                "success": False,
                "error": f"checkpoint_copy_failed: {str(e)}",
                "training_time_minutes": training_time / 60,
            }

        is_valid, msg = StageValidator.validate_finetuning_output(isolated_checkpoint)
        if not is_valid:
            logger.error(f"KTO checkpoint validation failed: {msg}")
            return {
                "success": False,
                "error": f"checkpoint_validation_failed: {msg}",
                "training_time_minutes": training_time / 60,
            }

        logger.info(f"KTO checkpoint validated and saved to: {isolated_checkpoint}")

        return {
            "success": True,
            "checkpoint_dir": str(isolated_checkpoint),
            "source_checkpoint": str(source_checkpoint),
            "training_time_minutes": training_time / 60,
        }


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Iterative KTO Fine-Tuning Pipeline")
    parser.add_argument("--llm_conf", type=str, default="nngpt_unique_arch_rag.json",
                        help="LLM config JSON (default: nngpt_unique_arch_rag.json → ABrain model)")
    parser.add_argument("--cycles", type=int, default=21,
                        help="Number of fine-tuning cycles (default: 21)")
    parser.add_argument("--models_per_cycle", type=int, default=150,
                        help="Number of models to generate per cycle (default: 150)")
    parser.add_argument("--samples_per_prompt", type=int, default=1)
    parser.add_argument("--accuracy_threshold", type=float, default=0.40,
                        help="Min first-epoch accuracy to count as desirable (default: 0.40)")
    parser.add_argument("--min_selected_k", type=int, default=15)
    parser.add_argument("--fallback_threshold", type=float, default=0.35)
    parser.add_argument("--adaptive_threshold", action="store_true", default=False)
    parser.add_argument("--novelty_check", action="store_true", default=True)
    parser.add_argument("--no_novelty_check", dest="novelty_check", action="store_false")
    parser.add_argument("--resume_from_cycle", type=int, default=None)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--use_optimized_training", action="store_true", default=True)
    parser.add_argument("--no_optimized_training", dest="use_optimized_training", action="store_false")
    parser.add_argument("--num_train_epochs", type=int, default=5)

    # KTO-specific
    parser.add_argument("--undesirable_floor_accuracy", type=float, default=0.10,
                        help="Models with success=True but accuracy < this are also undesirables")
    parser.add_argument("--undesirable_ratio", type=float, default=1.0,
                        help="Target |undesirable| / |desirable| in the KTO dataset (default: 1.0)")
    parser.add_argument("--max_undesirable_per_cycle", type=int, default=500)
    parser.add_argument("--kto_beta", type=float, default=0.1,
                        help="KTO KL-regularisation strength (default: 0.1)")
    parser.add_argument("--kto_desirable_weight", type=float, default=1.0)
    parser.add_argument("--kto_undesirable_weight", type=float, default=1.0)
    parser.add_argument("--kto_output_subdir", type=str, default="kto_pipeline",
                        help="Output subdirectory under out_dir (default: kto_pipeline)")

    args = parser.parse_args()

    if args.resume_from_cycle is not None:
        if args.resume_from_cycle < 1 or args.resume_from_cycle > args.cycles:
            print(f"[ERROR] Invalid resume_from_cycle {args.resume_from_cycle}; "
                  f"must be 1..{args.cycles}")
            sys.exit(1)

    pipeline = KTOIterativeFinetuner(
        llm_conf=args.llm_conf,
        cycles=args.cycles,
        models_per_cycle=args.models_per_cycle,
        samples_per_prompt=args.samples_per_prompt,
        accuracy_threshold=args.accuracy_threshold,
        min_selected_k=args.min_selected_k,
        fallback_threshold=args.fallback_threshold,
        adaptive_threshold=args.adaptive_threshold,
        novelty_check=args.novelty_check,
        resume_from_cycle=args.resume_from_cycle,
        max_retries=args.max_retries,
        use_optimized_training=args.use_optimized_training,
        num_train_epochs=args.num_train_epochs,
        undesirable_floor_accuracy=args.undesirable_floor_accuracy,
        undesirable_ratio=args.undesirable_ratio,
        max_undesirable_per_cycle=args.max_undesirable_per_cycle,
        kto_beta=args.kto_beta,
        kto_desirable_weight=args.kto_desirable_weight,
        kto_undesirable_weight=args.kto_undesirable_weight,
        kto_output_subdir=args.kto_output_subdir,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
