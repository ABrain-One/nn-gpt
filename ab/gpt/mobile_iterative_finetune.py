#!/usr/bin/env python3
"""
Standalone mobile-deployment extension for the iterative fine-tuning pipeline.

This module keeps the existing iterative pipeline untouched and adds extra
post-evaluation stages only when TuneNNGen is launched with --mobile_deployment.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ab.gpt.iterative_finetune import IterativeFinetuner, logger
from ab.gpt.iterative_pipeline.pipeline_validation import StageValidator
from ab.nn.util.Const import out_dir


class MobileDeploymentFinetuner(IterativeFinetuner):
    """Iterative pipeline with mobile deployment stages."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mobile_prompt_template = "mobile_deployment_rules.json"
        self.mobile_runs = 20
        self.mobile_eval_samples = 256
        self.mobile_artifacts_dir = self.output_dir / "mobile_deployment"
        self.mobile_artifacts_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Mobile deployment mode enabled (standalone extension path)")

    def run_finetuning(self, cycle: int, data_dir: Path) -> Dict[str, Any]:
        """
        Mobile-only checkpoint recovery.

        The base pipeline expects checkpoints under out/qlora-sft/final.
        In mobile runs, TuneNNGen may leave the latest adapter under
        out/nngpt/llm/epoch/A*/<model_name> instead. If the base path lookup
        fails with checkpoint_not_found, recover from that epoch directory.
        """
        result = super().run_finetuning(cycle, data_dir)
        if result.get("success", False):
            return result
        if result.get("error") != "checkpoint_not_found":
            return result

        fallback = self._recover_checkpoint_from_epoch_outputs(cycle)
        if fallback:
            logger.info(
                "Recovered mobile checkpoint from epoch outputs: %s",
                fallback,
            )
            return {
                "success": True,
                "checkpoint_dir": str(fallback),
                "training_time_minutes": result.get("training_time_minutes", 0),
                "source_checkpoint": "out/nngpt/llm/epoch/latest",
                "recovered_from_epoch_outputs": True,
            }
        return result

    def _recover_checkpoint_from_epoch_outputs(self, cycle: int) -> Optional[Path]:
        epoch_root = out_dir / "nngpt" / "llm" / "epoch"
        if not epoch_root.exists():
            return None

        def _epoch_index(p: Path) -> int:
            m = re.match(r"A(\d+)$", p.name)
            return int(m.group(1)) if m else -1

        epoch_dirs = sorted(
            [p for p in epoch_root.iterdir() if p.is_dir() and re.match(r"A\d+$", p.name)],
            key=_epoch_index,
            reverse=True,
        )
        for epoch_dir in epoch_dirs:
            candidates = sorted(
                epoch_dir.glob("**/adapter_config.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for cfg in candidates:
                source_checkpoint = cfg.parent
                is_valid, _ = StageValidator.validate_finetuning_output(source_checkpoint)
                if not is_valid:
                    continue
                isolated_checkpoint = self.output_dir / f"cycle_{cycle}" / "checkpoint"
                if isolated_checkpoint.exists():
                    shutil.rmtree(isolated_checkpoint)
                shutil.copytree(source_checkpoint, isolated_checkpoint)
                valid_copy, _ = StageValidator.validate_finetuning_output(isolated_checkpoint)
                if valid_copy:
                    return isolated_checkpoint
        return None

    def generate_models(
        self,
        cycle: int,
        checkpoint_path: str,
        starting_checksum: int = 0,
        data_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Override generation to use a mobile-focused prompt template."""
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"CYCLE {cycle}: MODEL GENERATION (MOBILE PROMPT)")
        logger.info("=" * 80)
        logger.info(f"Checkpoint: {checkpoint_path}")

        num_prompts = (self.models_per_cycle + self.samples_per_prompt - 1) // self.samples_per_prompt
        output_dir = self.output_dir / f"cycle_{cycle}" / "generation"
        output_dir.mkdir(parents=True, exist_ok=True)

        from ab.gpt.util.nn_sftcodegen_rag import main as sft_gen_main

        def run_generation():
            sft_gen_main(
                output_dir=str(output_dir),
                base_model=checkpoint_path,
                data_dir=str(data_dir) if data_dir else None,
                max_items=num_prompts,
                temperature=0.7,
                top_k=100,
                top_p=0.95,
                rejection_sampling=True,
                max_rejections=5,
                samples_per_prompt=self.samples_per_prompt,
                prompt_template=self.mobile_prompt_template,
            )

        try:
            run_generation()
        except Exception as e:
            logger.error(f"Model generation failed: {e}")
            return {"success": False, "error": f"generation_failed: {e}"}

        results_file = output_dir / "results.jsonl"
        if not results_file.exists():
            return {"success": False, "error": "results_not_found"}

        results = []
        with open(results_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))

        accepted = [r for r in results if r.get("ok", False)]
        return {
            "success": True,
            "output_dir": str(output_dir),
            "total_generated": len(results),
            "successful": len(accepted),
            "results": results,
        }

    def run_cycle(self, cycle: int) -> Dict[str, Any]:
        """Run a single cycle with mobile metrics feeding the next prompt loop."""
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"STARTING CYCLE {cycle} (MOBILE DEPLOYMENT)")
        logger.info("=" * 80)
        cycle_start = time.time()

        if cycle == 1:
            data_dir = self.base_data_dir
        else:
            data_dir = self.data_manager.get_training_data_dir(cycle - 1, self.output_dir)

        checkpoint_path = self.output_dir / f"cycle_{cycle}" / "checkpoint"
        if checkpoint_path.exists() and (checkpoint_path / "adapter_config.json").exists():
            logger.info(f"✓ Existing checkpoint found: {checkpoint_path}")
            logger.info("Skipping fine-tuning (checkpoint already exists)")
            ft_result = {"success": True, "checkpoint_dir": str(checkpoint_path), "training_time_minutes": 0}
        else:
            ft_result = self.run_finetuning(cycle, data_dir)
            if not ft_result.get("success", False):
                return {
                    "cycle": cycle,
                    "success": False,
                    "error": ft_result.get("error", "unknown"),
                }
            checkpoint_path = Path(ft_result["checkpoint_dir"])

        generation_dir = self.output_dir / f"cycle_{cycle}" / "generation"
        results_file = generation_dir / "results.jsonl"
        if results_file.exists() and (generation_dir / "accepted_code").exists():
            logger.info(f"✓ Existing generation results found: {generation_dir}")
            results = []
            with open(results_file, "r") as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
            gen_result = {
                "success": True,
                "output_dir": str(generation_dir),
                "total_generated": len(results),
                "successful": len([r for r in results if r.get("ok", False)]),
                "results": results,
            }
        else:
            gen_result = self.generate_models(cycle, str(checkpoint_path), data_dir=Path(data_dir))
            if not gen_result.get("success", False):
                return {
                    "cycle": cycle,
                    "success": False,
                    "error": gen_result.get("error", "unknown"),
                }

        starting_checksum = 0
        for prev_cycle in range(1, cycle):
            prev_results_file = self.output_dir / f"cycle_{prev_cycle}" / "cycle_results.json"
            if prev_results_file.exists():
                try:
                    prev_results = json.loads(prev_results_file.read_text())
                    starting_checksum += prev_results.get("generation", {}).get("total_generated", 0)
                except Exception:
                    pass

        eval_result = self.evaluate_models(cycle, Path(gen_result["output_dir"]), starting_checksum=starting_checksum)
        selected_models = self.filter_successful_novel(gen_result, eval_result, starting_checksum=starting_checksum)

        mobile_result = self._run_mobile_stage(cycle, eval_result.get("models", []))
        mobile_by_id = {m["model_id"]: m for m in mobile_result.get("models", []) if m.get("success")}
        for model in selected_models:
            mm = mobile_by_id.get(model.get("model_id"))
            if mm:
                model["mobile_metrics"] = {
                    "quantized_accuracy": mm.get("quantized_accuracy"),
                    "best_duration_ms": mm.get("best_duration_ms"),
                    "score": mm.get("score"),
                    "benchmark_json": mm.get("benchmark_json"),
                }

        if selected_models:
            chat_examples = self.convert_to_training_data(selected_models, cycle, str(checkpoint_path))
            augment_stats = self.data_manager.augment_training_data(chat_examples, cycle, self.output_dir)
            if self.novelty_check_enabled:
                for model_info in selected_models:
                    code_path = Path(model_info["code_file"])
                    self.novelty_checker.mark_as_seen(
                        code_path.read_text(),
                        model_id=model_info["model_id"],
                        source=f"cycle_{cycle}_selected",
                    )
                self.novelty_checker.save_cache()
        else:
            augment_stats = self.data_manager.augment_training_data([], cycle, self.output_dir)

        cycle_time = time.time() - cycle_start
        result = {
            "cycle": cycle,
            "success": True,
            "training": {
                "data_dir": str(data_dir),
                "total_examples": augment_stats.get("total_examples", 0),
                "new_examples_added": augment_stats.get("new_examples_added", 0),
                "training_time_minutes": ft_result.get("training_time_minutes", 0),
            },
            "generation": {
                "total_generated": gen_result.get("total_generated", 0),
                "successful": gen_result.get("successful", 0),
                "novel": len(selected_models) if self.novelty_check_enabled else gen_result.get("successful", 0),
                "selected_for_training": len(selected_models),
            },
            "evaluation": {
                "models_trained": eval_result.get("models_trained", 0),
                "best_accuracy": eval_result.get("best_accuracy", 0.0),
                "avg_accuracy": eval_result.get("avg_accuracy", 0.0),
                "success_rate": gen_result.get("successful", 0) / max(1, gen_result.get("total_generated", 1)),
            },
            "mobile": mobile_result,
            "cycle_time_minutes": cycle_time / 60,
        }
        cycle_results_file = self.output_dir / f"cycle_{cycle}" / "cycle_results.json"
        cycle_results_file.parent.mkdir(parents=True, exist_ok=True)
        cycle_results_file.write_text(json.dumps(result, indent=2, default=str))
        return result

    def convert_to_training_data(
        self,
        selected_models: List[Dict[str, Any]],
        cycle: int,
        checkpoint: str,
    ) -> List[Dict[str, Any]]:
        """Inject mobile score objective into next-cycle prompts."""
        chat_examples = []
        for model_info in selected_models:
            metadata = {
                "accuracy": model_info["accuracy"],
                "params": model_info["params"],
                "cycle": cycle,
                "checkpoint": checkpoint,
                "mobile_metrics": model_info.get("mobile_metrics"),
            }
            code_path = Path(model_info["code_file"])
            code = code_path.read_text()
            chat_example = self.data_manager.convert_code_to_chat_example(
                code, model_info["model_id"], metadata
            )
            mobile_metrics = model_info.get("mobile_metrics")
            if mobile_metrics:
                score = mobile_metrics.get("score")
                quant_acc = mobile_metrics.get("quantized_accuracy")
                duration_ms = mobile_metrics.get("best_duration_ms")
                boost = (
                    "\n\n**MOBILE DEPLOYMENT OBJECTIVE**\n"
                    f"- Previous quantized accuracy: {quant_acc}\n"
                    f"- Previous on-device latency (ms): {duration_ms}\n"
                    f"- Previous score (accuracy / latency): {score}\n"
                    "- Improve accuracy/latency score while keeping valid LEMUR format."
                )
                chat_example["messages"][1]["content"] += boost
                chat_example.setdefault("meta", {})["mobile_metrics"] = mobile_metrics
            chat_examples.append(chat_example)
        return chat_examples

    def _run_mobile_stage(self, cycle: int, model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert/evaluate/benchmark generated models for mobile deployment."""
        cycle_mobile_dir = self.mobile_artifacts_dir / f"cycle_{cycle}"
        tflite_int8_root = cycle_mobile_dir / "tflite" / "int8"
        weights_root = cycle_mobile_dir / "weights"
        tflite_int8_root.mkdir(parents=True, exist_ok=True)
        weights_root.mkdir(parents=True, exist_ok=True)

        processed = 0
        failures = 0
        per_model: List[Dict[str, Any]] = []
        for model in model_results:
            if not model.get("success"):
                continue
            model_id = str(model.get("model_id"))
            code_file = model.get("code_file")
            if not code_file:
                continue
            model_dir = Path(code_file).parent
            try:
                artifact = self._process_mobile_model(
                    cycle=cycle,
                    model_id=model_id,
                    model_dir=model_dir,
                    base_accuracy=float(model.get("accuracy", 0.0)),
                    tflite_int8_root=tflite_int8_root,
                    weights_root=weights_root,
                )
                per_model.append(artifact)
                processed += 1
                if not artifact.get("success", False):
                    failures += 1
            except Exception as e:
                failures += 1
                per_model.append(
                    {"model_id": model_id, "success": False, "error": str(e), "code_file": code_file}
                )

        summary = {
            "success": True,
            "processed_models": processed,
            "failed_models": failures,
            "models": per_model,
        }
        (cycle_mobile_dir / "mobile_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        return summary

    def _process_mobile_model(
        self,
        cycle: int,
        model_id: str,
        model_dir: Path,
        base_accuracy: float,
        tflite_int8_root: Path,
        weights_root: Path,
    ) -> Dict[str, Any]:
        """Process one model through quantization + benchmark."""
        checkpoint_path = self._resolve_mobile_checkpoint(cycle, model_id, model_dir)
        copied_weight = None
        if checkpoint_path:
            copied_weight = weights_root / f"{model_id}{checkpoint_path.suffix}"
            shutil.copy2(checkpoint_path, copied_weight)
        else:
            # Mobile-only fallback: persist a loadable checkpoint so downstream
            # conversion has an explicit weight file even when eval did not emit
            # .pth/.pt artifacts for this model directory.
            copied_weight = weights_root / f"{model_id}.pth"
            fallback_result = self._save_mobile_fallback_checkpoint(model_dir / "new_nn.py", copied_weight)
            if fallback_result.get("success"):
                checkpoint_path = copied_weight
            else:
                copied_weight = None

        tflite_path = tflite_int8_root / f"{model_id}.int8.tflite"
        quant_result = self._export_int8_tflite(model_dir / "new_nn.py", checkpoint_path, tflite_path)

        quantized_accuracy = self._evaluate_tflite_accuracy(tflite_path) if quant_result["success"] else None
        if quantized_accuracy is None:
            quantized_accuracy = base_accuracy

        if quant_result["success"] and tflite_path.exists():
            benchmark = self._benchmark_on_device(model_id, tflite_path)
        else:
            benchmark = {
                "device_type": "unknown",
                "error": quant_result.get("error", "tflite_export_failed"),
            }
        best_duration = benchmark.get("duration")
        score = None
        if best_duration and best_duration > 0:
            score = float(quantized_accuracy) / float(best_duration)

        stat_dir = tflite_int8_root / f"img-classification_cifar-10_acc_{model_id}"
        stat_dir.mkdir(parents=True, exist_ok=True)
        device_name = benchmark.get("device_type", "unknown_device").replace(" ", "_")
        stat_json = stat_dir / f"android_{device_name}.json"

        stat_payload = {
            "model_name": model_id,
            "device_type": benchmark.get("device_type", "unknown"),
            "os_version": benchmark.get("os_version", ""),
            "valid": bool(quant_result["success"]),
            "emulator": False,
            "iterations": self.mobile_runs,
            "duration": int(best_duration) if best_duration else 0,
            "unit": benchmark.get("unit", "CPU"),
            "cpu_duration": int(benchmark.get("cpu_avg", 0)),
            "cpu_min_duration": int(benchmark.get("cpu_min", 0)),
            "cpu_max_duration": int(benchmark.get("cpu_max", 0)),
            "cpu_std_dev": float(benchmark.get("cpu_std", 0.0)),
            "gpu_duration": int(benchmark.get("gpu_avg", 0)),
            "gpu_min_duration": int(benchmark.get("gpu_min", 0)),
            "gpu_max_duration": int(benchmark.get("gpu_max", 0)),
            "gpu_std_dev": float(benchmark.get("gpu_std", 0.0)),
            "npu_duration": int(benchmark.get("npu_avg", 0)),
            "npu_min_duration": int(benchmark.get("npu_min", 0)),
            "npu_max_duration": int(benchmark.get("npu_max", 0)),
            "npu_std_dev": float(benchmark.get("npu_std", 0.0)),
            "quantized_accuracy": float(quantized_accuracy),
            "accuracy_time_score": score,
            "tflite_path": str(tflite_path),
            "weights_path": str(copied_weight) if copied_weight else None,
            "cycle": cycle,
            "in_dim_0": 1,
            "in_dim_1": 32,
            "in_dim_2": 32,
            "in_dim_3": 3,
            "total_ram_kb": int(benchmark.get("total_ram_kb", 0)),
            "free_ram_kb": int(benchmark.get("free_ram_kb", 0)),
            "available_ram_kb": int(benchmark.get("available_ram_kb", 0)),
            "cached_kb": int(benchmark.get("cached_kb", 0)),
            "device_analytics": benchmark.get("device_analytics", {}),
        }
        stat_json.write_text(json.dumps(stat_payload, indent=2, default=str))

        return {
            "model_id": model_id,
            "success": quant_result["success"],
            "tflite_path": str(tflite_path),
            "weights_path": str(copied_weight) if copied_weight else None,
            "quantized_accuracy": float(quantized_accuracy),
            "best_duration_ms": best_duration,
            "score": score,
            "benchmark_json": str(stat_json),
            "benchmark": benchmark,
        }

    def _resolve_mobile_checkpoint(self, cycle: int, model_id: str, model_dir: Path) -> Optional[Path]:
        """
        Resolve model checkpoint for mobile deployment only.

        Resolution order:
        1) local model_dir (*.pth/*.pt)
        2) lookup by model_id in known mobile roots
        3) lookup by eval checksum in known mobile roots
        """
        direct = self._pick_checkpoint(model_dir)
        if direct:
            return direct

        eval_checksum = self._extract_eval_checksum(model_dir)
        search_roots = self._mobile_checkpoint_search_roots(cycle)

        by_model_id = self._find_checkpoint_in_roots(search_roots, model_id)
        if by_model_id:
            logger.info("Resolved mobile checkpoint for %s by model_id: %s", model_id, by_model_id)
            return by_model_id

        if eval_checksum:
            by_checksum = self._find_checkpoint_in_roots(search_roots, eval_checksum)
            if by_checksum:
                logger.info(
                    "Resolved mobile checkpoint for %s by checksum %s: %s",
                    model_id,
                    eval_checksum,
                    by_checksum,
                )
                return by_checksum

        logger.warning("No mobile checkpoint found for %s (searched local + fallback roots)", model_id)
        return None

    def _save_mobile_fallback_checkpoint(self, code_path: Path, output_checkpoint: Path) -> Dict[str, Any]:
        """Create a local mobile checkpoint when no trained weight file is available."""
        try:
            import torch

            model = self._load_model_from_code(code_path, checkpoint_path=None)
            output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_checkpoint)
            return {"success": True, "path": str(output_checkpoint)}
        except Exception as e:
            logger.warning("Failed to create fallback mobile checkpoint at %s: %s", output_checkpoint, e)
            return {"success": False, "error": str(e)}

    def _mobile_checkpoint_search_roots(self, cycle: int) -> List[Path]:
        """Known roots where training artifacts/checkpoints may exist for mobile runs."""
        roots = [
            self.output_dir / f"cycle_{cycle}" / "nneval",
            self.output_dir / f"cycle_{cycle}" / "generation",
            out_dir / "nngpt" / "new_lemur" / "train",
            out_dir / "nngpt" / "new_lemur",
            out_dir / "nngpt",
            out_dir,
        ]
        existing_roots = []
        seen = set()
        for root in roots:
            resolved = root.resolve()
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            if resolved.exists():
                existing_roots.append(resolved)
        return existing_roots

    @staticmethod
    def _extract_eval_checksum(model_dir: Path) -> Optional[str]:
        """Read eval checksum from eval_info.json if available."""
        eval_info_path = model_dir / "eval_info.json"
        if not eval_info_path.exists():
            return None
        try:
            payload = json.loads(eval_info_path.read_text(encoding="utf-8"))
            checksum = payload.get("eval_results", {}).get("checksum")
            if checksum:
                return str(checksum)
        except Exception:
            return None
        return None

    @staticmethod
    def _find_checkpoint_in_roots(search_roots: List[Path], needle: str) -> Optional[Path]:
        """Find newest checkpoint path where filename contains the given token."""
        candidates: List[Path] = []
        token = str(needle).strip()
        if not token:
            return None
        for root in search_roots:
            for ext in ("pth", "pt"):
                candidates.extend(root.rglob(f"*{token}*.{ext}"))
        if not candidates:
            return None
        # Prefer newest artifact if multiple checkpoints match.
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    @staticmethod
    def _pick_checkpoint(model_dir: Path) -> Optional[Path]:
        """Find a trained checkpoint file in model dir."""
        for pattern in ("*.pth", "*.pt"):
            files = sorted(model_dir.glob(pattern))
            if files:
                return files[0]
        return None

    @staticmethod
    def _load_model_from_code(code_path: Path, checkpoint_path: Optional[Path]):
        import torch

        spec = importlib.util.spec_from_file_location("mobile_model_module", str(code_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load code module from {code_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "Net"):
            raise RuntimeError(f"Model file missing Net class: {code_path}")

        prm = {"lr": 0.01, "momentum": 0.9, "dropout": 0.2, "batch": 16}
        model = module.Net((1, 3, 32, 32), (10,), prm, torch.device("cpu"))
        model.eval()
        if checkpoint_path and checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict):
                model.load_state_dict(state, strict=False)
        return model

    def _export_int8_tflite(
        self,
        code_path: Path,
        checkpoint_path: Optional[Path],
        output_path: Path,
    ) -> Dict[str, Any]:
        """Export model to INT8 tflite using ai_edge_torch conversion path."""
        try:
            import numpy as np
            import tensorflow as tf
            import torch
            try:
                import litert_torch as ai_edge_torch
            except Exception:
                import ai_edge_torch
        except Exception as e:
            return {"success": False, "error": f"mobile conversion deps missing: {e}"}

        try:
            model = self._load_model_from_code(code_path, checkpoint_path)
            dummy_input = (torch.randn(1, 3, 32, 32),)

            def representative_dataset():
                for _ in range(50):
                    yield [np.random.randn(1, 3, 32, 32).astype(np.float32)]

            output_path.parent.mkdir(parents=True, exist_ok=True)
            converter_flags = {
                "optimizations": [tf.lite.Optimize.DEFAULT],
                "representative_dataset": representative_dataset,
                "target_spec": {"supported_ops": [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]},
                "inference_input_type": tf.int8,
                "inference_output_type": tf.int8,
            }
            converted = self._ai_edge_convert_model(
                ai_edge_torch=ai_edge_torch,
                model=model,
                dummy_input=dummy_input,
                converter_flags=converter_flags,
            )
            self._persist_ai_edge_output(converted, output_path)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def _ai_edge_convert_model(
        ai_edge_torch: Any,
        model: Any,
        dummy_input: Tuple[Any, ...],
        converter_flags: Dict[str, Any],
    ) -> Any:
        """
        Convert via ai_edge_torch with compatibility across API versions.

        Supports:
        - ai_edge_torch.convert(...)
        - ai_edge_torch.<submodule>.convert(...) for known submodule layouts
        """
        convert_candidates = []
        top_level_convert = getattr(ai_edge_torch, "convert", None)
        if callable(top_level_convert):
            convert_candidates.append(("ai_edge_torch.convert", top_level_convert))

        for submodule_name in (
            "ai_edge_torch.converter",
            "ai_edge_torch.convert",
            "ai_edge_torch._convert",
        ):
            try:
                module = importlib.import_module(submodule_name)
            except Exception:
                continue
            convert_fn = getattr(module, "convert", None)
            if callable(convert_fn):
                convert_candidates.append((f"{submodule_name}.convert", convert_fn))

        seen = set()
        unique_candidates = []
        for label, fn in convert_candidates:
            fn_id = id(fn)
            if fn_id in seen:
                continue
            seen.add(fn_id)
            unique_candidates.append((label, fn))

        if not unique_candidates:
            attrs = [name for name in dir(ai_edge_torch) if not name.startswith("_")]
            raise RuntimeError(
                "No ai_edge_torch conversion function found. "
                f"Available attrs: {attrs[:25]}"
            )

        attempt_errors = []
        for label, convert_fn in unique_candidates:
            try:
                try:
                    return convert_fn(
                        model,
                        dummy_input,
                        _ai_edge_converter_flags=converter_flags,
                    )
                except TypeError:
                    # Some versions do not accept _ai_edge_converter_flags.
                    return convert_fn(model, dummy_input)
            except Exception as e:
                attempt_errors.append(f"{label}: {e}")

        raise RuntimeError("All ai_edge_torch conversion attempts failed: " + " | ".join(attempt_errors))

    @staticmethod
    def _persist_ai_edge_output(converted: Any, output_path: Path) -> None:
        """Persist converter output regardless of ai_edge_torch return type."""
        if hasattr(converted, "export") and callable(getattr(converted, "export")):
            converted.export(str(output_path))
            return

        if hasattr(converted, "save") and callable(getattr(converted, "save")):
            converted.save(str(output_path))
            return

        if isinstance(converted, (bytes, bytearray)):
            output_path.write_bytes(bytes(converted))
            return

        if isinstance(converted, str):
            src = Path(converted)
            if src.exists():
                shutil.copy2(src, output_path)
                return

        raise RuntimeError(
            "Unsupported ai_edge_torch conversion output type: "
            f"{type(converted).__name__}"
        )

    def _evaluate_tflite_accuracy(self, tflite_path: Path) -> Optional[float]:
        """Evaluate quantized tflite model on a small CIFAR-10 subset."""
        try:
            import numpy as np
            import tensorflow as tf
            import torchvision
            import torchvision.transforms as T
        except Exception:
            return None

        try:
            interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()[0]
            output_details = interpreter.get_output_details()[0]

            transform = T.Compose([T.Resize((32, 32)), T.ToTensor()])
            dataset = torchvision.datasets.CIFAR10(
                root=str(self.output_dir / "mobile_deployment" / "datasets"),
                train=False,
                download=True,
                transform=transform,
            )

            correct = 0
            total = 0
            for image, label in dataset:
                if total >= self.mobile_eval_samples:
                    break
                arr = image.numpy()  # C,H,W
                arr = arr.transpose(1, 2, 0)[None, ...]  # NHWC

                in_dtype = input_details["dtype"]
                in_scale, in_zero = input_details.get("quantization", (0.0, 0))
                if in_dtype == np.int8 and in_scale > 0:
                    arr = (arr / in_scale + in_zero).round().clip(-128, 127).astype(np.int8)
                else:
                    arr = arr.astype(np.float32)

                interpreter.set_tensor(input_details["index"], arr)
                interpreter.invoke()
                out = interpreter.get_tensor(output_details["index"])
                out_scale, out_zero = output_details.get("quantization", (0.0, 0))
                if out.dtype == np.int8 and out_scale > 0:
                    out = (out.astype(np.float32) - out_zero) * out_scale
                pred = int(out.argmax(axis=-1)[0])
                correct += int(pred == int(label))
                total += 1

            return float(correct) / max(1, total)
        except Exception:
            return None

    def _benchmark_on_device(self, model_id: str, tflite_path: Path) -> Dict[str, Any]:
        """Benchmark tflite model on Android device (cpu/gpu/npu)."""
        adb = shutil.which("adb")
        if adb is None:
            return {"device_type": "unknown", "error": "adb not found"}

        if subprocess.run([adb, "get-state"], capture_output=True, text=True).returncode != 0:
            return {"device_type": "unknown", "error": "no adb device"}

        remote_model = f"/data/local/tmp/{model_id}.tflite"
        subprocess.run([adb, "push", str(tflite_path), remote_model], capture_output=True, text=True)

        device_type = self._adb_shell(adb, "getprop ro.product.model").strip()
        os_version = (
            self._adb_shell(adb, "getprop ro.build.version.release").strip()
            + " | "
            + self._adb_shell(adb, "getprop ro.build.id").strip()
        )
        mem = self._get_android_memory(adb)
        device_analytics = self._get_device_analytics(adb)

        backend_map = {
            "cpu": "--use_xnnpack=false",
            "gpu": "--use_gpu=true",
            "npu": "--use_nnapi=true",
        }
        bench = {
            "device_type": device_type if device_type else "unknown",
            "os_version": os_version,
            "device_analytics": device_analytics,
        }
        bench.update(mem)
        options = {}
        for backend, flag in backend_map.items():
            cmd = f"/data/local/tmp/benchmark_model --graph={remote_model} --num_runs={self.mobile_runs} {flag}"
            out = self._adb_shell(adb, cmd)
            parsed = self._parse_benchmark_output(out)
            for key, value in parsed.items():
                bench[f"{backend}_{key}"] = value
            if parsed.get("avg", 0) > 0:
                options[backend.upper()] = parsed["avg"]
            else:
                bench[f"{backend}_error"] = self._extract_benchmark_error(out)

        self._adb_shell(adb, f"rm {remote_model}")
        if options:
            best_unit = min(options, key=options.get)
            bench["unit"] = best_unit
            bench["duration"] = options[best_unit]
        return bench

    @staticmethod
    def _adb_shell(adb: str, cmd: str) -> str:
        result = subprocess.run([adb, "shell", cmd], capture_output=True, text=True)
        return (result.stdout or "") + (result.stderr or "")

    @staticmethod
    def _parse_benchmark_output(output: str) -> Dict[str, float]:
        if not output:
            return {"avg": 0, "min": 0, "max": 0, "std": 0}
        # Prefer final inference timing summary when present; values are in microseconds.
        inf_avg_match = re.search(r"Inference \(avg\):\s*([\d\.]+)", output)
        parsed = {"avg": float(inf_avg_match.group(1)) * 1000.0 if inf_avg_match else 0.0}

        # count=... min=... max=... avg=... std=...
        # Benchmark emits warmup first and final run second; use the last one.
        payload = output.replace(" ", "")
        for key in ("min", "max", "std"):
            all_matches = re.findall(rf"{key}=([\d\.]+)", payload)
            parsed[key] = float(all_matches[-1]) * 1000.0 if all_matches else 0.0

        if parsed["avg"] <= 0:
            all_avg = re.findall(r"avg=([\d\.]+)", payload)
            parsed["avg"] = float(all_avg[-1]) * 1000.0 if all_avg else 0.0
        return parsed

    @staticmethod
    def _extract_benchmark_error(output: str) -> str:
        """Extract concise backend error details from benchmark output."""
        if not output or not output.strip():
            return "no output from benchmark_model"
        keywords = (
            "ERROR",
            "Error",
            "Failed",
            "unsupported",
            "Unsupported",
            "NNAPI",
            "delegate",
            "INVALID",
            "BAD_DATA",
        )
        lines = []
        for line in output.splitlines():
            line = line.strip()
            if line and any(k in line for k in keywords):
                lines.append(line)
        if lines:
            msg = " | ".join(lines[:3])
        else:
            nonempty = [ln.strip() for ln in output.splitlines() if ln.strip()]
            msg = nonempty[-1] if nonempty else "no output"
        if len(msg) > 500:
            msg = msg[:497] + "..."
        return msg

    @staticmethod
    def _get_android_memory(adb: str) -> Dict[str, int]:
        raw = MobileDeploymentFinetuner._adb_shell(adb, "cat /proc/meminfo")
        key_map = {
            "MemTotal": "total_ram_kb",
            "MemFree": "free_ram_kb",
            "MemAvailable": "available_ram_kb",
            "Cached": "cached_kb",
        }
        values: Dict[str, int] = {}
        for line in raw.splitlines():
            if ":" not in line:
                continue
            lhs, rhs = line.split(":", 1)
            lhs = lhs.strip()
            if lhs not in key_map:
                continue
            try:
                values[key_map[lhs]] = int(rhs.strip().split()[0])
            except Exception:
                continue
        return values

    @staticmethod
    def _get_device_analytics(adb: str) -> Dict[str, Any]:
        """Collect structured CPU/device metadata similar to dataset benchmark payloads."""
        raw = MobileDeploymentFinetuner._adb_shell(adb, "cat /proc/cpuinfo")
        processors: List[Dict[str, str]] = []
        current: Dict[str, str] = {}
        global_meta = {
            "hardware": "",
            "features": "",
            "cpu implementer": "",
            "cpu architecture": "",
            "cpu variant": "",
            "cpu part": "",
            "cpu revision": "",
        }

        for line in raw.splitlines():
            line = line.strip()
            if not line:
                if current:
                    processors.append(current)
                    current = {}
                continue
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k, v = k.strip().lower(), v.strip()
            if k == "processor" and v.isdigit():
                current["processor"] = v
            elif k in global_meta:
                global_meta[k] = v
                current[k] = v
            else:
                current[k] = v
        if current:
            processors.append(current)

        soc = (
            MobileDeploymentFinetuner._adb_shell(adb, "getprop ro.soc.model").strip()
            or MobileDeploymentFinetuner._adb_shell(adb, "getprop ro.board.platform").strip()
        )
        return {
            "timestamp": time.time(),
            "cpu_info": {
                "cpu_cores": len([p for p in processors if "processor" in p]),
                "processors": processors[:4],
                "arm_architecture": {
                    "hardware": global_meta["hardware"] or soc,
                    "features": global_meta["features"],
                    "cpu_implementer": global_meta["cpu implementer"],
                    "cpu_architecture": global_meta["cpu architecture"],
                    "cpu_variant": global_meta["cpu variant"],
                    "cpu_part": global_meta["cpu part"],
                    "cpu_revision": global_meta["cpu revision"],
                },
            },
        }
