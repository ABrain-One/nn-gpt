#!/usr/bin/env python
"""
Iterative Fine-Tuning Pipeline for Neural Architecture Generation

Implements a multi-cycle fine-tuning approach where:
1. Fine-tune model on current training data
2. Generate N models from the fine-tuned checkpoint
3. Evaluate models (compilation + first-epoch accuracy on CIFAR-10)
4. Filter for successful AND novel models
5. Convert selected models to training examples
6. Add to cumulative training data
7. Repeat for next cycle

Expected improvement: Both success rate and first-epoch accuracy should increase over cycles.
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional

from ab.gpt.iterative_pipeline.novelty_checker import NoveltyChecker
from ab.gpt.iterative_pipeline.training_data_manager import TrainingDataManager
from ab.gpt.iterative_pipeline.structural_reranker import StructuralReranker
from ab.gpt.iterative_pipeline.pipeline_validation import (
    PipelineValidator, RetryHandler, StageValidator, ErrorRecovery
)
from ab.gpt.iterative_pipeline.gpu_memory_manager import (
    ensure_gpu_memory, clear_gpu_cache, get_gpu_memory_info, check_gpu_memory,
    kill_gpu_processes
)
from ab.dup.preprocessing import curate_from_lemur
from ab.chatprep.prompt_builder import ChatPrepConfig
from ab.gpt.TuneNNGen import get_pipeline_defaults

# Setup logging - will be configured after output_dir is known
logger = logging.getLogger(__name__)


class IterativeFinetuner:
    """Orchestrates the iterative fine-tuning pipeline."""
    
    def __init__(
        self,
        base_data_dir: str,
        output_dir: str,
        llm_conf: str,
        cycles: int = 5,
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
    ):
        self.base_data_dir = Path(base_data_dir)
        self.output_dir = Path(output_dir)
        self.llm_conf = llm_conf
        self.cycles = cycles
        self.models_per_cycle = models_per_cycle
        self.samples_per_prompt = samples_per_prompt
        self.accuracy_threshold = accuracy_threshold
        self.min_selected_k = min_selected_k
        self.fallback_threshold = fallback_threshold
        self.adaptive_threshold_enabled = adaptive_threshold
        self.novelty_check_enabled = novelty_check
        self.resume_from_cycle = resume_from_cycle
        self.max_retries = max_retries
        self.use_optimized_training = use_optimized_training
        self.num_train_epochs = num_train_epochs
        
        # Initialize components
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging to use output directory (not relative path)
        # Remove any existing handlers to avoid duplicates
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler - log to output directory, not current working directory
        log_file = self.output_dir / "iterative_pipeline.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        self.novelty_checker = NoveltyChecker(self.output_dir / "seen_models.json")
        
        # Track results across cycles
        self.cycle_results = []
        
        # Resolve LLM config path (if relative, resolve to ab/gpt/conf/llm/)
        self.llm_conf = self._resolve_llm_config_path(self.llm_conf)
        
        # Ensure curation data exists in curation_output/ (create if missing)
        self._ensure_curation_output_data()
        
        # TrainingDataManager loads from base_data_dir (curation_output/chat_data)
        self.data_manager = TrainingDataManager(str(self.base_data_dir))
        
        # Initialize novelty checker with original training data
        if self.novelty_check_enabled:
            self._initialize_novelty_checker()
        
        # Validate prerequisites
        self._validate_prerequisites()
    
    def _initialize_novelty_checker(self):
        """Load original training data into novelty checker."""
        logger.info("Initializing novelty checker with original training data...")
        train_data = self.data_manager._load_jsonl(self.base_data_dir / "train.jsonl")
        
        for example in train_data:
            # Extract code from assistant message
            for msg in example.get("messages", []):
                if msg["role"] == "assistant":
                    content = msg["content"]
                    # Extract code from markdown fence
                    if "```python" in content:
                        code = content.split("```python")[1].split("```")[0].strip()
                        self.novelty_checker.add_training_data(code, source="original_training")
        
        stats = self.novelty_checker.get_stats()
        logger.info(f"Loaded {stats['total_seen']} models from training data")
        self.novelty_checker.save_cache()
    
    def _resolve_llm_config_path(self, config_path: str) -> str:
        """Resolve LLM config path to absolute path."""
        config_file = Path(config_path)
        
        # If it's already an absolute path and exists, use it
        if config_file.is_absolute() and config_file.exists():
            return str(config_file)
        
        # Try to resolve relative to ab/gpt/conf/llm/
        try:
            from ab.gpt.util.Const import conf_llm_dir, conf_test_dir, conf_train_dir
            resolved_path = conf_llm_dir / config_file.name
            if resolved_path.exists():
                logger.info(f"Resolved LLM config: {config_path} -> {resolved_path}")
                return str(resolved_path)
        except Exception as e:
            logger.warning(f"Failed to resolve config path using conf_llm_dir: {e}")
        
        # Try current directory
        if Path(config_path).exists():
            return config_path
        
        # Return as-is (will fail validation if doesn't exist)
        return config_path
    
    def _ensure_curation_output_data(self):
        """Ensure curation data exists in curation_output/, create if missing."""
        curation_output_dir = Path("curation_output")
        curation_chat_data = curation_output_dir / "chat_data"
        
        train_file = curation_chat_data / "train.jsonl"
        dev_file = curation_chat_data / "dev.jsonl"
        test_file = curation_chat_data / "test.jsonl"
        
        # Check if curation data already exists
        if train_file.exists() and test_file.exists():
            logger.info(f"✓ Curation data found in {curation_chat_data}")
            logger.info(f"  - Train: {train_file}")
            logger.info(f"  - Test: {test_file}")
            if dev_file.exists():
                logger.info(f"  - Dev: {dev_file}")
            return
        
        # Generate curation data if missing
        logger.info("")
        logger.info("="*80)
        logger.info("GENERATING CURATION DATA")
        logger.info("="*80)
        logger.info(f"Target: {curation_chat_data}")
        
        try:
            # Step 1: Curate from LEMUR
            logger.info("Step 1: Curating models from LEMUR...")
            curate_result = curate_from_lemur()
            logger.info(f"✓ Curation from LEMUR completed: {curate_result}")
            
            # Step 2: Build chat data (ChatPrepConfig writes to curation_output by default)
            logger.info("Step 2: Building chat data using ChatPrepConfig...")
            
            # Ensure directory exists
            curation_chat_data.mkdir(parents=True, exist_ok=True)
            
            # Build chat data (writes to curation_output/chat_data/)
            logger.info("  Building train/dev/test data...")
            config = ChatPrepConfig(
                accepted_dir="curation_output/accepted_code"
            )
            config.run()
            
            # Verify the output was created
            if train_file.exists() and test_file.exists():
                logger.info(f"✓ Curation data successfully created in {curation_chat_data}")
                logger.info("="*80)
            else:
                logger.error(f"✗ Curation data creation failed")
                raise RuntimeError(f"Curation data creation failed")
                
        except ImportError as e:
            logger.error(f"✗ Failed to import curation modules: {e}")
            logger.error("Please ensure ab.dup.preprocessing and ab.chatprep.prompt_builder are available")
            raise
        except Exception as e:
            logger.error(f"✗ Curation data creation failed: {e}")
            logger.error("Please create curation data manually or fix the issue above")
            raise
    
    
    def _validate_prerequisites(self):
        """Validate all prerequisites before starting pipeline."""
        logger.info("="*80)
        logger.info("VALIDATING PIPELINE PREREQUISITES")
        logger.info("="*80)
        
        # Training data is in base_data_dir (curation_output/chat_data)
        validator = PipelineValidator()
        results = validator.validate_all(self.base_data_dir, Path(self.llm_conf))
        
        all_passed = True
        for check_name, (passed, message) in results["results"].items():
            status = "✓" if passed else "✗"
            logger.info(f"{status} {check_name}: {message}")
            if not passed:
                all_passed = False
                logger.error(f"Prerequisite check failed: {check_name} - {message}")
        
        if not all_passed:
            logger.error("\n✗ Prerequisites validation failed. Cannot proceed.")
            raise RuntimeError("Prerequisites validation failed. Please fix issues before running pipeline.")
        
        logger.info("\n✓ All prerequisites validated successfully!")
        
        # Check for resume capability
        if self.resume_from_cycle is None:
            last_cycle = ErrorRecovery.find_last_successful_cycle(self.output_dir)
            if last_cycle is not None:
                can_resume, msg = ErrorRecovery.can_resume_from_cycle(self.output_dir, last_cycle)
                if can_resume:
                    logger.warning(f"\n[RESUME] Found previous run. Last successful cycle: {last_cycle}")
                    logger.warning(f"[RESUME] To resume from cycle {last_cycle + 1}, use: --resume_from_cycle {last_cycle + 1}")
                else:
                    logger.info(f"[INFO] Previous run found but cannot resume: {msg}")
    
    def run_finetuning(self, cycle: int, data_dir: Path) -> Dict[str, Any]:
        """
        Run fine-tuning for one cycle.
        
        Returns:
            Dictionary with training metrics (time, loss, etc.)
        """
        logger.info("")
        logger.info("="*80)
        logger.info(f"CYCLE {cycle}: FINE-TUNING")
        logger.info("="*80)
        logger.info(f"Training data: {data_dir}")
        
        # Set PyTorch memory allocator to reduce fragmentation (helps with CUDA OOM)
        import os
        if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            logger.info("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce memory fragmentation")
        
        # Clear GPU cache before fine-tuning
        logger.info("Clearing GPU cache before fine-tuning...")
        clear_gpu_cache()
        
        # Kill any other GPU processes (except this one)
        import os
        current_pid = str(os.getpid())
        logger.info("Checking for other GPU processes...")
        kill_gpu_processes(exclude_pids=[current_pid])
        clear_gpu_cache()
        
        # Check GPU memory availability
        has_memory, msg = check_gpu_memory(min_free_gb=8.0)  # Need at least 8GB for 7B model
        if not has_memory:
            logger.warning(f"GPU memory warning: {msg}")
            logger.info("Attempting aggressive memory cleanup...")
            if not ensure_gpu_memory(min_free_gb=8.0, aggressive=True):
                logger.error("Insufficient GPU memory for fine-tuning.")
                logger.error("Please:")
                logger.error("  1. Kill other GPU processes: nvidia-smi and kill PIDs")
                logger.error("  2. Restart the pipeline after freeing memory")
                logger.error("  3. Consider using DeepSpeed ZeRO-3 (set use_deepspeed=true in config)")
                return {
                    "success": False,
                    "error": f"insufficient_gpu_memory: {msg}",
                    "training_time_minutes": 0,
                }
        
        # Check if checkpoint already exists (from previous incomplete run)
        isolated_checkpoint = self.output_dir / f"cycle_{cycle}" / "checkpoint"
        if isolated_checkpoint.exists():
            # Validate existing checkpoint
            is_valid, msg = StageValidator.validate_finetuning_output(isolated_checkpoint)
            if is_valid:
                logger.info(f"✓ Existing checkpoint found and validated: {isolated_checkpoint}")
                logger.info("Skipping fine-tuning (checkpoint already exists)")
                return {
                    "success": True,
                    "checkpoint_dir": str(isolated_checkpoint),
                    "training_time_minutes": 0.0,  # No time spent since we skipped
                    "skipped": True,
                }
            else:
                logger.warning(f"Existing checkpoint found but validation failed: {msg}")
                logger.warning("Will re-run fine-tuning to create a new checkpoint")
        
        total, used, free = get_gpu_memory_info()
        logger.info(f"GPU memory before fine-tuning: {free:.2f}GB free / {total:.2f}GB total")
        
        # Output directory for this cycle's fine-tuning
        ft_output_dir = self.output_dir / f"cycle_{cycle}" / "finetuning_output"
        ft_output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "python", "-m", "ab.gpt.TuneNNGen",
            "--llm_conf", self.llm_conf,
            "--data_dir", str(data_dir),
        ]
        
        # Load previous cycle's checkpoint for continual learning (cycle 2+)
        if cycle > 1:
            prev_checkpoint = self.output_dir / f"cycle_{cycle-1}" / "checkpoint"
            if prev_checkpoint.exists():
                logger.info(f"Loading previous cycle checkpoint: {prev_checkpoint}")
                cmd.extend(["--peft", str(prev_checkpoint)])
            else:
                logger.warning(f"Previous cycle checkpoint not found: {prev_checkpoint}")
                logger.warning("Starting from base model instead")
        
        # Add optimized training parameters if enabled
        if self.use_optimized_training:
            # Get pipeline defaults from TuneNNGen.py (single source of truth)
            pipeline_defaults = get_pipeline_defaults()
            
            # Build command-line arguments from defaults dict
            optimized_params = []
            
            # Training hyperparameters
            optimized_params.extend(["--learning_rate", str(pipeline_defaults['learning_rate'])])
            optimized_params.extend(["--weight_decay", str(pipeline_defaults['weight_decay'])])
            optimized_params.extend(["--warmup_steps", str(pipeline_defaults['warmup_steps'])])
            optimized_params.extend(["--num_train_epochs", str(self.num_train_epochs)])  # Use instance value (may override default)
            optimized_params.extend(["--logging_steps", str(pipeline_defaults['logging_steps'])])
            optimized_params.extend(["--max_grad_norm", str(pipeline_defaults['max_grad_norm'])])
            
            # LoRA configuration
            optimized_params.extend(["--target_modules", pipeline_defaults['target_modules']])
            
            # Generation parameters
            optimized_params.extend(["--max_new_tokens", str(pipeline_defaults['max_new_tokens'])])
            optimized_params.extend(["--temperature", str(pipeline_defaults['temperature'])])
            optimized_params.extend(["--top_k", str(pipeline_defaults['top_k'])])
            
            # Evaluation and checkpointing
            optimized_params.extend(["--evaluation_strategy", pipeline_defaults['evaluation_strategy']])
            optimized_params.extend(["--eval_steps", str(pipeline_defaults['eval_steps'])])
            optimized_params.extend(["--per_device_eval_batch_size", str(pipeline_defaults['per_device_eval_batch_size'])])
            optimized_params.extend(["--save_strategy", pipeline_defaults['save_strategy']])
            optimized_params.extend(["--save_steps", str(pipeline_defaults['save_steps'])])
            optimized_params.extend(["--save_total_limit", str(pipeline_defaults['save_total_limit'])])
            if pipeline_defaults['load_best_model_at_end']:
                optimized_params.append("--load_best_model_at_end")
            optimized_params.extend(["--metric_for_best_model", pipeline_defaults['metric_for_best_model']])
            
            cmd.extend(optimized_params)
            logger.info("Using optimized training configuration (from TuneNNGen.py defaults)")
        else:
            logger.info("Using default training configuration")
        
        logger.info(f"Running fine-tuning: {' '.join(cmd)}")
        start_time = time.time()
        
        # Retry fine-tuning with exponential backoff
        def run_finetuning_cmd():
            # Clear GPU cache before each attempt
            clear_gpu_cache()
            
            # Stream output in real-time while still capturing return code
            result = subprocess.run(
                cmd,
                capture_output=False,  # Stream output to terminal in real-time
                text=True,
                check=False  # We'll check returncode manually
            )
            if result.returncode != 0:
                logger.error(f"Fine-tuning command failed with exit code {result.returncode}")
                logger.error("Check terminal output above for error details")
                logger.error("Common issues:")
                logger.error("  - CUDA out of memory: Free GPU memory or enable DeepSpeed")
                logger.error("  - Training errors: Check training data and config")
                clear_gpu_cache()  # Clear cache after failure
                raise subprocess.CalledProcessError(result.returncode, cmd)
            return result
        
        try:
            result = RetryHandler.retry_with_backoff(
                run_finetuning_cmd,
                max_retries=self.max_retries,
                initial_delay=30.0,  # 30 seconds initial delay
                backoff_factor=2.0,
                exceptions=(subprocess.CalledProcessError, OSError),
                operation_name=f"Fine-tuning cycle {cycle}"
            )
        except subprocess.CalledProcessError as e:
            training_time = time.time() - start_time
            logger.error(f"Fine-tuning failed after {self.max_retries} retries (exit code: {e.returncode})")
            logger.error("This cycle will be marked as failed. Pipeline will stop.")
            logger.error("To resume after fixing the issue, use: --resume_from_cycle {next_cycle}")
            return {
                "success": False,
                "error": f"fine-tuning_failed: exit_code_{e.returncode}",
                "training_time_minutes": training_time / 60,
            }
        except Exception as e:
            training_time = time.time() - start_time
            logger.error(f"Fine-tuning failed with unexpected error: {e}")
            logger.error("This cycle will be marked as failed. Pipeline will stop.")
            return {
                "success": False,
                "error": f"fine-tuning_failed: {str(e)}",
                "training_time_minutes": training_time / 60,
            }
        
        training_time = time.time() - start_time
        logger.info(f"Fine-tuning completed in {training_time/60:.1f} minutes")
        
        # Find the final checkpoint from fine-tuning
        # TuneNNGen saves to out/qlora-sft/final, but we'll copy it to isolated directory
        source_checkpoint = Path("out/qlora-sft/final")
        if not source_checkpoint.exists():
            logger.error(f"Checkpoint directory not found: {source_checkpoint}")
            return {
                "success": False,
                "error": "checkpoint_not_found",
                "training_time_minutes": training_time / 60,
            }
        
        # Copy to isolated checkpoint directory for this cycle
        isolated_checkpoint = self.output_dir / f"cycle_{cycle}" / "checkpoint"
        isolated_checkpoint.mkdir(parents=True, exist_ok=True)
        
        # Copy checkpoint files with retry
        def copy_checkpoint():
            logger.info(f"Copying checkpoint to isolated directory: {isolated_checkpoint}")
            if isolated_checkpoint.exists():
                shutil.rmtree(isolated_checkpoint)
            shutil.copytree(source_checkpoint, isolated_checkpoint)
            return True
        
        try:
            RetryHandler.retry_with_backoff(
                copy_checkpoint,
                max_retries=3,
                initial_delay=2.0,
                backoff_factor=1.5,
                exceptions=(OSError, shutil.Error),
                operation_name=f"Copy checkpoint cycle {cycle}"
            )
        except Exception as e:
            logger.error(f"Failed to copy checkpoint: {e}")
            return {
                "success": False,
                "error": f"checkpoint_copy_failed: {str(e)}",
                "training_time_minutes": training_time / 60,
            }
        
        # Validate checkpoint
        is_valid, msg = StageValidator.validate_finetuning_output(isolated_checkpoint)
        if not is_valid:
            logger.error(f"Checkpoint validation failed: {msg}")
            return {
                "success": False,
                "error": f"checkpoint_validation_failed: {msg}",
                "training_time_minutes": training_time / 60,
            }
        
        logger.info(f"Checkpoint validated and saved to: {isolated_checkpoint}")
        
        return {
            "success": True,
            "checkpoint_dir": str(isolated_checkpoint),
            "source_checkpoint": str(source_checkpoint),
            "training_time_minutes": training_time / 60,
        }
    
    def generate_models(self, cycle: int, checkpoint_path: str, starting_checksum: int = 0, data_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate models using Sft_Rag.py (nn_sftcodegen_rag).
        
        Args:
            cycle: Current cycle number
            checkpoint_path: Path to finetuned checkpoint (for pipeline mode)
            starting_checksum: Starting index for model naming (default: 0)
            data_dir: Directory containing test.jsonl for RAG examples (for pipeline mode)
        
        Returns:
            Dictionary with generation results (paths, counts, etc.)
        """
        logger.info("")
        logger.info("="*80)
        logger.info(f"CYCLE {cycle}: MODEL GENERATION")
        logger.info("="*80)
        logger.info(f"Checkpoint: {checkpoint_path}")
        # Calculate number of prompts needed based on models_per_cycle and samples_per_prompt
        num_prompts = (self.models_per_cycle + self.samples_per_prompt - 1) // self.samples_per_prompt
        logger.info(f"Generating {self.models_per_cycle} models ({self.samples_per_prompt} per prompt, {num_prompts} prompts)")
        
        # Output directory for generation (accepted_code will be created inside)
        output_dir = self.output_dir / f"cycle_{cycle}" / "generation"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Import the generation function directly
        from ab.gpt.util.nn_sftcodegen_rag import main as sft_gen_main
        
        logger.info(f"Calling nn_sftcodegen_rag with checkpoint: {checkpoint_path}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Generating {self.models_per_cycle} models ({self.samples_per_prompt} per prompt)")
        
        # Retry generation with exponential backoff
        def run_generation():
            try:
                # Pass checkpoint_path as base_model to use finetuned weights
                # Output will go to cycle_X/generation/
                sft_gen_main(
                    output_dir=str(output_dir),
                    base_model=checkpoint_path,  # Use the finetuned checkpoint from this cycle
                    data_dir=str(data_dir) if data_dir else None,  # Use cycle's test data for RAG examples
                    max_items=num_prompts,  # Number of prompts to process
                    temperature=0.7,
                    top_k=100,
                    top_p=0.95,
                    rejection_sampling=True,
                    max_rejections=5,
                    samples_per_prompt=self.samples_per_prompt,  # Generate N models per prompt
                )
            except Exception as e:
                logger.error(f"Generation function failed: {e}")
                raise
        
        try:
            RetryHandler.retry_with_backoff(
                run_generation,
                max_retries=self.max_retries,
                initial_delay=10.0,
                backoff_factor=2.0,
                exceptions=(Exception,),
                operation_name=f"Model generation cycle {cycle}"
            )
        except Exception as e:
            logger.error(f"Model generation failed after {self.max_retries} retries: {e}")
            logger.error("This cycle will be marked as failed. Pipeline will stop.")
            return {"success": False, "error": f"generation_failed: {str(e)}"}
        
        # Validate generation output
        is_valid, msg = StageValidator.validate_generation_output(
            output_dir,
            expected_count=self.models_per_cycle
        )
        if not is_valid:
            logger.warning(f"Generation validation warning: {msg}")
            # Don't fail completely, but log warning
        
        # Parse results
        results_file = output_dir / "results.jsonl"
        if not results_file.exists():
            logger.error(f"Results file not found: {results_file}")
            return {"success": False, "error": "results_not_found"}
        
        results = []
        with open(results_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        
        # Post-process code files (accepted_code directory)
        accepted_code_dir = output_dir / "accepted_code"
        if accepted_code_dir.exists():
            logger.info("Post-processing generated code files to fix missing imports...")
            self._fix_generated_code(accepted_code_dir)
            
            logger.info("Re-validating all generated code files...")
            results = self._revalidate_generated_models(results, accepted_code_dir)
        else:
            logger.warning(f"accepted_code directory not found: {accepted_code_dir}")
        
        # Update results.jsonl with corrected validation status
        with open(results_file, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')
        
        successful = [r for r in results if r.get("ok", False)]
        logger.info(f"After re-validation: {len(successful)}/{len(results)} models are valid")
        
        return {
            "success": True,
            "output_dir": str(output_dir),
            "total_generated": len(results),
            "successful": len(successful),
            "results": results,
        }
    
    def _fix_generated_code(self, accepted_code_dir: Path) -> None:
        """
        Post-process generated code files to fix common issues:
        - Add missing imports (torch, nn, F if functional API used)
        - Remove duplicate class definitions
        - Replace F.adaptive_avg_pool2d with nn.AdaptiveAvgPool2d
        - Remove duplicate function definitions
        
        Args:
            accepted_code_dir: Directory containing generated model subdirectories (B0, B1, etc.)
        """
        if not accepted_code_dir.exists():
            return
        
        logger.info("Post-processing generated code files to fix missing imports...")
        
        import re
        
        required_imports = "import torch\nimport torch.nn as nn\n"
        fixed_count = 0
        
        # Look for new_nn.py files in B* subdirectories
        code_files = list(accepted_code_dir.glob("B*/new_nn.py"))
        total_files = len(code_files)
        logger.info(f"Found {total_files} code files to process")
        
        for code_file in sorted(code_files):
            try:
                content = code_file.read_text()
                original_content = content
                
                # Check if imports are missing
                has_torch_import = "import torch" in content
                has_nn_import = "import torch.nn" in content or "import torch.nn as nn" in content
                has_f_import = "import torch.nn.functional as F" in content or "import torch.nn.functional" in content
                
                # Check if functional API is used (F.relu, F.max_pool2d, etc.)
                uses_functional_api = bool(re.search(r'\bF\.(relu|max_pool2d|avg_pool2d|adaptive_avg_pool2d|dropout|sigmoid|tanh|softmax)', content))
                
                # Fix missing imports
                if not has_torch_import or not has_nn_import:
                    logger.debug(f"  Missing imports in {code_file.name}, fixing...")
                    # Remove any existing partial imports
                    lines = content.split('\n')
                    filtered_lines = []
                    for line in lines:
                        if not (line.strip().startswith("import torch") or 
                                line.strip().startswith("from torch")):
                            filtered_lines.append(line)
                    
                    # Add required imports at the start
                    imports_to_add = required_imports
                    if uses_functional_api and not has_f_import:
                        imports_to_add += "import torch.nn.functional as F\n"
                    content = imports_to_add + '\n' + '\n'.join(filtered_lines)
                elif uses_functional_api and not has_f_import:
                    # Add F import if functional API is used but import is missing
                    logger.debug(f"  Missing F import in {code_file.name}, fixing...")
                    lines = content.split('\n')
                    # Find where to insert (after torch/nn imports)
                    insert_idx = 0
                    for i, line in enumerate(lines):
                        if line.strip().startswith("import torch"):
                            insert_idx = i + 1
                            # Check if next line is also an import
                            if i + 1 < len(lines) and lines[i + 1].strip().startswith("import"):
                                insert_idx = i + 2
                            break
                    lines.insert(insert_idx, "import torch.nn.functional as F")
                    content = '\n'.join(lines)
                
                # Remove duplicate class Net definitions (keep only the first one)
                if content.count("class Net(") > 1:
                    lines = content.split('\n')
                    new_lines = []
                    net_count = 0
                    in_net_class = False
                    skip_until_next_def = False
                    
                    for line in lines:
                        if line.strip().startswith("class Net("):
                            net_count += 1
                            if net_count == 1:
                                in_net_class = True
                                skip_until_next_def = False
                                new_lines.append(line)
                            else:
                                # Skip duplicate class definition
                                skip_until_next_def = True
                                in_net_class = False
                                continue
                        elif skip_until_next_def:
                            # Skip until we hit a top-level definition (class or def at column 0)
                            if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                                if line.strip().startswith("def ") or line.strip().startswith("class "):
                                    skip_until_next_def = False
                                    if not line.strip().startswith("class Net("):
                                        new_lines.append(line)
                        elif in_net_class:
                            # Check if we're leaving the class (next top-level definition)
                            if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                                if (line.strip().startswith("class ") or 
                                    (line.strip().startswith("def ") and not line.strip().startswith("def __"))):
                                    in_net_class = False
                                    if not line.strip().startswith("class Net("):
                                        new_lines.append(line)
                                else:
                                    new_lines.append(line)
                            else:
                                new_lines.append(line)
                        else:
                            # Outside Net class, keep all lines
                            new_lines.append(line)
                    
                    content = '\n'.join(new_lines)
                
                # Remove duplicate function definitions (e.g., supported_hyperparameters)
                if content.count("def supported_hyperparameters(") > 1:
                    lines = content.split('\n')
                    new_lines = []
                    seen_supported_hyperparameters = False
                    in_function = False
                    skip_lines = False
                    
                    for line in lines:
                        if re.search(r'^\s*def\s+supported_hyperparameters\s*\(', line):
                            if seen_supported_hyperparameters:
                                # Skip duplicate
                                skip_lines = True
                                in_function = True
                                continue
                            else:
                                seen_supported_hyperparameters = True
                                in_function = True
                                new_lines.append(line)
                        elif in_function and skip_lines:
                            # Skip until we hit a non-indented line (end of function)
                            if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                                skip_lines = False
                                in_function = False
                                new_lines.append(line)
                            # Also stop if we hit another function/class definition
                            elif re.search(r'^\s*(def|class)\s+', line):
                                skip_lines = False
                                in_function = False
                                new_lines.append(line)
                        else:
                            if in_function and not line.strip():
                                # Empty line might be end of function
                                in_function = False
                            new_lines.append(line)
                    
                    content = '\n'.join(new_lines)
                
                # Replace F.adaptive_avg_pool2d with nn.AdaptiveAvgPool2d
                # Pattern: x = F.adaptive_avg_pool2d(x, (1, 1))
                # Replace with: x = nn.AdaptiveAvgPool2d((1, 1))(x)
                if "F.adaptive_avg_pool2d" in content:
                    # Replace function call pattern
                    content = re.sub(
                        r'F\.adaptive_avg_pool2d\(([^,]+),\s*\(([^)]+)\)\)',
                        r'nn.AdaptiveAvgPool2d((\2))(\1)',
                        content
                    )
                
                # Add missing required methods/functions for evaluation
                has_supported_hyperparameters = re.search(r'\bdef\s+supported_hyperparameters\s*\(', content)
                has_train_setup = re.search(r'\bdef\s+train_setup\s*\(', content)
                has_learn = re.search(r'\bdef\s+learn\s*\(', content)
                
                if not has_supported_hyperparameters or not has_train_setup or not has_learn:
                    lines = content.split('\n')
                    
                    # Find the Net class boundaries
                    class_start = None
                    class_end = None
                    forward_end = None
                    in_class = False
                    indent_level = None
                    
                    for i, line in enumerate(lines):
                        if 'class Net(' in line:
                            class_start = i
                            in_class = True
                            # Determine indent level (should be 0, but check)
                            indent_level = len(line) - len(line.lstrip())
                        elif in_class:
                            # Check if we're still in the class
                            if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                                # Top-level definition - class ended
                                if line.strip().startswith('def ') or line.strip().startswith('class '):
                                    if not line.strip().startswith('def __'):
                                        class_end = i
                                        in_class = False
                                        break
                            # Find end of forward method
                            if re.search(r'\s+def\s+forward\s*\(', line):
                                # Find end of forward method (next method or end of class)
                                for j in range(i+1, len(lines)):
                                    if lines[j].strip():
                                        # Check if it's a method definition or end of class
                                        if not lines[j].startswith(' ') and not lines[j].startswith('\t'):
                                            forward_end = j
                                            break
                                        elif re.search(r'\s+def\s+', lines[j]):
                                            forward_end = j
                                            break
                                if forward_end:
                                    break
                    
                    # If we didn't find class end, it's at the end of file
                    if class_end is None:
                        class_end = len(lines)
                    
                    # If forward_end not found, use class_end
                    if forward_end is None:
                        forward_end = class_end
                    
                    # Build additions list
                    additions = []
                    
                    # Add train_setup and learn methods INSIDE the class (after forward)
                    if not has_train_setup:
                        additions.append('    def train_setup(self, prm):')
                        additions.append('        self.to(self.device)')
                        additions.append('        self.criteria = (nn.CrossEntropyLoss().to(self.device),)')
                        additions.append('        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm[\'lr\'], momentum=prm[\'momentum\'])')
                        additions.append('')
                    
                    if not has_learn:
                        additions.append('    def learn(self, train_data):')
                        additions.append('        self.train()')
                        additions.append('        for inputs, labels in train_data:')
                        additions.append('            inputs, labels = inputs.to(self.device), labels.to(self.device)')
                        additions.append('            self.optimizer.zero_grad()')
                        additions.append('            outputs = self(inputs)')
                        additions.append('            loss = self.criteria[0](outputs, labels)')
                        additions.append('            loss.backward()')
                        additions.append('            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)')
                        additions.append('            self.optimizer.step()')
                        additions.append('')
                    
                    # Insert methods INSIDE the class (after forward, before class_end)
                    if additions:
                        # Ensure we're inserting inside the class
                        insert_pos = min(forward_end, class_end)
                        lines[insert_pos:insert_pos] = additions
                        # Update class_end since we added lines
                        class_end += len(additions)
                    
                    # Add supported_hyperparameters function OUTSIDE Net class (after class ends)
                    if not has_supported_hyperparameters:
                        lines.insert(class_end, '')
                        lines.insert(class_end, 'def supported_hyperparameters():')
                        lines.insert(class_end + 1, "    return {'lr', 'momentum'}")
                    
                    content = '\n'.join(lines)
                
                # Only write if content changed
                if content != original_content:
                    code_file.write_text(content)
                    fixed_count += 1
                    logger.debug(f"  ✓ Fixed {code_file.name}")
                else:
                    logger.debug(f"  - {code_file.name} already has imports")
                    
            except Exception as e:
                logger.warning(f"Failed to fix {code_file.name}: {e}")
                logger.debug(traceback.format_exc())
        
        if fixed_count > 0:
            logger.info(f"Fixed {fixed_count}/{total_files} generated code files (added imports, removed duplicates)")
        else:
            logger.info(f"All {total_files} code files already have required imports")
    
    def _revalidate_generated_models(self, results: List[Dict[str, Any]], accepted_code_dir: Path) -> List[Dict[str, Any]]:
        """
        Re-validate all generated code files.
        
        Many models are incorrectly marked as failed during generation but have valid code.
        This method re-validates all code files and updates the 'ok' status.
        
        Args:
            results: List of result dictionaries from results.jsonl
            accepted_code_dir: Directory containing model subdirectories (B0, B1, etc.)
            
        Returns:
            Updated results list with corrected 'ok' status
        """
        import torch
        import torch.nn as nn
        
        if not accepted_code_dir.exists():
            logger.warning(f"Accepted code directory not found: {accepted_code_dir}")
            return results
        
        logger.info(f"Re-validating {len(results)} models from {accepted_code_dir}...")
        
        # Build a map of index -> result for easy lookup
        results_map = {i: r for i, r in enumerate(results)}
        updated_count = 0
        newly_valid_count = 0
        
        for i, result in enumerate(results):
            model_id = f"B{i}"  # Model directories are named B0, B1, etc.
            model_dir = accepted_code_dir / model_id
            code_file = model_dir / "new_nn.py"
            
            # Skip if already marked as ok and file exists
            if result.get("ok", False) and code_file.exists():
                continue
            
            # If code file doesn't exist, skip
            if not code_file.exists():
                continue
            
            # Re-validate this model
            try:
                code = code_file.read_text()
                
                # Build execution namespace (same as nn_sftcodegen_rag.py)
                g = {
                    "__name__": "__gen__",
                    "__file__": "<gen>",
                    "torch": torch,
                    "nn": nn,
                    "F": torch.nn.functional,
                    "Optional": type(None),
                    "Any": type(None),
                    "Callable": type(None),
                    "Tuple": tuple,
                    "List": list,
                    "Dict": dict,
                    "Tensor": torch.Tensor,
                }
                
                # Execute code
                exec(code, g)
                
                # Get Net class
                Net = g.get("Net")
                if not Net:
                    continue  # Still invalid
                
                # Check required functions/methods
                supported_hyperparameters = g.get("supported_hyperparameters")
                if not supported_hyperparameters:
                    continue  # Still invalid
                
                if not hasattr(Net, 'train_setup') or not hasattr(Net, 'learn'):
                    continue  # Still invalid
                
                # Try instantiation
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                C, H, W = 3, 32, 32
                num_classes = 10
                in_shape = (1, C, H, W)
                out_shape = (num_classes,)
                prm = {'lr': 0.01, 'batch': 64, 'dropout': 0.2, 'momentum': 0.9}
                
                try:
                    model = Net(in_shape, out_shape, prm, device)
                    model.to(device)
                    
                    # Try forward pass
                    x = torch.randn(1, 3, 32, 32).to(device)
                    with torch.no_grad():
                        output = model(x)
                    
                    # Check output shape
                    if output.shape != (1, num_classes):
                        continue  # Still invalid
                    
                    # Model is valid!
                    was_invalid = not result.get("ok", False)
                    result["ok"] = True
                    
                    # Update file path if it was None (model was marked as failed)
                    if result.get("file") is None:
                        result["file"] = str(code_file)
                    
                    updated_count += 1
                    
                    if was_invalid:
                        newly_valid_count += 1
                        logger.info(f"  ✓ {model_id}: Re-validated as valid (was marked as failed)")
                    
                except Exception as e:
                    # Instantiation or forward pass failed - still invalid
                    continue
                    
            except Exception as e:
                # Code execution failed - still invalid
                continue
        
        logger.info(f"Re-validation complete: {updated_count} models updated ({newly_valid_count} newly validated)")
        
        return results
    
    def evaluate_models(self, cycle: int, generation_dir: Path, starting_checksum: int = 0) -> Dict[str, Any]:
        """
        Evaluate generated models using NNEval.py.
        
        Args:
            cycle: Current cycle number
            generation_dir: Directory containing generated models
            starting_checksum: Starting index for model naming in evaluation (default: 0)
        
        Returns:
            Dictionary with evaluation results (accuracies, etc.)
        """
        logger.info("")
        logger.info("="*80)
        logger.info(f"CYCLE {cycle}: MODEL EVALUATION")
        logger.info("="*80)
        logger.info(f"Starting checksum: {starting_checksum}")
        
        # Prepare models for NNEval
        nneval_dir = self.output_dir / f"cycle_{cycle}" / "nneval"
        nneval_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if evaluation already exists - will be handled by evaluate_cycle_models.py
        # which will skip already-evaluated models and only evaluate missing ones
        evaluation_results_file = nneval_dir.parent / "evaluation_results.json"
        if evaluation_results_file.exists():
            logger.info(f"✓ Existing evaluation results found: {evaluation_results_file}")
            logger.info("Will check for missing models and only evaluate those")
        
        # Read generation results
        results_file = generation_dir / "results.jsonl"
        results = []
        with open(results_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        
        # Models are in generation_dir/accepted_code (B0, B1, etc.)
        accepted_code_dir = generation_dir / "accepted_code"
        if not accepted_code_dir.exists():
            logger.warning(f"Accepted code directory not found: {accepted_code_dir}")
            return {
                "success": True,
                "models_trained": 0,
                "accuracies": [],
            }
        
        # Rank models by structural quality before evaluation
        logger.info("Ranking models by structural quality...")
        reranker = StructuralReranker()
        
        valid_models = []
        for i, rec in enumerate(results):
            if not rec.get("ok", False):
                continue
            # Models are in accepted_code/B{i}/new_nn.py
            model_dir = accepted_code_dir / f"B{i}"
            code_path = model_dir / "new_nn.py"
            if not code_path.exists():
                continue
            valid_models.append((i, rec, code_path))
        
        if len(valid_models) == 0:
            logger.warning("No valid models to evaluate")
            return {
                "success": True,
                "models_trained": 0,
                "accuracies": [],
            }
        
        # Score all valid models
        scored_models = []
        for i, rec, code_path in valid_models:
            code = code_path.read_text()
            score_result = reranker.score_model(code, rec.get("id", f"gen_{i}"))
            scored_models.append((i, rec, code_path, score_result["score"], score_result))
        
        # Sort by score descending (best first)
        scored_models.sort(key=lambda x: x[3], reverse=True)
        
        # Save rankings
        rankings_file = generation_dir / "structural_rankings.json"
        rankings_data = [
            {
                "model_id": rec.get("id", f"gen_{idx}"),
                "file": str(code_path),
                "structural_score": score,
                "details": details
            }
            for idx, rec, code_path, score, details in scored_models
        ]
        with open(rankings_file, 'w') as f:
            json.dump(rankings_data, f, indent=2)
        logger.info(f"Saved structural rankings to {rankings_file}")
        
        # Log top models
        logger.info("Top 5 models by structural score:")
        for rank, (idx, rec, code_path, score, details) in enumerate(scored_models[:5], 1):
            logger.info(f"  {rank}. {code_path.name} - Score: {score:.2f}")
            if "breakdown" in details:
                logger.info(f"     Patterns: {details['breakdown']}")
        
        # Prepare models for evaluation (in ranked order)
        # Apply starting_checksum offset for model naming
        # IMPORTANT: Use original idx (not sorted position) to preserve mapping
        # with generation_results order expected by filter_successful_novel
        prepared = 0
        for idx, rec, code_path, score, details in scored_models:
            eval_idx = starting_checksum + idx
            model_dir = nneval_dir / f"gen_{eval_idx:04d}"
            model_dir.mkdir(exist_ok=True)
            
            # Copy code
            code = code_path.read_text()
            (model_dir / "new_nn.py").write_text(code)
            
            # Create metadata (include structural score)
            metadata = {
                "source": str(code_path),
                "model_id": rec.get("id", f"gen_{idx}"),
                "params": rec.get("params", 0),
                "class": rec.get("class", "Net"),
                "structural_score": score,
                "structural_patterns": details.get("patterns", {}),
            }
            (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
            
            prepared += 1
        
        logger.info(f"Prepared {prepared} models for training (ranked by structural quality)")
        
        if prepared == 0:
            return {
                "success": True,
                "models_trained": 0,
                "accuracies": [],
            }
        
        # Run evaluation using Eval API wrapper (bypasses NNEval argument bug)
        # NOTE: NNEval.py has an argument misalignment bug that prevents custom_synth_dir
        # from being passed correctly. We use a wrapper script that directly calls Eval API.
        cmd = [
            "python", "-m", "ab.gpt.iterative_pipeline.evaluate_cycle_models",
            "--cycle", str(cycle),
            "--nneval_dir", str(nneval_dir),
        ]
        
        logger.info(f"Running evaluation wrapper: {' '.join(cmd)}")
        
        # Retry evaluation with exponential backoff
        def run_evaluation_cmd():
            # Stream output in real-time
            result = subprocess.run(
                cmd,
                capture_output=False,  # Stream output to terminal in real-time
                text=True,
                check=False
            )
            if result.returncode != 0:
                logger.error(f"Evaluation wrapper failed with exit code {result.returncode}")
                logger.error("Check terminal output above for error details")
                logger.warning("Attempting to collect partial results if available...")
                # Don't raise immediately - allow partial result collection
                # The validation step will catch if no results are available
            return result
        
        evaluation_succeeded = True
        try:
            result = RetryHandler.retry_with_backoff(
                run_evaluation_cmd,
                max_retries=self.max_retries,
                initial_delay=60.0,  # 1 minute initial delay (evaluation takes time)
                backoff_factor=1.5,
                exceptions=(subprocess.CalledProcessError, OSError),
                operation_name=f"Model evaluation cycle {cycle}"
            )
            # Check if evaluation actually succeeded
            if result.returncode != 0:
                evaluation_succeeded = False
                logger.warning(f"Evaluation returned non-zero exit code: {result.returncode}")
                logger.warning("Attempting to collect partial results if available...")
        except subprocess.CalledProcessError as e:
            evaluation_succeeded = False
            logger.warning(f"Model evaluation failed after {self.max_retries} retries (exit code: {e.returncode})")
            logger.warning("Attempting to collect partial evaluation results...")
            # Continue to result collection - will handle empty results gracefully
        except FileNotFoundError:
            evaluation_succeeded = False
            logger.error("Evaluation wrapper module not found: ab.gpt.iterative_pipeline.evaluate_cycle_models")
            logger.error("Please ensure the module is properly installed")
            # Continue to result collection - will handle empty results gracefully
        except Exception as e:
            evaluation_succeeded = False
            logger.warning(f"Model evaluation failed with unexpected error: {e}")
            logger.warning("Attempting to collect partial evaluation results...")
            # Continue to result collection - will handle empty results gracefully
        
        # Validate evaluation output
        is_valid, msg = StageValidator.validate_evaluation_output(nneval_dir, min_expected=1)
        if not is_valid:
            if evaluation_succeeded:
                logger.warning(f"Evaluation validation warning: {msg}")
            else:
                logger.warning(f"Evaluation failed, but checking for partial results: {msg}")
            # Continue but log warning - will handle empty results gracefully
        
        # Load results from evaluation_results.json (written by evaluate_cycle_models.py)
        # This avoids loading stale results from previous runs
        evaluation_results_file = nneval_dir.parent / "evaluation_results.json"
        accuracies = []
        
        if evaluation_results_file.exists():
            try:
                eval_data = json.loads(evaluation_results_file.read_text())
                # Filter only successful evaluations with accuracy
                for model in eval_data.get("models", []):
                    if model.get("success", False) and "accuracy" in model:
                        accuracies.append({
                            "model_id": model["model_id"],
                            "accuracy": model["accuracy"],
                            "code_file": model["code_file"],
                            "metadata_file": "",  # Not used in current flow
                        })
                        logger.info(f"Model {model['model_id']}: {model['accuracy']*100:.2f}% accuracy")
            except Exception as e:
                logger.error(f"Failed to load evaluation_results.json: {e}")
                logger.warning("Falling back to scanning nneval directory...")
                
                # Fallback: Scan directory for 1.json files (old behavior)
                for model_dir in nneval_dir.glob("gen_*"):
                    if not model_dir.is_dir():
                        continue
                    
                    result_file = model_dir / "1.json"
                    if result_file.exists():
                        try:
                            data = json.loads(result_file.read_text())
                            if isinstance(data, list) and len(data) > 0:
                                acc = data[0].get("accuracy", data[0].get("acc", None))
                                if acc is not None:
                                    accuracies.append({
                                        "model_id": model_dir.name,
                                        "accuracy": acc,
                                        "metadata_file": str(model_dir / "metadata.json"),
                                        "code_file": str(model_dir / "new_nn.py"),
                                    })
                                    logger.info(f"Model {model_dir.name}: {acc*100:.2f}% accuracy")
                        except Exception as e2:
                            logger.debug(f"Failed to read {result_file}: {e2}")
        else:
            logger.warning(f"evaluation_results.json not found at {evaluation_results_file}")
            logger.warning("Expected file to be created by evaluate_cycle_models.py")
        
        if len(accuracies) == 0:
            logger.warning("No evaluation results found. This may indicate evaluation failed completely.")
            logger.warning("Cycle will continue but no models will be selected for training.")
        
        # Load the full evaluation_results.json to get the "models" format expected by filter_successful_novel
        evaluation_results_file = nneval_dir.parent / "evaluation_results.json"
        if evaluation_results_file.exists():
            try:
                eval_data = json.loads(evaluation_results_file.read_text())
                # Return in the format expected by filter_successful_novel
                return {
                    "success": True,
                    "models_trained": len(accuracies),
                    "accuracies": accuracies,  # Keep for backward compatibility
                    "models": eval_data.get("models", []),  # Add models list for filter_successful_novel
                    "best_accuracy": max([a["accuracy"] for a in accuracies]) if accuracies else 0.0,
                    "avg_accuracy": sum([a["accuracy"] for a in accuracies]) / len(accuracies) if accuracies else 0.0,
                }
            except Exception as e:
                logger.warning(f"Failed to load full evaluation_results.json: {e}")
        
        # Fallback: Convert accuracies to models format
        models_list = []
        for acc_info in accuracies:
            models_list.append({
                "model_id": acc_info["model_id"],
                "success": True,
                "accuracy": acc_info["accuracy"],
                "code_file": acc_info.get("code_file", ""),
            })
        
        return {
            "success": True,  # Always return success - empty results handled gracefully
            "models_trained": len(accuracies),
            "accuracies": accuracies,
            "models": models_list,  # Add models list for filter_successful_novel
            "best_accuracy": max([a["accuracy"] for a in accuracies]) if accuracies else 0.0,
            "avg_accuracy": sum([a["accuracy"] for a in accuracies]) / len(accuracies) if accuracies else 0.0,
        }
    
    def filter_successful_novel(
        self,
        generation_results: Dict[str, Any],
        evaluation_results: Dict[str, Any],
        starting_checksum: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Filter for successful AND novel models with adaptive threshold and top-K fallback.
        
        Selection strategy:
        1. Compute adaptive threshold if enabled (60-70th percentile)
        2. Select all models above threshold that are novel
        3. If selected < min_selected_k, add next-best novel models down to fallback_threshold
        
        Returns:
            List of selected models with metadata
        """
        import numpy as np
        
        # Load from evaluation_results["models"] (written by evaluate_cycle_models.py)
        # Filter only successful evaluations with accuracy
        accuracies = {a["model_id"]: a for a in evaluation_results.get("models", []) if a.get("success", False) and "accuracy" in a}
        
        # Build candidate list with novelty pre-check
        candidates = []
        results_list = generation_results.get("results", [])
        logger.info(f"Building candidates from {len(results_list)} generation results")
        logger.info(f"Accuracies dict has {len(accuracies)} successful evaluations")
        logger.info(f"Starting checksum: {starting_checksum}")
        
        if not results_list:
            logger.error("ERROR: generation_results.get('results', []) returned empty list!")
            logger.error(f"generation_results keys: {list(generation_results.keys())}")
            logger.error(f"generation_results type: {type(generation_results)}")
        
        for i, result in enumerate(results_list):
            if not result.get("ok", False):
                logger.debug(f"  gen_{i:04d}: skipped (ok=False)")
                continue
            
            # Use starting_checksum offset to match evaluation model_ids
            model_id = f"gen_{starting_checksum + i:04d}"
            if model_id not in accuracies:
                logger.debug(f"  gen_{i:04d} → {model_id}: skipped (not in accuracies)")
                continue
            
            acc_info = accuracies[model_id]
            accuracy = acc_info["accuracy"]
            code_file_str = acc_info["code_file"]
            # Resolve path (handle both absolute and relative paths)
            code_path = Path(code_file_str)
            if not code_path.is_absolute():
                # If relative, resolve relative to current working directory
                code_path = code_path.resolve()
            
            # Check novelty if enabled
            is_novel = True
            if self.novelty_check_enabled:
                try:
                    if not code_path.exists():
                        logger.warning(f"Model {model_id}: Code file not found: {code_path} (original: {code_file_str})")
                        is_novel = False
                    else:
                        code = code_path.read_text()
                        is_novel = self.novelty_checker.is_novel(code, model_id)
                        if not is_novel:
                            logger.info(f"Model {model_id} rejected: not novel (acc={accuracy*100:.2f}%)")
                        else:
                            logger.debug(f"Model {model_id} is novel (acc={accuracy*100:.2f}%)")
                except Exception as e:
                    logger.error(f"Error checking novelty for {model_id}: {e}")
                    logger.debug(traceback.format_exc())
                    is_novel = False  # Fail-safe: mark as non-novel on error
            
            candidates.append({
                "model_id": model_id,
                "code_file": str(code_path),  # Ensure it's a string, not Path
                "accuracy": accuracy,
                "params": result.get("params", 0),
                "metadata": {k: (str(v) if isinstance(v, Path) else v) for k, v in acc_info.items()},  # Convert any Path objects to strings
                "is_novel": is_novel,
            })
            logger.info(f"  Added candidate: {model_id}, acc={accuracy*100:.2f}%, novel={is_novel}, file={code_path.name}")
        
        logger.info(f"Built {len(candidates)} candidates from generation and evaluation results")
        
        # Sort by accuracy descending
        candidates.sort(key=lambda x: x["accuracy"], reverse=True)
        
        # Compute effective threshold
        effective_threshold = self.accuracy_threshold
        if self.adaptive_threshold_enabled and len(candidates) >= 10:
            all_accs = [c["accuracy"] for c in candidates]
            p65 = np.percentile(all_accs, 65)
            effective_threshold = max(self.accuracy_threshold, p65)
            logger.info(f"Adaptive threshold: {effective_threshold*100:.2f}% (65th percentile: {p65*100:.2f}%)")
        else:
            logger.info(f"Using fixed threshold: {effective_threshold*100:.2f}%")
        
        # Phase 1: Select all novel models above effective threshold
        selected = []
        logger.info(f"Filtering {len(candidates)} candidates with threshold {effective_threshold*100:.2f}%")
        for candidate in candidates:
            is_above_threshold = candidate["accuracy"] >= effective_threshold
            logger.debug(f"  {candidate['model_id']}: acc={candidate['accuracy']*100:.2f}%, novel={candidate['is_novel']}, above_threshold={is_above_threshold}")
            if candidate["is_novel"] and is_above_threshold:
                selected.append(candidate)
                logger.info(f"Model {candidate['model_id']} selected: acc={candidate['accuracy']*100:.2f}%, params={candidate['params']}")
        
        # Phase 2: Top-K fallback if needed
        fallback_added = 0
        if len(selected) < self.min_selected_k:
            logger.warning(f"Only {len(selected)} models above threshold (target: {self.min_selected_k})")
            logger.info(f"Applying top-K fallback (fallback threshold: {self.fallback_threshold*100:.2f}%)")
            
            for candidate in candidates:
                if len(selected) >= self.min_selected_k:
                    break
                
                # Skip if already selected or below fallback threshold
                if candidate in selected or candidate["accuracy"] < self.fallback_threshold:
                    continue
                
                # Only select novel models in fallback
                if candidate["is_novel"]:
                    selected.append(candidate)
                    fallback_added += 1
                    logger.info(f"Model {candidate['model_id']} selected (fallback): acc={candidate['accuracy']*100:.2f}%")
            
            if fallback_added > 0:
                logger.info(f"Added {fallback_added} models via fallback (total: {len(selected)})")
        
        # Re-sort final selection by accuracy
        selected.sort(key=lambda x: x["accuracy"], reverse=True)
        
        logger.info(f"Final selection: {len(selected)} models out of {len(accuracies)} evaluated")
        if len(selected) > 0:
            logger.info(f"Accuracy range: {selected[0]['accuracy']*100:.2f}% - {selected[-1]['accuracy']*100:.2f}%")
        else:
            logger.warning("No models selected - all evaluations failed or below threshold")
        
        # Save novelty checker cache
        if self.novelty_check_enabled:
            self.novelty_checker.save_cache()
        
        return selected
    
    def convert_to_training_data(
        self,
        selected_models: List[Dict[str, Any]],
        cycle: int,
        checkpoint: str,
    ) -> List[Dict[str, Any]]:
        """Convert selected models to chat training examples."""
        logger.info(f"Converting {len(selected_models)} models to training examples...")
        
        chat_examples = []
        for model_info in selected_models:
            code_path = Path(model_info["code_file"])
            code = code_path.read_text()
            
            metadata = {
                "accuracy": model_info["accuracy"],
                "params": model_info["params"],
                "cycle": cycle,
                "checkpoint": checkpoint,
            }
            
            chat_example = self.data_manager.convert_code_to_chat_example(
                code,
                model_info["model_id"],
                metadata
            )
            
            chat_examples.append(chat_example)
        
        return chat_examples
    
    def generate_final_report(self) -> Dict[str, Any]:
        """
        Generate analysis, plots, and summary report after pipeline completion.
        
        Returns:
            Dictionary with success status and list of generated files
        """
        logger.info("")
        logger.info("="*80)
        logger.info("GENERATING FINAL ANALYSIS AND REPORTS")
        logger.info("="*80)
        
        results = {
            "success": True,
            "files_generated": [],
            "errors": []
        }
        
        try:
            # Step 1: Run cycle analysis (generates cycle_analysis.json and plots)
            logger.info("Running cycle analysis...")
            analyze_cmd = [
                "python", "-m", "ab.gpt.iterative_pipeline.analyze_cycles",
                "--results_dir", str(self.output_dir)
            ]
            
            analyze_result = subprocess.run(
                analyze_cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if analyze_result.returncode == 0:
                logger.info("✓ Cycle analysis completed")
                results["files_generated"].extend([
                    str(self.output_dir / "cycle_analysis.json"),
                    str(self.output_dir / "cycle_metrics.csv"),
                    str(self.output_dir / "cycle_analysis.png"),
                    str(self.output_dir / "confidence_intervals.png")
                ])
            else:
                error_msg = f"Cycle analysis failed: {analyze_result.stderr}"
                logger.warning(error_msg)
                results["errors"].append(error_msg)
            
            # Step 2: Generate Wilson CI plots
            logger.info("Generating Wilson confidence interval plots...")
            wilson_cmd = [
                "python", "-m", "ab.gpt.iterative_pipeline.plot_wilson_ci",
                "--results_dir", str(self.output_dir)
            ]
            
            wilson_result = subprocess.run(
                wilson_cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if wilson_result.returncode == 0:
                logger.info("✓ Wilson CI plots generated")
                plots_dir = self.output_dir / "plots"
                results["files_generated"].extend([
                    str(plots_dir / "wilson_ci_generation_rate.png"),
                    str(plots_dir / "wilson_ci_accuracy_40.png")
                ])
            else:
                error_msg = f"Wilson CI plotting failed: {wilson_result.stderr}"
                logger.warning(error_msg)
                results["errors"].append(error_msg)
            
            # Step 3: Generate text summary
            logger.info("Creating text summary...")
            try:
                self._generate_text_summary()
                summary_file = self.output_dir / "pipeline_summary.txt"
                results["files_generated"].append(str(summary_file))
                logger.info(f"✓ Summary saved to: {summary_file}")
            except Exception as e:
                error_msg = f"Failed to create text summary: {e}"
                logger.warning(error_msg)
                results["errors"].append(error_msg)
            
            # Overall status
            if results["errors"]:
                logger.warning(f"Report generation completed with {len(results['errors'])} errors")
                results["success"] = False
            else:
                logger.info(f"✓ All reports generated successfully ({len(results['files_generated'])} files)")
            
            return results
            
        except Exception as e:
            logger.error(f"Unexpected error during report generation: {e}")
            results["success"] = False
            results["errors"].append(str(e))
            return results
    
    def _generate_text_summary(self):
        """Generate human-readable text summary of pipeline results."""
        summary_file = self.output_dir / "pipeline_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ITERATIVE FINE-TUNING PIPELINE - SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            
            # Configuration
            f.write("-"*80 + "\n")
            f.write("CONFIGURATION\n")
            f.write("-"*80 + "\n")
            f.write(f"Cycles: {self.cycles}\n")
            f.write(f"Models per cycle: {self.models_per_cycle}\n")
            f.write(f"Samples per prompt: {self.samples_per_prompt}\n")
            f.write(f"Accuracy threshold: {self.accuracy_threshold*100:.1f}%\n")
            f.write(f"Optimized training: {self.use_optimized_training}\n")
            f.write(f"Novelty checking: {self.novelty_check_enabled}\n")
            f.write(f"LLM config: {self.llm_conf}\n")
            f.write("\n")
            
            # Results per cycle
            f.write("-"*80 + "\n")
            f.write("CYCLE RESULTS\n")
            f.write("-"*80 + "\n\n")
            
            successful_cycles = [r for r in self.cycle_results if r.get("success")]
            
            for result in self.cycle_results:
                if result.get("success"):
                    cycle = result["cycle"]
                    eval_data = result.get("evaluation", {})
                    gen_data = result.get("generation", {})
                    train_data = result.get("training", {})
                    
                    f.write(f"Cycle {cycle}:\n")
                    f.write(f"  Generation:\n")
                    f.write(f"    Total generated: {gen_data.get('total_generated', 0)}\n")
                    f.write(f"    Success rate: {eval_data.get('success_rate', 0)*100:.1f}%\n")
                    f.write(f"  Evaluation:\n")
                    f.write(f"    Models trained: {eval_data.get('models_trained', 0)}\n")
                    f.write(f"    Best accuracy: {eval_data.get('best_accuracy', 0)*100:.2f}%\n")
                    f.write(f"    Avg accuracy: {eval_data.get('avg_accuracy', 0)*100:.2f}%\n")
                    f.write(f"  Selection:\n")
                    f.write(f"    Models selected: {gen_data.get('selected_for_training', 0)}\n")
                    f.write(f"    Novel models: {gen_data.get('novel', 0)}\n")
                    f.write(f"  Training Data:\n")
                    f.write(f"    New examples added: {train_data.get('new_examples_added', 0)}\n")
                    f.write(f"    Total examples: {train_data.get('total_examples', 0)}\n")
                    f.write(f"  Time: {result.get('cycle_time_minutes', 0):.1f} minutes\n\n")
                else:
                    cycle = result.get("cycle", "?")
                    error = result.get("error", "unknown")
                    f.write(f"Cycle {cycle}: FAILED ({error})\n\n")
            
            # Summary statistics
            if successful_cycles:
                f.write("-"*80 + "\n")
                f.write("SUMMARY STATISTICS\n")
                f.write("-"*80 + "\n")
                
                first = successful_cycles[0]
                last = successful_cycles[-1]
                
                f.write(f"Total successful cycles: {len(successful_cycles)}\n")
                f.write(f"Total pipeline time: {sum(r.get('cycle_time_minutes', 0) for r in successful_cycles) / 60:.1f} hours\n\n")
                
                f.write("Improvement (First → Last Cycle):\n")
                f.write(f"  Success rate: {first['evaluation']['success_rate']*100:.1f}% → {last['evaluation']['success_rate']*100:.1f}% ")
                f.write(f"({(last['evaluation']['success_rate'] - first['evaluation']['success_rate'])*100:+.1f}%)\n")
                f.write(f"  Best accuracy: {first['evaluation']['best_accuracy']*100:.2f}% → {last['evaluation']['best_accuracy']*100:.2f}% ")
                f.write(f"({(last['evaluation']['best_accuracy'] - first['evaluation']['best_accuracy'])*100:+.2f}%)\n")
                f.write(f"  Training examples: {first['training']['total_examples']} → {last['training']['total_examples']} ")
                f.write(f"(+{last['training']['total_examples'] - first['training']['total_examples']})\n")
            
            f.write("\n" + "-"*80 + "\n")
            f.write("GENERATED FILES\n")
            f.write("-"*80 + "\n")
            f.write(f"Detailed analysis: {self.output_dir / 'cycle_analysis.json'}\n")
            f.write(f"CSV metrics: {self.output_dir / 'cycle_metrics.csv'}\n")
            f.write(f"Main plots: {self.output_dir / 'cycle_analysis.png'}\n")
            f.write(f"Wilson CI plots: {self.output_dir / 'plots/'}\n")
            f.write(f"All results: {self.output_dir / 'all_cycles_results.json'}\n")
            f.write("\n" + "="*80 + "\n")
    
    def run_cycle(self, cycle: int) -> Dict[str, Any]:
        """Run a single fine-tuning cycle."""
        logger.info("")
        logger.info("="*80)
        logger.info(f"STARTING CYCLE {cycle}")
        logger.info("="*80)
        
        cycle_start = time.time()
        
        # Get training data directory for this cycle
        # For cycle 1, use base_data_dir (curation_output/chat_data)
        # For subsequent cycles, use augmented data from previous cycle
        if cycle == 1:
            data_dir = self.base_data_dir
        else:
            data_dir = self.data_manager.get_training_data_dir(cycle-1, self.output_dir)
        
        # Step 1: Fine-tune (skip if checkpoint exists)
        checkpoint_path = self.output_dir / f"cycle_{cycle}" / "checkpoint"
        if checkpoint_path.exists() and (checkpoint_path / "adapter_config.json").exists():
            logger.info(f"✓ Existing checkpoint found: {checkpoint_path}")
            logger.info("Skipping fine-tuning (checkpoint already exists)")
            ft_result = {"success": True, "checkpoint_dir": str(checkpoint_path)}
        else:
            ft_result = self.run_finetuning(cycle, data_dir)
            if not ft_result.get("success", False):
                return {
                    "cycle": cycle,
                    "success": False,
                    "error": ft_result.get("error", "unknown"),
                }
            checkpoint_path = Path(ft_result["checkpoint_dir"])
        
        # Step 2: Generate models (skip if generation results exist)
        generation_dir = self.output_dir / f"cycle_{cycle}" / "generation"
        results_file = generation_dir / "results.jsonl"
        if results_file.exists() and (generation_dir / "accepted_code").exists():
            logger.info(f"✓ Existing generation results found: {generation_dir}")
            logger.info("Skipping generation (results already exist)")
            # Load existing results
            results = []
            with open(results_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            results.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            gen_result = {
                "success": True,
                "output_dir": str(generation_dir),
                "total_generated": len(results),
                "successful": len([r for r in results if r.get("ok", False)]),
                "results": results,
            }
            logger.info(f"Loaded {len(results)} existing generation results")
        else:
            # Pass the same data_dir used for training so RAG examples match the training distribution
            gen_result = self.generate_models(cycle, str(checkpoint_path), data_dir=Path(data_dir))
            if not gen_result.get("success", False):
                return {
                    "cycle": cycle,
                    "success": False,
                    "error": gen_result.get("error", "unknown"),
                }
        
        # Step 3: Evaluate models
        # Calculate starting checksum based on previous cycles
        starting_checksum = 0
        for prev_cycle in range(1, cycle):
            prev_results_file = self.output_dir / f"cycle_{prev_cycle}" / "cycle_results.json"
            if prev_results_file.exists():
                try:
                    with open(prev_results_file, 'r') as f:
                        prev_results = json.load(f)
                        starting_checksum += prev_results.get("generation", {}).get("total_generated", 0)
                except Exception as e:
                    logger.warning(f"Could not read previous cycle results: {e}")
        
        logger.info(f"Starting checksum for cycle {cycle} evaluation: {starting_checksum}")
        eval_result = self.evaluate_models(cycle, Path(gen_result["output_dir"]), starting_checksum=starting_checksum)
        
        # Step 4: Filter successful & novel models
        # Ensure gen_result has 'results' key - reload if missing (safety check for resume scenarios)
        if "results" not in gen_result or not gen_result.get("results"):
            logger.warning("gen_result missing 'results' key or empty - reloading from results.jsonl")
            results_file = Path(gen_result.get("output_dir", "")) / "results.jsonl"
            if results_file.exists():
                results = []
                with open(results_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                results.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
                gen_result["results"] = results
                logger.info(f"Reloaded {len(results)} results from {results_file}")
            else:
                logger.error(f"Cannot reload results - file not found: {results_file}")
        
        logger.info(f"Filtering with {len(gen_result.get('results', []))} generation results")
        selected_models = self.filter_successful_novel(gen_result, eval_result, starting_checksum=starting_checksum)
        
        # Step 5: Convert to training data
        if selected_models:
            # Ensure checkpoint_path is a string (not Path) for JSON serialization
            checkpoint_str = str(checkpoint_path) if isinstance(checkpoint_path, Path) else checkpoint_path
            chat_examples = self.convert_to_training_data(
                selected_models,
                cycle,
                checkpoint_str
            )
            
            # Step 6: Update training data
            augment_stats = self.data_manager.augment_training_data(
                chat_examples,
                cycle,
                self.output_dir
            )
            
            # Step 6b: Mark selected models as seen in novelty checker
            # (Only models actually added to training should be in seen set)
            if self.novelty_check_enabled:
                for model_info in selected_models:
                    code_path = Path(model_info["code_file"])
                    code = code_path.read_text()
                    self.novelty_checker.mark_as_seen(
                        code, 
                        model_id=model_info["model_id"],
                        source=f"cycle_{cycle}_selected"
                    )
                logger.info(f"Marked {len(selected_models)} selected models as seen in novelty checker")
                self.novelty_checker.save_cache()
        else:
            logger.warning(f"No models selected for cycle {cycle}")
            # Still need to create training data directory for next cycle
            # (even if no new models are added, we need to copy previous cycle's data)
            augment_stats = self.data_manager.augment_training_data(
                [],  # Empty list - no new examples
                cycle,
                self.output_dir
            )
        
        cycle_time = time.time() - cycle_start
        
        # Compile results
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
            "cycle_time_minutes": cycle_time / 60,
        }
        
        # Save cycle results
        cycle_results_file = self.output_dir / f"cycle_{cycle}" / "cycle_results.json"
        cycle_results_file.parent.mkdir(parents=True, exist_ok=True)
        cycle_results_file.write_text(json.dumps(result, indent=2))
        
        return result
    
    def run(self):
        """Run the full iterative fine-tuning pipeline."""
        logger.info("="*80)
        logger.info("ITERATIVE FINE-TUNING PIPELINE")
        logger.info("="*80)
        logger.info(f"Base data: {self.base_data_dir}")
        logger.info(f"Output dir: {self.output_dir}")
        logger.info(f"Cycles: {self.cycles}")
        logger.info(f"Models per cycle: {self.models_per_cycle}")
        logger.info(f"Samples per prompt: {self.samples_per_prompt}")
        logger.info(f"Accuracy threshold: {self.accuracy_threshold*100:.1f}%")
        logger.info(f"Novelty check: {self.novelty_check_enabled}")
        logger.info(f"Max retries per operation: {self.max_retries}")
        if self.resume_from_cycle:
            logger.info(f"Resuming from cycle: {self.resume_from_cycle}")
        logger.info("="*80)
        
        # Determine starting cycle
        start_cycle = 1
        if self.resume_from_cycle:
            start_cycle = self.resume_from_cycle
            # Load previous results
            for cycle in range(1, start_cycle):
                cycle_file = self.output_dir / f"cycle_{cycle}" / "cycle_results.json"
                if cycle_file.exists():
                    try:
                        with open(cycle_file, 'r') as f:
                            self.cycle_results.append(json.load(f))
                    except Exception as e:
                        logger.warning(f"Failed to load results for cycle {cycle}: {e}")
            logger.info(f"Loaded {len(self.cycle_results)} previous cycle results")
        
        # Run cycles
        for cycle in range(start_cycle, self.cycles + 1):
            try:
                result = self.run_cycle(cycle)
                self.cycle_results.append(result)
                
                # Save intermediate results after each cycle
                all_results_file = self.output_dir / "all_cycles_results.json"
                all_results_file.write_text(json.dumps(self.cycle_results, indent=2))
                logger.info(f"Saved intermediate results to {all_results_file}")
                
                if not result.get("success", False):
                    error = result.get("error", "unknown")
                    logger.error(f"Cycle {cycle} failed: {error}")
                    logger.error("="*80)
                    logger.error("PIPELINE STOPPED DUE TO CYCLE FAILURE")
                    logger.error("="*80)
                    logger.error(f"Failed cycle: {cycle}")
                    logger.error(f"Error: {error}")
                    logger.error("")
                    logger.error("To resume after fixing the issue:")
                    logger.error(f"  python iterative_finetune.py --resume_from_cycle {cycle + 1} ...")
                    logger.error("")
                    logger.error("The pipeline has saved all results up to this point.")
                    logger.error("="*80)
                    break
                
                # Validate cycle completion
                cycle_file = self.output_dir / f"cycle_{cycle}" / "cycle_results.json"
                if not cycle_file.exists():
                    logger.error(f"Cycle {cycle} results file not found after completion!")
                    break
                    
            except KeyboardInterrupt:
                logger.warning(f"\nPipeline interrupted by user at cycle {cycle}")
                logger.info(f"Results saved. Resume with: --resume_from_cycle {cycle + 1}")
                break
            except Exception as e:
                logger.exception(f"Unexpected error in cycle {cycle}: {e}")
                logger.error(f"Pipeline stopped due to unexpected error. Resume with: --resume_from_cycle {cycle + 1}")
                break
        
        # Save all results
        all_results_file = self.output_dir / "all_cycles_results.json"
        all_results_file.write_text(json.dumps(self.cycle_results, indent=2))
        
        # Generate final analysis and reports
        logger.info("")
        logger.info("="*80)
        logger.info("GENERATING FINAL ANALYSIS")
        logger.info("="*80)
        
        try:
            report_result = self.generate_final_report()
            
            if report_result["success"]:
                logger.info("\n✓ All reports generated successfully!")
                logger.info("\nGenerated files:")
                for file_path in report_result.get("files_generated", []):
                    if Path(file_path).exists():
                        logger.info(f"  ✓ {file_path}")
            else:
                logger.warning("\n⚠ Report generation completed with errors")
                for error in report_result.get("errors", []):
                    logger.warning(f"  - {error}")
                logger.info("\nPipeline results are saved. You can generate reports manually:")
                logger.info(f"  python -m ab.gpt.iterative_pipeline.analyze_cycles --results_dir {self.output_dir}")
                logger.info(f"  python -m ab.gpt.iterative_pipeline.plot_wilson_ci --results_dir {self.output_dir}")
                
        except Exception as e:
            logger.error(f"\n✗ Failed to generate final reports: {e}")
            logger.info("\nPipeline results are saved. Generate reports manually:")
            logger.info(f"  python -m ab.gpt.iterative_pipeline.analyze_cycles --results_dir {self.output_dir}")
            logger.info(f"  python -m ab.gpt.iterative_pipeline.plot_wilson_ci --results_dir {self.output_dir}")
        
        logger.info("")
        logger.info("")
        logger.info("="*80)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*80)
        logger.info(f"Results saved to: {all_results_file}")
        logger.info("")
        logger.info("Summary:")
        for result in self.cycle_results:
            if result.get("success"):
                cycle = result["cycle"]
                success_rate = result["evaluation"]["success_rate"] * 100
                best_acc = result["evaluation"]["best_accuracy"] * 100
                avg_acc = result["evaluation"]["avg_accuracy"] * 100
                added = result["training"]["new_examples_added"]
                logger.info(f"  Cycle {cycle}: {success_rate:.1f}% success, "
                           f"best={best_acc:.2f}%, avg={avg_acc:.2f}%, added={added} examples")


def main():
    parser = argparse.ArgumentParser(description="Iterative Fine-Tuning Pipeline")
    parser.add_argument("--base_data_dir", type=str, required=True,
                        help="Path to original chat_data directory")
    parser.add_argument("--output_dir", type=str, default="out/iterative_cycles",
                        help="Output directory for all cycle results")
    parser.add_argument("--llm_conf", type=str, required=True,
                        help="LLM configuration file (e.g., ds_coder_7b_instruct.json)")
    parser.add_argument("--cycles", type=int, default=5,
                        help="Number of fine-tuning cycles to run")
    parser.add_argument("--models_per_cycle", type=int, default=150,
                        help="Number of models to generate per cycle (default: 150)")
    parser.add_argument("--samples_per_prompt", type=int, default=1,
                        help="Number of models to generate per prompt (default: 1). "
                             "Total models = prompts * samples_per_prompt")
    parser.add_argument("--accuracy_threshold", type=float, default=0.40,
                        help="Minimum first-epoch accuracy to select models (0.0-1.0, default: 0.40)")
    parser.add_argument("--min_selected_k", type=int, default=15,
                        help="Minimum models to select via fallback (default: 15)")
    parser.add_argument("--fallback_threshold", type=float, default=0.35,
                        help="Lower bound accuracy for fallback selection (default: 0.35)")
    parser.add_argument("--adaptive_threshold", action="store_true", default=False,
                        help="Enable adaptive threshold (60-70th percentile)")
    parser.add_argument("--novelty_check", action="store_true", default=True,
                        help="Enable novelty checking")
    parser.add_argument("--no_novelty_check", dest="novelty_check", action="store_false",
                        help="Disable novelty checking")
    parser.add_argument("--resume_from_cycle", type=int, default=None,
                        help="Resume pipeline from a specific cycle (1-5)")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Maximum retry attempts for transient failures (default: 3)")
    
    # Training optimization arguments
    parser.add_argument("--use_optimized_training", action="store_true", default=True,
                        help="Use optimized training hyperparameters for stability and quality (default: True)")
    parser.add_argument("--no_optimized_training", dest="use_optimized_training", action="store_false",
                        help="Use original default training hyperparameters")
    parser.add_argument("--num_train_epochs", type=int, default=5,
                        help="Number of fine-tuning epochs per cycle (default: 5)")
    
    args = parser.parse_args()
    
    # Validate resume cycle
    if args.resume_from_cycle is not None:
        if args.resume_from_cycle < 1 or args.resume_from_cycle > args.cycles:
            print(f"[ERROR] Invalid resume_from_cycle: {args.resume_from_cycle}. Must be between 1 and {args.cycles}")
            sys.exit(1)
    
    # Create pipeline
    pipeline = IterativeFinetuner(
        base_data_dir=args.base_data_dir,
        output_dir=args.output_dir,
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
    )
    
    # Run pipeline
    pipeline.run()


if __name__ == "__main__":
    main()

