#!/usr/bin/env python3
"""
Pipeline Validation and Error Handling Utilities

Provides robust validation, error handling, and retry mechanisms
for the iterative fine-tuning pipeline.
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

from ab.nn.util.Const import ab_root_path, out_dir

# Setup logging
# Logging will be configured by the main script to use the output directory
# Do not create log files here to avoid creating files in iterative_pipeline directory
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Only add console handler if no handlers exist (to avoid duplicate logs)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)


class PipelineValidator:
    """Validate pipeline prerequisites and environment."""
    
    @staticmethod
    def check_disk_space(min_gb: int = 50) -> Tuple[bool, str]:
        """Check if sufficient disk space is available."""
        try:
            import shutil
            stat = shutil.disk_usage(".")
            free_gb = stat.free / (1024**3)
            if free_gb < min_gb:
                return False, f"Insufficient disk space: {free_gb:.1f}GB available, need {min_gb}GB"
            return True, f"Disk space OK: {free_gb:.1f}GB available"
        except Exception as e:
            return False, f"Failed to check disk space: {e}"
    
    @staticmethod
    def check_gpu_available() -> Tuple[bool, str]:
        """Check if GPU is available."""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return True, f"GPU available: {gpu_name} ({gpu_memory:.1f}GB)"
            return False, "No GPU available (CUDA not available)"
        except Exception as e:
            return False, f"Failed to check GPU: {e}"
    
    @staticmethod
    def check_training_data(data_dir: Path) -> Tuple[bool, str]:
        """Validate training data directory."""
        if not data_dir.exists():
            return False, f"Training data directory not found: {data_dir}"
        
        # Check for both train.jsonl (old format) and NN_Rag_gen_train.jsonl (new format)
        train_file = data_dir / "train.jsonl"
        if not train_file.exists():
            train_file = data_dir / "NN_Rag_gen_train.jsonl"
        
        if not train_file.exists():
            return False, f"train.jsonl or NN_Rag_gen_train.jsonl not found in: {data_dir}"
        
        # Check file is readable and has content
        try:
            with open(train_file, 'r') as f:
                lines = [l for l in f if l.strip()]
                if len(lines) == 0:
                    return False, f"train.jsonl is empty: {train_file}"
                # Validate JSON
                for i, line in enumerate(lines[:10]):  # Check first 10 lines
                    json.loads(line)
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON in train.jsonl (line {i+1}): {e}"
        except Exception as e:
            return False, f"Failed to read train.jsonl: {e}"
        
        return True, f"Training data valid: {len(lines)} examples"
    
    @staticmethod
    def check_llm_config(config_file: Path) -> Tuple[bool, str]:
        """Validate LLM configuration file."""
        original_path = config_file
        
        # Resolve config path: if it's just a filename, look in ab/gpt/conf/llm/
        if not config_file.is_absolute():
            # Check if parent directory exists (means it's a relative path with directory)
            if config_file.parent.exists() and str(config_file.parent) != '.':
                # It's a relative path with directory, use as-is
                pass
            else:
                # It's just a filename, try to resolve relative to ab/gpt/conf/llm/
                try:
                    from ab.gpt.util.Const import conf_llm_dir
                    resolved_path = conf_llm_dir / config_file.name
                    if resolved_path.exists():
                        config_file = resolved_path
                        logger.info(f"Resolved LLM config: {original_path} -> {config_file}")
                except Exception as e:
                    logger.debug(f"Could not resolve using conf_llm_dir: {e}")
                
                # Also try current directory as fallback
                if not config_file.exists():
                    current_dir_path = Path(config_file.name)
                    if current_dir_path.exists():
                        config_file = current_dir_path
        
        if not config_file.exists():
            return False, f"LLM config not found: {original_path} (tried: {config_file})"
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            required_keys = ["base_model_name"]
            for key in required_keys:
                if key not in config:
                    return False, f"Missing required key in config: {key}"
            
            return True, f"LLM config valid: {config.get('base_model_name', 'unknown')}"
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON in config file: {e}"
        except Exception as e:
            return False, f"Failed to read config file: {e}"
    
    @staticmethod
    def check_infrastructure_files() -> Tuple[bool, List[str]]:
        """Check if all required infrastructure files exist."""
        # Pipeline files are in ab/gpt/iterative_pipeline/
        # iterative_finetune.py is at ab/gpt/iterative_finetune.py (one level up)
        gpt_dir = ab_root_path / 'ab' / 'gpt'
        pipeline_dir = gpt_dir / 'iterative_pipeline'

        # Files in the pipeline directory
        pipeline_files = [
            "novelty_checker.py",
            "training_data_manager.py",
            "aggregate_cycles.py",
            "plot_cycles.py",
        ]
        
        # Files in the gpt directory (one level up)
        gpt_files = [
            "iterative_finetune.py",
        ]
        
        missing = []
        
        # Check pipeline files
        for file in pipeline_files:
            if not Path(file).exists() and not (pipeline_dir / file).exists():
                missing.append(file)
        
        # Check gpt-level files
        for file in gpt_files:
            if not Path(file).exists() and not (gpt_dir / file).exists():
                missing.append(file)
        
        if missing:
            return False, missing
        return True, []
    
    @staticmethod
    def validate_all(base_data_dir: Path, llm_conf: Path) -> Dict[str, Any]:
        """Run all validation checks."""
        results = {
            "disk_space": PipelineValidator.check_disk_space(50),
            "gpu": PipelineValidator.check_gpu_available(),
            "training_data": PipelineValidator.check_training_data(base_data_dir),
            "llm_config": PipelineValidator.check_llm_config(llm_conf),
            "infrastructure": PipelineValidator.check_infrastructure_files(),
        }
        
        all_passed = all(result[0] for result in results.values())
        
        return {
            "all_passed": all_passed,
            "results": results,
        }


class RetryHandler:
    """Handle retries for transient failures."""
    
    @staticmethod
    def retry_with_backoff(
        func,
        max_retries: int = 3,
        initial_delay: float = 5.0,
        backoff_factor: float = 2.0,
        exceptions: tuple = (Exception,),
        operation_name: str = "operation"
    ) -> Any:
        """
        Retry a function with exponential backoff.
        
        Args:
            func: Function to retry
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            backoff_factor: Multiplier for delay after each retry
            exceptions: Tuple of exceptions to catch and retry
            operation_name: Name of operation for logging
        """
        delay = initial_delay
        
        for attempt in range(max_retries + 1):
            try:
                return func()
            except exceptions as e:
                if attempt == max_retries:
                    logger.error(f"{operation_name} failed after {max_retries} retries: {e}")
                    raise
                
                logger.warning(
                    f"{operation_name} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay *= backoff_factor
        
        raise RuntimeError(f"{operation_name} failed: max retries exceeded")


class StageValidator:
    """Validate inputs and outputs at each pipeline stage."""
    
    @staticmethod
    def validate_finetuning_output(checkpoint_dir: Path) -> Tuple[bool, str]:
        """Validate fine-tuning produced a valid checkpoint."""
        if not checkpoint_dir.exists():
            return False, f"Checkpoint directory not found: {checkpoint_dir}"
        
        # Check for adapter files
        adapter_files = list(checkpoint_dir.glob("adapter*.safetensors")) + \
                       list(checkpoint_dir.glob("adapter*.bin"))
        
        if not adapter_files:
            # Check for merged model files
            model_files = list(checkpoint_dir.glob("*.safetensors")) + \
                         list(checkpoint_dir.glob("*.bin"))
            if not model_files:
                return False, f"No adapter or model files found in {checkpoint_dir}"
        
        return True, f"Checkpoint valid: {len(adapter_files)} adapter files found"
    
    @staticmethod
    def validate_generation_output(generation_dir: Path, expected_count: int) -> Tuple[bool, str]:
        """Validate model generation produced expected results."""
        results_file = generation_dir / "results.jsonl"
        if not results_file.exists():
            return False, f"Results file not found: {results_file}"
        
        # Count successful generations
        try:
            with open(results_file, 'r') as f:
                results = []
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            results.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            
            successful = sum(1 for r in results if r.get("ok", False))
            
            if successful == 0:
                return False, f"No successful generations in {results_file}"
            
            if successful < expected_count * 0.1:  # At least 10% success rate
                return False, f"Very low success rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)"
            
            return True, f"Generation valid: {successful}/{len(results)} successful"
        except Exception as e:
            return False, f"Failed to validate generation output: {e}"
    
    @staticmethod
    def validate_evaluation_output(nneval_dir: Path, min_expected: int = 1) -> Tuple[bool, str]:
        """Validate model evaluation produced results."""
        if not nneval_dir.exists():
            return False, f"NNEval directory not found: {nneval_dir}"
        
        # Count models with results
        results_count = 0
        for model_dir in nneval_dir.glob("gen_*"):
            if not model_dir.is_dir():
                continue
            
            # Check for 1.json or eval_info.json
            if (model_dir / "1.json").exists() or (model_dir / "eval_info.json").exists():
                results_count += 1
        
        if results_count < min_expected:
            return False, f"Insufficient evaluation results: {results_count} < {min_expected}"
        
        return True, f"Evaluation valid: {results_count} models evaluated"
    
    @staticmethod
    def validate_training_data(data_dir: Path, expected_min: int) -> Tuple[bool, str]:
        """Validate augmented training data."""
        train_file = data_dir / "train.jsonl"
        if not train_file.exists():
            return False, f"Training data file not found: {train_file}"
        
        try:
            with open(train_file, 'r') as f:
                lines = [l for l in f if l.strip()]
                count = len(lines)
            
            if count < expected_min:
                return False, f"Insufficient training examples: {count} < {expected_min}"
            
            return True, f"Training data valid: {count} examples"
        except Exception as e:
            return False, f"Failed to validate training data: {e}"


class ErrorRecovery:
    """Handle error recovery and resume capability."""
    
    @staticmethod
    def find_last_successful_cycle(output_dir: Path) -> Optional[int]:
        """Find the last successfully completed cycle."""
        last_cycle = None
        
        for cycle in range(1, 6):
            cycle_dir = output_dir / f"cycle_{cycle}"
            results_file = cycle_dir / "cycle_results.json"
            
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        data = json.load(f)
                        if data.get("success", False):
                            last_cycle = cycle
                except Exception:
                    pass
        
        return last_cycle
    
    @staticmethod
    def can_resume_from_cycle(output_dir: Path, cycle: int) -> Tuple[bool, str]:
        """Check if we can resume from a specific cycle."""
        cycle_dir = output_dir / f"cycle_{cycle}"
        
        # Check if cycle has checkpoint
        checkpoint_dir = cycle_dir / "checkpoint"
        if not checkpoint_dir.exists():
            return False, f"No checkpoint found for cycle {cycle}"
        
        # Check if cycle has results
        results_file = cycle_dir / "cycle_results.json"
        if not results_file.exists():
            return False, f"No results file found for cycle {cycle}"
        
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
                if not data.get("success", False):
                    return False, f"Cycle {cycle} did not complete successfully"
        except Exception:
            return False, f"Failed to read results for cycle {cycle}"
        
        return True, f"Can resume from cycle {cycle + 1}"


def validate_pipeline_prerequisites(base_data_dir: str, llm_conf: str) -> bool:
    """Validate all prerequisites before starting pipeline."""
    logger.info("="*80)
    logger.info("VALIDATING PIPELINE PREREQUISITES")
    logger.info("="*80)
    
    validator = PipelineValidator()
    results = validator.validate_all(Path(base_data_dir), Path(llm_conf))
    
    all_passed = True
    for check_name, (passed, message) in results["results"].items():
        status = "✓" if passed else "✗"
        logger.info(f"{status} {check_name}: {message}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n✓ All prerequisites validated successfully!")
    else:
        logger.error("\n✗ Some prerequisites failed. Please fix issues before running pipeline.")
    
    return all_passed


if __name__ == "__main__":
    # Test validation
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_data_dir", type=str, default=out_dir / "curation_output/chat_data")
    parser.add_argument("--llm_conf", type=str, default=ab_root_path / "ab/gpt/conf/llm/ds_coder_7b_instruct.json")
    
    args = parser.parse_args()
    
    if validate_pipeline_prerequisites(args.base_data_dir, args.llm_conf):
        sys.exit(0)
    else:
        sys.exit(1)

