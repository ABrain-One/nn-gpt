"""
Main Pipeline Orchestrator.
Coordinates the prompt improvement cycle for generating vision models.
"""

import json
import os
import torch
import numpy as np
import random

# Set environment variables for reproducibility BEFORE importing other modules
os.environ['PYTHONHASHSEED'] = '43'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Required for CUDA deterministic ops

from datetime import datetime
from pathlib import Path
from typing import Optional

from ab.gpt.markov.config import PipelineConfig, DEFAULT_CONFIG
from ab.gpt.markov.llm_client import LLMClient
from ab.gpt.markov.code_generator import CodeGenerator, get_prompt_template
from ab.gpt.markov.code_extractor import CodeExtractor
from ab.gpt.markov.evaluator import Evaluator, EvaluationResult
from ab.gpt.markov.prompt_improver import PromptImprover


class Pipeline:
    """Main pipeline that orchestrates the prompt improvement cycle."""
    
    def __init__(self, config: PipelineConfig = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration. Uses default if not provided.
        """
        self.config = config or DEFAULT_CONFIG
        
        # Initialize components (lazy loading for LLM)
        self._llm_client = None
        self._code_generator = None
        self._code_extractor = CodeExtractor()
        self._evaluator = None
        self._prompt_improver = None
        
        # State
        self.current_prompt_template = get_prompt_template(self.config.dataset)  # Fixed template, not modified
        self.best_code: Optional[str] = None  # Best performing code so far (used as reference)
        self.best_accuracy: float = 0.0  # Best accuracy achieved
        self.best_iteration: int = 0  # Iteration that achieved best accuracy
        self.current_improvement_suggestions: Optional[str] = None  # Improvement suggestions from prompt improver
        self.last_generated_code: Optional[str] = None
        self.last_accuracy: Optional[float] = None
        self.iteration = 0
        self.results_history = []
        
        # Markov history for improvement tracking
        self.improvement_history: list[dict] = []  # History of improvements with results
        self.max_history_size: int = self.config.history_size  # Keep last N iterations of history
        self.pending_history_entry: Optional[dict] = None  # Entry waiting for result from next iteration
        
        # Setup output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_unsloth = self.config.use_unsloth

        if self.use_unsloth:
            try:
                from unsloth import FastModel
                UNSLOTH_AVAILABLE = True
                
                # ================= Dynamic injection of FastModel into LLM module =================
                import ab.gpt.util.LLM
                ab.gpt.util.LLM.FastModel = FastModel
                # ================================================
            except ImportError:
                UNSLOTH_AVAILABLE = False
    
    @property
    def llm_client(self) -> LLMClient:
        """Lazy load LLM client."""
        if self._llm_client is None:
            self._llm_client = LLMClient(
                model_name=self.config.model_name,
                use_remote=self.config.use_remote,
                use_unsloth=self.use_unsloth
            )
        return self._llm_client
    
    @property
    def code_generator(self) -> CodeGenerator:
        """Lazy load code generator."""
        if self._code_generator is None:
            self._code_generator = CodeGenerator(
                self.llm_client,
                initial_prompt_template=self.current_prompt_template,
            )
        return self._code_generator
    
    @property
    def evaluator(self) -> Evaluator:
        """Lazy load evaluator."""
        if self._evaluator is None:
            self._evaluator = Evaluator(
                epochs=self.config.train_epochs,
                batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                timeout=self.config.training_timeout,
                data_dir=self.config.data_dir,
                dataset=self.config.dataset
            )
        return self._evaluator
    
    @property
    def prompt_improver(self) -> PromptImprover:
        """Lazy load prompt improver."""
        if self._prompt_improver is None:
            self._prompt_improver = PromptImprover(self.llm_client, dataset=self.config.dataset)
        return self._prompt_improver
    
    def log_result(self, iteration: int, accuracy: Optional[float], success: bool, error: Optional[str] = None):
        """
        Log iteration result to file.
        
        Args:
            iteration: Current iteration number
            accuracy: Test accuracy (if successful)
            success: Whether training was successful
            error: Error message (if failed)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        result_entry = {
            'iteration': iteration,
            'timestamp': timestamp,
            'success': success,
            'accuracy': accuracy,
            'error': error
        }
        self.results_history.append(result_entry)
        
        # Append to results log
        log_path = self.output_dir / self.config.results_log
        with open(log_path, 'a') as f:
            if accuracy is not None:
                f.write(f"iteration: {iteration}, accuracy: {accuracy:.4f}, timestamp: {timestamp}\n")
            else:
                f.write(f"iteration: {iteration}, error: {error}, timestamp: {timestamp}\n")
    
    def log_suggestions(self, iteration: int, suggestions: str, reason: str = None, inspiration: str = None):
        """
        Log improvement suggestions to file.
        
        Args:
            iteration: Current iteration number
            suggestions: The improvement suggestions
            reason: Reason for the suggestions (if applicable)
            inspiration: Source of inspiration (if applicable)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_path = self.output_dir / self.config.prompts_log
        with open(log_path, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Iteration: {iteration}\n")
            f.write(f"Timestamp: {timestamp}\n")
            if reason:
                f.write(f"Reason: {reason}\n")
            if inspiration:
                f.write(f"Inspiration: {inspiration}\n")
            f.write(f"\nImprovement Suggestions:\n{suggestions}\n")
    
    def save_code(self, iteration: int, code: str):
        """
        Save generated code to file.
        
        Args:
            iteration: Current iteration number
            code: Generated model code
        """
        if self.config.save_models:
            code_dir = self.output_dir / "generated_models"
            code_dir.mkdir(parents=True, exist_ok=True)
            
            code_path = code_dir / f"model_iter_{iteration}.py"
            with open(code_path, 'w') as f:
                f.write(code)
    
    def run_iteration(self) -> tuple[bool, Optional[float], str, Optional[str]]:
        """
        Run a single iteration of the pipeline.
        
        Returns:
            Tuple of (success, accuracy, feedback, generated_code)
        """
        self.iteration += 1
        
        # Step 1: Generate code (with best code as reference if enabled)
        print("\n[Step 1] Generating code...")        
        response = self.code_generator.generate(
            prompt_template=self.current_prompt_template,
            reference_code=self.best_code if self.config.use_reference_code else None,
            improvement_suggestions=self.current_improvement_suggestions,
            current_code=self.last_generated_code,
            current_accuracy=self.last_accuracy
        )
        
        # Step 2: Extract code
        print("\n[Step 2] Extracting code...")
        code, extract_message = self._code_extractor.extract(response)
        
        if code is None:
            feedback = f"Code extraction failed: {extract_message}"
            self.log_result(self.iteration, None, False, feedback)
            return False, None, feedback, None
        
        # Save the generated code
        self.save_code(self.iteration, code)
        
        # Step 3.1: Quick validation
        print("\n[Step 3.1] Quick validation...")
        is_valid, valid_message = self.evaluator.quick_validate(code)
        print(f"Validation result: {valid_message}")
        
        if not is_valid:
            feedback = f"Code validation failed: {valid_message}"
            self.log_result(self.iteration, None, False, feedback)
            return False, None, feedback, code
        
        # Step 3.2: Full training and evaluation
        print("\n[Step 3.2] Training and evaluating model...")
        result = self.evaluator.train_and_evaluate(code)
        
        if result.success:
            accuracy = result.accuracy
            feedback = result.get_feedback()
            print(f"Training successful! Accuracy: {accuracy * 100:.2f}%")
            self.log_result(self.iteration, accuracy, True)
            return True, accuracy, feedback, code
        else:
            feedback = result.get_feedback()
            print(f"Training failed: {result.error}")
            self.log_result(self.iteration, None, False, result.error)
            return False, None, feedback, code
    
    def update_history_with_result(self, accuracy: Optional[float], feedback: str):
        """
        Update the pending history entry with the result from the current iteration.
        
        Args:
            accuracy: The accuracy achieved (or None if failed)
            feedback: The feedback/error message
        """
        if self.pending_history_entry is not None:
            # Format the result
            if accuracy is not None:
                result_str = f"accuracy: {accuracy*100:.2f}%"
            else:
                # Truncate long error messages
                result_str = f"error: {feedback[:200]}"
            
            self.pending_history_entry['result'] = result_str
            
            # Add to history
            self.improvement_history.append(self.pending_history_entry)
            
            # Keep only the most recent entries
            if len(self.improvement_history) > self.max_history_size:
                self.improvement_history = self.improvement_history[-self.max_history_size:]
            
            print(f"[History] Updated entry for iteration {self.pending_history_entry['iteration']} with result: {result_str}")
            self.pending_history_entry = None
    
    def generate_suggestions(self, current_code: str, current_accuracy: Optional[float], feedback: str, current_iteration: int):
        """
        Generate improvement suggestions based on feedback.
        
        Args:
            current_code: Current generated code (may be None if extraction failed)
            current_accuracy: Current accuracy (may be None if failed)
            feedback: Feedback from evaluator
            current_iteration: Current iteration number
        """
        print("\n[Generating improvement suggestions based on feedback...]")
        print(f"[History] Current history size: {len(self.improvement_history)}")
        
        improvement = self.prompt_improver.improve(
            best_code=self.best_code if self.config.use_reference_code else None,
            best_accuracy=self.best_accuracy if self.config.use_reference_code else 0.0,
            current_code=current_code or "No code generated",
            current_accuracy=current_accuracy,
            feedback=feedback,
            history=self.improvement_history if self.config.use_history else None,
            output_dir=self.output_dir,
            current_iteration=current_iteration
        )
        
        if improvement['improvement_suggestions']:
            print(f"Reason: {improvement['reason']}")
            print(f"Inspiration: {improvement['inspiration']}")
            suggestions_str = str(improvement['improvement_suggestions']) if not isinstance(improvement['improvement_suggestions'], str) else improvement['improvement_suggestions']
            print(f"Improvement suggestions: {suggestions_str[:200]}...")
            
            # Update improvement suggestions for next iteration
            self.current_improvement_suggestions = suggestions_str
            self.code_generator.update_improvement_suggestions(self.current_improvement_suggestions)
            
            # Create pending history entry (result will be filled after next iteration)
            self.pending_history_entry = {
                'iteration': self.iteration,
                'problem': str(improvement['reason']),
                'suggestion': suggestions_str
            }
            print(f"[History] Created pending entry for iteration {self.iteration}")
            
            self.log_suggestions(
                self.iteration,
                self.current_improvement_suggestions,
                reason=improvement['reason'],
                inspiration=improvement['inspiration']
            )
        else:
            print("Warning: Failed to generate improvement suggestions")
    
    def run(self) -> dict:
        """
        Run the full pipeline until target accuracy is reached or max iterations.
        
        Returns:
            Dictionary with final results
        """
        print(f"\nStarting Prompt Improvement Pipeline")
        print(f"Target accuracy: {self.config.target_accuracy * 100:.1f}%")
        print(f"Max iterations: {self.config.max_iterations}")
        
        # Log initial state
        self.log_suggestions(0, "No suggestions yet - first iteration", reason="Initial state")
        
        while self.iteration < self.config.max_iterations:
            # Run iteration
            success, accuracy, feedback, generated_code = self.run_iteration()
            
            # Update history with result from this iteration (for the previous suggestion)
            self.update_history_with_result(accuracy, feedback)
            
            # Save the current iteration code and accuracy for the next generation
            self.last_generated_code = generated_code
            self.last_accuracy = accuracy
            
            # Track best result and update best_code
            if success and accuracy is not None:
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_iteration = self.iteration
                    self.best_code = generated_code
                    print(f"[NEW BEST] Updated best code: iteration {self.best_iteration}, accuracy {self.best_accuracy*100:.2f}%")
                
                # Check if target reached
                if accuracy >= self.config.target_accuracy:
                    print(f"\n{'='*60}")
                    print(f"Target accuracy reached!")
                    print(f"Final accuracy: {accuracy * 100:.2f}%")
                    print(f"{'='*60}")
                    break
            
            # Generate suggestions for next iteration (if not last)
            if self.iteration < self.config.max_iterations:
                if self.config.use_prompt_improver:
                    if generated_code is None:
                        print("\n[Using Step 2 extraction error as feedback for suggestions]")
                    self.generate_suggestions(generated_code, accuracy, feedback, self.iteration)
                else:
                    print("\n[Prompt Improver disabled for this run]")
                    self.current_improvement_suggestions = None
                    self.code_generator.update_improvement_suggestions(None)
        
        # Final summary
        final_result = {
            'iterations': self.iteration,
            'best_accuracy': self.best_accuracy,
            'best_iteration': self.best_iteration,
            'target_reached': self.best_accuracy >= self.config.target_accuracy,
            'results_history': self.results_history
        }
        
        print(f"\n{'='*60}")
        print("Pipeline Complete")
        print(f"Total iterations: {self.iteration}")
        print(f"Best accuracy: {self.best_accuracy * 100:.2f}% (iteration {self.best_iteration})")
        print(f"Target reached: {final_result['target_reached']}")
        print(f"{'='*60}")
        
        # Save final summary
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(final_result, f, indent=2)
        
        return final_result


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Prompt Improvement Pipeline')
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help='HuggingFace model name or remote model ID')
    parser.add_argument('--remote', action='store_true',
                        help='Use remote model API (reads SiliconCloud_Key environment variable)')
    parser.add_argument('--dataset', type=str, default='imagenette',
                        choices=['imagenette', 'cifar10', 'cifar100'],
                        help='Dataset to use for training (imagenette, cifar10, or cifar100)')
    parser.add_argument('--target-accuracy', type=float, default=0.8,
                        help='Target accuracy to reach')
    parser.add_argument('--max-iterations', type=int, default=1000,
                        help='Maximum number of iterations')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Training epochs per iteration')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Training batch size')
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Output directory')
    parser.add_argument('--test', action='store_true',
                        help='Run with test configuration')
    parser.add_argument('--use-unsloth', action='store_true',
                        help='Use Unsloth for faster training')
    
    # Ablation study flags
    parser.add_argument('--no-improver', action='store_true',
                        help='Disable prompt improver (Random Search Baseline)')
    parser.add_argument('--no-reference', action='store_true',
                        help='Disable reference code (Best Implementation)')
    parser.add_argument('--no-history', action='store_true',
                        help='Disable history memory')
    parser.add_argument('--history-size', type=int, default=5,
                        help='Number of previous iterations to keep in history')
    
    
    args = parser.parse_args()
    
    # Create configuration
    if args.test:
        from config import TEST_CONFIG
        config = TEST_CONFIG
        config.model_name = args.model
        config.use_remote = args.remote
        config.dataset = args.dataset
    else:
        config = PipelineConfig(
            model_name=args.model,
            use_remote=args.remote,
            dataset=args.dataset,
            target_accuracy=args.target_accuracy,
            max_iterations=args.max_iterations,
            train_epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            use_prompt_improver=not args.no_improver,
            use_reference_code=not args.no_reference,
            use_history=not args.no_history,
            history_size=args.history_size,
            use_unsloth=args.use_unsloth
        )
    
    # Run pipeline
    pipeline = Pipeline(config)
    result = pipeline.run()
    
    return result


if __name__ == "__main__":
    # set the seed
    torch.manual_seed(43)
    torch.cuda.manual_seed(43)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    np.random.seed(43)
    random.seed(43)

    main()
