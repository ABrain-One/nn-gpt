"""
State definition for LangGraph multi-agent workflow.

Integrates with TuneNNGen.py pipeline.

Based on:
- TuneNNGen.py: LLM fine-tuning and model generation
- NNEval.py: Model training and eval_info.json structure
- TuneAccPrediction.py: Predictor requirements
"""

from typing import TypedDict, Optional, Dict, Any, List


class AgentState(TypedDict, total=False):
    """State shared between Manager, Generator, and Predictor agents."""
    
    # ============ EXPERIMENT METADATA ============
    experiment_id: str                # Experiment identifier
    base_out_dir: str                 # Base output directory
    
    # ============ TUNENNGEN PARAMETERS (Pass-through) ============
    num_train_epochs: int              # LLM fine-tuning epochs
    nn_train_epochs: int              # NN training epochs (2 for early metrics)
    nn_gen_conf_id: str               # Config ID (e.g., 'improve_classification_only')
    temperature: float                # LLM temperature
    top_k: int                        # LLM top_k
    top_p: float                      # LLM top_p
    max_new_tokens: int               # Max tokens for LLM generation
    save_llm_output: bool             # Save full LLM output
    
    # ============ GENERATOR OUTPUT ============
    # File paths
    model_file_path: Optional[str]    # Path to new_nn.py
    eval_file_path: Optional[str]     # Path to eval_info.json
    
    # From eval_info.json
    model_code: Optional[str]         # Generated model code (from new_nn.py)
    eval_results: Optional[List[Any]] # [nn_name, accuracy, acc_to_time, score]
    eval_args: Optional[Dict[str, Any]] # Hyperparameters used (from eval_info.json)
    cli_args: Optional[Dict[str, Any]] # Task, dataset, metric, etc.
    
    # Extracted from eval_results
    nn_name: Optional[str]            # Generated NN name
    accuracy: Optional[float]         # Final accuracy after nn_train_epochs
    accuracy_to_time: Optional[float] # Accuracy/time metric
    score: Optional[float]            # Composite score (eval_results[3])
    
    # Hyperparameters (extracted from eval_args or hp.txt)
    hyperparameters: Optional[Dict[str, Any]]  # Training hyperparameters
    
    # Additional metadata from cli_args
    task: Optional[str]               # Task type (e.g., 'img-classification')
    dataset: Optional[str]             # Dataset name (e.g., 'cifar-10')
    metric: Optional[str]              # Metric name (e.g., 'acc')
    
    # ============ PREDICTOR INPUT (CRITICAL) ============
    # Per-epoch accuracies (MUST be extracted from database or training logs)
    epoch_1_accuracy: Optional[float] # Accuracy after epoch 1
    epoch_2_accuracy: Optional[float] # Accuracy after epoch 2
    
    # Additional predictor inputs (from database or eval_info.json)
    transform_code: Optional[str]     # Data transformation code
    metric_code: Optional[str]        # Metric calculation code
    max_epoch: Optional[int]          # Maximum epoch for prediction
    best_accuracy: Optional[float]    # Best accuracy (for addon in prompt)
    best_epoch: Optional[int]         # Best epoch (for addon in prompt)
    
    # ============ PREDICTOR OUTPUT ============
    predicted_best_accuracy: Optional[float]    # Predicted best accuracy
    predicted_best_epoch: Optional[int]         # Predicted best epoch
    
    # ============ WORKFLOW CONTROL ============
    gpu_available: bool               # GPU availability
    next_action: str                  # "generate", "predict", "end"
    use_predictor: bool               # Enable predictor?
    
    # ============ STATUS ============
    status: str                       # "success", "error", "pending"
    error_message: Optional[str]      # Error details
    
    # ============ HISTORY/DEBUGGING ============
    history: Optional[List[str]]      # Execution history for debugging
    metrics_source: Optional[str]    # "eval_info.json", "database", "none"

