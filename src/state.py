from typing import TypedDict, Optional, Dict, Any


class AgentState(TypedDict):
    """State shared between Manager, Generator, and Predictor agents."""
    
    # ============ GENERATOR INPUT ============
    epoch: int                          # Which epoch to generate
    conf_key: str                       # Config for generation
    base_model_name: str                # Base model identifier
    
    # ============ GENERATOR → PREDICTOR ============
    model_code: Optional[str]           # Generated model code (from new_nn.py)
    accuracy: Optional[float]           # 2-epoch accuracy result
    loss: Optional[float]               # 2-epoch loss
    metrics: Optional[Dict[str, Any]]   # All 2-epoch metrics from eval_info.json
    
    # ============ PREDICTOR OUTPUT ============
    predicted_final_accuracy: Optional[float]   # Predictor's prediction
    estimated_epochs: Optional[int]             # How many epochs to reach target
    
    # ============ MANAGER CONTROL ============
    gpu_available: bool                 # Is GPU free?
    next_action: str                    # "generate", "predict", "wait", "end"
    
    # ============ DATABASE CACHE ============
    found_in_cache: bool                # Found in database?
    
    # ============ TRACKING ============
    experiment_id: Optional[str]        # For logging
    status: str                         # "success", "error", "cached"