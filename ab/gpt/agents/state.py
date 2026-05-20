"""
Shared state for LangGraph agents.
Contains runtime resources + loop control.
Business logic reads from here.
"""

from typing import TypedDict, Any, Optional, Tuple


class AgentState(TypedDict, total=False):
    # ---- Loop Control ----
    current_epoch: int
    llm_tune_epochs: int
    skip_epoch: int
    next_action: str
    status: str
    use_predictor: bool

    # ---- Generation Inputs ----
    experiment_id: str
    nn_name_prefix: Optional[str]
    nn_train_epochs: int
    conf_keys: Tuple
    prompt_dict: dict
    test_nn: int
    max_new_tokens: int
    save_llm_output: bool
    prompt_batch: int

    # ---- Finetune Config ----
    train_config_path: str
    base_model_name: str
    only_best_accuracy: bool
    max_prompts: Optional[int]
    trans_mode: bool
    classification_mode: bool
    context_length: Optional[int]
    use_unsloth: bool
    unsloth_max_input_length: Optional[int]
    trainer_resume_checkpoint: Optional[str]

    # ---- Sampling ----
    temperature: float
    top_k: int
    top_p: float

    # ---- Runtime Resources (built once in tune()) ----
    model: Any
    tokenizer: Any
    model_loader: Any
    lora_tuner: Any
    chat_bot: Any

    # ---- Optional outputs (predictor / metrics) ----
    accuracy: float
    predicted_best_accuracy: float
    predicted_best_epoch: int
    epoch_1_accuracy: float
    epoch_2_accuracy: float
    error_message: str

    # ---- Predictor inputs (collected by evaluate_step, names match LEMUR DB columns) ----
    nn_code: str
    prm: dict
    task: str
    dataset: str
    metric: str
    transform_code: str
    nn: str
