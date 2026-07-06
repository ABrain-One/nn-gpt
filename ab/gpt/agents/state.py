"""
Shared state for LangGraph agents.
Contains runtime resources + loop control.
Business logic reads from here.
"""

from pathlib import Path
from typing import Any, Optional, Tuple
from pydantic import BaseModel, field_validator


class AgentState(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    @field_validator('train_config_path', mode='before')
    @classmethod
    def coerce_path_to_str(cls, v):
        return str(v) if isinstance(v, Path) else v

    def __getitem__(self, item):
        return getattr(self, item)

    def get(self, item, default=None):
        value = getattr(self, item, None)
        return value if value is not None else default

    # ---- Loop Control ----
    current_epoch: Optional[int] = None
    llm_tune_epochs: Optional[int] = None
    skip_epoch: Optional[int] = None
    next_action: Optional[str] = None
    status: Optional[str] = None
    use_predictor: Optional[bool] = None
    use_backbone: Optional[bool] = None
    enable_merge: Optional[bool] = None
    sft_nn_prefixes: Optional[Any] = None
    sft_dataset: Optional[str] = None

    # ---- Generation Inputs ----
    experiment_id: Optional[str] = None
    nn_name_prefix: Optional[str] = None
    nn_train_epochs: Optional[int] = None
    conf_keys: Optional[Tuple] = None
    prompt_dict: Optional[dict] = None
    test_nn: Optional[int] = None
    max_new_tokens: Optional[int] = None
    save_llm_output: Optional[bool] = None
    prompt_batch: Optional[int] = None

    # ---- Finetune Config ----
    train_config_path: Optional[str] = None
    base_model_name: Optional[str] = None
    only_best_accuracy: Optional[bool] = None
    max_prompts: Optional[int] = None
    trans_mode: Optional[bool] = None
    classification_mode: Optional[bool] = None
    context_length: Optional[int] = None
    use_unsloth: Optional[bool] = None
    unsloth_max_input_length: Optional[int] = None
    trainer_resume_checkpoint: Optional[str] = None

    # ---- Sampling ----
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None

    # ---- Runtime Resources (built once in tune()) ----
    model: Optional[Any] = None
    tokenizer: Optional[Any] = None
    model_loader: Optional[Any] = None
    lora_tuner: Optional[Any] = None
    chat_bot: Optional[Any] = None

    # ---- Optional outputs (predictor / metrics) ----
    accuracy: Optional[float] = None
    predicted_best_accuracy: Optional[float] = None
    predicted_best_epoch: Optional[int] = None
    epoch_1_accuracy: Optional[float] = None
    epoch_2_accuracy: Optional[float] = None
    epoch_3_accuracy: Optional[float] = None
    error_message: Optional[str] = None

    # ---- Predictor inputs (collected by evaluate_step, names match LEMUR DB columns) ----
    nn_code: Optional[str] = None
    prm: Optional[dict] = None
    task: Optional[str] = None
    dataset: Optional[str] = None
    metric: Optional[str] = None
    transform_code: Optional[str] = None
    nn: Optional[str] = None
