"""
Manager Agent - Controls the execution flow of the pipeline.

Responsible for routing:
finetune → generate → evaluate → (predict?) → finetune → ...
"""

import os
from typing import Dict, Any
from ab.gpt.act.agents.state import AgentState


def _skip_closing_eval() -> bool:
    return os.environ.get("NNGPT_SKIP_CLOSING_EVAL", "").strip().lower() in {"1", "true", "yes", "on"}


def manager_node(state: AgentState) -> Dict[str, Any]:
    """Decide which step should run next."""

    epoch = state.get("current_epoch", 0)
    total_epochs = state.get("llm_tune_epochs", 1)
    next_action = state.get("next_action", "generate")

    # Stop condition. The cycle order is generate -> evaluate -> fine-tune, so
    # when the counter runs out the most recent fine-tune has not been measured
    # by anything. Run one closing generate+evaluate pass (no fine-tune after
    # it) so the last adapter is validated and recorded, then stop.
    if epoch >= total_epochs:
        if not state.get("final_eval_done") and _skip_closing_eval():
            print(f"[MANAGER] All {total_epochs} fine-tuning cycles complete. "
                  f"Closing validation pass skipped (NNGPT_SKIP_CLOSING_EVAL=1). Ending.")
            return {"next_action": "end"}
        if not state.get("final_eval_done"):
            print(f"[MANAGER] All {total_epochs} fine-tuning cycles complete. "
                  f"Running a closing validation pass for the final adapter.")
            return {"next_action": "generate", "final_eval": True}
        print(f"[MANAGER] All {total_epochs} epochs complete. Ending.")
        return {"next_action": "end"}

    # Generation
    if next_action == "generate":
        print(f"[MANAGER] Epoch {epoch}: starting generation")
        return {"next_action": "generate"}

    # Evaluation (always follows generation via direct edge — manager handles it only on skip)
    if next_action == "evaluate":
        print(f"[MANAGER] Epoch {epoch}: evaluating generated models")
        return {"next_action": "evaluate"}

    # Predictor
    if next_action == "predict":
        print(f"[MANAGER] Epoch {epoch}: evaluator done → predictor running (CPU)")
        return {"next_action": "predict"}

    # Finetuning
    if next_action == "finetune":
        print(f"[MANAGER] Epoch {epoch}: finetuner gets GPU")
        return {"next_action": "finetune"}

    # Fallback
    print(f"[MANAGER] Unknown next_action '{next_action}', defaulting to generate")
    return {"next_action": "generate"}