"""
Predictor Agent - Predicts final accuracy (optional).

This agent DOES NOT implement prediction logic.
It only calls predict_best_accuracy() from the predictor module.
All inputs come from state — collected by evaluate_step() after 3 full epochs.
"""

from typing import Dict, Any
from ab.gpt.agents.state import AgentState


def predictor_node(state: AgentState) -> Dict[str, Any]:
    """Predict final accuracy from early-epoch metrics."""

    print("[PREDICTOR] Running...")

    try:
        from ab.gpt.AccPredictor import predict_best_accuracy

        nn_code     = state.get("nn_code")
        epoch_1_acc = state.get("epoch_1_accuracy")
        epoch_2_acc = state.get("epoch_2_accuracy")
        epoch_3_acc = state.get("epoch_3_accuracy")

        if nn_code and epoch_1_acc is not None and epoch_2_acc is not None and epoch_3_acc is not None:
            pred_acc, pred_epoch = predict_best_accuracy(
                nn_code=nn_code,
                task=state.get("task", ""),
                dataset=state.get("dataset", ""),
                metric=state.get("metric", ""),
                epoch_1_accuracy=epoch_1_acc,
                epoch_2_accuracy=epoch_2_acc,
                epoch_3_accuracy=epoch_3_acc,
            )

            print(f"[PREDICTOR] predicted accuracy={pred_acc:.4f}, best_epoch={pred_epoch}")
            return {
                "status": "success",
                "predicted_best_accuracy": pred_acc,
                "predicted_best_epoch": pred_epoch,
                "next_action": "finetune",
            }

        return {
            "status": "partial_success",
            "error_message": "Predictor needs nn_code and 3 epoch accuracies",
            "next_action": "finetune",
        }

    except ImportError:
        return {
            "status": "partial_success",
            "error_message": "AccPredictor not available yet",
            "next_action": "finetune",
        }

    except Exception as e:
        return {
            "status": "partial_success",
            "error_message": str(e),
            "next_action": "finetune",
        }
