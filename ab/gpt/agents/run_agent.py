from langgraph.graph import StateGraph, END
from ab.gpt.agents.state import AgentState
from ab.gpt.agents.manager import manager_node
from ab.gpt.util.Tune import generate_step, evaluate_step, finetune_step


def run_agent_controller(initial_state: dict):
    """
    Builds and runs the LangGraph workflow.
    All pipeline logic lives in Tune.py.
    Nodes are thin wrappers only.

    Uses MemorySaver checkpointing: if the run crashes mid-epoch,
    re-invoking with the same experiment_id resumes from the last
    completed node instead of restarting from epoch 0.
    """
    use_predictor = initial_state.get("use_predictor", False)

    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("manager",   manager_node)
    workflow.add_node("generator", generate_step)
    workflow.add_node("evaluator", evaluate_step)
    workflow.add_node("finetuner", finetune_step)

    if use_predictor:
        from ab.gpt.agents.predictor import predictor_node
        workflow.add_node("predictor", predictor_node)

    # Entry point — always start at manager
    workflow.set_entry_point("manager")

    # Router reads next_action from state
    def route(state: AgentState) -> str:
        return state.get("next_action", "end")

    # Build edge map
    edge_map = {
        "generate":  "generator",
        "evaluate":  "evaluator",
        "finetune":  "finetuner",
        "end":       END,
    }
    if use_predictor:
        edge_map["predict"] = "predictor"

    # Manager uses route function + edge map to decide next node
    workflow.add_conditional_edges("manager", route, edge_map)

    # generator routes based on next_action: "evaluate" → evaluator, "finetune" → finetuner
    workflow.add_conditional_edges("generator", route, edge_map)

    # evaluator, finetuner, predictor all return to manager
    workflow.add_edge("evaluator", "manager")
    workflow.add_edge("finetuner", "manager")
    if use_predictor:
        workflow.add_edge("predictor", "manager")

    app = workflow.compile()
    config = {}

    print("[CONTROLLER] LangGraph pipeline starting...")
    final_state = app.invoke(initial_state, config)
    print("[CONTROLLER] Pipeline complete.")
    return final_state
