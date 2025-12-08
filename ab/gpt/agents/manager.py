"""
Manager Agent - Coordinates workflow and GPU resources.

Decides workflow routing:
- "generate": Run generator to create model
- "predict": Run predictor to forecast accuracy  
- "end": Workflow complete

LangGraph Best Practices:
- Returns only changed state fields
- Handles errors gracefully
- Validates prerequisites before routing
"""

from typing import Dict, Any
from ab.gpt.agents.state import AgentState


def manager_node(state: AgentState) -> Dict[str, Any]:
    """
    Manager agent coordinates GPU resources and workflow routing.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with next_action set (only changed fields)
    """
    
    print("🎛️ Manager: Coordinating workflow...")
    
    # ============================================================
    # 1. ERROR HANDLING (Check first!)
    # ============================================================
    if state.get('status') == 'error':
        print("❌ Manager: Error detected → Ending workflow")
        return {
            "next_action": "end",
            "gpu_available": True,  # Release GPU on error
        }
    
    # ============================================================
    # 2. RELEASE GPU IF PREVIOUS AGENT COMPLETED
    # ============================================================
    # If GPU was reserved but agent completed successfully, release it
    if (state.get('gpu_available') == False 
        and state.get('status') == 'success'):
        print("✅ Manager: Previous agent completed → Releasing GPU")
        # Continue to routing logic below
    
    # ============================================================
    # 3. VALIDATE GENERATOR OUTPUT
    # ============================================================
    has_model = (
        state.get('model_code') is not None 
        and state.get('model_code', '').strip() != ''
        and state.get('status') != 'error'
    )
    
    # ============================================================
    # 4. VALIDATE PREDICTOR OUTPUT
    # ============================================================
    has_prediction = (
        state.get('predicted_best_accuracy') is not None
        and state.get('predicted_best_epoch') is not None
    )
    
    # ============================================================
    # 5. CHECK PREDICTOR PREREQUISITES
    # ============================================================
    use_predictor = state.get('use_predictor', False)
    can_predict = (
        has_model
        and state.get('epoch_1_accuracy') is not None
        and state.get('epoch_2_accuracy') is not None
    )
    
    # ============================================================
    # 6. DECISION LOGIC
    # ============================================================
    gpu_available = state.get('gpu_available', True)
    
    if not has_model:
        # Need to generate model
        if gpu_available:
            print("✅ Manager: GPU available → Assigning to Generator")
            return {
                "next_action": "generate",
                "gpu_available": False,  # Reserve GPU
            }
        else:
            print("⏳ Manager: GPU busy → Retrying...")
            # Keep current state, don't change next_action
            # Graph will retry or use conditional edges
            return {}
    
    elif has_model and use_predictor and not has_prediction:
        # Have model, predictor enabled, need prediction
        if not can_predict:
            print("⚠️ Manager: Predictor prerequisites missing → Ending")
            return {
                "next_action": "end",
                "gpu_available": True,
            }
        
        if gpu_available:
            print("✅ Manager: GPU available → Assigning to Predictor")
            return {
                "next_action": "predict",
                "gpu_available": False,  # Reserve GPU
            }
        else:
            print("⏳ Manager: GPU busy → Retrying...")
            return {}
    
    else:
        # Everything done!
        print("✅ Manager: Workflow complete!")
        return {
            "next_action": "end",
            "gpu_available": True,  # Release GPU
        }

