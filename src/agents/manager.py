from typing import Dict, Any


def manager_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Manager agent - coordinates GPU resources and workflow
    
    Decides what should happen next based on current state:
    - generate: Run generator to create model
    - predict: Run predictor to forecast accuracy
    - end: Workflow complete
    """
    
    print(f"🎛️ Manager: Coordinating workflow...")
    
    # Check what's been done
    has_model = state.get('model_code') is not None
    has_prediction = state.get('predicted_final_accuracy') is not None
    
    # Decision logic
    if not has_model:
        # Need to generate model
        if state.get('gpu_available', True):
            print("✅ Manager: GPU available → Assigning to Generator")
            state['next_action'] = 'generate'
            state['gpu_available'] = False  # Reserve GPU
        else:
            print("⏳ Manager: GPU busy → Waiting...")
            state['next_action'] = 'wait'
    
    elif has_model and not has_prediction:
        # Have model, need prediction
        if state.get('gpu_available', True):
            print("✅ Manager: GPU available → Assigning to Predictor")
            state['next_action'] = 'predict'
            state['gpu_available'] = False  # Reserve GPU
        else:
            print("⏳ Manager: GPU busy → Waiting...")
            state['next_action'] = 'wait'
    
    else:
        # Everything done!
        print("✅ Manager: Workflow complete!")
        state['next_action'] = 'end'
        state['gpu_available'] = True  # Release GPU
    
    print(f"📊 Manager decision: {state['next_action']}")
    
    return state