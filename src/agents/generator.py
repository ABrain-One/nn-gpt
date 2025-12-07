from pathlib import Path
import json
from typing import Dict, Any


def generator_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generator agent - wraps TuneNNGen.py
    Calls existing pipeline to generate model and get 2-epoch results
    """
    
    print(f"🤖 Generator: Starting for epoch {state['epoch']}")
    
    # 1. Call TuneNNGen.py (generates model + trains for 2 epochs)
    from ab.gpt.TuneNNGen import main as tune_nn_gen
    
    try:
        tune_nn_gen()
        print("✅ TuneNNGen.py completed!")
        
    except Exception as e:
        print(f"❌ Error in TuneNNGen: {e}")
        state['status'] = 'error'
        return state
    
    # 2. Read results from output directory
    output_dir = Path(f"out/nngpt/llm/epoch/A{state['epoch']}/synth_nn/B0/")
    
    try:
        # Read model code
        model_code = (output_dir / "new_nn.py").read_text(encoding='utf-8')
        print(f"📄 Model code read: {len(model_code)} characters")
        
        # Read training metrics
        metrics = json.loads((output_dir / "eval_info.json").read_text(encoding='utf-8'))
        print(f"📊 Metrics loaded: accuracy={metrics.get('accuracy')}")
        
    except Exception as e:
        print(f"❌ Error reading results: {e}")
        state['status'] = 'error'
        return state
    
    # 3. Update state with results
    state['model_code'] = model_code
    state['accuracy'] = metrics.get('accuracy')
    state['loss'] = metrics.get('loss')
    state['metrics'] = metrics
    state['status'] = 'success'
    state['found_in_cache'] = False
    
    print(f"✅ Generator complete! Accuracy: {state['accuracy']}")
    
    return state