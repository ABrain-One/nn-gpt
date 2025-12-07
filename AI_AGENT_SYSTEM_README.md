# AI Agent System - LangGraph Multi-Agent Orchestration

Multi-agent system for automated neural architecture generation using LangGraph framework to orchestrate existing nn-gpt pipeline.

## Overview

This system implements LangGraph-based multi-agent workflow:
1. **Manager Agent**: Coordinates GPU resources and workflow routing
2. **Generator Agent**: Wraps existing TuneNNGen.py pipeline
3. **Predictor Agent**: Placeholder for future accuracy prediction (in development)

## Current Status

- ✅ **Manager Agent**: Complete - GPU coordination and workflow routing
- ✅ **Generator Agent**: Complete - Wraps TuneNNGen.py in LangGraph node
- ✅ **LangGraph Integration**: Complete - StateGraph workflow orchestration
- ⏳ **Predictor Agent**: Placeholder (awaiting teammate's fine-tuned model)

## Architecture

### LangGraph Workflow
```
Manager Agent
    ↓
  (Decides: generate/predict/end)
    ↓
Generator Agent
    ↓
TuneNNGen.py (existing pipeline)
    ↓
  - LLM generates architecture
  - Trains for 2 epochs
  - Returns metrics
    ↓
State updated with results
    ↓
END
```

### Key Design Principles
- **Wrap, don't recreate**: Uses existing TuneNNGen.py without modification
- **Minimal code**: ~150 lines total for complete multi-agent system
- **Clean state management**: TypedDict interface for agent communication
- **Modern framework**: LangGraph for agent orchestration

## Implementation

### State Management (`src/state.py`)
Defines shared state between agents:
- Generator inputs: `epoch`, `conf_key`, `base_model_name`
- Generator outputs: `model_code`, `accuracy`, `loss`, `metrics`
- Manager control: `gpu_available`, `next_action`
- Tracking: `experiment_id`, `status`, `found_in_cache`

### Manager Agent (`src/agents/manager.py`)
Coordinates workflow:
- Checks GPU availability
- Routes to generator or predictor
- Manages workflow state

### Generator Agent (`src/agents/generator.py`)
Wraps existing pipeline:
- Calls `TuneNNGen.main()` from nn-gpt repo
- Reads generated model from `out/nngpt/llm/epoch/A{epoch}/synth_nn/B0/`
- Extracts metrics from `eval_info.json`
- Updates state with results

### Main Workflow (`src/main.py`)
LangGraph orchestration:
- Creates StateGraph with agent nodes
- Defines Manager → Generator flow
- Compiles and executes workflow

## Setup

### Prerequisites
- Python 3.10+
- CUDA-capable GPU
- nn-gpt and nn-dataset repositories cloned

### Installation
```bash
# Install LangGraph and dependencies
pip install langgraph langchain

# From nn-gpt root directory
cd nn-gpt
```

## Usage

### Run Complete Workflow
```bash
python src/main.py
```

### Expected Output
```
🤖 AI AGENT SYSTEM - Neural Architecture Generation
============================================================
�� Starting workflow with initial state...
Epoch: 0
Config: improve_classification_only

🎛️ Manager: Coordinating workflow...
✅ Manager: GPU available → Assigning to Generator
📊 Manager decision: generate

🤖 Generator: Starting for epoch 0
[TuneNNGen.py output...]
✅ TuneNNGen.py completed!
📄 Model code read: XXXX characters
📊 Metrics loaded: accuracy=0.XX
✅ Generator complete! Accuracy: 0.XX

============================================================
✅ WORKFLOW COMPLETE!
============================================================
```

## Project Structure
```
nn-gpt/
├── src/
│   ├── state.py              - State definition (TypedDict)
│   ├── main.py               - LangGraph workflow entry point
│   └── agents/
│       ├── generator.py      - Wraps TuneNNGen.py
│       ├── manager.py        - GPU coordinator
│       └── predictor.py      - Placeholder
├── ab/gpt/
│   └── TuneNNGen.py         - Existing pipeline (unchanged)
└── out/                      - Generated outputs
```

## Configuration

Generator uses TuneNNGen.py defaults:
- LLM: OlympicCoder-7B (from nn-gpt)
- Training epochs: 1 (controlled by `NN_TRAIN_EPOCHS`)
- LLM fine-tuning epochs: 3 (controlled by `NUM_TRAIN_EPOCHS`)

All configuration is in existing TuneNNGen.py - no duplication.

## Integration Benefits

**vs. Previous Approach:**
- ❌ Old: Recreated entire pipeline (496 lines)
- ✅ New: Wraps existing pipeline (~40 lines per agent)

**Advantages:**
- Minimal code maintenance
- No duplication of logic
- Easy to extend with new agents
- Modern agent framework (LangGraph)

## Future Work

- [ ] Complete predictor agent (awaiting fine-tuned model)
- [ ] Add conditional routing based on predictions
- [ ] Implement database caching layer
- [ ] Add retry logic for failed generations
- [ ] Support for multiple LLM backends

## Development

This project integrates LangGraph multi-agent orchestration into existing nn-gpt pipeline.

**Framework**: LangGraph (2024)
**Integration**: nn-gpt + nn-dataset repositories
**Approach**: Wrapper pattern (minimal modification to existing code)

## Acknowledgments

Built on top of ABrain's nn-gpt and nn-dataset frameworks.
