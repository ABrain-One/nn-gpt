# AI Agent System / NNGPT

Neural network generation and fine-tuning pipeline with optional LangGraph multi-agent orchestration.

---

## LangGraph Multi-Agent Integration

The pipeline supports an optional **LangGraph multi-agent workflow** that coordinates:

- **Manager agent**: Routes workflow and manages GPU resources
- **Generator agent**: Generates neural network models (calls `nn_gen()` from `Tune.py`)
- **Predictor agent** (optional): Predicts final accuracy from early-epoch metrics (uses `TuneAccPrediction.py`)

All orchestration logic lives in **one place**: `tune_with_agents()` in `ab/gpt/util/Tune.py`. The entry point remains `TuneNNGen.py`.

### How to Run

**Traditional pipeline (no agents):**
```bash
python -m ab.gpt.TuneNNGen --test_nn 1
```

**LangGraph workflow (agents, no predictor):**
```bash
python -m ab.gpt.TuneNNGen --use_agents --test_nn 1
```

**LangGraph workflow (agents + predictor):**
```bash
python -m ab.gpt.TuneNNGen --use_agents --use_predictor --test_nn 1
```

**Note:** The predictor requires a fine-tuned model. Train it first with:
```bash
python -m ab.gpt.TuneAccPrediction
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--use_agents` | False | Enable LangGraph multi-agent workflow |
| `--use_predictor` | False | Enable predictor agent (requires `--use_agents`) |

### File Structure

| File | Role |
|------|------|
| `ab/gpt/TuneNNGen.py` | Entry point; routes to `tune()` or `tune_with_agents()` |
| `ab/gpt/util/Tune.py` | Contains `tune()`, `tune_with_agents()`, `nn_gen()`, and shared helpers |
| `ab/gpt/agents/manager.py` | Manager agent (routing, GPU) |
| `ab/gpt/agents/generator.py` | Generator agent (calls `nn_gen()` from Tune.py) |
| `ab/gpt/agents/predictor.py` | Predictor agent (calls TuneAccPrediction functions) |
| `ab/gpt/agents/state.py` | Shared state definition |

### Design Principles

- **Single source of code**: Orchestration in `tune_with_agents()` in `Tune.py` only
- **No duplication**: Agents call existing functions (`nn_gen()`, `load_llm_and_chatbot()`, etc.)
- **Resource management**: GPU released after each agent; manager coordinates access
- **Backward compatible**: Default `use_agents=False` keeps original pipeline unchanged

### Dependencies

Requires `langgraph`. See `requirements.txt`.
