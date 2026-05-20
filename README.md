# <img src='https://abrain.one/img/lemur-nn-icon-64x64.png' width='32px'/> GPT-Driven Neural Network Generator

<sub><a href='https://pypi.python.org/pypi/nn-gpt'><img src='https://img.shields.io/pypi/v/nn-gpt.svg'/></a> <a href="https://pepy.tech/project/nn-gpt"><img alt="GitHub release" src="https://static.pepy.tech/badge/nn-gpt"></a><br/>
short alias  <a href='https://pypi.python.org/pypi/lmurg'>lmurg</a> 
</sub> 
<br/>
<img src='https://abrain.one/img/nngpt-logo-tr.png' width='25%'/>
<h3>📖 Overview</h3>

This Python-based <a href='https://github.com/ABrain-One/nn-gpt'>NNGPT</a> project leverages large language models (LLMs) to automate the creation of neural network architectures, streamlining the design process for machine learning practitioners. It leverages various neural networks from the <a href="https://github.com/ABrain-One/nn-dataset">LEMUR Dataset</a> to fine-tune LLMs and provide insights into potential architectures during the creation of new neural network models.

## LangGraph Multi-Agent Workflow

NNGPT supports an optional LangGraph-based multi-agent orchestration mode. The agent system integrates directly inside `tune()` — no separate entry point, no duplicated logic.

### Design Principle

All pipeline logic remains in `ab/gpt/util/Tune.py` as the **single source of truth**. Agent nodes are thin wrappers only — they read from state and call the existing functions. No logic is reimplemented inside any agent file.

### Agent Flow

The professor-specified flow is: **Finetuner → Generator → Evaluator → Predictor**


- **manager** — controls routing, checks epoch stop condition, decides next node
- **generator** — calls `nn_gen()` / `trans_gen()`; skips if epoch < skip_epoch; skips evaluator if no code generated
- **evaluator** — calls `_evaluate_epoch()`; stores accuracy and all predictor inputs in state
- **finetuner** — calls `_finetune_epoch()`; increments epoch counter, returns to manager
- **predictor** — optional; activates after epoch 1 and epoch 2 accuracies are both available

Any future improvement to `nn_gen()`, `trans_gen()`, `_evaluate_epoch()`, or `_finetune_epoch()` automatically applies to both classic and agent modes.

### Crash Recovery

Agent mode uses LangGraph `MemorySaver` checkpointing. If the pipeline crashes mid-epoch (e.g. GPU OOM), re-running with the same `nn_name_prefix` resumes from the last completed node — no restart from epoch 0.

### Usage

The agent mode is enabled by default.

To use the accuracy predictor agent:

```bash
python -m ab.gpt.TuneNNGen --use_predictor
```

### Agent Files

| File | Purpose |
|---|---|
| `ab/gpt/agents/run_agent.py` | Builds and runs the LangGraph StateGraph |
| `ab/gpt/agents/manager.py` | Routing logic and epoch stop condition |
| `ab/gpt/agents/predictor.py` | Optional accuracy prediction node |
| `ab/gpt/agents/state.py` | Shared `AgentState` TypedDict — field names match LEMUR DB columns |
| `ab/gpt/util/Tune.py` | Single source of truth: `nn_gen`, `trans_gen`, `_evaluate_epoch`, `_finetune_epoch`, `generate_step`, `evaluate_step`, `finetune_step` |
| `ab/gpt/util/AccPredictor.py` | Accuracy predictor interface (to be implemented) |

## Create and Activate a Virtual Environment (recommended)
For Linux/Mac:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python3 -m pip install --upgrade pip
   ```
For Windows:
   ```bash
   python3 -m venv .venv
   .venv\Scripts\activate
   python3 -m pip install --upgrade pip
   ```

It is assumed that CUDA 13.0 is installed; otherwise, consider replacing 'cu130' with the appropriate version. Most LLM usage scenarios require GPUs with at least 24 GB of memory.

## Environment for NNGPT Developers
### Pip package manager

Create a virtual environment, activate it, and run the following command to install all the project dependencies:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
pip install -r req-no-isolation.txt --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cu130
```

If there are installation problems, install the dependencies from the 'requirements.txt' file one by one.

## Update of NN Dataset
To get the latest code and statistics, install the most recent version of the LEMUR Dataset from GitHub:
```bash
rm -rf db
pip uninstall -y nn-dataset
pip install --no-cache-dir git+https://github.com/ABrain-One/nn-dataset --extra-index-url https://download.pytorch.org/whl/cu130
```
Installing the stable version:
```bash
pip uninstall -y nn-dataset 
pip install nn-dataset --extra-index-url https://download.pytorch.org/whl/cu130
```
Adding functionality to export data to Excel files and generate plots for <a href='https://github.com/ABrain-One/nn-stat'>analyzing neural network performance</a>:
```bash
pip install nn-stat --extra-index-url https://download.pytorch.org/whl/cu130
```
and export/generate:
```bash
python -m ab.stat.export
```

## Installation of NNGPT with pip

```bash
   pip install nn-gpt --extra-index-url https://download.pytorch.org/whl/cu130
   pip install nn-gpt[flash] --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cu130
   ```

## Use

- **`ab.gpt.NNAlter*.py`** – Generates modified neural network models.  
  Use the `-e` argument to set the number of epochs for the initial CV model generation.

- **`ab.gpt.NNEval.py`** – Evaluates the models generated in the previous step.

- **`ab.gpt.TuneNNGen*.py`** – Performs fine-tuning and evaluation of an LLM. For evaluation purposes, the LLM generates neural network models, which are then trained to assess improvements in the LLM’s performance on this task. The -s flag allows skipping model generation for the specified number of epochs.

- **`ab.gpt.TuneNNGen_delta.py`** – Delta-based fine-tuning entry point (see [arXiv:2605.04903](https://arxiv.org/abs/2605.04903)). The LLM generates compact unified diffs (deltas) to refine baseline architectures instead of full code. Uses paper-aligned hyperparameters (lr=1e-5, temperature=0.35, top-k=50, LoRA with `lm_head`). Calls `TuneNNGen.main()` with delta defaults — no upstream behavior is changed.
  ```bash
  python -m ab.gpt.TuneNNGen_delta
  python -m ab.gpt.TuneNNGen_delta --llm_conf qwen2.5_coder_7b_instruct.json
  ```

<a href='https://huggingface.co/ABrain'><strong>Fine-tuned LLMs</strong></a>

### 🐳 Docker
All versions of this project are compatible with <a href='https://hub.docker.com/r/abrainone/ai-linux' target='_blank'>AI Linux</a> and can be seamlessly executed within the AI Linux Docker container.

Installing the latest version of the project from GitHub
```bash
docker run --rm -u $(id -u):ab -v $(pwd):/a/mm abrainone/ai-linux:llm bash -c "[ -d nn-gpt ] && git -C nn-gpt pull || git -c advice.detachedHead=false clone --depth 1 https://github.com/ABrain-One/nn-gpt"
```

Running script
```bash
docker run --rm -u $(id -u):ab --shm-size=16G -v $(pwd)/nn-gpt:/a/mm abrainone/ai-linux:llm bash -c "python -m ab.gpt.TuneNNGen_8B"
```

If recently added dependencies are missing in the <a href='https://hub.docker.com/r/abrainone/ai-linux' target='_blank'>AI Linux</a>, you can create a container from the Docker image ```abrainone/ai-linux:llm```, install the missing packages (preferably using ```pip install <package name>```), and then create a new image from the container using ```docker commit <container name> <new image name>```. You can use this new image locally or push it to the registry for deployment on the computer cluster.

## Citation

The original version of this project was created at the Computer Vision Laboratory of the University of Würzburg by the authors mentioned below. If you find this project to be useful for your research, please consider citing our articles for <a target='_blank' href='https://arxiv.org/pdf/2511.20333'>NNGPT</a>, <a target='_blank' href='https://arxiv.org/pdf/2601.02997'>architecture design</a>, <a target='_blank' href='https://openaccess.thecvf.com/content/ICCV2025W/AIM/papers/Kochnev_Optuna_vs_Code_Llama_Are_LLMs_a_New_Paradigm_for_ICCVW_2025_paper.pdf'>hyperparameter tuning</a> and <a target='_blank' href='https://arxiv.org/abs/2605.04903'>delta-based NAS</a> with LLMs:
```bibtex

@InProceedings{ABrain.NNGPT,
	title = {{NNGPT}: Rethinking {AutoML} with Large Language Models},
	author = {Kochnev, Roman and Khalid, Waleed and Uzun, Tolgay Atinc and Zhang, Xi and Dhameliya, Yashkumar Sanjaybhai and Qin, Furui and Vysyaraju, Chandini and Duvvuri, Raghuvir and Goyal, Avi and Ignatov, Dmitry and Timofte, Radu},
	booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},	
	year={2026},
    note={to appear}
}

@InProceedings{ABrain.Architect,
	title={From Memorization to Creativity: {LLM} as a Designer of Novel Neural Architectures},
	author={Khalid, Waleed and Ignatov, Dmitry and Timofte, Radu},
	booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},	
	year={2026},
    note={to appear}
}

@InProceedings{ABrain.HPGPT,
	title={Optuna vs Code Llama: Are {LLMs} a New Paradigm for Hyperparameter Tuning?},
	author={Kochnev, Roman and Goodarzi, Arash Torabi and Bentyn, Zofia Antonina and Ignatov, Dmitry and Timofte, Radu},
	booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)},
	url={https://openaccess.thecvf.com/content/ICCV2025W/AIM/papers/Kochnev_Optuna_vs_Code_Llama_Are_LLMs_a_New_Paradigm_for_ICCVW_2025_paper.pdf},
	pages = {5664--5674},
	year={2025}
}

@Article{ABrain.DeltaNAS,
	title={Delta-Based Neural Architecture Search: {LLM} Fine-Tuning via Code Diffs},
	author={Adhikari, Santosh Premi and Timofte, Radu and Ignatov, Dmitry},
	journal={arXiv preprint arXiv:2605.04903},
	year={2026}
}

```
## Licenses

This project is distributed under the following licensing terms:
<ul><li>models with pretrained weights under the legacy <a href="https://github.com/ABrain-One/nn-dataset/blob/main/Doc/Licenses/LICENSE-DEEPSEEK-LLM-V2">DeepSeek LLM V2</a> license</li>
<li> all neural network models and their weights not covered by the above licenses, as well as all other files and assets in this project, are subject to the <a href="LICENSE">MIT license</a></li> 
</ul>

#### The idea and leadership of Dr. Ignatov
