<img src="https://abrain.one/img/lemur-nn-icon-64x64.png" width=32 /> Alter — LLM-Based Modification of Neural Networks
===

Part of the [NNGPT](https://github.com/ABrain-One/nn-gpt) project.
Package: `ab.gpt.act.alter.*` · Core implementation: `ab/gpt/util/nn/AlterNN.py`

---

## 1. What is "Alter"?

**Alter is the *generation* stage of NNGPT.** It takes neural networks that already
exist in the [LEMUR Dataset](https://github.com/ABrain-One/nn-dataset), shows them to a
Large Language Model (LLM) together with their measured accuracy, and asks the LLM to
write an **improved version** of each architecture. The answers are saved as ready-to-run
PyTorch files.

In one sentence:

> *Alter = (existing models + their statistics) → LLM → (new candidate models on disk).*

Its place in the NNGPT pipeline:

````
   LEMUR Dataset            ab.gpt.act.alter.*            evaluation              fine-tuning
 (models + accuracy)  ──►   generate new models   ──►   train & measure   ──►   improve the LLM
                                 (THIS MODULE)              (NNEval)              (TuneNNGen*)
````

Alter only **writes code**. It never trains the generated networks — that is the job of the
evaluation step. This separation makes experiments cheap: generation needs minutes,
training needs hours.

### Why "alter" and not "create from scratch"?

Asking an LLM for a completely new architecture usually produces code that does not run.
Starting from a *known-good* model and asking for a *modification* keeps the output inside
the LEMUR interface (class `Net`, `train_setup`, `learn`, `supported_hyperparameters`) and
makes a much larger share of generations trainable. Alter additionally supplies
**supporting models** — other architectures sampled from LEMUR — so the LLM can transfer
ideas between architectures instead of only perturbing one.

---

## 2. Quick start

### 2.1 Local LLM (default, needs a GPU)

````bash
python -m ab.gpt.act.alter.olymp_7B_code -e 1
````

Loads `open-r1/OlympicCoder-7B` on the local GPU and runs **1** generation epoch.
A GPU with **≥ 24 GB** of memory is recommended.

### 2.2 Remote LLM served by vLLM (no local GPU needed)

````bash
python -m ab.gpt.act.alter.olymp_7B_code -e 1 --vllm_url http://<SERVER_HOST>:<PORT>/v1
````

With `--vllm_url`, the local model is **not loaded at all**; the prompts are sent to an
OpenAI-compatible [vLLM](https://docs.vllm.ai) server. This is how large models
(e.g. 70B) can be used for network modification on a machine that could never hold them.

Replace `<SERVER_HOST>` and `<PORT>` with the address of your own server (ask your lab
administrator; in a cluster this is typically an internal IP and a node port).
`/v1` is appended automatically if you leave it out.

Check that the server answers before starting a long run:

````bash
curl http://<SERVER_HOST>:<PORT>/v1/models
````

To serve a model yourself:

````bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
       --port <PORT> --tensor-parallel-size <N_GPUS>
````

---

## 3. What happens during a run — step by step

For every epoch `e = 0 … epochs-1`:

1. **Load the prompt configuration.**
   A JSON file from `ab/gpt/conf/prompt/test/` (default `NN_alter.json`) defines the
   system message, the prompt text, and which database columns are inserted into it.

2. **Query the LEMUR Dataset.**
   `ab.nn.api.data(only_best_accuracy=True, task=…, nn_prefixes=…)` returns, for the
   configured task (e.g. `img-classification`), the **best** recorded run of every model.
   One row per architecture is sampled, so one prompt per architecture is produced.

3. **Sample supporting models.**
   For each main model, `n` further models are drawn from the `addon_task` pool
   (never the model itself). Their code and accuracy are appended to the prompt as
   *Supporting Model 1 … n*.

4. **Format the prompt.**
   Placeholders such as `{nn_code}`, `{accuracy}` and `{supporting_models_prompt}` are
   substituted. Missing placeholders produce a warning, not a crash.

5. **Generate.**
   * *Local path*: the prompt is rendered with the tokenizer chat template and
     `model.generate(...)` is called (`max_new_tokens = 8192`, `do_sample=True`,
     `temperature`, `top_k`, `top_p=0.95`). With `batch_size > 1`, prompts are
     left-padded and generated in batches.
   * *vLLM path*: the pairs *(system, user)* are sent to `/v1/chat/completions` through a
     thread pool (`vllm_workers` parallel requests) and results are consumed as they
     complete.

6. **Extract the code.**
   `extract_code()` searches the answer, in order, for `<nn>…</nn>` tags, then a
```` ```python ```` fence, then a bare fence, then a plain `class Net … def forward`.
   A guard rejects fragments shorter than 50 non-whitespace characters, pure `...`
   echoes, and anything without both `class` and `def`.

7. **Save.**
   A valid answer is written to a new folder `B<index>` (see §6). An invalid answer
   prints `[INFO]Response Invalid!` and is **skipped**, so `B` indices count only
   usable generations.

> **Note.** `alter()` deletes the whole epoch directory tree at start-up
> (`shutil.rmtree(epoch_dir())`). **Copy or rename results you want to keep before
> launching a new run.**

---

## 4. Entry scripts in this directory

Every file here is a small, readable launcher: it parses command-line arguments and calls
`alter()` with one LLM and one prompt configuration. Reading one of them is the fastest
way to understand the module.

Reference example — `olymp_7B_code.py`:

```python
alter(args.epochs, 'NN_alter.json', 'open-r1/OlympicCoder-7B',
      vllm_url=args.vllm_url, vllm_model=args.vllm_model, vllm_workers=args.vllm_workers)
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `-e`, `--epochs` | int | `8` | Number of generation epochs (one full pass over the selected models per epoch). |
| `--vllm_url` | str | `None` | Base URL of an OpenAI-compatible vLLM server, e.g. `http://<SERVER_HOST>:<PORT>/v1`. If omitted, the local model is used. |
| `--vllm_model` | str | first served model | Served model id, e.g. `meta-llama/Llama-3.3-70B-Instruct`. |
| `--vllm_workers` | int | `4` | Number of requests sent to the server in parallel. |

To create your own experiment, copy this script and change the model name or the
configuration file.

---

## 5. The `alter()` API

```python
from ab.gpt.util.nn.AlterNN import alter

alter(epochs, test_conf, llm_name, gguf_file=None, n=1,
      temperature=0.6, top_k=50, **kwargs)
```

### Positional and common parameters

| Parameter | Default | Description |
|---|---|---|
| `epochs` | — | Number of generation epochs. |
| `test_conf` | — | Prompt configuration file name inside `ab/gpt/conf/prompt/test/`. |
| `llm_name` | — | Hugging Face model id for the **local** path (ignored when `vllm_url` is set). |
| `gguf_file` | `None` | Optional GGUF weight file for the local loader. |
| `n` | `1` | Number of supporting models added to each prompt. |
| `temperature` | `0.6` | Sampling temperature. |
| `top_k` | `50` | Top-*k* sampling. |

### Keyword arguments (`**kwargs`)

| Keyword | Default | Description |
|---|---|---|
| `nn_prefixes` | from config | Tuple/list of architecture name prefixes, e.g. `('AlexNet', 'VGG')`. Filtering happens at SQL level — the fastest way to shorten a test run. |
| `batch_size` | `1` | Local path only: prompts generated per batch. |
| `load_in_4bit` | `False` | Local path only: 4-bit quantised loading. |
| `inference_gpt_oss` | `False` | Local path only: skip prompts longer than the limit below (avoids O(n²) attention OOM). |
| `inference_gpt_oss_max_input_length` | unlimited | Token limit used by the option above. |
| `vllm_url` | `None` | **Switch to the remote path.** OpenAI-compatible base URL. |
| `vllm_model` | first served | Served model id. |
| `vllm_workers` | `4` | Parallel requests. |
| `vllm_max_tokens` | `None` | Output limit; `None` lets the server use the remaining context. |

### Local vs. vLLM

| | Local path | vLLM path (`vllm_url` given) |
|---|---|---|
| Model loading | `LLM(llm_name)` into local GPU memory | none — model lives on the server |
| Local GPU memory | ≈ model size (≥ 24 GB typical) | **0 GB** |
| Practical model size | ≤ ~13B on one consumer GPU | 70B and beyond |
| Parallelism | `batch_size` | `vllm_workers` concurrent HTTP requests |
| Output limit | `max_new_tokens = 8192` | `vllm_max_tokens`, else server default |
| Saving | identical | identical |

Behaviour without `vllm_url` is **unchanged** — the remote path is purely additive.

**Client details** (`ab/gpt/util/llm/VLLMClient.py`): request timeout 600 s, up to 3
attempts with a growing pause (5 s, 10 s, 15 s). Connection errors are retried; HTTP 4xx
responses (for example a prompt longer than the served context) are **not**, because
retrying cannot help. A failed prompt is reported as `[VLLM] …` and skipped; the run
continues. Requires the `httpx` package.

---

## 6. Output — what is produced and where

Generated files are written under the LEMUR output root (`<project_root>/out/`):

```
out/nngpt/llm/epoch/A<epoch>/synth_nn/B<index>/
```

* `A<epoch>` — generation epoch, starting at `A0`.
* `B<index>` — one **successfully extracted** model, starting at `B0`.

Contents of a `B<index>` folder:

| File | Content |
|---|---|
| `new_nn.py` | **The generated model.** Extracted Python code, ready for evaluation. |
| `full_output.txt` | The complete, unmodified LLM answer (reasoning, explanations, code). Useful for debugging and for reporting in a thesis. |
| `original_<NN>.py` | The LEMUR source model the generation started from. |
| `dataframe.df` | Pickled pandas row (task, dataset, metric, hyper-parameters, accuracy) passed on to the evaluator. |

Concrete example:

```
out/nngpt/llm/epoch/A0/synth_nn/B0/new_nn.py
out/nngpt/llm/epoch/A0/synth_nn/B0/full_output.txt
out/nngpt/llm/epoch/A0/synth_nn/B0/original_AlexNet.py
out/nngpt/llm/epoch/A0/synth_nn/B0/dataframe.df
```

Points worth remembering:

* **Invalid answers create no folder.** If 10 prompts are sent and 8 answers contain
  usable code, you get `B0 … B7`.
* **In vLLM mode the `B` numbering follows completion order**, not prompt order — results
  are saved as they arrive. The accompanying `original_*.py` and `dataframe.df` always
  belong to the same generation, so nothing is mismatched.
* Console output during a run: a `tqdm` progress bar, plus `Response Available!`,
  `[INFO]Saving code to: …` and `[EXTRACT] ✓ Found NN code: N chars` per generation.

---

## 7. Prompt configuration

Default file: `ab/gpt/conf/prompt/test/NN_alter.json`. Structure:

| Key | Meaning |
|---|---|
| `task` | LEMUR task used for the **main** models, e.g. `img-classification`. |
| `addon_task` | Task pool used for the **supporting** models. |
| `input_list` | Mapping *placeholder → database column* for the main model (`nn_code`, `accuracy`). |
| `addon_list` | Same mapping for supporting models (`addon_nn_code`, `addon_accuracy`). |
| `nn_prefixes` | Optional architecture-name filter (overridden by the `nn_prefixes` argument). |
| `system` | Lines joined into the system message. |
| `prompt` | Lines joined into the user message; may contain `{placeholders}` and `{supporting_models_prompt}`. |

The supplied template instructs the LLM to keep the class name `Net`, keep the signatures
of `__init__`, `forward`, `train_setup(device)` and `learn(data, target, device)`, expose
`supported_hyperparameters() → ['lr', 'momentum']`, use plain PyTorch only (no
`torchvision` pre-built models), and target 32×32 RGB inputs with 10 classes.

To run a different experiment, add a new JSON file to the same folder and pass its name as
`test_conf`.

---

## 8. Delta-based variant

`AlterNN.py` also provides `alter_delta(...)`, which asks the LLM for a **unified diff**
instead of a whole file. The diff is validated and applied to the baseline model; the
patch is stored as `delta.diff` next to `new_nn.py`. If delta extraction or application
fails, the function falls back to normal full-code extraction. It requires a
delta-enabled configuration (`use_delta: true`, e.g. `NN_gen_delta.json`) and otherwise
falls back to `alter()`. See *Delta-Based Neural Architecture Search*
([arXiv:2605.04903](https://arxiv.org/abs/2605.04903)).

---

## 9. After generation

Generated models are candidates, not results. The next step trains them and records their
accuracy (see the project README, `ab.gpt.NNEval`), after which validated models can be
contributed back to LEMUR and used for LLM fine-tuning (`ab.gpt.TuneNNGen*`). This closes
the NNGPT loop: better models → better training data → a better generator.

---

## 10. Troubleshooting

| Symptom | Cause and remedy |
|---|---|
| `CUDA out of memory` on the local path | Use `--vllm_url`, or `load_in_4bit=True`, or a smaller model, or `batch_size=1`. |
| `vLLM request failed: HTTP 400` | Prompt longer than the served context. Reduce `n`, or serve the model with a larger `--max-model-len`. Not retried by design. |
| `vLLM request failed` (connection) | Server unreachable. Verify with `curl http://<SERVER_HOST>:<PORT>/v1/models`; check VPN, firewall and port. |
| Many `[INFO]Response Invalid!` lines | The model is not returning extractable code. Inspect `full_output.txt`, lower `temperature`, or choose a code-specialised model. |
| Run takes very long | Restrict the models: `nn_prefixes=('AlexNet', 'VGG')` and `-e 1`. |
| Previous results disappeared | The epoch directory is cleared at start-up. Back up `out/nngpt/llm/epoch/` first. |
| `ModuleNotFoundError: httpx` | `pip install httpx` (required only for the vLLM path). |

**Reference measurement.** On a lab workstation, `alter()` with `vllm_url` against
Llama-3.3-70B-Instruct, restricted to `nn_prefixes=('AlexNet','VGG')`: 8 of 8 prompts
answered and saved in 86 s, with no local GPU memory used.

---

## 11. Citation and license

If this module contributes to your research, please cite NNGPT:

```bibtex
@InProceedings{ABrain.NNGPT,
    title = {{NNGPT}: Rethinking {AutoML} with Large Language Models},
    author = {Kochnev, Roman and Khalid, Waleed and Uzun, Tolgay Atinc and Zhang, Xi and Dhameliya, Yashkumar Sanjaybhai and Qin, Furui and Vysyaraju, Chandini and Duvvuri, Raghuvir and Goyal, Avi and Ignatov, Dmitry and Timofte, Radu},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
    pages = {5664--5674},
    year = {2026}
}
```

Licensed under the MIT license, as part of the NNGPT project.

#### The idea and leadership of Dr. Ignatov
````
