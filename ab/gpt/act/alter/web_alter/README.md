A pipeline that alters an existing LEMUR model in two stages:

1. **Plan:** the LLM reads the model and decides ONE architecture change as JSON (change, location, modules, reason). With `--extra_info`, the same prompt also contains web suggestions for that network or family, and the LLM may use one of them or none.
2. **Implement:** the LLM applies exactly that plan as a whole-file edit. A runtime smoke test (build the model, train on 2 real batches, check the output shape) catches broken code, followed by up to 2 repair rounds.

Without `--extra_info` the pipeline is the baseline, so web vs. no web differs only in the suggestions block.

Only new files; no existing file is changed.

## How to run
```bash
# baseline
python -m ab.gpt.act.alter.web_alter.exp3.staged_alter --set SET.csv --vllm_url http://<server>/v1 --samples 5 --seed 3000 --out OUT --name staged_base
# with web knowledge: JSON {nn or family: text}
python -m ab.gpt.act.alter.web_alter.exp3.staged_alter --set SET.csv --vllm_url http://<server>/v1 --extra_info notes.json --samples 5 --seed 3000 --out OUT --name staged_web
# train the generated models (Eval), chunk by chunk
python -m ab.gpt.act.alter.web_alter.run_eval_chunks OUT/staged_base OUT/staged_web
```
`--set` is a CSV with a column `nn` listing the LEMUR models to alter. `--only`, `--samples`, `--workers` and the sampling settings are listed in `--help`. Any OpenAI-compatible vLLM endpoint works; tested with Llama-3.3-70B-Instruct-FP8.

## Files
- `ab/gpt/act/alter/web_alter/exp3/`: the two-stage pipeline (`staged_alter`, `edit_alter`, `comprehension`)
- `ab/gpt/act/alter/web_alter/exp2/`: the code the implement stage imports, and the smoke test
- `ab/gpt/act/alter/web_alter/__init__.py` + `conf/prompt/test/NN_alter_web.json`: job building and vLLM backend
- `build_extra_info.py`, `retrieve.py`, `blocks.py`: turn retrieved web notes into `--extra_info`
- `run_eval_chunks.py`: chunked training of generated models

## Results so far (CIFAR-10, 26 networks, 2 epochs)
- Web vs. baseline: +1.1 points on average (95% CI -0.2 to +2.6), 14 networks better and 12 worse; not yet significant.
- RegNet with the web suggestion (ReLU -> SiLU): confirmed on new samples, +5.4 points, 5 of 5 better.
- Test of this branch: RegNet, one sample, web condition; smoke test passed on the first attempt, trained for 1 epoch (53.8%).

## Known limitations
- CIFAR-10 must already be in the repo's `data/` folder. A first-time download counts toward the 300 s smoke timeout and is treated as a code error, which triggers needless repairs.
- The stage-1 prompt targets the accuracy of each model after a short training; it does not yet target the best model.
- The web retrieval itself (search, page distillation) lives in a separate repository; this PR takes the resulting notes as JSON.

## Next steps
Target the best model instead of the mean, start from strong LEMUR models, add few-shot diffs of similar models (as in `qwen3_5_9B_edit_improve.py`) together with the web notes, and switch to a 27B Qwen on the DGX.
