# Transform and Augmentation Pipelines

Everything is under `ab/gpt/brute/trans/`. There are two independent tracks that share one LLM fine-tuning loop.

| Track | What is searched | Folder |
|---|---|---|
| Transform | Per-image torchvision preprocessing pipelines | `trans/transform` |
| Augment | Batch-level augmentation *schedules* | `trans/augment/` |


## 0. Rules that apply to both

**Search with the validation split, report with the test split.** 
   - Default run = search phase, scored on the 10% validation slice.
   - `--final_judgment` = scored once on the real test set 


## 1. Transform pipeline

### 1.1 Build a corpus
```bash
python -m ab.gpt.brute.trans.GenerateTransform -t 3 -n 200 -o ab/gpt/brute/trans/transform/out
```
Writes `-n` standalone `.py` files, each defining `transform(norm)`. It **wipes the output folder first**.

### 1.2 Evaluate candidates

```bash
python -m ab.gpt.brute.trans.transform.TransformEval --num_epochs 20 --seed 42
python -m ab.gpt.brute.trans.transform.TransformEval --num_epochs 20 --seed 23 --final_judgment
```

### 1.3 LLM loop (generate, evaluate, fine-tune with OlympicCoder-7b)
```bash
python -m ab.gpt.act.tune.code_7B_olympic_trans_ft
```


## 2. Augment pipeline

### 2.1 What a schedule is
```json
[["mixup", 0.4, 0.3], ["cutmix", 1.0, 0.5], ["none", null, 0.2]]
```
`[method, alpha, fraction]`, applied **in order within every epoch**. Fractions sum to exactly 1.0. Batch
`i` of `n` uses the segment containing `(i + 0.5) / n`. The schema is in `ScheduleValidate.py`.

### 2.2 Generate candidate configs
```bash
# grid
python -m ab.gpt.brute.trans.augment.ConfigGen -t 3 -n 40 --seed 42
# off-grid (continuous alphas)
python -m ab.gpt.brute.trans.augment.ConfigGen --offgrid -n 60 --seed 42
# reorderings of the best existing schedules (order as the only variable)
python -m ab.gpt.brute.trans.augment.PermuteGen --per_tier 5 --max_per_source 5 --seed 42
```

### 2.3 Evaluate
```bash
python -m ab.gpt.brute.trans.augment.AugEval --config_file <file>.json --seed 42 --num_epochs 30
```

### 2.4 TPE Pipeline
```bash
python -m ab.gpt.brute.trans.augment.TPE --n_trials 60 -e 30 --seed 42 --study tpe_imagenette_resnet
```
Trials are scored on the validation split. After the study ends, the best trial is re-scored once on the test
split. `--n_trials` counts *additional* trials; `--n_trials 0` just re-judges an existing study.

### 2.5 LLM loop
```bash
python -m ab.gpt.act.tune.code_7B_olympic_trans_ft
```
### 2.5 LLM loop

Both tracks share one entry point:

```bash
python -m ab.gpt.act.tune.code_7B_olympic_trans_ft
```

Which track it runs is set by two flags:

| Track     | `trans_mode` | `aug_mode` |
|-----------|--------------|------------|
| Transform | `True`       | `False`    |
| Augment   | `False`      | `True`     |