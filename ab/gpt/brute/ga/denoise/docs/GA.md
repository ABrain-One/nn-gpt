# GA search — current state and the fixes that must not be reverted

Reference sheet for the genetic-algorithm arm. **§1 is a do-not-revert list.** Several items here
are corrections to behaviour that looked correct and was measured to be wrong; reintroducing the
old behaviour would silently degrade the corpus.

Last updated 2026-08-07.

---

## 1. DO NOT REVERT

| # | Fix | What breaks if reverted |
|---|---|---|
| 1 | **Zero-cost predictor is disabled in all 14 launchers.** | It ranks **ρ = −0.178** anti-correlated with truth (n=326, `predictor_tracking_history.csv`) while its own k-fold CV claimed +0.693. With `--predict-keep 0.5` it trained the top half by that ranking — i.e. **preferentially discarded the better half of every generation.** |
| 2 | **`arch_uid2` is THE distinctness identity, everywhere.** | Mixing identities silently re-admits duplicates. The graded cosine check is a *tightener*, never the definition. |
| 3 | **Rationale headers sit BELOW the shebang.** | An earlier edit put the predictor-disabled header above `#!/usr/bin/env bash` in all 14 launchers, so line 1 was a comment and `./launcher.sh` no longer ran under bash. Fixed 2026-08-07; keep `head -1` as `#!`. |
| 4 | **Never run `json_train_to_db`.** | Wipes uncommitted models. |
| 5 | **Pod-local SQLite copies use the `backup()` API, never `cp`.** | A plain `cp` of a live DB yields a corrupt file. |
| 6 | **`merge_shard_dbs.py`: dry-run first, max 8 shards per call.** | Single-writer discipline; more shards per call has failed. |
| 7 | **Never edit a script while a pod or coordinator is executing it.** | Kill the job first. |
| 8 | **`nn-dataset/data/midd` belongs to another session.** | Do not modify; the `midd-x30` pod is theirs. |
| 9 | **Leave the 132 final-epoch-only models alone.** | Already pushed. |
| 10 | **git commit/push is user-only.** | Never `git add -A` or `commit -a` in nn-dataset. |

Verify #1 and #3 at any time:

```bash
cd nn-gpt/ab/gpt/brute/ga/denoise
for f in growth_loop.sh launch_*.sh; do head -1 "$f" | grep -q '^#!' || echo "SHEBANG BROKEN: $f"; done
grep -l 'predict-keep' growth_loop.sh launch_*.sh   # expect: no output
```

---

## 2. Environment

| thing | value |
|---|---|
| host root | `/shared/ssd/home/b-b-ahmed/thesis` |
| pod path for the same tree | `/a/mm` |
| main DB | `nn-dataset/db/ab.nn.db` |
| K8s identity | `runAsUser: 1067`, `runAsGroup: 1376` |
| GPU quota | `requests.nvidia.com/gpu` **16**, no scopes — H100 counts against it |
| torch on host | only `thesis/venv_tflite/bin/python` (2.8.0+cpu) |

Operational gotchas that have cost hours:

- **`kubectl get pods` STATUS prints `Completed`; `.status.phase` is `Succeeded`.** Match both, and
  prefer `-o jsonpath='{.items[0].status.phase}'`.
- **Delete a job before re-applying the same name.**
- **`| tee` makes the pipeline exit status tee's (0).** Use `set -o pipefail` or a job that
  "succeeded" while crashing will be reported as success.
- **`pkill -f <pattern>` self-matches the shell running it.** Use `ps | grep 'patt[e]rn'` and kill
  by PID.
- **`prefix='rx'` in `check_nn` RENAMES the model**, so checkpoints land in
  `<root>/out/ckpt/rx-<hash>/`, not `<family>-<hash>/`. Match on the trailing hash when copying out
  (`Util.py:198`, `Const.py:147`).

---

## 3. Metrics

- `PSNR = accuracy × 48` (`ab/nn/metric/psnr.py`).
- **No-op (identity) PSNR = 33.021 dB** on the AWGN val crop, **32.34 dB** full-frame. A model
  reporting exactly 33.021 has collapsed to the identity — treat as degenerate, not as a result.
- **MAI 2021 denoising Final Score = 2^(2·PSNR) / (C · runtime)**, C = 2⁶⁴ (recovered, 7/7 published
  rows within 0.6%).
- **MAI 2021 classification = 2^(Top-1 + Top-3) / (C · runtime)**, C = 2¹⁸⁵ (8/8 rows ≤2.7%).
- **Score′ = 2^(2·(PSNR − PSNR_noop)) / runtime_ms** — our proposal, referenced to the no-op so the
  exponent is a *gain* rather than an absolute.
- Exchange rate implied by the MAI form: **+0.5 dB ⟺ 2× runtime**.

**Run-to-run noise floor is context-dependent — quote the right one.**

| source | floor |
|---|---|
| confirmation run, `psnrrgb`, 30 ep — AWGN | 0.207 dB |
| same — MIDD | 0.231 dB |
| same — SIDD | 0.182 dB |
| skip ablation, pooled within-cell, `psnr`, 50 ep | **0.142 dB** |

**There is no seeding in training.** No `manual_seed` in `Train.py`/`api.py`/`Util.py`; the train
loader at `Util.py:179` is `shuffle=True` with no generator. Test data *is* seeded
(`denoise_awgn.py:98`, `manual_seed(1000003+idx)`). So repeats differ, and any single-run difference
below the floor is not a finding.

`Util.py:203-208` documents that crop-PSNR checkpoint selection can **inverse-rank** against
full-resolution (SplitterNetX: 37.908/37.920 crop → 37.927/**37.562** full-res).

---

## 4. Selection: use `predict_final.py`, never a learned predictor

```python
rank_for_selection(candidates)   # epoch-3 measured PSNR only, calls no model
estimate_final_db(...)           # -> (dB, best_epoch, mae_dB), uses the fine-tuned predictor
provenance()                     # what produced each number
```

`CALIBRATION_MAE_DB = 0.509`, `SELECTION_RHO = 0.728`. Verified to reproduce the measured 82%
top-50 overlap exactly and to retain the best model. Ranking table in `LLM.md §7`.

---

## 5. The NPU deployability gate

`npu_gate.py` — **empirical, not static.** Modes: `--validate --convert --profile --verdict
--stage-ids --filter-ids`. Driver: `npu_gate_stage.sh` (3-phase; poll horizon 90 min).

- Device `Samsung Galaxy S26 Ultra`, 256×256, default ceiling **10.0 ms**.
- Validated **10/10** against known verdicts; re-measure stability median 0.2%, max 1.0%.
- **Fails OPEN when inconclusive.** `cmd_verdict` computes
  `coverage = measured / total` where `measured` excludes `reason == "not-profiled"`, and sets
  `inconclusive = coverage < --min-coverage` (default 0.8). `cmd_filter_ids` then passes everything.
  Rows that pass carry **no** `reason` key — use `r.get("reason")`, not `r["reason"]`.

**Static operator rules are NOT a substitute and are never used as a gate:**

| rule | vs "0% on NPU" | vs "latency > 10 ms" |
|---|---|---|
| `matmul` or `PixelUnshuffle` present | 100% / 100% | **36% recall / 50% precision** |
| any of {matmul, Softmax, normalize, sqrt, gelu, zeros} | 100% / 100% | 36% / 80% |

The perfect column is an artifact of labelling against `npu_time_frac`; against *latency* the same
rules miss two of three undeployable models, because the DenoiseGA slowness cluster is depth/width,
invisible to operator signatures. A 36%-recall gate is worse than none — it licenses the belief that
survivors are deployable. The operator table is emitted as **prompt guidance** only.

### Wiring

- `run3_evolve.sh` stage **`[k.3b]`**, between preflight and the 1-epoch screen. `SCREEN_IDS` is set
  on all three paths. `NPU_GATE=0` disables. Fails open loudly.
- `promote.py`: `--npu-verdict`, `--npu-max-ms 10` (default). `npu-slow` is **hard**;
  `NPU-UNMEASURED?` is **soft** via `_is_soft`. Loader honours `inconclusive`. Adds an `ms` report
  column and an `npu_ms` record field. Stage first with
  `npu_gate_stage.sh <sources> <work>`, then pass `--npu-verdict <work>/verdict.json`.

---

## 6. On-device measurement

`bulk_profile.py`. `DEVICES = ["Samsung Galaxy S26 Ultra", "Samsung Galaxy S24 (Family)"]`,
round-robin. **Device is recorded as a covariate and NOT normalised**: rank agreement ρ=0.9919 but a
per-model offset of 1.23–2.41×, so cross-device latency is not interchangeable.
`stratify()` is proportional-by-family via sha1 hash. Collector is a 16-thread pool — serial was the
bottleneck (469 ms/job × 789 = 6.2 min/pass; parallel = 88 s). Any "farm rate" measured before that
change is contaminated.

Latency findings live in `ABLATIONS.md §5`.

---

## 7. Where the artifacts are

| what | where | count |
|---|---|---|
| checkpoints | `sweep_out/ckpt/` | 1,660 (1,298 DenoiseGA) |
| checkpoints | `nn-dataset/out/ckpt/` | 19 |
| generated sources | `sweep_out/tflite_pool/codes.jsonl` | 2,869 |
| candidate sources | `sweep_out/llmprod/{a1,a5,a10,a12}/run/cand/*/candidate.py` | 1,699 |
| profiled latencies | `sweep_out/aihub/bulk_results.json` + `results_s26u.json` | 552 |

**`DnSeedGA` has zero checkpoints.** Known outstanding issue: **31 code-hash duplicate groups in
`ab/nn/nn/`** need cleanup before publishing.
