"""Design-space improvement fine-tuning on the LEMUR GenFractalNet corpus.

Replaces ``qwen3_5_9B_channel_alter.py``. What was wrong and what changed.

1. **The supervised target was not reachable from the prompt.**
   ``NN_gen_train_nn_improve.json`` sets ``no_repeat: ["nn"]``, so in
   ``join_nn_query_legacy`` the partner is resolved by ``ORDER BY RANDOM()``
   among any *different* model scoring higher. The prompt asks the model to
   edit the baseline; the label is an unrelated architecture. No function maps
   one to the other, so SFT can only learn "ignore the input, emit a memorised
   network". Here the target is the baseline with at most three *named* design
   decisions changed.

2. **The improvement threshold sat inside the noise floor.** Over 1,218 repeat
   groups on cifar-10 at epoch 1 -- same model, every hyperparameter identical
   -- the median run-to-run sd is 0.025 and 41.8% of groups span more than
   0.08. Among 9,664 pairs whose true improvement is zero, noise alone clears
   the production ``improve: 0.05`` threshold **16.4%** of the time. About one
   training pair in six was pure noise. Pairing now runs on a fitted factor
   model and requires the measurement to agree with it.

3. **Train and test asked different questions.** The old test prompt asked,
   unpaired, for "the model that achieves a maximum value for the metric after
   the first epoch", while training asked to raise a specific baseline. Both
   are now rendered from one pair of strings in ``design_pairs.py``.

4. **The functional output came last.** With ``<reason>`` first, an
   un-fine-tuned model writes a long rationale, exhausts ``max_new_tokens`` and
   never reaches the model file: measured at cycle 0, 19 of 41 generations
   produced no ``<nn>`` at all. The order is now ``<hp> <tr> <nn> <reason>``.

Why GenFractalNet, and why 13 factors
-------------------------------------
3,846 models trained on cifar-10 for one epoch, each a distinct point in a
13-factor space of 829,440 possibilities. Nine factors are declared in the file
header; four more -- Drop Path, Fractal Columns, Start Channels, Total Blocks
-- live in the body. With only the header nine, 602 design points collide while
their code still differs, which would make every "only these decisions changed"
claim in the corpus false. With all thirteen there are zero collisions.

Adjusted effects (regression, interactions averaged out). Raw marginal means
are confounded at 0.46% sampling density: LR and Momentum look like +0.069 /
+0.058 marginally but are +0.017 / +0.008 adjusted, and Drop Path, invisible in
the header, is the third-largest factor:

    Optimizer +0.169   Norm Type +0.123   Drop Path +0.102
    Start Channels +0.064   FC Dropout +0.047   Pooling +0.041
    Total Blocks +0.038   Conv Type +0.034   Fractal Columns +0.031
    Kernel +0.022   Activation +0.022   LR +0.017   Momentum +0.008

Median observed accuracy is 0.472 and the best observed model reaches 0.730 --
roughly 0.26 of real headroom against a 0.025 noise floor.

Run
---
    python -m ab.gpt.act.tune.qwen3_5_9B_design_improve
"""

import argparse
import os
from pathlib import Path

import ab.gpt.act.tune.Tune as TuneNNGen
from ab.gpt.util.Const import conf_test_dir, nngpt_dir
from ab.gpt.util.data import design_pairs

# --------------------------------------------------------------------------
# Measurement protocol
# --------------------------------------------------------------------------
TASK = "img-classification"
DATASET = "cifar-10"
METRIC = "acc"
EPOCH = 1

#: Only the design-space family. Mixing families in would reintroduce the
#: unrelated-target problem this script exists to remove.
NN_PREFIXES = ("GenFractalNet-",)

# --------------------------------------------------------------------------
# Pairing
# --------------------------------------------------------------------------

#: Pairing runs on the fitted factor model, not on raw observations.
#:
#: Each design point was trained once, and seed noise at a one-epoch budget is
#: large (see the module docstring). A ridge fit on the 13 factors plus all
#: two-way interactions reaches 5-fold CV RMSE 0.046 (R2 0.86) against an
#: observed spread of 0.126, and its fitted gains correlate 0.885 with observed
#: gains -- so the fitted value estimates a design point far better than its
#: own single training run.
MIN_FITTED_GAIN = 0.12

#: ...but the measurement still has to agree, which is what guards against the
#: factor model being misspecified. Neither signal is trusted alone.
MIN_OBSERVED_GAIN = 0.05

#: At most three of the thirteen named decisions change between baseline and
#: target, so the target stays reachable from the prompt by a small edit.
MAX_HAMMING = 3

# --------------------------------------------------------------------------
# Cycle budget
# --------------------------------------------------------------------------

#: Held-out baselines used for generation, spread across the score range.
#:
#: 30 keeps a cycle near 45 minutes of training. Paired over 30 held-out
#: baselines the standard error of the mean delta is about 0.019, so a +0.05
#: effect is 2.7 sigma inside a single cycle. n=40 would give 3.1 sigma for 33%
#: more wall clock, which is not the right trade: the seeds are pinned, so the
#: cycle-0-to-cycle-N comparison is paired per baseline and cancels baseline
#: difficulty, and the 16-cycle slope pools 480 observations either way.
TEST_NN = 30

#: Generate this many times TEST_NN candidates; the validity gate then trains
#: the first TEST_NN that pass.
#:
#: What this buys is sample size, not saved GPU time. Eval.evaluate() already
#: rejects structurally broken code before training, so an invalid generation
#: costs almost nothing -- but it also produces no paired observation. At the
#: observed valid rate, generating exactly TEST_NN prompts yields roughly half
#: the usable baselines and pushes the standard error from 0.019 to 0.027.
GENERATION_OVERSAMPLE = 2

#: One run per candidate. For a fixed budget of N trainings split as n
#: baselines x r repeats, the between-baseline term of Var(mean paired delta)
#: scales with r, so r=1 is optimal for any sigma_b > 0. There is also no
#: estimator mismatch to correct: GenFractalNet models have a single recorded
#: trial each, so the baseline's stored accuracy is already one unbiased run,
#: measured exactly as the candidate is.

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

#: The corpus is run *input*, so it must live outside ``nngpt_dir``.
#: ``EpochStats.reset_run_dir`` deletes that whole directory at the start of
#: every run -- "every file under out/nngpt belongs to one run" -- which would
#: remove the corpus between ``prepare()`` writing it and SFT reading it.
CORPUS_DIR = Path(nngpt_dir).parent / "design_improve"

#: Same reason, and it additionally keeps the pinned test baselines stable
#: across restarts, which is what makes cycle 0 and cycle N comparable.
SEED_FILE = CORPUS_DIR / "seeds.json"

TEST_CONF = "NN_design_improve_test.json"
CONF_KEY = "design_improve_test"

#: Never read. ``Tune.py`` builds ``conf_train_dir / llm_tune_conf`` as a Path,
#: but with ``data_dir`` set ``NNGenPrompt.get_raw_dataset`` returns from
#: ``_raw_dataset_from_disk`` before opening it -- the corpus on disk already
#: holds fully rendered system/user/assistant turns, so there is nothing left
#: to template. An existing upstream filename is used so the reference is never
#: dangling.


def prepare(offline_root: str = None) -> dict:
    """Build the corpus, the generation prompt config and the pinned seeds.

    The corpus is rebuilt from LEMUR on every start. Caching it would save a
    minute or two and risk training on a stale corpus after a LEMUR update
    while the printed statistics describe the old one.
    """
    stats = design_pairs.build(
        out_dir=str(CORPUS_DIR),
        task=TASK, dataset=DATASET, metric=METRIC, epoch=EPOCH,
        nn_prefixes=NN_PREFIXES,
        min_fitted_gain=MIN_FITTED_GAIN,
        min_observed_gain=MIN_OBSERVED_GAIN,
        max_hamming=MAX_HAMMING,
        # Anything already pinned as a test baseline is held out
        # unconditionally, so the test set is excluded by construction and
        # stays excluded however the corpus changes. On the first build the
        # file does not exist yet and the hashed holdout decides;
        # emit_pinned_seeds below then writes it from that holdout.
        pinned_seed_file=str(SEED_FILE),
        offline_root=offline_root,
    )
    print(f"[design-improve] corpus: {stats}")

    # Written from the same constants the corpus was rendered with, so the
    # generation prompt is byte-identical to the training prompt.
    design_pairs.emit_test_config(
        str(Path(conf_test_dir) / TEST_CONF), key=CONF_KEY,
        task=TASK, dataset=DATASET, metric=METRIC, epoch=EPOCH,
        nn_prefixes=NN_PREFIXES)

    # ab/gpt/util/eval/Seed.py already pins a {"nn": [...]} list for a whole
    # run; pointing it at held-out baselines is all that is needed to evaluate
    # on models the fine-tune has never seen.
    design_pairs.emit_pinned_seeds(str(CORPUS_DIR), str(SEED_FILE), n=TEST_NN)
    os.environ["NNGPT_SEED_FILE"] = str(SEED_FILE)
    os.environ["NNGPT_PIN_SEEDS"] = "1"

    # The design point *includes* lr and momentum, so the baseline's prm must
    # not overwrite what the model chose -- that would silently delete two of
    # the thirteen factors it is being trained to reason about.
    os.environ["NNGPT_EVAL_PRM_MODE"] = "generated"

    # Train only the generations that can actually run, and only TEST_NN of
    # them, so oversampling does not put the cycle over budget.
    os.environ["NNGPT_VALIDITY_GATE"] = "1"
    os.environ["NNGPT_VALIDITY_KEEP"] = str(TEST_NN)

    # The baseline file appears in both the prompt and the target, so loss over
    # the prompt mostly teaches the model to copy text it was handed. Routed
    # through the environment rather than a keyword because
    # ab/gpt/act/tune/Tune.py dispatches to tune() implementations in Tune,
    # Tune_Onnx and Tune_prun that do not share a signature; that module
    # already configures seed pinning the same way and says why.
    os.environ["NNGPT_TRAIN_ON_COMPLETIONS_ONLY"] = "1"

    # Cap the vLLM sequence window independently of the trainer's context.
    # VLLMSleep._engine_window puts it plainly: KV cache is linear in
    # max_model_len, and on a card where the weights take most of the budget
    # that one number decides whether anything can batch. Measured over the
    # corpus: prompt max 2,640 tokens plus a 4,096 generation cap -> 6,736
    # needed, so 8,192 leaves margin. Reserving more KV per sequence than that
    # buys nothing and makes generation close to serial.
    os.environ.setdefault("NNGPT_VLLM_MAX_MODEL_LEN", "16*1024")
    return stats


def main():
    prepare(offline_root=None)

    TuneNNGen.main(
        llm_conf='qwen3.5_9b.json',
        nn_gen_conf=TEST_CONF,
        nn_gen_conf_id=CONF_KEY,

        # The SFT corpus is read from disk, so NNGenPrompt bypasses the LEMUR
        # join entirely -- no random partner, no enhance_nn threshold.
        data_dir=str(CORPUS_DIR),

        # Over-generate, then train only the statically valid ones. The
        # expensive stage still sees exactly TEST_NN models.
        test_nn=TEST_NN * GENERATION_OVERSAMPLE,

        skip_epoches=-1,          # generate at cycle 0 for the pre-tune control
        nn_name_prefix='llm',
        cycles=16,
        num_train_epochs=1,
        context_length=16*1024,
        sft_max_length=6 * 1024,
        max_new_tokens=12 * 1024,
        # ~750 examples per cycle over a ~5.7k corpus.
        max_prompts=768,
        only_best_accuracy=True,
        # 5e-5 on a 9B model over a narrow corpus overwrites the pretrained
        # coding ability that makes the first-cycle generations valid at all.
        learning_rate=1e-5,
        lora_alpha=64,

        suppress_thinking=True,
        use_unsloth=True,
        use_vllm=True,
    )


if __name__ == "__main__":
    main()
