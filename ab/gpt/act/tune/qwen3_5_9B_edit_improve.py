"""Code-edit improvement fine-tuning over any architecture in LEMUR.

The sibling script ``qwen3_5_9B_design_improve.py`` reads a model as a point in
a closed 13-factor space. That is only definable for one generator template,
and it buys a factor model whose estimate of a design is far better than that
design's own single training run -- which is what keeps measurement noise out
of the training pairs.

This script gives that up in exchange for generality. A model is just a file;
an improvement is a small set of edits to it; and a pair is two models that sit
in the same architecture bucket, differ by at most a few dozen lines, and
differ in measured accuracy. Nothing here knows what a fractal column is, so it
works on AirNet, DPN, AlexNet or anything else the dataset holds.

What that costs, stated plainly: without a factor model the only guard against
noise is the size of the gain, and at this budget noise alone clears 0.05 about
one time in six. Expect a noisier corpus than the design-space one and judge
the run on the tail -- cycle_max and models above the corpus ceiling -- rather
than on the mean.

Why edits rather than whole files: a file is ~700 lines and most of a
generation's token budget goes into reproducing text the model was handed,
which is also the only place a reply can be truncated. An edit is a few dozen
lines. Why search/replace rather than a unified diff: a diff carries line
numbers and @@ hunks that have to be exactly right, and the model has no
reliable way to count lines. A <find> block is matched literally; getting it
wrong is detected rather than silently applied at the wrong offset.

Run
---
    python -m ab.gpt.act.tune.qwen3_5_9B_edit_improve
"""

import json
import os
import shutil
from pathlib import Path

import ab.gpt.act.tune.Tune as TuneNNGen
from ab.gpt.util.Const import conf_test_dir, nngpt_dir
from ab.gpt.util.data import edit_pairs

TASK = "img-classification"
DATASET = "cifar-10"
METRIC = "acc"
EPOCH = 1

#: Which architectures to train on. The mechanism is architecture-agnostic --
#: nothing below knows what a fractal column is -- so this is only a choice of
#: experiment. Set it to None to use every architecture in the dataset for this
#: pipeline (21,637 cifar-10 models across 2,343 architectures, against 3,846
#: for GenFractalNet alone).
#: No prefix filter: the similarity band below is the selector, and it draws
#: comparable architectures from the whole dataset. Restricting to
#: "GenFractalNet-" as well removed every model the band returned. Set this
#: only to confine the corpus to particular families, which costs the
#: architectural diversity this pipeline exists to use.
NN_PREFIXES = None

#: Architecture-similarity band the training neighbourhoods are drawn from.
#: Measured on this corpus: "high" yields 13,857 improving pairs across 289
#: distinct architectures and every one of them is expressible as a <=40-line
#: edit, so no diff work is wasted. "low" offers 600 architectures but only 26%
#: of its improving pairs are small enough. The previous class-name grouping
#: produced 13 groups from 3,909 models.
BANDS = ("high",)

#: A pair must clear this measured gain. Without a factor model this is the
#: only noise guard there is, so it is deliberately not lowered.
MIN_OBSERVED_GAIN = 0.05

#: An improvement the model can state as edits. Above this the reply stops
#: being a patch and becomes a rewrite, which is the failure this avoids.
MAX_CHANGED_LINES = 40

#: Prefix this run writes its own evaluated models back under. Must match the
#: nn_name_prefix passed to TuneNNGen.main below.
NN_NAME_PREFIX = "llm"

TEST_NN = 30

#: How many of the 30 test baselines may come from one architecture group --
#: the MinHash similarity cluster the corpus recorded, or the name prefix when
#: it recorded none.
#:
#: Measured without it: 29 of 30 were rl-bb-init-*, one search family's
#: near-identical variants. Nothing is learned from such a test set that
#: generalises, and the fine-tune that follows reproduces that family's
#: template -- 60 of 60 valid generations of one run shared a class skeleton
#: and differed by a backbone string. The cap is relaxed automatically if the
#: corpus cannot fill 30 slots under it.
TEST_NN_FAMILY_CAP = 3

#: Every similarity cluster in the corpus contributes at least this many
#: baselines to the holdout, so the cap above has diverse candidates to pick
#: from in the first place.
HOLDOUT_MIN_PER_FAMILY = 2

#: Neighbour examples per prompt. Four of them, next to a ~700-line baseline in
#: a 16k window, is what forced the answer budget down to 256 tokens and cost
#: one run 71 truncated replies. Two leaves room to answer.
CONTEXT_EXAMPLES = 2

#: Context window and answer budget.
#:
#: Raised from 16384/4096. A ~700-line baseline is ~9k tokens, and with two
#: examples and the now-required <tr> block the prompt reaches ~11k, leaving
#: under 5k for an answer -- which produced seven truncated replies in a
#: thirty-prompt cycle. Buying room is the right fix; cutting examples trades
#: away the context the model edits from. Qwen3.5 handles far more than this,
#: so the cost is KV cache on a card that also holds the parked trainer: if
#: the engine cannot allocate, lower max_num_seqs before lowering this.
CONTEXT_LENGTH = 24 * 1024
MAX_NEW_TOKENS = 10240
SFT_MAX_LENGTH = 20 * 1024

#: Smallest answer worth asking for, in tokens. The upstream default is 1280
#: and exists precisely to drop a prompt that cannot be answered; lowering it
#: to 256 did not make those prompts answerable, it only moved the failure from
#: "prompt dropped" to "reply cut off mid-class".
MIN_ANSWER_TOKENS = 8192

#: Re-measure this many of a cycle's best candidates, this many times each,
#: before reporting the cycle's result.
#:
#: OFF. The ladder re-trains the top candidates with nothing changed but the
#: random seed, to separate a better architecture from a luckier draw -- the
#: same model measured 0.9562 and 0.9621 on this budget. That costs
#: STAGE2_TOPK * STAGE2_REPEATS extra trainings per cycle, which at 10 minutes
#: each is most of an hour, and it is being given up deliberately.
#:
#: Two things follow automatically from the 0 and are worth knowing:
#: screening results go back to being written to LEMUR (the suppression below
#: is conditional on this value), so the corpus keeps growing from every
#: evaluated candidate rather than only from re-measured ones; and
#: MIN_OBSERVED_GAIN stays at its configured 0.05, because the noise floor in
#: edit_pairs has no measured repeat spread to raise it with.
#:
#: Set back to 5 to turn the ladder on again; nothing else has to change.
STAGE2_TOPK = 0
STAGE2_REPEATS = 3

CORPUS_DIR = Path(nngpt_dir).parent / "edit_improve"
SEED_FILE = CORPUS_DIR / "seeds.json"
TEST_CONF = "NN_edit_improve_test.json"
CONF_KEY = "edit_improve_test"


def emit_test_config(path, menu_unused=None):
    """Generation prompt, rendered from the same strings as the corpus."""
    cfg = {CONF_KEY: {
        "comment": ["Generated by qwen3_5_9B_edit_improve.emit_test_config.",
                    "Do not hand-edit: it must stay identical to the prompt the",
                    "SFT corpus was rendered with, or train and test diverge."],
        "system": edit_pairs.SYSTEM_PROMPT.format(
            dataset=DATASET, metric=METRIC, epoch=EPOCH).split("\n"),
        "task": TASK, "dataset": DATASET, "metric": METRIC, "epoch": EPOCH,
        "min_accuracy": 0.0, "no_repeat": [], "keep_same": [],
        # "transform"/"hyper"/"examples"/"target" are injected by the edit
        # branch of nn_gen, not taken from a dataset column.
        "input_list": [{"para": p, "value": p} for p in (
            "nn_code", "accuracy", "epoch", "dataset", "task", "metric", "prm")],
        "prompt": edit_pairs.USER_PROMPT.split("\n"),
    }}
    if NN_PREFIXES:
        cfg[CONF_KEY]["nn_prefixes"] = list(NN_PREFIXES)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return path


def emit_pinned_seeds(corpus_dir, seed_file, n=TEST_NN,
                      family_cap=TEST_NN_FAMILY_CAP):
    """Test baselines: strong, spread over the range, and spread over families.

    Two objectives at once. The objective of the run is the best model a cycle
    produces, and a baseline far below the ceiling cannot reach it by editing a
    few lines -- so the set is drawn mostly from the top of the holdout. But a
    set drawn only by accuracy is drawn from whichever family happens to own
    the top of the corpus, and measured that was 29 of 30 from one family: a
    benchmark that says nothing about any other architecture, and a fine-tuning
    signal that collapses onto one template.

    So the pick is capped per family, strongest first within each family, and
    the cap is relaxed one step at a time only if 30 slots cannot be filled
    under it. The previous version's "two thirds from the top quartile" was
    also unreachable as written: with 37 holdout rows the quartile holds 9, so
    min((2*n)//3, len(quartile)) gave 9 of 30, not 20.
    """
    from ab.gpt.util.data.family import describe, family_of

    rows = [json.loads(l) for l in
            open(Path(corpus_dir) / "holdout_baselines.jsonl", encoding="utf-8")]

    def group_of(row):
        """What the cap counts. The similarity cluster the corpus recorded, and
        only failing that the name prefix -- prefixes are a heuristic, and this
        is the one decision that determines whether the 30 test baselines
        measure anything beyond one search family's neighbourhood."""
        return row.get("cluster") or family_of(row["nn"])
    rows.sort(key=lambda r: r["observed"])
    if n >= len(rows):
        chosen = list(rows)
    else:
        n_top = max(1, (2 * n) // 3)
        # Deep enough to actually hold n_top rows, which the quartile need not.
        band = rows[-max(len(rows) // 4 or 1, n_top):]
        tail = [r for r in rows if r not in band]

        def pick(pool, want, cap):
            """`want` rows from `pool`, strongest first, <= cap per family."""
            out, taken = [], {}
            for r in sorted(pool, key=lambda r: -r["observed"]):
                fam = group_of(r)
                if cap and taken.get(fam, 0) >= cap:
                    continue
                out.append(r)
                taken[fam] = taken.get(fam, 0) + 1
                if len(out) >= want:
                    break
            return out

        chosen, cap = [], family_cap
        while True:
            chosen = pick(band, n_top, cap)
            seen = {r["nn"] for r in chosen}
            # The tail keeps the set from being only near-ceiling models: a
            # baseline with room to improve is where a large gain can show.
            for r in pick([r for r in tail if r["nn"] not in seen],
                          n - len(chosen), cap):
                chosen.append(r)
                seen.add(r["nn"])
            if len(chosen) >= n:
                break
            # Still short: either take more from the band under the same cap,
            # or, if the corpus simply has too few families, loosen it.
            extra = pick([r for r in rows if r["nn"] not in seen],
                         n - len(chosen), cap * 2 if cap else 0)
            chosen.extend(extra)
            if len(chosen) >= n or not cap:
                break
            cap *= 2
            print(f"[edit-improve] family cap relaxed to {cap} to fill "
                  f"{n} test baselines", flush=True)
        chosen = chosen[:n]
    names = [r["nn"] for r in chosen]
    print(f"[edit-improve] pinned test set: {describe(names)}; "
          f"{len({group_of(r) for r in chosen})} similarity cluster(s)",
          flush=True)
    Path(seed_file).parent.mkdir(parents=True, exist_ok=True)
    Path(seed_file).write_text(
        json.dumps({"nn": names, "pinned_at_epoch": 0}, indent=2),
        encoding="utf-8")
    return seed_file


#: Generated transforms written before names were derived from content.
#:
#: They were named gen_<md5 of the raw text>, so the same pipeline written with
#: a different blank line became a different transform: one cycle shipped six
#: files for three pipelines, three of which the dataset already had under
#: norm_32_flip. Nothing will ever refer to these names again -- a statistics
#: row naming one is a row for a transform that is a duplicate -- so they are
#: removed from both the submission and the installed dataset.
LEGACY_TRANSFORM_GLOB = "gen_[0-9a-f]*.py"


def sweep_legacy_transforms() -> int:
    """Delete gen_<hash>.py transforms left by the pre-uid naming."""
    import importlib.util
    from ab.gpt.util.Const import new_lemur_transform_dir

    dirs = [Path(new_lemur_transform_dir)]
    try:
        spec = importlib.util.find_spec("ab.nn")
        roots = [Path(spec.origin).parent] if (spec and spec.origin) else []
        roots += [Path(p) for p in (spec.submodule_search_locations or ())] if spec else []
        dirs += [r / "transform" for r in roots]
    except Exception:
        pass
    removed = []
    for d in dirs:
        try:
            if not d.is_dir():
                continue
            for f in sorted(d.glob(LEGACY_TRANSFORM_GLOB)):
                f.unlink()
                removed.append(str(f))
        except Exception as exc:
            print(f"[edit-improve] could not sweep {d}: {exc}", flush=True)
    if removed:
        print(f"[edit-improve] removed {len(removed)} transform(s) named by the "
              f"old raw-text hash: {', '.join(Path(r).name for r in removed)}",
              flush=True)
    return len(removed)


def prepare(offline_root=None) -> dict:
    if CORPUS_DIR.exists():
        print(f"[edit-improve] removing the previous corpus: {CORPUS_DIR}", flush=True)
        shutil.rmtree(CORPUS_DIR, ignore_errors=True)

    # Read by edit_pairs.render_context for BOTH the corpus and the test
    # prompts, so set it before the corpus is built or the two diverge.
    os.environ["NNGPT_CONTEXT_EXAMPLES"] = str(CONTEXT_EXAMPLES)
    edit_pairs.CONTEXT_EXAMPLES = CONTEXT_EXAMPLES
    # Also read by build(), so it has to be set before the call below and not
    # with the rest of the environment further down. This run's own
    # generations are training data, never test baselines: a benchmark made of
    # your own output measures editing your own output, and it changes every
    # cycle, which is the opposite of a frozen test set.
    os.environ.setdefault("NNGPT_HOLDOUT_EXCLUDE_PREFIXES", NN_NAME_PREFIX)

    stats = edit_pairs.build(
        out_dir=str(CORPUS_DIR), task=TASK, dataset=DATASET, metric=METRIC,
        epoch=EPOCH, nn_prefixes=NN_PREFIXES,
        min_observed_gain=MIN_OBSERVED_GAIN,
        max_changed_lines=MAX_CHANGED_LINES, bands=BANDS,
        min_holdout=TEST_NN, holdout_min_per_family=HOLDOUT_MIN_PER_FAMILY,
        pinned_seed_file=str(SEED_FILE), offline_root=offline_root)
    print(f"[edit-improve] corpus: {stats}")

    emit_test_config(str(Path(conf_test_dir) / TEST_CONF))
    emit_pinned_seeds(str(CORPUS_DIR), str(SEED_FILE), n=TEST_NN)
    edit_pairs.emit_seed_context(
        str(SEED_FILE), str(CORPUS_DIR / "seed_context.json"),
        task=TASK, dataset=DATASET, metric=METRIC, epoch=EPOCH,
        band=BANDS[0], nn_prefixes=NN_PREFIXES)
    pinned = json.loads(Path(SEED_FILE).read_text(encoding="utf-8"))["nn"]
    if len(pinned) != TEST_NN:
        raise RuntimeError(f"seed file holds {len(pinned)}, expected {TEST_NN}")

    os.environ["NNGPT_EDIT_MODE"] = "1"
    os.environ["NNGPT_SEED_FILE"] = str(SEED_FILE)
    os.environ["NNGPT_PIN_SEEDS"] = "1"
    # ab/gpt/act/tune/Tune.py resets the pinned list on a non-resume start,
    # which would delete the held-out test set written above and redraw it over
    # the whole corpus -- measuring the run on models it was trained on.
    os.environ["NNGPT_SEED_FILE_EXTERNAL"] = "1"
    os.environ["NNGPT_SEED_POOL_FACTOR"] = "1"
    # Was "256". The floor is what drops a prompt that leaves no room for an
    # answer; set below the length of an answer it makes the prompt fit and the
    # reply truncate instead, which is strictly worse -- the cycle pays the full
    # generation cost for a reply extraction cannot use. CONTEXT_EXAMPLES above
    # is the dial that makes prompts shorter without taking answer room away.
    os.environ["NNGPT_MIN_ANSWER_TOKENS"] = str(MIN_ANSWER_TOKENS)
    os.environ["NNGPT_DESIGN_CEILING"] = str(stats["observed_max"])
    os.environ["NNGPT_DESIGN_CEILING_SCRATCH"] = str(
        stats.get("observed_max_scratch", stats["observed_max"]))
    os.environ["NNGPT_DESIGN_CEILING_PRETRAINED"] = str(
        stats.get("observed_max_pretrained", stats["observed_max"]))
    os.environ["NNGPT_EDIT_KNOWN_UIDS"] = str(CORPUS_DIR / "known_uids.json")
    os.environ["NNGPT_EDIT_SEED_CONTEXT"] = str(CORPUS_DIR / "seed_context.json")
    os.environ["NNGPT_PARK_TRAINER_FOR_EVAL"] = "1"
    os.environ["NNGPT_TRAIN_ON_COMPLETIONS_ONLY"] = "1"
    os.environ["NNGPT_VALIDITY_GATE"] = "1"
    os.environ["NNGPT_VALIDITY_KEEP"] = str(TEST_NN)
    # The build check belongs to the evaluation worker, which constructs the
    # model on the real device, with the real dataset and the real prm, moments
    # before training it. Building it here as well meant a second, approximate
    # build per candidate in a separate CPU process -- and a third during
    # generation, where the engine is resident and the trainer is parked in
    # RAM. That third one is what the kernel killed the run over.
    os.environ["NNGPT_VALIDITY_SMOKE"] = "1"
    # Keep the code of rejected generations. The gate otherwise deletes it and
    # keeps only the reason, and a reason with no code cannot become a training
    # prompt -- which is what the "will_it_run" auxiliary task needs. These are
    # the model's own mistakes, and nothing in the corpus has ever told it
    # which of its outputs fail to build.
    os.environ.setdefault("NNGPT_VALIDITY_KEEP_REJECTED", "1")
    os.environ.setdefault("NNGPT_SMOKE_BUDGET_SECONDS", "900")

    # The <hp> block the system prompt offers is only honoured in "generated"
    # mode. The default, "seed", does prm.update(baseline_prm) AFTER reading
    # hp.txt, so the baseline's own values win and the model cannot change the
    # learning rate at all: all 60 evaluated models of one run trained at
    # lr=0.01, momentum=0.9, dropout=0.2 -- the values of the family that
    # supplied nearly every baseline -- and two diverged to 0.13 there.
    os.environ.setdefault("NNGPT_EVAL_PRM_MODE", "generated")
    # Make a generated <tr> module resolvable by name so evaluation can train
    # with it. The transform slot was read 0 times in 210 generations, so data
    # augmentation -- the cheapest accuracy there is on a one-epoch CIFAR
    # budget -- was never searched. Writes the module into the installed
    # ab/nn/transform package; if that is not writable the run says so once and
    # keeps the baseline pipeline.
    os.environ.setdefault("NNGPT_INSTALL_GENERATED_TRANSFORMS", "1")

    # Draw each cycle's rows spread over families and weighted by the measured
    # gain, instead of uniformly over rows. No pair is removed from the corpus;
    # only the mix of what is drawn from it changes, so the trainer still sees
    # 2048 rows and cannot overfit a narrower set.
    os.environ.setdefault("NNGPT_SFT_MAX_FAMILY_SHARE", "0.25")
    os.environ.setdefault("NNGPT_SFT_GAIN_SCALE", "12")

    # Re-measure the best few candidates before reporting the cycle's result.
    os.environ["NNGPT_STAGE2_TOPK"] = str(STAGE2_TOPK)
    os.environ["NNGPT_STAGE2_REPEATS"] = str(STAGE2_REPEATS)
    # Same epoch count as stage 1 on purpose: a repeat at a different epoch
    # writes a LEMUR row the corpus builder (which queries epoch=EPOCH) will
    # never see, so it would inform the report and not the next cycle.
    os.environ["NNGPT_STAGE2_EPOCHS"] = str(EPOCH)
    if STAGE2_TOPK > 0:
        # Screening results stay out of LEMUR; the stage-2 pass turns this
        # back on for itself. A pair built on a single one-epoch run selects on
        # noise as often as on architecture -- the same model measured 0.9562
        # and 0.9621 -- and that noise was going straight into the next
        # cycle's training corpus.
        os.environ.setdefault("NNGPT_EVAL_SAVE_TO_DB", "0")

    # Show the best result the run has produced so far in every prompt, and
    # keep an archive of the champions. The objective is the maximum, and
    # nothing used to tell the model what the maximum currently was: the run's
    # best, 0.9708, was reached in cycle 0 and never beaten.
    sweep_legacy_transforms()
    os.environ.setdefault("NNGPT_ELITE_IN_PROMPT", "1")
    # One more turn, with the traceback, for a candidate that does not build.
    # It earns its place: measured over six cycles it rewrote 12 to 16 of every
    # 15 to 22 broken candidates, and the one cycle that ran without it trained
    # 3 of 21 instead of the usual 10 or 11. Turning it off to save memory cost
    # far more than it saved.
    #
    # What made it dangerous was not the turn itself but what it used to do: a
    # full build, forward and backward at the record's own resolution, in a
    # separate process, while the engine was resident and the trainer parked in
    # RAM. It now asks only whether the file builds -- construct, train_setup,
    # one forward without gradients at batch 1, measured at 0.95 GiB against
    # 2.05 for the full rehearsal -- under a ceiling set below the host's real
    # headroom. The validity gate no longer builds at all, so this is the only
    # build before training, where there used to be three.
    os.environ.setdefault("NNGPT_REPAIR_TURN", "1")
    # Reject a candidate that merely restates one already accepted this cycle.
    os.environ.setdefault("NNGPT_NEAR_DUPLICATE_JACCARD", "0.95")
    # Retry an out-of-memory evaluation once at half the batch size.
    os.environ.setdefault("NNGPT_OOM_RETRY", "1")
    # Best-of-N is OFF: it multiplies the evaluation budget, not just
    # generation. Set NNGPT_BEST_OF_N=2 and raise NNGPT_VALIDITY_KEEP with it
    # if there is time for twice as many trainings per cycle.
    os.environ.setdefault("NNGPT_BEST_OF_N", "1")
    # Force the reply to open with a tag. A reasoning model expands to fill
    # whatever budget it is given, and prose before the first block is the one
    # thing that makes a reply unusable. A single "<" is deliberately weaker
    # than prefilling "<edit>\n<find>\n": that would decide the format for the
    # model, and the whole point of training both forms is that it picks the
    # shorter one. Set this to the stronger value to force edits only.
    os.environ["NNGPT_ASSISTANT_PREFILL"] = "<"

    rebuild = dict(out_dir=str(CORPUS_DIR), task=TASK, dataset=DATASET,
                   metric=METRIC, epoch=EPOCH,
                   nn_prefixes=list(NN_PREFIXES) if NN_PREFIXES else None,
                   min_observed_gain=MIN_OBSERVED_GAIN,
                   max_changed_lines=MAX_CHANGED_LINES, bands=list(BANDS),
                   min_holdout=TEST_NN,
                   holdout_min_per_family=HOLDOUT_MIN_PER_FAMILY,
                   pinned_seed_file=str(SEED_FILE))
    rebuild_path = CORPUS_DIR / "corpus_build_args.json"
    rebuild_path.write_text(json.dumps(rebuild, indent=2), encoding="utf-8")
    os.environ["NNGPT_EDIT_REBUILD"] = str(rebuild_path)
    os.environ.setdefault("NNGPT_VLLM_MAX_MODEL_LEN", f"{CONTEXT_LENGTH // 1024}*1024")
    return stats


def main():
    prepare(offline_root=None)
    TuneNNGen.main(
        llm_conf='qwen3.5_9b.json',
        nn_gen_conf=TEST_CONF,
        nn_gen_conf_id=CONF_KEY,
        data_dir=str(CORPUS_DIR),
        test_nn=TEST_NN,
        skip_epoches=-1,
        # Evaluated results are written back under this prefix, and NN_PREFIXES
        # above must accept it or the rebuild cannot see them. It did not: three
        # cycles wrote 19 models as llm-* while the query asked for
        # GenFractalNet-* only, so records stayed at 3,909 and every cycle
        # retrained on the same 2,141 rows. Keeping a distinct prefix also keeps
        # this run's own output telling apart from the seed corpus.
        nn_name_prefix=NN_NAME_PREFIX,
        cycles=16,
        num_train_epochs=1,
        context_length=CONTEXT_LENGTH,
        sft_max_length=SFT_MAX_LENGTH,
        # An edit is short, but the baseline in the prompt is not, and at cycle
        # 0 the model has never seen the format and reasons first.
        max_new_tokens=MAX_NEW_TOKENS,
        # Rows per cycle, now sampled rather than taken from the head. One
        # measured cycle spent 42 minutes generating and training networks
        # against 8 minutes fine-tuning; the corpus is the cheap half and was
        # being starved. Raising this rebalances without touching the
        # expensive half.
        # Rows per cycle. 2048 at 8-step accumulation is 231 optimiser steps,
        # which was thin for one task and is far too thin for ten: the
        # auxiliary tasks need enough gradient to register at all. Doubling
        # this doubles the steps and costs only fine-tuning time, which is the
        # cheap half of the cycle.
        max_prompts=2048,
        only_best_accuracy=False,
        # Was 1e-5, which is a full-finetune rate, not a LoRA one. Measured
        # consequence: training loss sat at 0.0501 and flat from step 96 to
        # step 231 with a gradient norm of 0.21 -- the adapter barely moved,
        # and no cycle ever beat the 0.9708 the base model reached before any
        # fine-tuning. Standard for r=32 is 1e-4 to 3e-4; this is the bottom of
        # that range, because the risk of going higher is the adapter damaging
        # the base model's ability to write code at all. Watch valid_rate: if
        # it falls, this is why.
        learning_rate=1e-4,
        # Cosine decayed to 7.5e-7 by step 192 of 231, so the second half of
        # every cycle was a no-op that still cost 76 minutes. A run that
        # fine-tunes 16 times wants a rate that stays up within each cycle.
        lr_scheduler='constant_with_warmup',
        # Five steps of warmup made sense at 231 steps; at ~460 it is a
        # rounding error, and a constant schedule needs a real ramp to avoid
        # a large first update.
        warmup_steps=20,
        lora_alpha=64,
        suppress_thinking=True,
        use_unsloth=True,
        use_vllm=True,
    )


if __name__ == "__main__":
    main()
