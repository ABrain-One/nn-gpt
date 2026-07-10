"""
Configuration builder for the edge k4 generation pipeline.

Constructs SFTConfig/LoraConfig with edge-pipeline settings and delegates to the
shared curriculum tuner (ab.gpt.util.Tune_Curriculum.tune), which is used as-is.

Differences from the CurriculumGen_L3_k4 configuration this replaces:
- SFTConfig.max_length is configurable (sft_max_length) so k=4 training prompts
  (~10-14k tokens) are not truncated to trl's 1024-token default
- packing_strategy='wrapped' instead of trl's default 'bfd': BFD packing forces
  padding-free mode, which rejects the custom data collator used during
  fine-tuning (trl>=0.24 raises ValueError); wrapped packing keeps the collator
- num_cycles (outer generate/evaluate/finetune epochs) is explicit
"""

import json
from typing import Literal

import torch

from ab.gpt.util.Const import conf_llm_dir, nngpt_dir, NN_TRAIN_EPOCHS
from ab.nn.util.Const import out_dir

# --- LoRA defaults (aligned with the lab's curriculum experiments) ---
START_LAYER = 0
END_LAYER = 24
R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj')
TASK_TYPE = 'CAUSAL_LM'
BiasType = Literal['none', 'all', 'lora_only']
BIAS: BiasType = 'none'

# --- Training defaults (aligned with curriculum standalone mode) ---
NUM_TRAIN_EPOCHS = 1
LR_SCHEDULER = 'cosine'
LEARNING_RATE = 1e-6
MAX_GRAD_NORM = 1.0
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_RATIO = 0.05
LOGGING_STEPS = 96
OPTIMIZER = 'paged_adamw_8bit'

# --- Edge pipeline defaults ---
NUM_CYCLES = 10          # outer generate/evaluate/finetune epochs
SFT_MAX_LENGTH = 16 * 1024
MAX_NEW_TOKENS = 16 * 1024
MAX_PROMPTS = 4 * 1024
TEMPERATURE = 1.0
TOP_K = 50
TOP_P = 0.9

RUN_META = out_dir / 'nngpt' / 'run_config.json'


def persist_run_config(llm_conf: str, enable_merge: bool = False) -> None:
    """Record the active LLM config so downstream tools (merge) can find it."""
    RUN_META.parent.mkdir(parents=True, exist_ok=True)
    run_config = {'llm_conf': llm_conf, 'enable_merge': enable_merge}
    llm_conf_path = conf_llm_dir / llm_conf
    if llm_conf_path.exists():
        try:
            with open(llm_conf_path) as f:
                run_config['base_model_name'] = json.load(f).get('base_model_name')
        except Exception as e:
            print(f'Failed to read base_model_name from {llm_conf}: {e}')
    with open(RUN_META, 'w') as f:
        json.dump(run_config, f, indent=2)
    print(f'Run config saved: {RUN_META}')


def _dtype_flags() -> dict:
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return {'bf16': bf16_ok, 'fp16': not bf16_ok}


def _force_flash_attention() -> bool:
    """Route model loading to FlashAttention-2 when available.

    The shared loader (ab/gpt/util/LLM.py) does not expose an attention
    implementation setting, and transformers defaults to SDPA. On the k=4
    prompts (~12k tokens) SDPA falls off its fused kernels and materializes
    the full attention matrix (~16 GB), which OOMs on 24 GB GPUs. Wrapping
    from_pretrained at runtime injects the setting for this process only,
    without modifying any shared file.
    """
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        print('[EDGE] flash_attn not installed — long-prompt generation may OOM on <40GB GPUs '
              '(install per req-no-isolation.txt)')
        return False

    # transformers verifies flash_attn via package METADATA, which some images
    # lack even though the module imports. Synthesize a minimal dist-info next
    # to the working directory (on sys.path when running via python -m).
    import importlib.metadata as _md
    try:
        _md.version('flash_attn')
    except _md.PackageNotFoundError:
        import os as _os
        version = getattr(flash_attn, '__version__', '2.8.3')
        dist_dir = _os.path.join(_os.getcwd(), f'flash_attn-{version}.dist-info')
        try:
            _os.makedirs(dist_dir, exist_ok=True)
            with open(_os.path.join(dist_dir, 'METADATA'), 'w') as f:
                f.write(f'Metadata-Version: 2.1\nName: flash_attn\nVersion: {version}\n')
            _md.version('flash_attn')  # re-check
            print(f'[EDGE] Synthesized flash_attn dist-info (v{version}) — image lacked package metadata')
        except Exception as e:
            print(f'[EDGE] flash_attn metadata unavailable ({e}) — keeping default attention '
                  '(needs a high-memory GPU for 12k-token prompts)')
            return False

    # transformers freezes its availability flag when first imported — this
    # function must therefore run before anything imports transformers.
    from transformers.utils import is_flash_attn_2_available
    if not is_flash_attn_2_available():
        # Some cluster images ship a stub flash_attn AND hard-disable this check
        # (is_flash_attn_2_available() { return False }). A real installation can
        # be provided via PYTHONPATH (see k8s manifests: fa2_real + cu13rt mounts).
        # Trust an override only after the real compiled kernel actually loads —
        # the stub has no flash_attn_2_cuda, and this import also validates that
        # libcudart is resolvable via LD_LIBRARY_PATH.
        try:
            import flash_attn_2_cuda  # noqa: F401
        except ImportError:
            print('[EDGE] transformers reports FlashAttention-2 unavailable and no compiled '
                  'kernel is loadable — keeping default attention (needs a high-memory GPU '
                  'for 12k-token prompts)')
            return False
        import transformers.utils as _tu
        import transformers.utils.import_utils as _iu
        _iu.is_flash_attn_2_available = lambda: True
        _tu.is_flash_attn_2_available = lambda: True
        print('[EDGE] Compiled FlashAttention-2 kernel verified — overriding the image\'s '
              'disabled availability check')
    from transformers import AutoModelForCausalLM
    original = AutoModelForCausalLM.from_pretrained.__func__

    def _with_flash_attention(cls, *args, **kwargs):
        kwargs.setdefault('attn_implementation', 'flash_attention_2')
        return original(cls, *args, **kwargs)

    AutoModelForCausalLM.from_pretrained = classmethod(_with_flash_attention)
    print('[EDGE] Model loading configured for FlashAttention-2')
    return True


def _pre_eval_novelty_filter(models_dir) -> None:
    """Reject near-duplicate generations before evaluation.

    Renames new_nn.py to new_nn.py.rejected for models whose MinHash Jaccard
    similarity is >= 0.8 vs any in-prompt reference (copying) or >= 0.95 vs a
    sibling generation (duplicate), so NNEval never spends GPU time training
    them. Thresholds and helpers are shared with EdgeScore.
    """
    import pandas as pd
    from ab.gpt.edge.EdgeScore import (
        code_minhash, max_similarity, REFERENCE_KEYS,
        SIMILARITY_THRESHOLD_DEFAULT, SIBLING_SIMILARITY_THRESHOLD,
    )

    ref_hashes: dict = {}
    siblings: list = []
    rejected = 0
    for b_dir in sorted(models_dir.glob('B*')):
        nn_file = b_dir / 'new_nn.py'
        if not nn_file.is_file():
            continue
        code = nn_file.read_text(encoding='utf-8', errors='replace')

        df_path = b_dir / 'dataframe.df'
        if df_path.is_file():
            try:
                row = pd.read_pickle(df_path)
                for key in REFERENCE_KEYS:
                    ref = row.get(key) if hasattr(row, 'get') else None
                    if isinstance(ref, str) and ref.strip() and hash(ref) not in ref_hashes:
                        ref_hashes[hash(ref)] = (f'reference:{key}', code_minhash(ref))
            except Exception:
                pass

        mh = code_minhash(code)
        ref_sim, ref_to = max_similarity(mh, list(ref_hashes.values()))
        sib_sim, sib_to = max_similarity(mh, siblings)

        reject_reason = None
        if ref_sim is not None and ref_sim >= SIMILARITY_THRESHOLD_DEFAULT:
            reject_reason = f'copy of {ref_to} (sim={ref_sim})'
        elif sib_sim is not None and sib_sim >= SIBLING_SIMILARITY_THRESHOLD:
            reject_reason = f'duplicate of {sib_to} (sim={sib_sim})'

        if reject_reason:
            nn_file.rename(b_dir / 'new_nn.py.rejected')
            (b_dir / 'rejection_reason.txt').write_text(reject_reason)
            rejected += 1
            print(f'[EDGE] Pre-eval reject {b_dir.name}: {reject_reason}')
        else:
            siblings.append((b_dir.name, mh))

    if rejected:
        print(f'[EDGE] Pre-eval novelty filter: rejected {rejected} model(s) before evaluation')


def _install_trl_compat() -> None:
    """Compatibility layer for older trl (image ships 0.11.x).

    The shared trainer (ab/gpt/util/LoRA.py) targets modern trl:
      (a) `trl.trainer.sft_trainer.DataCollatorForLanguageModeling` with
          kwargs (pad_token_id, completion_only_loss, ...) — in old trl that
          name is transformers' collator with a different signature;
      (b) `SFTTrainer(processing_class=...)` — old trl uses `tokenizer=`.
    Inject adapted classes before LoRA.py binds these names. No-ops on
    modern trl. Must run before ab.gpt.util.Tune_Curriculum is imported.
    """
    import inspect
    import trl
    import trl.trainer.sft_trainer as st

    collator = getattr(st, 'DataCollatorForLanguageModeling', None)
    needs_collator = True
    if collator is not None:
        try:
            needs_collator = 'pad_token_id' not in inspect.signature(collator.__init__).parameters
        except (ValueError, TypeError):
            pass
    if needs_collator:
        import torch

        class _CompatLMCollator:
            """Right-padding causal-LM collator with the modern-trl signature."""

            def __init__(self, pad_token_id=0, completion_only_loss=True,
                         pad_to_multiple_of=8, return_tensors='pt', **_):
                self.pad_token_id = pad_token_id
                self.pad_to_multiple_of = pad_to_multiple_of

            def __call__(self, examples):
                ids = [torch.as_tensor(e['input_ids'], dtype=torch.long) for e in examples]
                max_len = max(len(x) for x in ids)
                if self.pad_to_multiple_of:
                    m = self.pad_to_multiple_of
                    max_len = ((max_len + m - 1) // m) * m
                batch = len(ids)
                input_ids = torch.full((batch, max_len), self.pad_token_id, dtype=torch.long)
                attention = torch.zeros((batch, max_len), dtype=torch.long)
                labels = torch.full((batch, max_len), -100, dtype=torch.long)
                for i, x in enumerate(ids):
                    input_ids[i, :len(x)] = x
                    attention[i, :len(x)] = 1
                    labels[i, :len(x)] = x
                return {'input_ids': input_ids, 'attention_mask': attention, 'labels': labels}

        st.DataCollatorForLanguageModeling = _CompatLMCollator
        print('[EDGE] trl compat: injected padding LM collator (old trl signature mismatch)')

    trainer_params = inspect.signature(st.SFTTrainer.__init__).parameters
    if 'processing_class' not in trainer_params:
        original_trainer = st.SFTTrainer
        accepted = set(trainer_params)

        class _CompatSFTTrainer(original_trainer):
            def __init__(self, *args, **kwargs):
                if 'processing_class' in kwargs:
                    kwargs['tokenizer'] = kwargs.pop('processing_class')
                dropped = [k for k in list(kwargs) if k not in accepted]
                for k in dropped:
                    kwargs.pop(k)
                if dropped:
                    print(f'[EDGE] trl compat: SFTTrainer dropped kwargs {sorted(dropped)}')
                super().__init__(*args, **kwargs)

        st.SFTTrainer = _CompatSFTTrainer
        trl.SFTTrainer = _CompatSFTTrainer
        print('[EDGE] trl compat: SFTTrainer processing_class→tokenizer adapter installed')


def _shim_nneval_kwargs() -> None:
    """Route NNEval.main through a wrapper that (a) drops keyword arguments the
    installed signature does not accept (version drift: the shared tuner passes
    use_sequential=True), and (b) runs the pre-eval novelty filter so duplicate
    generations never consume GPU evaluation time.
    """
    import inspect
    import ab.gpt.NNEval as NNEval
    from ab.gpt.util.Const import epoch_dir, synth_dir

    original = NNEval.main
    params = inspect.signature(original).parameters
    has_var_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    accepted = set(params)

    def _edge_main(*args, **kwargs):
        if not has_var_kwargs:
            dropped = [k for k in list(kwargs) if k not in accepted]
            for k in dropped:
                kwargs.pop(k)
            if dropped:
                print(f'[EDGE] NNEval.main: dropped unsupported kwargs {sorted(dropped)}')
        try:
            only_epoch = kwargs.get('only_epoch', args[2] if len(args) > 2 else None)
            if only_epoch is not None:
                _pre_eval_novelty_filter(synth_dir(epoch_dir(only_epoch)))
        except Exception as e:
            print(f'[EDGE] Pre-eval novelty filter skipped: {e}')
        return original(*args, **kwargs)

    NNEval.main = _edge_main


# Similarity bands (Jaccard vs anchor), matching the lab's curriculum levels.
_SIM_BANDS = {
    'high': (0.95, 1.0000001),
    'medium': (0.85, 0.95),
    'low': (0.60, 0.85),
    'very_low': (0.0, 0.60),
    'very_low_near': (0.30, 0.85),
}


def _install_reference_filter(max_params: int, min_acc: float,
                              prefixes: tuple = ()) -> None:
    """Constrain the k-reference pool to edge-appropriate models
    (params <= max_params AND accuracy >= min_acc), as agreed with the advisor.

    The stock anchor-band query ranks the whole DB by accuracy before slicing,
    so <=6M models (best 0.939 on cifar-10) never surface among the 0.95+
    heavyweights — post-filtering its output yields zero rows, and JoinConf has
    no params/accuracy fields to push the filter into SQL. This wrapper
    intercepts anchor-band reference queries only and rebuilds the result:
    plain best-accuracy query -> params+accuracy filter -> anchor = best
    filtered model -> Jaccard vs anchor from the stored nn_minhash signatures
    -> band selection. Returned frame matches the tall-mode shape the
    curriculum prompt builder expects (anchor_nn, anchor_jaccard, nn_code,
    prm, transform_code, ...). All other lemur.data calls pass through.
    """
    if not (max_params or min_acc):
        return
    import numpy as np
    import sqlite3
    import ab.nn.api as lemur
    from ab.nn.util.Const import db_file

    original = lemur.data
    aux: dict = {}

    def _load_aux():
        con = sqlite3.connect(str(db_file))
        aux['params'] = dict(con.execute(
            'SELECT nn_name, MIN(total_params) FROM nn_stat GROUP BY nn_name'))
        aux['sigs'] = {nn: np.frombuffer(hv, dtype=np.uint32)
                       for nn, hv in con.execute('SELECT nn, hashvalues FROM nn_minhash')}
        con.close()

    def _filtered_data(*args, **kwargs):
        sql = kwargs.get('sql')
        if getattr(sql, 'similarity_mode', None) != 'anchor_band_db_minhash':
            return original(*args, **kwargs)

        if not aux:
            _load_aux()
        k = int(getattr(sql, 'num_joint_nns', 2) or 2)
        band = getattr(sql, 'similarity_band', None) or 'very_low_near'
        band_min, band_max = _SIM_BANDS.get(band, (0.0, 1.0000001))
        max_rows = kwargs.get('max_rows') or 400

        pool = original(
            only_best_accuracy=True,
            task=kwargs.get('task'),
            dataset=kwargs.get('dataset'),
            metric=kwargs.get('metric'),
            nn_prefixes=tuple(prefixes) or kwargs.get('nn_prefixes') or (),
            max_rows=100000,
        )
        pool = pool[pool['nn'].isin(aux['sigs'])].copy()
        pool['params'] = pool['nn'].map(aux['params'])
        if max_params:
            pool = pool[pool['params'].notna() & (pool['params'] <= max_params)]
        if min_acc:
            pool = pool[pool['accuracy'] >= min_acc]
        if len(pool) <= k:
            raise ValueError(
                f'[EDGE] Reference filter left only {len(pool)} models '
                f'(params<={max_params}, acc>={min_acc}) — relax the thresholds '
                f'(EDGE_REF_MAX_PARAMS / EDGE_REF_MIN_ACC).')

        # one row per model (best accuracy variant)
        pool = (pool.sort_values('accuracy', ascending=False)
                    .drop_duplicates('nn').reset_index(drop=True))
        anchor = pool.iloc[0]['nn']
        anchor_sig = aux['sigs'][anchor]
        pool['anchor_jaccard'] = pool['nn'].map(
            lambda n: float((aux['sigs'][n] == anchor_sig).mean()))

        # Band selection is RANK-based within the filtered pool, because the
        # absolute Jaccard bands were calibrated on the unfiltered DB: inside
        # the edge pool the similarity distribution is bimodal (one family at
        # >0.85, everything else <0.2, nothing between), so absolute ranges
        # come up empty. Rank semantics preserve the curriculum meaning:
        #   high   -> most similar to the anchor (easy imitation)
        #   medium -> middle of the similarity ranking
        #   low*   -> most dissimilar (hard synthesis)
        members = pool[pool['nn'] != anchor].copy()
        if band == 'high':
            out = members.sort_values(['anchor_jaccard', 'accuracy'],
                                      ascending=[False, False])
        elif band in ('low', 'very_low', 'very_low_near'):
            out = members.sort_values(['anchor_jaccard', 'accuracy'],
                                      ascending=[True, False])
        else:  # medium: closest to the pool's median similarity
            median_j = members['anchor_jaccard'].median()
            out = members.assign(_d=(members['anchor_jaccard'] - median_j).abs()) \
                         .sort_values(['_d', 'accuracy'], ascending=[True, False]) \
                         .drop(columns=['_d'])
        out = out.head(max_rows).copy()
        out['anchor_nn'] = anchor
        print(f"[EDGE] Reference filter: pool={len(pool)} band[{band}] rows={len(out)} "
              f"jaccard=[{out['anchor_jaccard'].min():.3f}, {out['anchor_jaccard'].max():.3f}] "
              f"anchor={anchor} (params<={max_params}, acc>={min_acc})")
        return out.reset_index(drop=True)

    _filtered_data.cache_clear = getattr(original, 'cache_clear', lambda: None)
    lemur.data = _filtered_data


def _write_run_manifest(config: dict) -> None:
    """Persist the full run configuration + environment for reproducibility
    (report/preprint appendix). One timestamped file per run in out/edge/runs/."""
    import subprocess
    from datetime import datetime

    manifest = {'timestamp': datetime.now().isoformat(), 'config': config}
    try:
        manifest['git_commit'] = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10
        ).stdout.strip()
        manifest['git_branch'] = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        pass
    try:
        import transformers, trl, peft as peft_lib
        manifest['versions'] = {
            'torch': torch.__version__,
            'transformers': transformers.__version__,
            'trl': trl.__version__,
            'peft': peft_lib.__version__,
        }
        if torch.cuda.is_available():
            manifest['gpu'] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    runs_dir = out_dir / 'edge' / 'runs'
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    manifest_file = runs_dir / f'run_{stamp}.json'
    manifest_file.write_text(json.dumps(manifest, indent=2))
    print(f'[EDGE] Run manifest saved: {manifest_file}')


def main(llm_conf: str = 'ds_coder_7b_olympic.json',
         llm_tune_conf: str = 'edge/curriculum_k4_train.json',
         nn_gen_conf: str = 'edge/curriculum_k4.json',
         nn_gen_conf_id: str = 'curriculum_edge_k4',
         test_nn: int = 5,
         skip_epoches: int = 0,
         nn_name_prefix: str = 'edge',
         num_cycles: int = NUM_CYCLES,
         nn_train_epochs: int = NN_TRAIN_EPOCHS,
         context_length: int = 16384,
         sft_max_length: int = SFT_MAX_LENGTH,
         max_new_tokens: int = MAX_NEW_TOKENS,
         max_prompts: int = MAX_PROMPTS,
         temperature: float = TEMPERATURE,
         top_k: int = TOP_K,
         top_p: float = TOP_P,
         num_train_epochs: int = NUM_TRAIN_EPOCHS,
         learning_rate: float = LEARNING_RATE,
         per_device_train_batch_size: int = PER_DEVICE_TRAIN_BATCH_SIZE,
         gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS,
         r: int = R,
         lora_alpha: float = LORA_ALPHA,
         lora_dropout: float = LORA_DROPOUT,
         target_modules: tuple = TARGET_MODULES,
         tune_layers=range(START_LAYER, END_LAYER),
         peft: str = None,
         enable_merge: bool = False,
         prompt_batch: int = 1,
         ref_max_params: int = 6_000_000,
         ref_min_acc: float = 0.85,
         ref_prefixes: tuple = ()) -> None:

    # Must run before anything imports transformers (see _force_flash_attention).
    _force_flash_attention()
    # Must run before ab.gpt.util.Tune_Curriculum (and thus LoRA.py) is imported.
    _install_trl_compat()

    persist_run_config(llm_conf, enable_merge)
    _write_run_manifest(dict(
        llm_conf=llm_conf, llm_tune_conf=llm_tune_conf, nn_gen_conf=nn_gen_conf,
        nn_gen_conf_id=nn_gen_conf_id, test_nn=test_nn, skip_epoches=skip_epoches,
        nn_name_prefix=nn_name_prefix, num_cycles=num_cycles, nn_train_epochs=nn_train_epochs,
        context_length=context_length, sft_max_length=sft_max_length,
        max_new_tokens=max_new_tokens, max_prompts=max_prompts,
        temperature=temperature, top_k=top_k, top_p=top_p,
        num_train_epochs=num_train_epochs, learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=list(target_modules), tune_layers=list(tune_layers),
        peft=peft, enable_merge=enable_merge, prompt_batch=prompt_batch,
    ))

    _shim_nneval_kwargs()
    _install_reference_filter(ref_max_params, ref_min_acc, ref_prefixes)

    from peft import LoraConfig
    from trl import SFTConfig
    from ab.gpt.util.Tune_Curriculum import tune

    import dataclasses
    sft_fields = {f.name for f in dataclasses.fields(SFTConfig)}
    sft_kwargs = dict(
        num_train_epochs=num_train_epochs,
        lr_scheduler_type=LR_SCHEDULER,
        max_grad_norm=MAX_GRAD_NORM,
        report_to=[],
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=WARMUP_RATIO,
        learning_rate=learning_rate,
        logging_steps=LOGGING_STEPS,
        output_dir=str(nngpt_dir / 'outputs'),
        optim=OPTIMIZER,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        max_length=sft_max_length,
        packing_strategy='wrapped',
        **_dtype_flags(),
    )
    # trl version adaptation: older releases name the length field
    # 'max_seq_length' and have no 'packing_strategy' (their packing does not
    # force padding-free mode, so dropping it is safe there).
    if 'max_length' not in sft_fields and 'max_seq_length' in sft_fields:
        sft_kwargs['max_seq_length'] = sft_kwargs.pop('max_length')
    dropped = [k for k in list(sft_kwargs) if k not in sft_fields]
    for key in dropped:
        sft_kwargs.pop(key)
    if dropped:
        print(f'[EDGE] SFTConfig ({SFTConfig.__module__}): dropped unsupported fields {sorted(dropped)}')

    training_args = SFTConfig(**sft_kwargs)
    # The shared trainer reads 'max_length' via getattr regardless of trl
    # version — pin it so fine-tuning never falls back to its 4096 default.
    setattr(training_args, 'max_length', sft_max_length)

    peft_config = LoraConfig(
        r=int(r),
        lora_alpha=int(lora_alpha),
        target_modules=target_modules,
        layers_to_transform=list(tune_layers),
        lora_dropout=float(lora_dropout),
        bias=BIAS,
        task_type=TASK_TYPE,
    )

    print(f'''[EDGE] Pipeline configuration:
llm_conf={llm_conf}, llm_tune_conf={llm_tune_conf}, nn_gen_conf={nn_gen_conf}, nn_gen_conf_id={nn_gen_conf_id},
test_nn={test_nn}, num_cycles={num_cycles}, nn_train_epochs={nn_train_epochs}, skip_epoches={skip_epoches},
context_length={context_length}, sft_max_length={sft_max_length}, max_new_tokens={max_new_tokens},
temperature={temperature}, top_k={top_k}, top_p={top_p}, lr={learning_rate},
lora: r={r}, alpha={lora_alpha}, dropout={lora_dropout}, layers={list(tune_layers)},
packing_strategy=wrapped (avoids trl padding-free/custom-collator conflict)''')

    tune(
        test_nn,
        nn_train_epochs,
        skip_epoches,
        peft,
        llm_tune_conf,
        nn_gen_conf,
        nn_gen_conf_id,
        llm_conf,
        training_args,
        peft_config,
        max_prompts=max_prompts,
        save_llm_output=True,
        max_new_tokens=max_new_tokens,
        nn_name_prefix=nn_name_prefix,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        prompt_batch=prompt_batch,
        use_agents=False,
        enable_merge=enable_merge,
        context_length=context_length,
        num_cycles=num_cycles,
    )

    print('\n[EDGE] Run complete. Score the generated models before relaunching:')
    print('  python -m ab.gpt.edge.EdgeScore')


if __name__ == '__main__':
    main()
