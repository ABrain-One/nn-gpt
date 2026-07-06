"""
ab/gpt/util/KTO.py - KTO (Kahneman-Tversky Optimization) trainer wrapper.

Analog of ab/gpt/util/LoRA.py for KTO training. Wraps TRL's KTOTrainer so
the iterative pipeline can swap SFT for KTO with the same LoRA-attachment
and checkpoint/resume semantics as the existing SFT path.

KTO differs from SFT in three key ways:
  1. Dataset format: {prompt: str, completion: str, label: bool}
     instead of {text: str}.  label=True means "desirable" example,
     label=False means "undesirable" example.
  2. Loss function: KL-regularised Kahneman-Tversky value function with
     separate weights for the two classes (desirable_weight,
     undesirable_weight).
  3. Reference model: KTO computes per-example log-probs against a frozen
     reference.  When using PEFT, TRL automatically uses the base model
     (with adapters disabled) as the reference, so no separate ref_model
     instance is needed.

Reference: https://huggingface.co/papers/2402.01306 (KTO paper)
           https://huggingface.co/docs/trl/main/en/kto_trainer (TRL docs)
"""

from os import makedirs
import inspect
from pathlib import Path
from typing import Optional

from ab.nn.util.Util import release_memory
from ab.gpt.util.LoRA import find_all_linear_names, print_trainable_parameters
import ab.gpt.util.training_runtime as TrainingRuntime
from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# Import KTO machinery from TRL.  KTOTrainer requires trl >= 0.7.0.
try:
    from trl import KTOTrainer, KTOConfig
except ImportError as e:
    raise ImportError(
        "trl.KTOTrainer / KTOConfig not available.  Upgrade trl: "
        "`pip install -U trl`.  Original error: " + str(e)
    )


def kto_lora_config(target_modules, r=16, lora_alpha=16, lora_dropout=0.05,
                    bias="none", task_type="CAUSAL_LM", layers_to_transform=None):
    """LoRA config for KTO — lower-rank defaults than SFT for drift control."""
    kwargs = dict(
        r=r, lora_alpha=lora_alpha, target_modules=list(target_modules),
        lora_dropout=lora_dropout, bias=bias, task_type=task_type,
    )
    if layers_to_transform is not None:
        kwargs["layers_to_transform"] = list(layers_to_transform)
    return LoraConfig(**kwargs)


class KTO:
    """
    KTO trainer wrapper.  Mirrors the LoRA class interface so callers
    (TuneNNGenKTO.py, KTOIterativeFinetuner) can use it as a drop-in
    replacement for LoRA when training_mode == "kto".
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        training_args: TrainingArguments,
        access_token=None,
        peft_config=None,
        use_unsloth=False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.access_token = access_token
        self._use_unsloth = use_unsloth

        if peft_config is None:
            self.peft_config = kto_lora_config(find_all_linear_names(self.model))
        else:
            self.peft_config = peft_config

        if use_unsloth:
            try:
                inner_unsloth_available = True
                try:
                    from unsloth import FastModel
                except ImportError:
                    inner_unsloth_available = False

                if not inner_unsloth_available:
                    raise ImportError("Unsloth not installed")

                self.peft_model = FastModel.get_peft_model(
                    self.model,
                    r=self.peft_config.r,
                    lora_alpha=self.peft_config.lora_alpha,
                    lora_dropout=self.peft_config.lora_dropout,
                    target_modules=list(self.peft_config.target_modules),
                    bias="none",
                    use_gradient_checkpointing="unsloth",
                    random_state=42,
                )
                print("[KTO] Using Unsloth's FastModel.get_peft_model() for bfloat16 compatibility")
            except Exception as e:
                print(f"[KTO] Unsloth get_peft_model failed: {e}, falling back to standard PEFT")
                use_unsloth = False
                self._use_unsloth = False

        if not use_unsloth:
            # Standard PEFT flow for non-Unsloth models
            self.model = prepare_model_for_kbit_training(self.model)
            self.model.gradient_checkpointing_enable()
            print("[KTO] Gradient checkpointing enabled")
            self.peft_model = get_peft_model(self.model, self.peft_config)

        self.peft_model._hf_peft_config_loaded = True
        print(f"[KTO] Adapters attached. Effective target_modules: {self.peft_config.target_modules}")
        print("[KTO] Trainable parameter summary:")
        print_trainable_parameters(self.peft_model)

    def train(
        self,
        dataset: Dataset,
        tokenizer,
        output_dir: str,
        resume_from_checkpoint: Optional[str] = None,
        runtime_state_hooks: Optional[TrainingRuntime.RuntimeStateHooks] = None,
        checkpoint_label: str = "kto_trainer",
        beta: float = 0.1,
        desirable_weight: float = 1.0,
        undesirable_weight: float = 1.0,
        max_prompt_length: int = 2048,
        max_completion_length: int = 2048,
    ):
        """
        Train the model using KTOTrainer.

        Args:
            dataset: HF Dataset with columns {"prompt": str, "completion": str, "label": bool}
            tokenizer: Tokenizer instance (also stored as self.tokenizer)
            output_dir: Directory to save the trained adapter
            resume_from_checkpoint: Optional path to a trainer checkpoint for resume
            runtime_state_hooks: Optional pipeline-level state-persistence hooks
            checkpoint_label: Label used in resume error messages
            beta: KTO KL-regularisation strength (paper recommends 0.1)
            desirable_weight: per-class loss weight for label=True examples
            undesirable_weight: per-class loss weight for label=False examples
            max_prompt_length: max tokens kept from the prompt (truncated from left)
            max_completion_length: max tokens kept from the completion
        """
        self.peft_model.config.use_cache = False

        # KTOTrainer requires the exact three columns: prompt, completion, label.
        # If the caller passed metadata fields too, strip them now so the trainer
        # doesn't choke on unexpected columns.
        required_cols = {"prompt", "completion", "label"}
        if hasattr(dataset, "column_names"):
            extra = [c for c in dataset.column_names if c not in required_cols]
            if extra:
                print(f"[KTO] Removing non-KTO columns from dataset: {extra}")
                dataset = dataset.remove_columns(extra)
            missing = required_cols - set(dataset.column_names)
            if missing:
                raise ValueError(
                    f"KTO dataset missing required columns: {missing}. "
                    f"Got columns: {dataset.column_names}"
                )

        # Split dataset for train/eval
        dataset = dataset.train_test_split(test_size=0.1, seed=43)
        train_dataset = dataset['train']
        eval_dataset = dataset['test']

        # Report label balance
        train_labels = [int(bool(x)) for x in train_dataset["label"]]
        eval_labels = [int(bool(x)) for x in eval_dataset["label"]]
        n_train_des = sum(train_labels)
        n_train_und = len(train_labels) - n_train_des
        n_eval_des = sum(eval_labels)
        n_eval_und = len(eval_labels) - n_eval_des
        print(f"[KTO] Train split: {len(train_dataset)} examples "
              f"({n_train_des} desirable, {n_train_und} undesirable)")
        print(f"[KTO] Eval split:  {len(eval_dataset)} examples "
              f"({n_eval_des} desirable, {n_eval_und} undesirable)")

        # Build KTOConfig from the TrainingArguments instance the caller supplied.
        # KTOConfig is a subclass of TrainingArguments, so .to_dict() round-tripping
        # is safe.  We then layer the KTO-specific fields on top.
        if isinstance(self.training_args, KTOConfig):
            kto_config = self.training_args
        else:
            base_kwargs = self.training_args.to_dict()
            # Filter to KTOConfig accepted args to be safe across TRL minor versions
            kto_sig = inspect.signature(KTOConfig.__init__)
            accepted = set(kto_sig.parameters.keys())
            filtered = {k: v for k, v in base_kwargs.items() if k in accepted}
            kto_config = KTOConfig(**filtered)

        # KTO-specific knobs
        kto_config.beta = beta
        kto_config.desirable_weight = desirable_weight
        kto_config.undesirable_weight = undesirable_weight
        kto_config.max_prompt_length = max_prompt_length
        kto_config.max_completion_length = max_completion_length
        kto_config.max_length = max_prompt_length + max_completion_length
        # KTOTrainer accepts these columns directly; do NOT remove unused columns
        kto_config.remove_unused_columns = False

        # Tokenizer config — KTOTrainer tokenises prompt + completion separately
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "right"

        # Build the KTOTrainer.  In TRL >= 0.10 the parameter is "processing_class";
        # in earlier versions it was "tokenizer".  Detect at runtime.
        kto_init_sig = inspect.signature(KTOTrainer.__init__)
        kto_kwargs = dict(
            model=self.peft_model,
            args=kto_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        if "processing_class" in kto_init_sig.parameters:
            kto_kwargs["processing_class"] = self.tokenizer
        elif "tokenizer" in kto_init_sig.parameters:
            kto_kwargs["tokenizer"] = self.tokenizer
        else:
            raise RuntimeError(
                "Installed KTOTrainer accepts neither 'processing_class' nor 'tokenizer' — "
                "TRL version may be incompatible."
            )

        # When using PEFT, KTOTrainer can use the base model (adapters disabled)
        # as the implicit reference, so we don't supply ref_model.  Explicit None.
        if "ref_model" in kto_init_sig.parameters:
            kto_kwargs["ref_model"] = None

        trainer = KTOTrainer(**kto_kwargs)

        # Verify dtypes before training
        dtypes = {}
        for _, p in self.peft_model.named_parameters():
            dtype = p.dtype
            dtypes[dtype] = dtypes.get(dtype, 0) + p.numel()
        total = sum(dtypes.values())
        for k, v in dtypes.items():
            print(k, v, v / total)

        if runtime_state_hooks is not None:
            TrainingRuntime.restore_or_reset_runtime_state(
                Path(resume_from_checkpoint).expanduser().resolve() if resume_from_checkpoint else None,
                runtime_state_hooks,
            )
            runtime_callback = TrainingRuntime.build_trainer_checkpoint_callback(runtime_state_hooks)
            if runtime_callback is not None:
                trainer.add_callback(runtime_callback)

        # Train
        print("[KTO] Training...")
        if resume_from_checkpoint:
            resolved_resume_checkpoint = Path(resume_from_checkpoint).expanduser().resolve()
            if not resolved_resume_checkpoint.exists():
                raise FileNotFoundError(
                    f"{checkpoint_label.capitalize()} resume checkpoint not found: "
                    f"{resolved_resume_checkpoint}"
                )
            train_signature = inspect.signature(trainer.train)
            if "resume_from_checkpoint" not in train_signature.parameters:
                raise RuntimeError(
                    f"Installed trainer does not support resume_from_checkpoint for "
                    f"{checkpoint_label} resume."
                )
            train_result = trainer.train(resume_from_checkpoint=str(resolved_resume_checkpoint))
        else:
            train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics(split="train", metrics=metrics)
        trainer.save_metrics(split="train", metrics=metrics)
        trainer.save_state()
        print(metrics)

        # Re-enable cache for inference
        self.peft_model.config.use_cache = True

        # Save adapter + tokenizer
        print(f"[KTO] Saving final adapter to {output_dir}")
        makedirs(output_dir, exist_ok=True)
        trainer.model.save_pretrained(output_dir, access_token=self.access_token)
        self.tokenizer.save_pretrained(output_dir)
        print(f"[KTO] Tokenizer saved to {output_dir}")

        release_memory()
        return self.peft_model
