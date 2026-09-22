"""
ab/gpt/util/DPO.py — DPO (Direct Preference Optimization) trainer wrapper.

Analog of ab/gpt/util/KTO.py, but for DPO's *paired* preference format
{prompt, chosen, rejected} instead of KTO's unpaired {prompt, completion, label}.
Wraps TRL's DPOTrainer with the same LoRA-attachment and checkpoint/resume
semantics as the KTO path, so the self-contained pipeline can swap KTO for DPO.

DivPO (Diverse Preference Optimization) is implemented entirely at the DATA level
(diverse pair selection, in the pipeline) — the DPO loss itself is unchanged, so
this wrapper is plain DPO. See ab/gpt/TuneNNGenDPO.py and the pipeline's
build_divpo_dataset() for the DivPO selection rule.

Reference: https://huggingface.co/papers/2305.18290 (DPO)
           https://arxiv.org/abs/2501.18101 (DivPO — diverse pair selection)
"""

from os import makedirs
import inspect
from pathlib import Path
from typing import Optional

from ab.nn.util.Util import release_memory
from ab.gpt.util.llm.LoRA import find_all_linear_names, print_trainable_parameters
from ab.gpt.util.llm.KTO import kto_lora_config  # shared LoRA-config builder (drift-control defaults)
import ab.gpt.util.training_runtime as TrainingRuntime
from datasets import Dataset
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# DPOTrainer requires trl >= 0.7.0 (present in the container's trl 0.22.2).
try:
    from trl import DPOTrainer, DPOConfig
except ImportError as e:
    raise ImportError(
        "trl.DPOTrainer / DPOConfig not available.  Upgrade trl: "
        "`pip install -U trl`.  Original error: " + str(e)
    )


class DPO:
    """DPO trainer wrapper.  Mirrors the KTO class interface so the pipeline can
    swap KTO for DPO with the same LoRA-attachment and save semantics."""

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
                try:
                    from unsloth import FastModel
                    inner_unsloth_available = True
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
                print("[DPO] Using Unsloth's FastModel.get_peft_model() for bfloat16 compatibility")
            except Exception as e:  # noqa: BLE001
                print(f"[DPO] Unsloth get_peft_model failed: {e}, falling back to standard PEFT")
                use_unsloth = False
                self._use_unsloth = False

        if not use_unsloth:
            self.model = prepare_model_for_kbit_training(self.model)
            self.model.gradient_checkpointing_enable()
            print("[DPO] Gradient checkpointing enabled")
            self.peft_model = get_peft_model(self.model, self.peft_config)

        self.peft_model._hf_peft_config_loaded = True
        print(f"[DPO] Adapters attached. Effective target_modules: {self.peft_config.target_modules}")
        print("[DPO] Trainable parameter summary:")
        print_trainable_parameters(self.peft_model)

    def train(
        self,
        dataset: Dataset,
        tokenizer,
        output_dir: str,
        resume_from_checkpoint: Optional[str] = None,
        runtime_state_hooks: Optional[TrainingRuntime.RuntimeStateHooks] = None,
        checkpoint_label: str = "dpo_trainer",
        beta: float = 0.1,
        max_prompt_length: int = 2048,
        max_completion_length: int = 2048,
    ):
        """Train via DPOTrainer.

        Args:
            dataset: HF Dataset with columns {"prompt": str, "chosen": str, "rejected": str}
            beta: DPO KL-regularisation strength (higher → closer to the diverse
                  reference/base model; a lever against diversity collapse)
        """
        self.peft_model.config.use_cache = False

        # DPOTrainer needs exactly prompt/chosen/rejected; strip metadata columns.
        required_cols = {"prompt", "chosen", "rejected"}
        if hasattr(dataset, "column_names"):
            extra = [c for c in dataset.column_names if c not in required_cols]
            if extra:
                print(f"[DPO] Removing non-DPO columns from dataset: {extra}")
                dataset = dataset.remove_columns(extra)
            missing = required_cols - set(dataset.column_names)
            if missing:
                raise ValueError(
                    f"DPO dataset missing required columns: {missing}. "
                    f"Got columns: {dataset.column_names}"
                )

        dataset = dataset.train_test_split(test_size=0.1, seed=43)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
        print(f"[DPO] Train split: {len(train_dataset)} pairs | Eval split: {len(eval_dataset)} pairs")

        # Build DPOConfig from the supplied TrainingArguments (subclass round-trip),
        # then layer the DPO-specific fields on top (mirrors the KTO wrapper).
        if isinstance(self.training_args, DPOConfig):
            dpo_config = self.training_args
        else:
            base_kwargs = self.training_args.to_dict()
            accepted = set(inspect.signature(DPOConfig.__init__).parameters.keys())
            filtered = {k: v for k, v in base_kwargs.items() if k in accepted}
            dpo_config = DPOConfig(**filtered)

        dpo_config.beta = beta
        dpo_config.max_prompt_length = max_prompt_length
        dpo_config.max_length = max_prompt_length + max_completion_length
        # max_completion_length exists in newer TRL; set defensively.
        try:
            dpo_config.max_completion_length = max_completion_length
        except Exception:  # noqa: BLE001
            pass
        dpo_config.remove_unused_columns = False

        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "right"

        dpo_init_sig = inspect.signature(DPOTrainer.__init__)
        dpo_kwargs = dict(
            model=self.peft_model,
            args=dpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        if "processing_class" in dpo_init_sig.parameters:
            dpo_kwargs["processing_class"] = self.tokenizer
        elif "tokenizer" in dpo_init_sig.parameters:
            dpo_kwargs["tokenizer"] = self.tokenizer
        else:
            raise RuntimeError(
                "Installed DPOTrainer accepts neither 'processing_class' nor 'tokenizer' — "
                "TRL version may be incompatible."
            )
        # With PEFT, DPOTrainer uses the base model (adapters disabled) as the
        # implicit reference, so no separate ref_model is needed.
        if "ref_model" in dpo_init_sig.parameters:
            dpo_kwargs["ref_model"] = None

        trainer = DPOTrainer(**dpo_kwargs)
        print(f"[DPO] Trainer ready (beta={beta}, max_prompt_length={max_prompt_length}, "
              f"max_length={dpo_config.max_length})")

        # Verify dtypes before training
        dtypes = {}
        for _, p in self.peft_model.named_parameters():
            dtypes[p.dtype] = dtypes.get(p.dtype, 0) + p.numel()
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

        print("[DPO] Training...")
        if resume_from_checkpoint:
            resolved = Path(resume_from_checkpoint).expanduser().resolve()
            if not resolved.exists():
                raise FileNotFoundError(
                    f"{checkpoint_label.capitalize()} resume checkpoint not found: {resolved}")
            if "resume_from_checkpoint" not in inspect.signature(trainer.train).parameters:
                raise RuntimeError(
                    f"Installed trainer does not support resume_from_checkpoint for {checkpoint_label}.")
            train_result = trainer.train(resume_from_checkpoint=str(resolved))
        else:
            train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics(split="train", metrics=metrics)
        trainer.save_metrics(split="train", metrics=metrics)
        trainer.save_state()
        print(metrics)

        self.peft_model.config.use_cache = True

        print(f"[DPO] Saving final adapter to {output_dir}")
        makedirs(output_dir, exist_ok=True)
        trainer.model.save_pretrained(output_dir, access_token=self.access_token)
        self.tokenizer.save_pretrained(output_dir)
        print(f"[DPO] Tokenizer saved to {output_dir}")

        release_memory()
        return self.peft_model
