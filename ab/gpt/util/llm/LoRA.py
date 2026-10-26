import os
from os import makedirs
import inspect
from pathlib import Path
from typing import Optional

import torch
from ab.nn.util.Util import release_memory
import ab.gpt.util.training_runtime as TrainingRuntime
from datasets import Dataset
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizerBase
)
from trl import SFTTrainer, SFTConfig
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

# Try to import DataCollatorForCompletionOnlyLM (may not be available in all trl versions)
try:
    from trl.trainer.sft_trainer import DataCollatorForCompletionOnlyLM
except ImportError:
    try:
        from trl import DataCollatorForCompletionOnlyLM
    except ImportError:
        try:
            from trl.trainer import DataCollatorForCompletionOnlyLM
        except ImportError:
            # If not available, set to None and fall back to DataCollatorForLanguageModeling
            DataCollatorForCompletionOnlyLM = None


def _build_lm_collator(tokenizer, completion_only_loss=True, pad_to_multiple_of=8):
    """Construct DataCollatorForLanguageModeling whatever signature is installed.

    The name ``trl.trainer.sft_trainer.DataCollatorForLanguageModeling`` has
    meant at least three different classes:

    * TRL <=1.14   ``pad_token_id``, ``completion_only_loss``, ...
    * newer TRL    ``pad_token_id`` (required), no ``completion_only_loss``
    * transformers ``tokenizer``, ``mlm``, ...

    Branching on one of those shapes breaks on the others, so nothing is
    assumed: every keyword the installed class actually declares is filled from
    a pool of known values, and anything it does not declare is dropped. A
    required parameter with no known value is reported rather than guessed.

    Where the collator has no ``completion_only_loss``, masking is not lost --
    SFTTrainer applies it from the prompt/completion columns instead.
    """
    import inspect

    try:
        params = inspect.signature(DataCollatorForLanguageModeling.__init__).parameters
    except (TypeError, ValueError):
        params = {}

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    known = {
        'tokenizer': tokenizer,
        'pad_token_id': pad_token_id,
        'completion_only_loss': completion_only_loss,
        'mlm': False,
        'pad_to_multiple_of': pad_to_multiple_of,
        'return_tensors': 'pt',
    }
    accepted = {name: value for name, value in known.items() if name in params}

    missing = [
        name for name, prm in params.items()
        if name != 'self'
        and prm.default is inspect.Parameter.empty
        and prm.kind not in (inspect.Parameter.VAR_POSITIONAL,
                             inspect.Parameter.VAR_KEYWORD)
        and name not in accepted
    ]
    if missing:
        print(f"[LoRA] DataCollatorForLanguageModeling requires {missing}, which "
              f"this code has no value for. Construction will probably fail; "
              f"the installed trl/transformers version is not one this was "
              f"written against.")

    print(f"[LoRA] Collator kwargs in use: {', '.join(sorted(accepted)) or 'none'}"
          + ("" if 'completion_only_loss' in accepted else
             " (no completion_only_loss on this version; SFTTrainer masks the "
             "completion from the prompt/completion columns)"))
    return DataCollatorForLanguageModeling(**accepted)


# When using deepspeed, no Training arguments' initialization after model initialization, if pre-trained model is used
# Reference: https://huggingface.co/docs/transformers/deepspeed?zero-config=ZeRO-3
# With the claim: "The TrainingArguments object must be created before calling the model from_pretrained()"

def find_all_linear_names(model):
    cls = torch.nn.modules.linear.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )


def create_peft_config(modules):
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    :param modules: Names of the modules to apply Lora to
    """
    return LoraConfig(
        r=32,  # dimension of the updated matrices
        lora_alpha=64,  # parameter for scaling
        target_modules=modules,
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=True
    )


class LoRA:
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizerBase,
                 training_args: TrainingArguments,
                 access_token=None,
                 peft_config=None,
                 use_unsloth=False,
                 init_adapter=None,
                 ):
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.access_token = access_token
        self._use_unsloth = use_unsloth

        if peft_config is None:
            modules = find_all_linear_names(self.model)
            self.peft_config = create_peft_config(modules)
        else:
            self.peft_config = peft_config

        if init_adapter and use_unsloth:
            raise ValueError("init_adapter (NNGPT_CARRY_ADAPTER) is not supported with unsloth")

        if use_unsloth:
            # Use Unsloth's native LoRA attachment (keeps bfloat16 dtypes)
            try:
                # Late-import Unsloth only if requested and available
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
                    use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
                    random_state=42,
                )
                print("[LoRA] Using Unsloth's FastModel.get_peft_model() for bfloat16 compatibility")
            except Exception as e:
                print(f"[LoRA] Unsloth get_peft_model failed: {e}, falling back to standard PEFT")
                use_unsloth = False
                self._use_unsloth = False

        if not use_unsloth:
            # Standard PEFT flow for non-Unsloth models
            self.model = prepare_model_for_kbit_training(self.model)
            self.model.gradient_checkpointing_enable()
            print("[LoRA] Gradient checkpointing enabled")
            if init_adapter:
                # Continue training the previous cycle's adapter on the frozen base,
                # after the same k-bit preparation a fresh adapter gets.
                self.peft_model = PeftModel.from_pretrained(self.model, init_adapter, is_trainable=True)
                print(f"[LoRA] Continuing adapter from {init_adapter}")
            else:
                self.peft_model = get_peft_model(self.model, self.peft_config)

        self.peft_model._hf_peft_config_loaded = True
        # Log trainable parameters immediately after adapters are attached
        target_modules = getattr(self.peft_config, 'target_modules', 'N/A (prefix tuning)')
        print(f"[LoRA] Adapters attached. Effective target_modules: {target_modules}")
        print("[LoRA] Trainable parameter summary:")
        print_trainable_parameters(self.peft_model)

    def train(
            self,
            dataset: Dataset,
            tokenizer,
            output_dir: str,
            train_on_completions_only=False,
            response_template=None,
            resume_from_checkpoint: Optional[str] = None,
            runtime_state_hooks: Optional[TrainingRuntime.RuntimeStateHooks] = None,
            checkpoint_label: str = "trainer",
    ):
        """
        Train the model using SFTTrainer.

        Args:
            dataset: Dataset with pre-rendered text (from chat template) or raw text
            tokenizer: Tokenizer instance
            output_dir: Directory to save the trained model
            train_on_completions_only: If True, mask loss on system+user tokens (train only on assistant responses)
            response_template: String that precedes assistant answer in rendered text (e.g., "Assistant:" for DeepSeek)
        """
        self.peft_model.config.use_cache = False

        # Check if dataset has pre-rendered text or prompt/completion pairs.
        column_names = set(dataset.column_names) if hasattr(dataset, 'column_names') else set()
        has_text_field = "text" in column_names
        has_prompt_completion = {"prompt", "completion"}.issubset(column_names)
        if has_prompt_completion:
            train_on_completions_only = True

        # Say plainly which branch the data selected and why. "Dataset does not
        # have 'text' field" alone does not reveal what the dataset DID carry,
        # which makes a half-applied change look like a change that does not work.
        print(f"[LoRA] Dataset columns: {sorted(column_names)}")
        if has_prompt_completion:
            print("[LoRA] -> prompt/completion branch: loss masked to the completion")
        elif has_text_field:
            print("[LoRA] -> text branch: loss over the whole sequence "
                  "(prompt included) unless a response template is found")
        else:
            print("[LoRA] -> NEITHER 'prompt'+'completion' NOR 'text' present. "
                  "The loss will cover the whole sequence, prompt included. "
                  "If you expected completion-only training, Prompt.get_dataset "
                  "is not returning prompt/completion columns.")

        # Split The dataset (NNGPT_SEED_SFT_SPLIT: seeded from training_args.seed, so the
        # rows each cycle trains on can be reconstructed; unseeded otherwise)
        if os.environ.get("NNGPT_SEED_SFT_SPLIT", "").strip().lower() in {"1", "true", "yes", "on"}:
            dataset = dataset.train_test_split(test_size=0.1, seed=self.training_args.seed)
        else:
            dataset = dataset.train_test_split(test_size=0.1)
        train_dataset = dataset['train']
        eval_dataset = dataset['test']

        # build the trainer
        print("Parameter configuration of the model")
        print_trainable_parameters(self.peft_model)

        # Determine data collator and trainer kwargs
        if train_on_completions_only and not has_prompt_completion:
            if response_template is None:
                # Try to auto-detect response template from first example
                if has_text_field and len(train_dataset) > 0:
                    sample_text = train_dataset[0]["text"]
                    print(f"[INFO] Inspecting sample text to detect response template:")
                    print(f"Sample (first 500 chars):\n{sample_text[:500]}")

                    # Common patterns for DeepSeek and similar models
                    possible_templates = ["Assistant:", "assistant:", "<|assistant|>", "### Assistant:"]
                    for template in possible_templates:
                        if template in sample_text:
                            response_template = template
                            print(f"[INFO] Auto-detected response template: '{response_template}'")
                            break

                    if response_template is None:
                        print("[WARN] Could not auto-detect response template. Please set response_template manually.")
                        print("[WARN] Falling back to training on all tokens.")
                        train_on_completions_only = False
                else:
                    print("[WARN] Cannot auto-detect response template. Dataset missing 'text' field or is empty.")
                    print("[WARN] Falling back to training on all tokens.")
                    train_on_completions_only = False

            if train_on_completions_only and response_template and DataCollatorForCompletionOnlyLM is not None:
                print(f"[INFO] Using completion-only training with response template: '{response_template}'")
                collator = DataCollatorForCompletionOnlyLM(
                    response_template=response_template,
                    tokenizer=self.tokenizer
                )
            else:
                if train_on_completions_only and DataCollatorForCompletionOnlyLM is None:
                    print("[WARN] DataCollatorForCompletionOnlyLM not available, using DataCollatorForLanguageModeling with completion_only_loss=True")
                collator = _build_lm_collator(
                    self.tokenizer,
                    completion_only_loss=bool(train_on_completions_only) or True,
                )
        else:
            collator = _build_lm_collator(self.tokenizer, completion_only_loss=True)

        # Use SFTTrainer for pre-rendered text or prompt/completion pairs.
        if has_text_field or has_prompt_completion:
            if has_prompt_completion:
                print("[INFO] Using SFTTrainer with prompt/completion dataset")
            else:
                print("[INFO] Using SFTTrainer with pre-rendered text (dataset_text_field='text')")
            use_packing = False if has_prompt_completion else not train_on_completions_only
            print(f"[INFO] Packing enabled: {use_packing} ({'Simple mode: full text' if use_packing else 'Precise mode: completions only'})")
            sft_max_length = getattr(self.training_args, "max_length", None) or 4096
            print(f"[INFO] SFT max_length: {sft_max_length}")

            # SFTTrainer truncates prompt+completion at max_length, from the
            # right. The completion ends with the <nn> block, so a budget below
            # the sample length removes the network code and trains the model to
            # emit <hp> and <tr> and then stop. Measure it rather than let it
            # happen silently.
            if has_prompt_completion:
                try:
                    over = longest = 0
                    _sample = train_dataset.select(range(min(128, len(train_dataset))))
                    for _p, _c in zip(_sample["prompt"], _sample["completion"]):
                        _n = len(self.tokenizer(_p + _c, add_special_tokens=False)["input_ids"])
                        longest = max(longest, _n)
                        over += _n > sft_max_length
                    if over:
                        print(f"[CONFIG WARNING] {over}/{len(_sample)} sampled rows exceed "
                              f"max_length={sft_max_length} (longest {longest} tokens). "
                              f"SFTTrainer truncates from the RIGHT and the completion ends "
                              f"with the <nn> block, so those rows are trained with the "
                              f"network code cut off. Raise sft_max_length to at least "
                              f"{longest}, and keep it >= context_length.")
                    else:
                        print(f"[INFO] No sampled row exceeds max_length "
                              f"(longest {longest} tokens); nothing is truncated.")
                except Exception as _exc:
                    print(f"[INFO] Could not measure truncation: {_exc}")

            # Configure tokenizer for SFT: truncate from left (keep assistant response), pad on right
            # This ensures sequences are truncated correctly when SFTTrainer tokenizes
            # Remember what we are about to overwrite. This tokenizer is SHARED
            # with the ChatBot used for generation: leaving model_max_length at
            # sft_max_length makes ChatBot's input budget
            # (model_max_length - max_new_tokens) collapse to ~0 after the first
            # fine-tune, so the prompt is truncated to almost nothing and the
            # model free-associates instead of doing the task. Restored below.
            self._tok_state = {
                "truncation_side": getattr(self.tokenizer, "truncation_side", "right"),
                "padding_side": getattr(self.tokenizer, "padding_side", "right"),
                "model_max_length": getattr(self.tokenizer, "model_max_length", None),
            }
            self.tokenizer.truncation_side = "left"
            self.tokenizer.padding_side = "right"
            self.tokenizer.model_max_length = sft_max_length

            # Suppress sequence length warnings - SFTTrainer will handle truncation correctly
            import warnings
            warnings.filterwarnings("ignore", message=".*sequence length.*longer than.*maximum.*")
            warnings.filterwarnings("ignore", message=".*Token indices sequence length.*")

            # Set packing, max_length, and dataset_text_field in SFTConfig to avoid warnings
            # Convert to SFTConfig if not already, or set attributes directly
            if isinstance(self.training_args, SFTConfig):
                self.training_args.remove_unused_columns = False  # critical when using raw text
                self.training_args.packing = use_packing  # Simple: True, Precise: False
                self.training_args.max_length = sft_max_length
                if has_text_field:
                    self.training_args.dataset_text_field = "text"  # Feed Dataset with {"text": ...} format
            else:
                # If training_args is TrainingArguments, create SFTConfig with all attributes
                sft_config = SFTConfig(**self.training_args.to_dict())
                sft_config.remove_unused_columns = False  # critical when using raw text
                sft_config.packing = use_packing  # Simple: True, Precise: False
                sft_config.max_length = sft_max_length
                if has_text_field:
                    sft_config.dataset_text_field = "text"  # Feed Dataset with {"text": ...} format
                self.training_args = sft_config

            # SFTTrainer will handle tokenization and truncation internally
            # Since chat_template already added special tokens, tokenizer will use add_special_tokens=False internally
            trainer = SFTTrainer(
                model=self.peft_model,
                processing_class=self.tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                args=self.training_args,
                data_collator=collator  # Simple: DataCollatorForLanguageModeling, Precise: DataCollatorForCompletionOnlyLM
                # packing, max_length, and dataset_text_field are in training_args (SFTConfig)
                # SFTTrainer will handle truncation based on max_length and tokenizer settings
            )
        else:
            print("[WARN] Dataset does not have 'text' field. Using standard Trainer.")
            # Fallback to original Trainer if dataset format is different
            from transformers import Trainer
            trainer = Trainer(
                model=self.peft_model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                args=self.training_args,
                data_collator=collator
            )

        # verifying the datatypes before training
        dtypes = {}
        for _, p in self.peft_model.named_parameters():
            dtype = p.dtype
            if dtype not in dtypes:
                dtypes[dtype] = 0
            dtypes[dtype] += p.numel()
        total = 0
        for k, v in dtypes.items():
            total += v
        for k, v in dtypes.items():
            print(k, v, v / total)
        do_train = True

        if runtime_state_hooks is not None:
            # Pipelines opt into runtime state persistence; LoRA only wires the shared contract.
            TrainingRuntime.restore_or_reset_runtime_state(
                Path(resume_from_checkpoint).expanduser().resolve() if resume_from_checkpoint else None,
                runtime_state_hooks,
            )
            runtime_callback = TrainingRuntime.build_trainer_checkpoint_callback(runtime_state_hooks)
            if runtime_callback is not None:
                trainer.add_callback(runtime_callback)

        # starting training
        print("Training...")
        if do_train:
            if resume_from_checkpoint:
                # Trainer resume is separate from runtime state restore and remains an explicit one-shot input.
                resolved_resume_checkpoint = Path(resume_from_checkpoint).expanduser().resolve()
                if not resolved_resume_checkpoint.exists():
                    raise FileNotFoundError(
                        f"{checkpoint_label.capitalize()} resume checkpoint not found: {resolved_resume_checkpoint}"
                    )
                train_signature = inspect.signature(trainer.train)
                if "resume_from_checkpoint" not in train_signature.parameters:
                    raise RuntimeError(
                        f"Installed trainer does not support resume_from_checkpoint for {checkpoint_label} resume."
                    )
                train_result = trainer.train(resume_from_checkpoint=str(resolved_resume_checkpoint))
            else:
                train_result = trainer.train()
            metrics = train_result.metrics
            trainer.log_metrics(split="train", metrics=metrics)
            trainer.save_metrics(split="train", metrics=metrics)
            trainer.save_state()
            print(metrics)

        # prepare the model for usage
        self.peft_model.config.use_cache = True

        # Saving model
        print("Saving last checkpoint of the model...")
        makedirs(output_dir, exist_ok=True)
        trainer.model.save_pretrained(output_dir, access_token=self.access_token)

        # Always save tokenizer alongside adapter
        self.tokenizer.save_pretrained(output_dir)
        print(f"Tokenizer saved to {output_dir}")

        # Free memory for merging weights
        # del self.model  (peft_model still holds a ref, so this would not free GPU memory)
        del trainer
        release_memory()
        # Put the shared tokenizer back the way generation needs it.
        state = getattr(self, "_tok_state", None)
        if state:
            for attr, value in state.items():
                if value is not None:
                    setattr(self.tokenizer, attr, value)
            print(f"[INFO] Tokenizer restored after training: "
                  f"model_max_length={self.tokenizer.model_max_length}, "
                  f"truncation_side={self.tokenizer.truncation_side}")
            self._tok_state = None

        return self.peft_model