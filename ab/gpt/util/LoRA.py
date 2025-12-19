from os import makedirs

import torch
from ab.nn.util.Util import release_memory
from datasets import Dataset
from peft import (
    LoraConfig,
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
                 peft_config=None
                 ):
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.access_token = access_token
        if peft_config is None:
            modules = find_all_linear_names(self.model)
            self.peft_config = create_peft_config(modules)
        else:
            self.peft_config = peft_config
        self.model = prepare_model_for_kbit_training(self.model)
        self.model.gradient_checkpointing_enable()
        self.peft_model = get_peft_model(self.model, self.peft_config)
        
        # Log trainable parameters immediately after adapters are attached
        print(f"[LoRA] Adapters attached. Effective target_modules: {self.peft_config.target_modules}")
        print("[LoRA] Trainable parameter summary:")
        print_trainable_parameters(self.peft_model)

    def train(self, dataset: Dataset, tokenizer, output_dir: str, train_on_completions_only=False, response_template=None):
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
        
        # Check if dataset has "text" field (pre-rendered) or needs formatting
        has_text_field = "text" in dataset.column_names if hasattr(dataset, 'column_names') else False
        
        # Split The dataset
        dataset = dataset.train_test_split(test_size=0.1)
        train_dataset = dataset['train']
        eval_dataset = dataset['test']

        # build the trainer
        print("Parameter configuration of the model")
        print_trainable_parameters(self.peft_model)

        # Determine data collator and trainer kwargs
        if train_on_completions_only:
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
                # TRL's DataCollatorForLanguageModeling uses pad_token_id and completion_only_loss
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                collator = DataCollatorForLanguageModeling(
                    pad_token_id=pad_token_id,
                    completion_only_loss=train_on_completions_only if train_on_completions_only else True,
                    pad_to_multiple_of=8,
                    return_tensors="pt"
                )
        else:
            # TRL's DataCollatorForLanguageModeling uses pad_token_id
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            collator = DataCollatorForLanguageModeling(
                pad_token_id=pad_token_id,
                completion_only_loss=True,
                pad_to_multiple_of=8,
                return_tensors="pt"
            )

        # Use SFTTrainer for pre-rendered text, or fallback to Trainer if needed
        # TL;DR: Feed Dataset.from_list([...{"text": ...}...]) to SFTTrainer with dataset_text_field="text"
        # Choose:
        #   Simple: train on full text (packing=True)
        #   Precise: DataCollatorForCompletionOnlyLM with response_template + packing=False
        if has_text_field:
            print("[INFO] Using SFTTrainer with pre-rendered text (dataset_text_field='text')")
            # Packing: enable when NOT using completion-only masking (completion-only requires packing=False)
            # Simple mode: packing=True (train on full text)
            # Precise mode: packing=False + DataCollatorForCompletionOnlyLM (train on completions only)
            use_packing = not train_on_completions_only
            print(f"[INFO] Packing enabled: {use_packing} ({'Simple mode: full text' if use_packing else 'Precise mode: completions only'})")
            
            # Configure tokenizer for SFT: truncate from left (keep assistant response), pad on right
            # This ensures sequences are truncated correctly when SFTTrainer tokenizes
            self.tokenizer.truncation_side = "left"
            self.tokenizer.padding_side = "right"
            self.tokenizer.model_max_length = 4096  # DeepSeek-Coder-7B-Instruct-v1.5 has ~4K context
            
            # Suppress sequence length warnings - SFTTrainer will handle truncation correctly
            import warnings
            warnings.filterwarnings("ignore", message=".*sequence length.*longer than.*maximum.*")
            warnings.filterwarnings("ignore", message=".*Token indices sequence length.*")
            
            # Set packing, max_seq_length, and dataset_text_field in SFTConfig to avoid warnings
            # Convert to SFTConfig if not already, or set attributes directly
            if isinstance(self.training_args, SFTConfig):
                self.training_args.remove_unused_columns = False  # critical when using raw text
                self.training_args.packing = use_packing  # Simple: True, Precise: False
                self.training_args.max_seq_length = 4096  # DeepSeek-Coder-7B-Instruct-v1.5 has ~4K context (4k/4.1k)
                self.training_args.dataset_text_field = "text"  # Feed Dataset with {"text": ...} format
            else:
                # If training_args is TrainingArguments, create SFTConfig with all attributes
                sft_config = SFTConfig(**self.training_args.to_dict())
                sft_config.remove_unused_columns = False  # critical when using raw text
                sft_config.packing = use_packing  # Simple: True, Precise: False
                sft_config.max_seq_length = 4096  # DeepSeek-Coder-7B-Instruct-v1.5 has ~4K context (4k/4.1k)
                sft_config.dataset_text_field = "text"  # Feed Dataset with {"text": ...} format
                self.training_args = sft_config
            
            # SFTTrainer will handle tokenization and truncation internally
            # Since chat_template already added special tokens, tokenizer will use add_special_tokens=False internally
            trainer = SFTTrainer(
                model=self.peft_model,
                tokenizer=self.tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                args=self.training_args,
                data_collator=collator  # Simple: DataCollatorForLanguageModeling, Precise: DataCollatorForCompletionOnlyLM
                # packing, max_seq_length, and dataset_text_field are in training_args (SFTConfig)
                # SFTTrainer will handle truncation based on max_seq_length and tokenizer settings
            )
        else:
            print("[WARN] Dataset does not have 'text' field. Using standard Trainer.")
            # Fallback to original Trainer if dataset format is different
            from transformers import Trainer
            from ab.gpt.util.Metrics import Metrics

            # metric_for_best_model is set in TuneNNGen.py to the raw metric name (e.g., "bleu")
            test_metric = getattr(self.training_args, "metric_for_best_model", None)

            # Transformers sometimes refers to "eval_bleu" in logs/errors; normalize to "bleu"
            if isinstance(test_metric, str) and test_metric.startswith("eval_"):
                test_metric = test_metric[len("eval_"):]

            metrics_obj = Metrics(self.tokenizer)
            compute_metrics = metrics_obj.init_compute_metrics(test_metric)

            # Reduce memory: don't keep full logits for metrics
            def preprocess_logits_for_metrics(logits, labels):
                if isinstance(logits, tuple):
                    logits = logits[0]
                return logits.argmax(dim=-1).detach().to("cpu")

            trainer = Trainer(
                model=self.peft_model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                args=self.training_args,
                data_collator=collator,
                compute_metrics=compute_metrics,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics if compute_metrics else None,
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

        # starting training
        print("Training...")
        if do_train:
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

        # Free memory for merging weights
        # del self.model
        # del trainer
        release_memory()
        return self.peft_model
