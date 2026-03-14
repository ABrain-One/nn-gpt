# TuneNNGen_20B_oss.py 的顶部

# Unsloth conditional import
try:
    from unsloth import FastModel
    UNSLOTH_AVAILABLE = True
    
    # ================= Dynamic injection of FastModel into LLM module =================
    import ab.gpt.util.LLM
    ab.gpt.util.LLM.FastModel = FastModel
    # ================================================

except ImportError:
    UNSLOTH_AVAILABLE = False

    
import argparse
import ab.gpt.TuneNNGen as TuneNNGen
from ab.gpt.NNEval import NN_TRAIN_EPOCHS
from ab.gpt.util.Const import new_out_file


if __name__ == '__main__':
    TARGET_MODULES_STR = ','.join(TuneNNGen.TARGET_MODULES)
    parser = argparse.ArgumentParser(description='TuneNNGen for 20B OSS model.')
    
    # Standalone-specific parameters
    parser.add_argument('-ne', '--num_train_epochs', type=int, default=TuneNNGen.NUM_TRAIN_EPOCHS,
                        help=f'Number of LLM fine-tuning epochs (default: {TuneNNGen.NUM_TRAIN_EPOCHS}).')
    parser.add_argument('-ls', '--lr_scheduler', type=str, default=TuneNNGen.LR_SCHEDULER,
                        help=f'Name of learning rate scheduler for LLM fine-tuning (default: {TuneNNGen.LR_SCHEDULER}).')
    parser.add_argument('-g', '--max_grad_norm', type=float, default=TuneNNGen.MAX_GRAD_NORM,
                        help=f'Upper limit on the backpropagation gradients for LLM fine-tuning (default: {TuneNNGen.MAX_GRAD_NORM}).')
    parser.add_argument('--test_metric', type=str, default=TuneNNGen.TEST_METRIC,
                        help=f'Test metric for LLM fine-tuning implemented in transformers package (default: {TuneNNGen.TEST_METRIC}).')
    
    # LoRA configuration
    parser.add_argument('-s', '--start_layer', type=int, default=TuneNNGen.START_LAYER,
                        help=f'Index of the first fine-tuned layer in the LLM (default: {TuneNNGen.START_LAYER}).')
    parser.add_argument('-e', '--end_layer', type=int, default=TuneNNGen.END_LAYER,
                        help=f'Index of the last fine-tuned layer in the LLM (default: {TuneNNGen.END_LAYER}).')
    parser.add_argument('-r', '--r', type=int, default=TuneNNGen.R,
                        help=f'Dimension of the updated matrices (default: {TuneNNGen.R}).')
    parser.add_argument('-a', '--lora_alpha', type=float, default=TuneNNGen.LORA_ALPHA,
                        help=f'LoRA alpha parameter for scaling (default: {TuneNNGen.LORA_ALPHA}).')
    parser.add_argument('-d', '--lora_dropout', type=float, default=TuneNNGen.LORA_DROPOUT,
                        help=f'LoRA dropout probability for layers (default: {TuneNNGen.LORA_DROPOUT}).')
    parser.add_argument('-t', '--target_modules', type=lambda s: s.split(','), default=TuneNNGen.TARGET_MODULES,
                        help=f'Target modules separated by comma (default: {TARGET_MODULES_STR})')
    parser.add_argument('-l', '--learning_rate', type=float, default=TuneNNGen.LEARNING_RATE,
                        help=f'Learning rate (default: {TuneNNGen.LEARNING_RATE}).')
    parser.add_argument('-y', '--task_type', type=str, default=TuneNNGen.TASK_TYPE,
                        help=f'LLM task type (default: {TuneNNGen.TASK_TYPE}).')
    parser.add_argument('-b', '--bias', type=str, default=TuneNNGen.BIAS,
                        help=f'Bias type (default: {TuneNNGen.BIAS}).')
    
    # Config files
    parser.add_argument('--llm_tune_conf', type=str, default=TuneNNGen.LLM_TUNE_CONF,
                        help=f'Config with a prompt for LLM fine-tuning (default: {TuneNNGen.LLM_TUNE_CONF}).')
    parser.add_argument('--nn_gen_conf', type=str, default=TuneNNGen.NN_GEN_CONF,
                        help=f'Config with a prompt for generation of neural networks by LLM (default: {TuneNNGen.NN_GEN_CONF}).')
    parser.add_argument('--nn_gen_conf_id', type=str, default=TuneNNGen.NN_GEN_CONF_ID,
                        help=f'Specifies prompt in the config for neural network generation by LLM (default: {TuneNNGen.NN_GEN_CONF_ID}).')
    parser.add_argument('--llm_conf', type=str, default='gpt_oss_20b.json',
                        help=f'Config of LLM (default: gpt_oss_20b.json).')
    
    # Training configuration
    parser.add_argument('-n', '--test_nn', type=int, default=TuneNNGen.TEST_NN,
                        help=f'Count of neural networks generated or modified by the LLM before and between fine-tuning epochs to monitor training progress (default: {TuneNNGen.TEST_NN}).')
    parser.add_argument('--nn_train_epochs', type=int, default=NN_TRAIN_EPOCHS,
                        help=f'Number of training epochs for the generated neural network (default: {NN_TRAIN_EPOCHS}).')
    parser.add_argument('-m', '--max_prompts', type=int, default=TuneNNGen.MAX_PROMPTS,
                        help=f'Max prompts for LLM fine-tuning; excess is truncated (default: {TuneNNGen.MAX_PROMPTS}).')
    parser.add_argument('--max_new_tokens', type=int, default=TuneNNGen.MAX_NEW_TOKENS,
                        help=f'Max number of tokens in LLM output (default: {TuneNNGen.MAX_NEW_TOKENS}).')
    parser.add_argument('--save_llm_output', type=bool, default=TuneNNGen.SAVE_LLM_OUTPUT,
                        help=f'Save full output of LLM in the file {new_out_file} (default: {TuneNNGen.SAVE_LLM_OUTPUT}).')
    parser.add_argument('--use_deepspeed', type=bool, default=TuneNNGen.USE_DEEPSPEED,
                        help=f'Utilize DeepSpeed optimizations for LLM fine-tuning (default: {TuneNNGen.USE_DEEPSPEED}).')
    parser.add_argument('--per_device_train_batch_size', type=int, default=TuneNNGen.PER_DEVICE_TRAIN_BATCH_SIZE,
                        help=f'Per device train batch size (default: {TuneNNGen.PER_DEVICE_TRAIN_BATCH_SIZE}).')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=TuneNNGen.GRADIENT_ACCUMULATION_STEPS,
                        help=f'Gradient accumulation steps (default: {TuneNNGen.GRADIENT_ACCUMULATION_STEPS}).')
    parser.add_argument('--warmup_ratio', type=float, default=TuneNNGen.WARMUP_RATIO,
                        help=f'Warmup step ratio for one fine-tuning epoch (default: {TuneNNGen.WARMUP_RATIO}).')
    parser.add_argument('--logging_steps', type=int, default=TuneNNGen.LOGGING_STEPS,
                        help=f'Logging steps (default: {TuneNNGen.LOGGING_STEPS}).')
    parser.add_argument('--optimizer', type=str, default=TuneNNGen.OPTIMIZER,
                        help=f'Optimizer for LLM fine-tuning (default: {TuneNNGen.OPTIMIZER}).')
    parser.add_argument('-k', '--skip_epoches', type=int, default=TuneNNGen.SKIP_EPOCHES,
                        help='Number of epoches to skip the neural network generation.')
    parser.add_argument('--peft', type=str, default=None, help='Path to saved LoRA layers.')
    parser.add_argument("--data_dir", type=str, default=None,
        help="Folder with train.jsonl/dev.jsonl/test.jsonl (produced by chat prep).")
    parser.add_argument('--nn_name_prefix', type=str, default=TuneNNGen.NN_NAME_PREFIX,
                        help=f'Neural network name prefix (default: {TuneNNGen.NN_NAME_PREFIX}).')
    parser.add_argument('--temperature', type=float, default=TuneNNGen.TEMPERATURE,
                        help=f'LLM temperature controls randomness in output generation (default: {TuneNNGen.TEMPERATURE}).')
    parser.add_argument('--top_k', type=int, default=TuneNNGen.TOP_K,
                        help=f'LLM top_k limits token selection in output generation (default: {TuneNNGen.TOP_K}).')
    parser.add_argument('--top_p', type=float, default=TuneNNGen.TOP_P,
                        help=f'LLM top_p controls token diversity in output generation (default: {TuneNNGen.TOP_P}).')
    
    # Pipeline-specific overrides (optional - for backward compatibility)
    parser.add_argument('--evaluation_strategy', type=str, default=None,
                        help=f"[Pipeline] Evaluation strategy during training (default: None, uses epoch-based for standalone).")
    parser.add_argument('--per_device_eval_batch_size', type=int, default=None,
                        help=f"[Pipeline] Per device eval batch size (default: None, uses same as train batch size).")
    parser.add_argument('--eval_steps', type=int, default=None,
                        help=f"[Pipeline] Evaluate every N steps (default: None).")
    parser.add_argument('--save_strategy', type=str, default=None,
                        help=f"[Pipeline] Save strategy during training (default: None).")
    parser.add_argument('--save_steps', type=int, default=None,
                        help=f"[Pipeline] Save checkpoint every N steps (default: None).")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help=f"[Pipeline] Maximum number of checkpoints to keep (default: None).")
    parser.add_argument('--load_best_model_at_end', action=argparse.BooleanOptionalAction, default=False,
                        help=f"[Pipeline] Load best model at end of training (default: False).")
    parser.add_argument('--metric_for_best_model', type=str, default=None,
                        help=f"[Pipeline] Metric to use for selecting best model (default: None).")
    parser.add_argument('--warmup_steps', type=int, default=None,
                        help=f"[Pipeline] Warmup steps override (default: None, uses warmup_ratio for standalone).")
    parser.add_argument('--weight_decay', type=float, default=None,
                        help=f"[Pipeline] Weight decay for regularization (default: None).")

    args = parser.parse_args()
    print("args:", args)
    TuneNNGen.main(num_train_epochs=args.num_train_epochs,
         lr_scheduler=args.lr_scheduler,
         max_grad_norm=args.max_grad_norm,
         tune_layers=range(args.start_layer, args.end_layer),
         r=args.r,
         lora_alpha=args.lora_alpha,
         lora_dropout=args.lora_dropout,
         task_type=args.task_type,
         bias=args.bias,
         target_modules=args.target_modules,
         learning_rate=args.learning_rate,
         llm_tune_conf=args.llm_tune_conf,
         nn_gen_conf=args.nn_gen_conf,
         nn_gen_conf_id=args.nn_gen_conf_id,
         llm_conf=args.llm_conf,
         test_nn=args.test_nn,
         per_device_train_batch_size=args.per_device_train_batch_size,
         gradient_accumulation_steps=args.gradient_accumulation_steps,
         warmup_ratio=args.warmup_ratio,
         logging_steps=args.logging_steps,
         optimizer=args.optimizer,
         peft=args.peft,
         skip_epoches=args.skip_epoches,
         max_prompts=args.max_prompts,
         max_new_tokens=args.max_new_tokens,
         use_deepspeed=args.use_deepspeed,
         save_llm_output=args.save_llm_output,
         nn_name_prefix=args.nn_name_prefix,
         nn_train_epochs=args.nn_train_epochs,
         temperature=args.temperature,
         top_k=args.top_k,
         top_p=args.top_p,
         test_metric=args.test_metric,
         data_dir=args.data_dir,
         # Pipeline overrides (optional)
         evaluation_strategy=args.evaluation_strategy,
         eval_steps=args.eval_steps,
         per_device_eval_batch_size=args.per_device_eval_batch_size,
         save_strategy=args.save_strategy,
         save_steps=args.save_steps,
         save_total_limit=args.save_total_limit,
         load_best_model_at_end=args.load_best_model_at_end,
         metric_for_best_model=args.metric_for_best_model,
         warmup_steps=args.warmup_steps,
         weight_decay=args.weight_decay,
         )
