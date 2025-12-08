import argparse
from typing import Literal

from peft import LoraConfig
from transformers import TrainingArguments

from ab.gpt.NNEval import NN_TRAIN_EPOCHS
from ab.gpt.util.Const import nngpt_dir, new_out_file
from ab.gpt.util.Tune import tune, ds_conf

# --- Default Evaluation Parameters ---
# These will be used as defaults for argparse arguments
START_LAYER = 0
END_LAYER = 24
TUNE_LAYERS = range(START_LAYER, END_LAYER)
R = 32  # dimension of the updated matrices
LORA_ALPHA = 32  # parameter for scaling
LORA_DROPOUT = 0.05  # dropout probability for layers
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
TASK_TYPE = "CAUSAL_LM"
BiasType = Literal["none", "all", "lora_only"]
BIAS: BiasType = "none"

LEARNING_RATE = 1e-6 # 1e-5

PEFT = None
SKIP_EPOCHES = -1

NUM_TRAIN_EPOCHS = 3
LR_SCHEDULER = 'cosine'
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
WARMUP_RATIO = 0.05
TEST_NN = 10
LOGGING_STEPS = 128
MAX_GRAD_NORM = 1.0
OPTIMIZER = 'paged_adamw_8bit'  # 'adamw_torch'
LLM_TUNE_CONF = 'NN_gen.json'
NN_GEN_CONF = 'NN_gen.json'
NN_GEN_CONF_ID = 'improve_classification_only'
LLM_CONF = 'ds_coder_7b_olympic.json'
MAX_PROMPTS = 4 * 1024
MAX_NEW_TOKENS = 16 * 1024
SAVE_LLM_OUTPUT = True
USE_DEEPSPEED = False
NN_NAME_PREFIX = None
TEMPERATURE = 0.8
TOP_K = 70
TOP_P = 0.9
TEST_METRIC = None  # 'bleu'

# --- LangGraph Agent Flags ---
USE_AGENTS = False
USE_PREDICTOR = False


def main(num_train_epochs=NUM_TRAIN_EPOCHS, lr_scheduler=LR_SCHEDULER, max_grad_norm=MAX_GRAD_NORM, test_metric=TEST_METRIC,
         tune_layers=TUNE_LAYERS, r=R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, target_modules=TARGET_MODULES,
         task_type=TASK_TYPE, bias=BIAS, learning_rate=LEARNING_RATE, llm_tune_conf=LLM_TUNE_CONF, nn_gen_conf=NN_GEN_CONF, nn_gen_conf_id=NN_GEN_CONF_ID,
         llm_conf=LLM_CONF, test_nn=TEST_NN, peft=PEFT, skip_epoches=SKIP_EPOCHES, per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
         gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, warmup_ratio=WARMUP_RATIO, logging_steps=LOGGING_STEPS, optimizer=OPTIMIZER,
         max_prompts=MAX_PROMPTS, save_llm_output=SAVE_LLM_OUTPUT, max_new_tokens=MAX_NEW_TOKENS, use_deepspeed=USE_DEEPSPEED, nn_name_prefix=NN_NAME_PREFIX,
         nn_train_epochs=NN_TRAIN_EPOCHS, temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P, use_agents=USE_AGENTS, use_predictor=USE_PREDICTOR):
    """
    TuneNNGen main function with optional LangGraph multi-agent orchestration.
    
    Args:
        use_agents: If True, use LangGraph workflow with agents. If False, use traditional pipeline.
        use_predictor: If True (and use_agents=True), enable predictor agent.
        ... (all other parameters same as before)
    
    Returns:
        If use_agents=True: Final state dictionary
        If use_agents=False: None (existing behavior)
    """
    
    # ============================================================
    # LANGGRAPH WORKFLOW (when use_agents=True)
    # ============================================================
    if use_agents:
        try:
            from langgraph.graph import StateGraph, END
            from ab.gpt.agents.state import AgentState
            from ab.gpt.agents.manager import manager_node
            from ab.gpt.agents.generator import generator_node
            
            # Import predictor only if enabled
            if use_predictor:
                from ab.gpt.agents.predictor import predictor_node
            
            print("=" * 60)
            print("🚀 Starting LangGraph Multi-Agent Workflow")
            print("=" * 60)
            print(f"  use_agents: {use_agents}")
            print(f"  use_predictor: {use_predictor}")
            print("=" * 60)
            
            # ============================================================
            # CREATE STATEGRAPH WORKFLOW
            # ============================================================
            workflow = StateGraph(AgentState)
            
            # Add nodes
            workflow.add_node("manager", manager_node)
            workflow.add_node("generator", generator_node)
            if use_predictor:
                workflow.add_node("predictor", predictor_node)
            
            # Set entry point
            workflow.set_entry_point("manager")
            
            # ============================================================
            # CONDITIONAL ROUTING FUNCTION
            # ============================================================
            def should_continue(state: AgentState) -> str:
                """Route based on manager's next_action decision"""
                return state.get('next_action', 'end')
            
            # ============================================================
            # DEFINE CONDITIONAL EDGES
            # ============================================================
            edges = {
                "generate": "generator",
                "end": END,
            }
            if use_predictor:
                edges["predict"] = "predictor"
            
            workflow.add_conditional_edges(
                "manager",
                should_continue,
                edges
            )
            
            # ============================================================
            # AFTER AGENTS, RETURN TO MANAGER
            # ============================================================
            workflow.add_edge("generator", "manager")
            if use_predictor:
                workflow.add_edge("predictor", "manager")
            
            # ============================================================
            # COMPILE WORKFLOW
            # ============================================================
            app = workflow.compile()
            
            # ============================================================
            # INITIALIZE STATE FROM PARAMETERS
            # ============================================================
            initial_state: AgentState = {
                # Experiment metadata
                "experiment_id": nn_name_prefix or 'exp_default',
                "base_out_dir": str(nngpt_dir),
                
                # TuneNNGen parameters
                "num_train_epochs": num_train_epochs,
                "nn_train_epochs": nn_train_epochs,
                "nn_gen_conf_id": nn_gen_conf_id,
                "temperature": temperature,
                "top_k": int(top_k) if isinstance(top_k, (int, str)) else top_k,
                "top_p": float(top_p) if isinstance(top_p, (int, str, float)) else top_p,
                "max_new_tokens": max_new_tokens,
                "save_llm_output": save_llm_output,
                
                # Additional parameters needed by generator
                "llm_conf": llm_conf,
                "nn_gen_conf": nn_gen_conf,
                
                # Workflow control
                "use_predictor": use_predictor,
                "gpu_available": True,
                "next_action": "generate",
                "status": "pending",
            }
            
            # ============================================================
            # RUN WORKFLOW
            # ============================================================
            print("\n🔄 Executing LangGraph workflow...\n")
            final_state = app.invoke(initial_state)
            
            # ============================================================
            # RETURN RESULTS
            # ============================================================
            print("\n" + "=" * 60)
            print("✅ LangGraph Workflow Completed!")
            print("=" * 60)
            print(f"  Status: {final_state.get('status', 'unknown')}")
            if final_state.get('model_code'):
                print(f"  Model generated: {final_state.get('nn_name', 'N/A')}")
                print(f"  Accuracy: {final_state.get('accuracy', 'N/A')}")
            if final_state.get('predicted_best_accuracy'):
                print(f"  Predicted accuracy: {final_state.get('predicted_best_accuracy', 'N/A')}")
                print(f"  Predicted epoch: {final_state.get('predicted_best_epoch', 'N/A')}")
            print("=" * 60)
            
            return final_state
            
        except ImportError as e:
            print(f"❌ Error: LangGraph not available. Install with: pip install langgraph")
            print(f"   Details: {e}")
            print("⚠️ Falling back to traditional pipeline...")
            # Fall through to traditional pipeline
        except Exception as e:
            print(f"❌ Error in LangGraph workflow: {e}")
            import traceback
            traceback.print_exc()
            print("⚠️ Falling back to traditional pipeline...")
            # Fall through to traditional pipeline
    
    # ============================================================
    # TRADITIONAL PIPELINE (existing behavior - unchanged)
    # ============================================================
    print(f'''All hyperparameters: 
num_train_epochs={num_train_epochs}, lr_scheduler={lr_scheduler}, max_grad_norm={max_grad_norm}, tune_layers={tune_layers}, test_metric={test_metric}, 
r={r}, lora_alpha={lora_alpha}, lora_dropout={lora_dropout}, target_modules={target_modules}, task_type={task_type}, bias={bias}, 
learning_rate={learning_rate}, llm_tune_conf={llm_tune_conf}, nn_gen_conf={nn_gen_conf}, nn_gen_conf_id={nn_gen_conf_id},
llm_conf={llm_conf}, test_nn={test_nn}, nn_train_epochs={nn_train_epochs}, peft={peft}, skip_epoches={skip_epoches}, 
per_device_train_batch_size={per_device_train_batch_size}, gradient_accumulation_steps={gradient_accumulation_steps}, warmup_ratio={warmup_ratio}, 
logging_steps={logging_steps}, optimizer={optimizer}, max_prompts={max_prompts}, save_llm_output={save_llm_output}, max_new_tokens={max_new_tokens}, 
use_deepspeed={use_deepspeed}, nn_name_prefix={nn_name_prefix}, temperature={temperature}, top_k={top_k}, top_p={top_p} ''')
    test_prm = {
        'metric_for_best_model': test_metric,
        'greater_is_better': True,

        'eval_strategy': "epoch",
        'save_strategy': 'epoch',
        'save_total_limit': 3,
        'load_best_model_at_end': False} if test_metric else {}

    training_args = TrainingArguments(
        num_train_epochs=num_train_epochs,
        lr_scheduler_type=lr_scheduler,
        max_grad_norm=max_grad_norm,
        report_to=None,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=logging_steps,
        output_dir=nngpt_dir / 'outputs',
        optim=optimizer,
        deepspeed=ds_conf if use_deepspeed else None,
        gradient_checkpointing=True,
        **test_prm)

    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        layers_to_transform=list(tune_layers),
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type)

    tune(test_nn, nn_train_epochs, skip_epoches, peft, llm_tune_conf, nn_gen_conf, nn_gen_conf_id, llm_conf, training_args, peft_config,
         max_prompts=max_prompts, save_llm_output=save_llm_output, max_new_tokens=max_new_tokens, nn_name_prefix=nn_name_prefix,
         temperature=temperature, top_k=top_k, top_p=top_p, test_metric=test_metric)


if __name__ == '__main__':
    TARGET_MODULES_STR = ','.join(TARGET_MODULES)
    parser = argparse.ArgumentParser(description="Evaluate Neural Networks generated by NNAlter.py.")
    parser.add_argument('-ne', '--num_train_epochs', type=int, default=NUM_TRAIN_EPOCHS,
                        help=f"Number of LLM fine-tuning epochs (default: {NUM_TRAIN_EPOCHS}).")
    parser.add_argument('-ls', '--lr_scheduler', type=str, default=LR_SCHEDULER,
                        help=f"Name of learning rate scheduler for LLM fine-tuning (default: {LR_SCHEDULER}).")
    parser.add_argument('-g', '--max_grad_norm', type=float, default=MAX_GRAD_NORM,
                        help=f"Upper limit on the  backpropagation gradients for LLM fine-tuning (default: {MAX_GRAD_NORM}).")
    parser.add_argument('-s', '--start_layer', type=int, default=START_LAYER,
                        help=f"Index of the first fine-tuned layer in the LLM (default: {START_LAYER}).")
    parser.add_argument('-e', '--end_layer', type=int, default=END_LAYER,
                        help=f"Index of the last fine-tuned layer in the LLM (default: {END_LAYER}).")
    parser.add_argument('-r', '--r', type=int, default=R,
                        help=f"Dimension of the updated matrices (default: {R}).")
    parser.add_argument('-a', '--lora_alpha', type=float, default=LORA_ALPHA,
                        help=f"LoRA alpha parameter for scaling (default: {LORA_ALPHA}).")
    parser.add_argument('-d', '--lora_dropout', type=float, default=LORA_DROPOUT,
                        help=f"LoRA dropout probability for layers (default: {LORA_DROPOUT}).")
    parser.add_argument('-t', '--target_modules', type=lambda s: s.split(','), default=TARGET_MODULES,
                        help=f'Target modules separated by comma (default: {TARGET_MODULES_STR})')
    parser.add_argument('-l', '--learning_rate', type=float, default=LEARNING_RATE,
                        help=f"Learning rate (default: {LEARNING_RATE}).")
    parser.add_argument('-y', '--task_type', type=str, default=TASK_TYPE,
                        help=f"LLM task type (default: {TASK_TYPE}).")
    parser.add_argument('-b', '--bias', type=str, default=BIAS,
                        help=f"Bias type (default: {BIAS}).")
    parser.add_argument('--llm_tune_conf', type=str, default=LLM_TUNE_CONF,
                        help=f"Config with a prompt for LLM fine-tuning (default: {LLM_TUNE_CONF}).")
    parser.add_argument('--nn_gen_conf', type=str, default=NN_GEN_CONF,
                        help=f"Config with a prompt for generation of neural networks by LLM (default: {NN_GEN_CONF}).")
    parser.add_argument('--nn_gen_conf_id', type=str, default=NN_GEN_CONF_ID,
                        help=f"Specifies prompt in the config for neural network generation by LLM (default: {NN_GEN_CONF_ID}).")
    parser.add_argument('--llm_conf', type=str, default=LLM_CONF,
                        help=f"Config of LLM (default: {LLM_CONF}).")
    parser.add_argument('-n', '--test_nn', type=int, default=TEST_NN,
                        help=f"Count of neural networks generated or modified by the LLM before and between fine-tuning epochs to monitor training progress (default: {TEST_NN}).")
    parser.add_argument('--nn_train_epochs', type=int, default=NN_TRAIN_EPOCHS,
                        help=f"Number of training epochs for the generated neural network (default: {NN_TRAIN_EPOCHS}).")
    parser.add_argument('-m', '--max_prompts', type=int, default=MAX_PROMPTS,
                        help=f"Max prompts for LLM fine-tuning; excess is truncated (default: {MAX_PROMPTS}).")
    parser.add_argument('--max_new_tokens', type=int, default=MAX_NEW_TOKENS,
                        help=f"Max number of tokens in LLM output (default: {MAX_NEW_TOKENS}).")
    parser.add_argument('--save_llm_output', type=bool, default=SAVE_LLM_OUTPUT,
                        help=f"Save full output of LLM in the file {new_out_file} (default: {SAVE_LLM_OUTPUT}).")
    parser.add_argument('--use_deepspeed', type=bool, default=USE_DEEPSPEED,
                        help=f"Utilize DeepSpeed optimizations for LLM fine-tuning (default: {USE_DEEPSPEED}).")
    parser.add_argument('--per_device_train_batch_size', type=int, default=PER_DEVICE_TRAIN_BATCH_SIZE,
                        help=f"Per device train batch size (default: {PER_DEVICE_TRAIN_BATCH_SIZE}).")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=GRADIENT_ACCUMULATION_STEPS,
                        help=f"Gradient accumulation steps (default: {GRADIENT_ACCUMULATION_STEPS}).")
    parser.add_argument('--warmup_ratio', type=float, default=WARMUP_RATIO,
                        help=f"Warmup step ratio for one fine-tuning epoch (default: {WARMUP_RATIO}).")
    parser.add_argument('--logging_steps', type=int, default=LOGGING_STEPS,
                        help=f"Logging steps (default: {LOGGING_STEPS}).")
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER,
                        help=f"Optimizer for LLM fine-tuning (default: {OPTIMIZER}).")
    parser.add_argument('-k', '--skip_epoches', type=int, default=SKIP_EPOCHES,
                        help='Number of epoches to skip the neural network generation.')
    parser.add_argument('--peft', type=str, default=None, help='Path to saved LoRA layers.')
    parser.add_argument('--nn_name_prefix', type=str, default=NN_NAME_PREFIX,
                        help=f"Neural network name prefix (default: {NN_NAME_PREFIX}).")
    parser.add_argument('--temperature', type=float, default=TEMPERATURE,
                        help=f"LLM temperature controls randomness in output generation (default: {TEMPERATURE}).")
    parser.add_argument('--top_k', type=str, default=TOP_K,
                        help=f"LLM top_k limits token selection in output generation (default: {TOP_K}).")
    parser.add_argument('--top_p', type=str, default=TOP_P,
                        help=f"LLM top_p controls token diversity in output generation (default: {TOP_P}).")
    parser.add_argument('--test_metric', type=str, default=TEST_METRIC,
                        help=f"Test metric for LLM fine-tuning implemented in transformers package (default: {TEST_METRIC}).")
    
    # ============================================================
    # LANGGRAPH AGENT FLAGS
    # ============================================================
    parser.add_argument('--use_agents', action='store_true', default=USE_AGENTS,
                        help='Enable LangGraph multi-agent workflow (default: False).')
    parser.add_argument('--use_predictor', action='store_true', default=USE_PREDICTOR,
                        help='Enable accuracy prediction agent (requires --use_agents) (default: False).')

    args = parser.parse_args()
    main(num_train_epochs=args.num_train_epochs,
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
         use_agents=args.use_agents,
         use_predictor=args.use_predictor,
         )
