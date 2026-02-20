"""
Generator Agent - Integrated into TuneNNGen pipeline.

Calls lower-level functions (nn_gen) instead of full pipeline.
Truly integrated, not separate.

Location: ab/gpt/agents/generator.py (inside pipeline codebase)
"""

from typing import Dict, Any
from pathlib import Path
import json

from ab.gpt.agents.state import AgentState
from ab.gpt.util.Tune import nn_gen, load_llm_and_chatbot, load_prompt_config, read_eval_info
from ab.gpt.util.Const import conf_test_dir, conf_llm_dir, epoch_dir, synth_dir
from ab.gpt.util.Util import extract_code, extract_hyperparam
from ab.gpt.util.Code import improve_code
import ast

# Lazy import for database
try:
    import ab.nn.api as lemur
    from ab.nn.util.Util import uuid4, read_py_file_as_string
    LEMUR_AVAILABLE = True
except ImportError:
    LEMUR_AVAILABLE = False
    lemur = None


def generator_node(state: AgentState) -> Dict[str, Any]:
    """
    Generator agent - integrated into pipeline.
    
    Calls nn_gen() directly (not TuneNNGen.main()) to generate ONE model.
    This is truly integrated - uses internal pipeline functions.
    
    Args:
        state: Current workflow state with TuneNNGen parameters
        
    Returns:
        Updated state with model_code, metrics, and GPU released
    """
    
    print("🤖 Generator: Starting generation (integrated into pipeline)...")
    
    try:
        # ============ EXTRACT PARAMETERS FROM STATE ============
        nn_train_epochs = state.get('nn_train_epochs', 2)
        nn_gen_conf_id = state.get('nn_gen_conf_id', 'improve_classification_only')
        temperature = state.get('temperature', 0.8)
        top_k = state.get('top_k', 70)
        top_p = state.get('top_p', 0.9)
        max_new_tokens = state.get('max_new_tokens', 12 * 1024)
        save_llm_output = state.get('save_llm_output', True)
        base_out_dir_str = state.get('base_out_dir', './outputs')
        # Handle both Path and string
        base_out_dir = Path(base_out_dir_str) if isinstance(base_out_dir_str, str) else base_out_dir_str
        llm_conf = state.get('llm_conf', 'nngpt_ds_coder_1.3b_instruct.json')
        nn_gen_conf = state.get('nn_gen_conf', 'NN_gen.json')
        experiment_id = state.get('experiment_id', 'exp_default')
        
        # ============ SETUP OUTPUT DIRECTORY ============
        # base_out_dir is already nngpt_dir (out/nngpt), so just add llm/epoch/A0
        epoch = 0
        out_path = base_out_dir / 'llm' / 'epoch' / f'A{epoch}'
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Debug: print the actual path being used
        print(f"📁 Generator: base_out_dir: {base_out_dir}")
        print(f"📁 Generator: out_path: {out_path}")
        
        print(f"📁 Generator: Output path: {out_path}")
        
        # ============ LOAD LLM AND CHATBOT ============
        # Use extracted function from Tune.py to avoid code duplication
        print("📦 Generator: Loading LLM and tokenizer...")
        model, tokenizer, chat_bot, model_loader = load_llm_and_chatbot(
            llm_conf,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            llm_path=None,  # No LoRA path for agent workflow
            use_deepspeed=None,  # Read from config
            context_length=None,  # Read from config
            access_token=None  # Read from file if needed
        )
        print("✅ Generator: LLM and ChatBot loaded")
        
        # ============ LOAD PROMPT CONFIG ============
        # Use extracted function from Tune.py to avoid code duplication
        prompt_dict = load_prompt_config(nn_gen_conf)
        conf_keys = (nn_gen_conf_id,)  # Single config key
        
        # ============ CALL nn_gen() DIRECTLY ============
        print(f"📞 Generator: Calling nn_gen() with test_nn=1 (integrated)...")
        
        nn_gen(
            epoch=epoch,
            out_path=out_path,
            chat_bot=chat_bot,
            conf_keys=conf_keys,
            nn_train_epochs=nn_train_epochs,
            prompt_dict=prompt_dict,
            test_nn=1,  # Generate only ONE model (not 10!)
            max_new_tokens=max_new_tokens,
            save_llm_output=save_llm_output,
            nn_name_prefix=experiment_id,
        )
        
        print("✅ Generator: nn_gen() completed!")
        
        # ============ READ RESULTS ============
        output_dir = synth_dir(out_path) / 'B0'  # First (and only) model
        
        if not output_dir.exists():
            raise FileNotFoundError(f"Output directory not found: {output_dir}")
        
        # Read model code (with fallback to full_output.txt)
        model_file = output_dir / 'new_nn.py'
        full_output_file = output_dir / 'full_output.txt'
        
        # Debug: Check what files exist
        print(f"📁 Generator: Checking output_dir: {output_dir}")
        print(f"📁 Generator: output_dir exists: {output_dir.exists()}")
        if output_dir.exists():
            print(f"📁 Generator: Files in output_dir: {[f.name for f in output_dir.iterdir()]}")
        
        model_code = None
        
        # Try to read from new_nn.py first
        if model_file.exists():
            try:
                model_code = model_file.read_text(encoding='utf-8')
                if model_code and model_code.strip():
                    # Validate syntax - if invalid, extract from full_output.txt
                    try:
                        ast.parse(model_code)
                        print(f"📄 Generator: Model code read from new_nn.py ({len(model_code)} chars)")
                    except SyntaxError as e:
                        print(f"⚠️ Generator: new_nn.py has syntax errors: {e}")
                        print(f"⚠️ Generator: Will extract from full_output.txt instead...")
                        model_code = None  # Reset to try fallback extraction
                    except Exception as e:
                        print(f"⚠️ Generator: Error validating new_nn.py syntax: {e}")
                        model_code = None  # Reset to try fallback
                else:
                    print(f"⚠️ Generator: new_nn.py exists but is empty ({len(model_code) if model_code else 0} chars)")
                    model_code = None  # Reset to try fallback
            except Exception as e:
                print(f"⚠️ Generator: Error reading new_nn.py: {e}")
                model_code = None
        
        # Fallback: extract from full_output.txt if new_nn.py is missing/empty
        if not model_code and full_output_file.exists():
            print("⚠️ Generator: Extracting model code from full_output.txt...")
            try:
                full_output = full_output_file.read_text(encoding='utf-8')
                print(f"📄 Generator: full_output.txt size: {len(full_output)} chars")
                
                # Try different extraction methods
                model_code = extract_code(full_output)
                
                # Check if standard extraction worked and is reasonable
                if not model_code or not model_code.strip() or len(model_code.strip()) < 200:
                    # Try alternative extraction: look for class Net or import torch
                    print("⚠️ Generator: Standard extraction failed or too short, trying alternative methods...")
                    
                    # Method 1: Find 'import torch' as start marker (most reliable)
                    if 'import torch' in full_output:
                        start_idx = full_output.find('import torch')
                        # Find end - look for common end markers, but search far enough
                        end_markers = ['\n\n```', '\n\n</nn>', '\n\n</', '\n\n# Explanation', '\n\nNote:', '\n\nThis', '\n\n---', '\n\n----']
                        end_idx = len(full_output)
                        # Search further out (at least 3000 chars) to get full code
                        for marker in end_markers:
                            marker_idx = full_output.find(marker, start_idx + 1000)
                            if marker_idx > start_idx and marker_idx < end_idx:
                                end_idx = marker_idx
                        
                        potential_code = full_output[start_idx:end_idx].strip()
                        # Clean up - remove trailing empty lines and common suffixes
                        potential_code = potential_code.rstrip()
                        # Remove trailing markers if present
                        for suffix in ['```', '----', '</nn>', '</>']:
                            if potential_code.endswith(suffix):
                                potential_code = potential_code[:-len(suffix)].strip()
                        
                        if len(potential_code) > 300:  # Reasonable code length
                            model_code = potential_code
                            print(f"✅ Generator: Extracted code using 'import torch' marker ({len(model_code)} chars)")
                    
                    # Method 2: Look for 'class Net' if Method 1 failed
                    if (not model_code or len(model_code.strip()) < 200) and 'class Net' in full_output:
                        start_idx = full_output.find('class Net')
                        # Find end - look for end markers
                        end_markers = ['\n\n```', '\n\n</nn>', '\n\n</', '\n\n#', '\n\nNote:', '\n\nThis', '\n\n----']
                        end_idx = len(full_output)
                        for marker in end_markers:
                            marker_idx = full_output.find(marker, start_idx + 1000)
                            if marker_idx > start_idx and marker_idx < end_idx:
                                end_idx = marker_idx
                        
                        # Include imports before class Net if they exist nearby
                        # Look backwards for import statements
                        import_start = full_output.rfind('import', max(0, start_idx - 2000), start_idx)
                        if import_start > 0 and (start_idx - import_start) < 2000:
                            start_idx = import_start
                        
                        potential_code = full_output[start_idx:end_idx].strip()
                        potential_code = potential_code.rstrip()
                        # Remove trailing markers
                        for suffix in ['```', '----', '</nn>', '</>']:
                            if potential_code.endswith(suffix):
                                potential_code = potential_code[:-len(suffix)].strip()
                        
                        if len(potential_code) > 300:
                            model_code = potential_code
                            print(f"✅ Generator: Extracted code using 'class Net' marker ({len(model_code)} chars)")
                
                if model_code and model_code.strip():
                    # Clean up code using improve_code() (same as extract_code uses)
                    cleaned_code = improve_code(model_code)
                    if cleaned_code and cleaned_code.strip() and len(cleaned_code) > 100:
                        model_code = cleaned_code
                        print(f"✅ Generator: Code cleaned using improve_code() ({len(model_code)} chars)")
                    else:
                        print(f"⚠️ Generator: improve_code() didn't help, using raw extraction")
                    
                    # Validate syntax before saving
                    try:
                        ast.parse(model_code)
                        print(f"✅ Generator: Code syntax validated")
                    except SyntaxError as e:
                        print(f"⚠️ Generator: Code has syntax errors: {e}")
                        print(f"⚠️ Generator: Attempting basic fixes...")
                        # Try basic fixes: remove leading non-Python text
                        lines = model_code.split('\n')
                        # Find first line that looks like Python (import or class or def)
                        start_idx = 0
                        for i, line in enumerate(lines):
                            stripped = line.strip()
                            if stripped.startswith('import ') or stripped.startswith('from ') or stripped.startswith('class ') or stripped.startswith('def '):
                                start_idx = i
                                break
                        if start_idx > 0:
                            model_code = '\n'.join(lines[start_idx:])
                            print(f"⚠️ Generator: Removed {start_idx} leading non-Python lines")
                            try:
                                ast.parse(model_code)
                                print(f"✅ Generator: Code fixed and validated")
                            except:
                                print(f"⚠️ Generator: Could not fix syntax errors, saving anyway")
                    
                    # Only save if code is reasonably long
                    if len(model_code.strip()) < 200:
                        print(f"⚠️ Generator: Extracted code too short ({len(model_code)} chars), may be invalid")
                    
                    # Save extracted code to new_nn.py
                    model_file.parent.mkdir(parents=True, exist_ok=True)
                    model_file.write_text(model_code, encoding='utf-8')
                    print(f"✅ Generator: Extracted and saved model code to new_nn.py ({len(model_code)} chars)")
                else:
                    # Last resort: save first 10000 chars for debugging
                    print(f"❌ Generator: Could not extract code. Showing first 500 chars of full_output:")
                    print(full_output[:500])
                    raise FileNotFoundError(
                        f"Could not extract model code from full_output.txt. "
                        f"File exists: {full_output_file.exists()}, "
                        f"Size: {len(full_output) if full_output_file.exists() else 0} chars"
                    )
            except Exception as e:
                print(f"❌ Generator: Error extracting from full_output.txt: {e}")
                raise
        
        # Final check
        if not model_code or not model_code.strip():
            raise FileNotFoundError(
                f"Model code is empty. "
                f"new_nn.py exists: {model_file.exists()}, "
                f"full_output.txt exists: {full_output_file.exists() if full_output_file else False}"
            )
        
        # Read eval_info.json using extracted function from Tune.py
        eval_file = output_dir / 'eval_info.json'
        eval_info = read_eval_info(output_dir)
        if not eval_info:
            # Try to get from database
            print("⚠️ Generator: eval_info.json not found, querying database...")
            eval_info = _get_metrics_from_database(model_code, output_dir)
        
        print(f"📊 Generator: Eval info loaded")
        
        # ============ EXTRACT DATA ============
        eval_results = eval_info.get('eval_results', [])
        eval_args = eval_info.get('eval_args', {})
        cli_args = eval_info.get('cli_args', {})
        
        # Parse eval_results: [nn_name, accuracy, acc_to_time, score]
        nn_name = eval_results[0] if len(eval_results) > 0 else None
        accuracy = eval_results[1] if len(eval_results) > 1 else None
        accuracy_to_time = eval_results[2] if len(eval_results) > 2 else None
        score = eval_results[3] if len(eval_results) > 3 else None
        
        print(f"✅ Generator: Extracted - NN: {nn_name}, Accuracy: {accuracy}")
        
        # ============ QUERY DATABASE FOR EPOCH ACCURACIES ============
        epoch_1_accuracy, epoch_2_accuracy = _get_epoch_accuracies(model_code)
        
        if epoch_1_accuracy is None or epoch_2_accuracy is None:
            print("⚠️ Generator: Per-epoch accuracies not found in database")
        
        # ============ DETERMINE STATUS ============
        # If we have model code, generation was successful
        # But if we don't have metrics, training failed/incomplete
        has_model = model_code and model_code.strip()
        has_metrics = accuracy is not None or (eval_results and len(eval_results) > 1)
        
        if has_model and has_metrics:
            status = 'success'
            error_message = None
        elif has_model and not has_metrics:
            status = 'partial_success'  # Model generated but training failed
            error_message = 'Model generated successfully but training failed or timed out. No metrics available.'
        else:
            status = 'error'
            error_message = 'Model code could not be extracted'
        
        # ============ UPDATE STATE ============
        return {
            # Model and results
            'model_code': model_code,
            'eval_results': eval_results if eval_results else None,
            'eval_args': eval_args if eval_args else None,
            'cli_args': cli_args if cli_args else None,
            
            # Extracted fields
            'nn_name': nn_name,
            'accuracy': accuracy,
            'accuracy_to_time': accuracy_to_time,
            'score': score,
            'hyperparameters': eval_args.get('prm', {}) if eval_args else {},
            'task': cli_args.get('task') if cli_args else None,
            'dataset': cli_args.get('dataset') if cli_args else None,
            'metric': cli_args.get('metric') if cli_args else None,
            
            # Epoch accuracies (CRITICAL for Predictor)
            'epoch_1_accuracy': epoch_1_accuracy,
            'epoch_2_accuracy': epoch_2_accuracy,
            
            # File paths
            'model_file_path': str(model_file),
            'eval_file_path': str(eval_file) if eval_file.exists() else None,
            
            # Status
            'status': status,
            'gpu_available': True,  # Release GPU
            'error_message': error_message,
            'metrics_source': 'eval_info.json' if eval_file.exists() else ('database' if eval_info else 'none'),
        }
        
    except Exception as e:
        print(f"❌ Generator: Error - {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'error',
            'error_message': str(e),
            'gpu_available': True,  # Release GPU on error
        }


def _get_epoch_accuracies(model_code: str) -> tuple:
    """Query database for epoch 1 and 2 accuracies (CRITICAL for Predictor)"""
    if not LEMUR_AVAILABLE:
        return None, None
    
    try:
        checksum = uuid4(model_code)
        df = lemur.data()
        matches = df[df['nn_id'] == checksum]
        
        if matches.empty:
            print(f"⚠️ Generator: No database entries found for checksum: {checksum}")
            return None, None
        
        epoch_1_data = matches[matches['epoch'] == 1]
        epoch_2_data = matches[matches['epoch'] == 2]
        
        epoch_1_acc = float(epoch_1_data['accuracy'].iloc[0]) if not epoch_1_data.empty else None
        epoch_2_acc = float(epoch_2_data['accuracy'].iloc[0]) if not epoch_2_data.empty else None
        
        if epoch_1_acc is not None and epoch_2_acc is not None:
            print(f"✅ Generator: Found epoch accuracies - Epoch 1: {epoch_1_acc}, Epoch 2: {epoch_2_acc}")
        
        return epoch_1_acc, epoch_2_acc
    except Exception as e:
        print(f"⚠️ Generator: Database query failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def _get_metrics_from_database(model_code: str, model_dir: Path) -> Dict[str, Any]:
    """Get metrics from database if eval_info.json doesn't exist"""
    if not LEMUR_AVAILABLE:
        return {}
    
    try:
        checksum = uuid4(model_code)
        df = lemur.data()
        matches = df[df['nn_id'] == checksum]
        
        if matches.empty:
            print(f"⚠️ Generator: No database entries found for checksum: {checksum}")
            return {}
        
        # Get best accuracy
        best_row = matches.loc[matches['accuracy'].idxmax()]
        
        duration = best_row.get('duration', 1.0)
        accuracy = float(best_row.get('accuracy', 0))
        
        return {
            "eval_args": {
                "task": best_row.get('task'),
                "dataset": best_row.get('dataset'),
                "metric": best_row.get('metric'),
                "prm": best_row.get('prm', {}),
            },
            "eval_results": [
                best_row.get('nn', checksum),
                accuracy,
                accuracy / duration if duration > 0 else accuracy,
                accuracy,
            ],
            "cli_args": {
                "task": best_row.get('task'),
                "dataset": best_row.get('dataset'),
                "metric": best_row.get('metric'),
            },
            "source": "database",
        }
    except Exception as e:
        print(f"⚠️ Generator: Database query failed: {e}")
        import traceback
        traceback.print_exc()
        return {}

