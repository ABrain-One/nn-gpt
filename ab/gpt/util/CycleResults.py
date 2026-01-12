import json
import os


def generate_cycle_results(cycle, models_base_dir, eval_results_list, model_dirs_list, successful_models, 
                            failed_models, cycle_time_minutes, current_alter_epoch_path):
    """
    Generate cycle-level results JSON with aggregated metrics.
    
    Args:
        cycle: Cycle number (finetuning iteration, separate from epoch)
        models_base_dir: Base directory containing generated models
        eval_results_list: List of successful evaluation results
        model_dirs_list: List of all model directories
        successful_models: List of successfully evaluated model directories
        failed_models: List of failed model directories
        cycle_time_minutes: Total cycle time in minutes
        current_alter_epoch_path: Path to the current epoch directory
    
    Returns:
        Dictionary with cycle results in the specified JSON format
    """
    # Calculate evaluation metrics
    models_trained = len(successful_models)
    accuracies = [r['accuracy'] for r in eval_results_list if r['accuracy'] is not None]
    best_accuracy = max(accuracies) if accuracies else None
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else None
    total_attempted = len(model_dirs_list)
    success_rate = models_trained / total_attempted if total_attempted > 0 else 0.0
    
    # Calculate generation metrics
    total_generated = len(model_dirs_list)
    successful = len(successful_models)
    # Novel models are those that were successfully evaluated (not duplicates)
    novel = successful
    
    # Calculate training metrics (set to None - not available from evaluation context)
    training_data_dir = None
    total_examples = None
    new_examples_added = None
    training_time_minutes = None
    
    # Determine success
    success = models_trained > 0
    
    # Build results dictionary
    results = {
        "cycle": cycle,
        "success": success,
        "training": {
            "data_dir": training_data_dir if training_data_dir else None,
            "total_examples": total_examples,
            "new_examples_added": new_examples_added,
            "training_time_minutes": training_time_minutes
        },
        "generation": {
            "total_generated": total_generated,
            "successful": successful,
            "novel": novel
        },
        "evaluation": {
            "models_trained": models_trained,
            "best_accuracy": best_accuracy,
            "avg_accuracy": avg_accuracy,
            "success_rate": success_rate
        },
        "cycle_time_minutes": cycle_time_minutes
    }
    
    return results


def collect_cycle_metrics(models_base_dir, current_alter_epoch_path):
    """
    Collect metrics from a cycle by scanning model directories and existing eval_info.json files.
    
    Args:
        models_base_dir: Base directory containing generated models
        current_alter_epoch_path: Path to the current epoch directory
    
    Returns:
        Tuple of (eval_results_list, model_dirs_list, successful_models, failed_models)
    """
    eval_results_list = []
    model_dirs_list = []
    successful_models = []
    failed_models = []
    existing_eval_files = {}
    
    if not models_base_dir.exists():
        return eval_results_list, model_dirs_list, successful_models, failed_models
    
    for model_id in os.listdir(models_base_dir):
        model_dir_path = models_base_dir / model_id
        if not model_dir_path.is_dir():
            continue
        
        # Track all model directories for generation metrics
        model_dirs_list.append(model_dir_path)
        
        eval_info_path = model_dir_path / 'eval_info.json'
        error_path = model_dir_path / 'error.txt'
        
        # Check for existing evaluation results
        if eval_info_path.exists():
            try:
                with open(eval_info_path, 'r') as f:
                    existing_eval_data = json.load(f)
                    existing_eval_files[model_id] = existing_eval_data
                    # If we have existing successful eval, track it
                    if 'eval_results' in existing_eval_data:
                        eval_res = existing_eval_data['eval_results']
                        if isinstance(eval_res, (tuple, list)) and len(eval_res) >= 2:
                            eval_results_list.append({
                                'model_id': model_id,
                                'model_dir': str(model_dir_path),
                                'accuracy': eval_res[1] if len(eval_res) > 1 else None,
                                'eval_results': eval_res
                            })
                            successful_models.append(model_dir_path)
            except:
                pass
        
        # Check for errors
        if error_path.exists() and model_id not in existing_eval_files:
            failed_models.append(model_dir_path)
    
    return eval_results_list, model_dirs_list, successful_models, failed_models


def save_cycle_results(cycle_results, output_path):
    """
    Save cycle results to a JSON file.
    
    Args:
        cycle_results: Dictionary with cycle results
        output_path: Path where to save the JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(cycle_results, f, indent=2, default=str)

