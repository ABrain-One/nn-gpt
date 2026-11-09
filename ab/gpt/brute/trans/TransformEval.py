import os
import json
import shutil
from pathlib import Path
import ab.nn.api as api

from ab.gpt.util.Const import nngpt_dir, epoch_dir, synth_dir, transformer_file, trans_dir


# DEFAULT CONFIGURATION

RESULT_DIR = trans_dir/'test'
TRANSFORM_DIR = trans_dir/'test'

TASK = "img-classification"
DATASET = "cifar-10"
METRIC = "acc"
MODEL_FILTER = "ResNet"
THRESHOLD= 0.40

#FT_MODE = True  #'False' for evaluation of transforms in a single folder
 


# HYPERPARAMETERS
DEFAULTS = {
    'lr': 0.01,
    'batch': 64,
    'dropout': 0.2,
    'momentum': 0.9,
    'epoch': 1
}


def get_best_model_from_db():
    """
    Get best ResNet model from DB for CIFAR-10
    """
    df = api.data(
        only_best_accuracy=True,
        task=TASK,
        dataset=DATASET,
        metric=METRIC,
        nn=MODEL_FILTER,
        max_rows=1
    )
    
    if not df.empty:
        best_model = df.iloc[0].to_dict()
        print(f"Found best model: {best_model['nn']} with accuracy: {best_model['accuracy']}")
        return best_model
    else:
        raise ValueError("No matching models found in database")



def get_candidates(epoch_num= None, FT_MODE= False): 
    """
    Scan for transform files based on the mode
    """

    candidates = []
    
    if not FT_MODE:
        # Look for .py files directly
      
        search_dir = TRANSFORM_DIR

        print(f"Scanning in: {search_dir}")
        for entry in search_dir.glob("*.py"):
            candidates.append({
                'name': entry.stem,             # Filename is the ID
                'transform_param': entry.stem,  # API expects filename
                'context_dir': search_dir,      # API looks here
                'source_file': entry
            })

    else:
        # # Iterate through all subdirs in the path
        base_input_dir = synth_dir(epoch_dir(epoch_num)) 
        
        if not base_input_dir.exists():
            raise ValueError(f"Directory {base_input_dir} does not exist")
            
        for entry in os.scandir(base_input_dir):
            if entry.is_dir():
                tr_file = Path(entry.path) / "tr.py" 
                if tr_file.exists():
                    candidates.append({
                        'name': entry.name,      # Dir name is the ID
                        'transform_param': 'tr', # Fixed param name
                        'context_dir': Path(entry.path), # API looks inside subdir
                        'source_file': tr_file
                    })
    
    return candidates




def run_eval(epoch_num= None, FT_MODE= False):
    
    # Get model and candidates
    best_model = get_best_model_from_db()
    candidates = get_candidates(epoch_num, FT_MODE)
    
    print(f"Found {len(candidates)} transforms to evaluate.")

    for i, cand in enumerate(candidates, 1):
        print(f"\n[{i}/{len(candidates)}] Evaluating: {cand['name']}")
        
        # Parameters
        prm = DEFAULTS.copy()
        prm.update({
            'transform': cand['transform_param']
        })

        context_dir= Path(cand['context_dir'])
        try:
            # Run evaluation
            result = api.check_nn(
                nn_code=best_model['nn_code'],
                task=TASK,
                dataset=DATASET,
                metric=METRIC,
                prm=prm,
                save_to_db=False,
                prefix=f"{best_model['nn']}_eval_{cand['name']}",
                save_path=context_dir,
                transform_dir=context_dir
            )

            if result:
                _, accuracy, time_metric, _ = result
                print(f"  Result: Acc={accuracy}, Time={time_metric}")

                # Save JSON result
                result_data = {
                    "accuracy": float(accuracy),
                    "batch": prm['batch'],
                    "duration": prm['duration'],
                    "lr": prm['lr'],
                    "momentum": prm['momentum'],
                    "transform": cand['name'],
                    "uid": prm['uid']
                
                }
                
                json_path = context_dir/f"{cand['name']}.json"
                with open(json_path, 'w') as f:
                    json.dump(result_data, f, indent=2)

                if FT_MODE:
                    if THRESHOLD > 0 and float(accuracy) >= THRESHOLD:
                        print(f"  [SUCCESS] Accuracy >= {THRESHOLD}")
                        # Copy .py file
                        shutil.copy(cand['source_file'], TRANSFORM_DIR/f"A{epoch_num}{cand['name']}.py")
                        # Copy .json file
                        shutil.copy(json_path, RESULT_DIR/f"A{epoch_num}{cand['name']}.json")

        except Exception as e:
            print(f"  Error: {e}")
            error_data = {"accuracy": 0.0, "transform": cand['name'], "error": str(e)}
            with open(context_dir / f"{cand['name']}.json", 'w') as f:
                json.dump(error_data, f, indent=2)


if __name__ == "__main__":
    #run_eval()
    run_eval(epoch_num=0, FT_MODE=True)