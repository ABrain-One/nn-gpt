"""
Train wrapper that evaluates candidates on a held-out split, 
instead of ab.nn.util.Train's single test_dataset.

train_set is split once per (task, dataset, transform, transform_dir) into
train_sub / val_sub (default 90/10, seeded). Only call use_val_split=False is scored against
the real test_set.

The split is cached per (task, dataset, transform, transform_dir, val_fraction,
split_seed) so every caller across an entire search run gets back the same validation set. 
Use: check_nn_split(nn_code, task, dataset, metric, prm, use_val_split=True)

Final judgment -- computed once
use check_nn_split(nn_code, task, dataset, metric, prm, use_val_split=False)

"""

import functools
from typing import Optional, Union

import torch
from torch.utils.data import random_split

import ab.nn.util.CodeEval as codeEvaluator
import ab.nn.util.db.Write as DB_Write
from ab.nn.util.Const import default_epoch_limit_minutes
from ab.nn.util.Loader import load_dataset
from ab.nn.util.Train import Train
from ab.nn.util.Util import *     
from ab.nn.util.db.Util import *   

DEFAULT_VAL_FRACTION = 0.1
DEFAULT_SPLIT_SEED = 42


@functools.lru_cache(maxsize=64)
def _load_and_split(task: str, dataset: str, transform_name: str, transform_dir: Optional[str],
                     val_fraction: float, split_seed: int):
    """
    load train/test once per (task, dataset, transform, transform_dir) and
    cache the train_sub/val_sub split so every caller during a search run the identical validation set.
    """
    out_shape, minimum_accuracy, train_set, test_set = load_dataset(
        task, dataset, transform_name, transform_dir)

    n_val = max(1, int(round(len(train_set) * val_fraction)))
    n_train = len(train_set) - n_val
    if n_train <= 0:
        raise ValueError(
            f"val_fraction={val_fraction} leaves no training data "
            f"for {len(train_set)}-example train set")

    generator = torch.Generator().manual_seed(split_seed)
    train_sub, val_sub = random_split(train_set, [n_train, n_val], generator=generator)

    return out_shape, minimum_accuracy, test_set, train_sub, val_sub


def clear_split_cache():
    """ for a fresh split rather than reuse the cache."""
    _load_and_split.cache_clear()


def check_nn_split(
    nn_code: str,
    task: str,
    dataset: str,
    metric: str,
    prm: dict,
    use_val_split: bool,
    save_to_db: bool = False,
    prefix: Union[str, None] = None,
    save_path: Union[str, None] = None,
    export_onnx: bool = False,
    epoch_limit_minutes: int = default_epoch_limit_minutes,
    layer_analysis: bool = False,
    transform_dir: Optional[str] = None,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[str, float, float, float]:
    """
    Drop-in replacement for ab.nn.api.check_nn().

    use_val_split=True  -> train on train_sub, evaluate on val_sub.
                            
    use_val_split=False -> train on train_sub, evaluate on the real test_set. 
   
    """
    model_name = uuid4(nn_code)
    if prefix:
        model_name = prefix + "-" + model_name

    tmp_modul = ".".join((out, 'nn', 'tmp'))
    tmp_modul_name = ".".join((tmp_modul, model_name))
    tmp_dir = ab_root_path / tmp_modul.replace('.', '/')
    create_file(tmp_dir, '__init__.py')
    temp_file_path = tmp_dir / f"{model_name}.py"

    trainer = None
    accuracy, accuracy_to_time, duration = 0.0, 0.0, 0
    res = None
    try:
        with open(temp_file_path, 'w') as f:
            f.write(nn_code)
        res = codeEvaluator.evaluate_single_file(temp_file_path)

        out_shape, minimum_accuracy, test_set, train_sub, val_sub = _load_and_split(
            task, dataset, prm['transform'], transform_dir, val_fraction, split_seed)

        eval_dataset = val_sub if use_val_split else test_set
        num_workers = prm.get('num_workers', 1)

        trainer = Train(
            config=(task, dataset, metric, model_name),
            out_shape=out_shape,
            minimum_accuracy=minimum_accuracy,
            batch=prm['batch'],
            nn_module=tmp_modul_name,
            task=task,
            train_dataset=train_sub,
            test_dataset=eval_dataset,
            metric=metric,
            num_workers=num_workers,
            prm=prm,
            save_to_db=save_to_db,
            layer_analysis=layer_analysis,
            is_code=True)
        trainer.nn_code = nn_code

        epoch = prm['epoch']
        accuracy, accuracy_to_time, duration = trainer.train_n_eval(
            epoch, epoch_limit_minutes, False, export_onnx, train_sub, save_path=save_path)

        if save_to_db:
            if good(accuracy, minimum_accuracy, duration):
                model_name = DB_Write.save_nn(nn_code, task, dataset, metric, epoch, prm, force_name=model_name)
                print(f"Model saved to database with accuracy: {accuracy}")
            else:
                print(f"Model accuracy {accuracy} is below the minimum threshold {minimum_accuracy}. Not saved.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during training: {e}")
        raise
    finally:
        remove(temp_file_path)
        # Unlike ab.nn.util.Train.train_new(), do not del train_set/test_set
        try:
            if trainer:
                del trainer.model
        except AttributeError:
            pass
        release_memory()

    return model_name, accuracy, accuracy_to_time, res['score'] if res else 0.0


def score_selected_winners(winners: list[dict], val_fraction: float = DEFAULT_VAL_FRACTION,
                            split_seed: int = DEFAULT_SPLIT_SEED) -> list[dict]:
    """
    Run the final pass

    winners: list of dicts, each with the same keys check_nn_split() takes:
        {'nn_code', 'task', 'dataset', 'metric', 'prm', 'prefix' (optional)}
    Returns the same list with 'final_accuracy', 'final_model_name',
    'final_code_score' added

    """
    scored = []
    for w in winners:
        model_name, accuracy, accuracy_to_time, code_score = check_nn_split(
            w['nn_code'], w['task'], w['dataset'], w['metric'], w['prm'],
            use_val_split=False,  # the real test set
            save_to_db=False,
            prefix=w.get('prefix'),
            val_fraction=val_fraction,
            split_seed=split_seed,
        )
        scored.append({
            **w,
            'final_model_name': model_name,
            'final_accuracy': accuracy,
            'final_accuracy_to_time': accuracy_to_time,
            'final_code_score': code_score,
        })
        print(f"[FINAL] {w.get('prefix', w['dataset'])}: accuracy={accuracy:.4f} "
              f"(search-time accuracy was on a validation split, not this test set)")
    return scored
