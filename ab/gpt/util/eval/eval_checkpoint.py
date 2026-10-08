"""Persist trained PyTorch checkpoints during NNEval for mobile deployment."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import ab.nn.util.CodeEval as codeEvaluator
import ab.nn.util.Train as train_runtime
import ab.nn.util.db.Write as DB_Write
from ab.nn.util.Const import ab_root_path, default_epoch_limit_minutes
from ab.nn.util.Loader import load_dataset
from ab.nn.util.Train import Train
from ab.nn.util.Util import (
    create_file,
    export_torch_weights,
    good,
    release_memory,
    remove,
    uuid4,
)

EVAL_CHECKPOINT_NAME = "eval_checkpoint.pth"


def eval_checkpoint_path(model_dir: Union[str, Path]) -> Path:
    return Path(model_dir) / EVAL_CHECKPOINT_NAME


def train_and_eval_with_checkpoint(
    nn_code: str,
    task: str,
    dataset: str,
    metric: str,
    prm: dict,
    checkpoint_path: Union[str, Path],
    *,
    save_to_db: bool = False,
    prefix: Optional[str] = None,
    save_path: Optional[str] = None,
    export_onnx: bool = False,
    epoch_limit_minutes=default_epoch_limit_minutes,
    transform_dir=None,
) -> tuple[str, float, float, float]:
    """
    Train/evaluate NN code and persist state_dict to checkpoint_path.

    Mirrors ab.nn.api.check_nn / Train.train_new but saves weights before cleanup.
    """
    model_name = uuid4(nn_code)
    if prefix:
        model_name = prefix + "-" + model_name

    tmp_modul = ".".join((train_runtime.out, "nn", "tmp"))
    tmp_modul_name = ".".join((tmp_modul, model_name))
    tmp_dir = ab_root_path / tmp_modul.replace(".", "/")
    create_file(tmp_dir, "__init__.py")
    temp_file_path = tmp_dir / f"{model_name}.py"
    trainer = None
    accuracy = 0.0
    accuracy_to_time = 0.0
    res = {"score": 0.0}
    try:
        with open(temp_file_path, "w", encoding="utf-8") as handle:
            handle.write(nn_code)
        res = codeEvaluator.evaluate_single_file(temp_file_path)
        out_shape, minimum_accuracy, train_set, test_set = load_dataset(
            task, dataset, prm["transform"], transform_dir
        )
        num_workers = prm.get("num_workers", 1)
        trainer = Train(
            config=(task, dataset, metric, model_name),
            out_shape=out_shape,
            minimum_accuracy=minimum_accuracy,
            batch=prm["batch"],
            nn_module=tmp_modul_name,
            task=task,
            train_dataset=train_set,
            test_dataset=test_set,
            metric=metric,
            num_workers=num_workers,
            prm=prm,
            save_to_db=save_to_db,
            is_code=True,
        )
        epoch = prm["epoch"]
        accuracy, accuracy_to_time, duration = trainer.train_n_eval(
            epoch,
            epoch_limit_minutes,
            False,
            export_onnx,
            train_set,
            save_path=save_path,
        )

        ckpt_path = Path(checkpoint_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        export_torch_weights(trainer.model, str(ckpt_path))

        if save_to_db:
            if good(accuracy, minimum_accuracy, duration):
                model_name = DB_Write.save_nn(
                    nn_code, task, dataset, metric, epoch, prm, force_name=model_name
                )
                print(f"Model saved to database with accuracy: {accuracy}")
            else:
                print(
                    f"Model accuracy {accuracy} is below the minimum threshold "
                    f"{minimum_accuracy}. Not saved."
                )
    finally:
        remove(temp_file_path)

        try:
            del train_set
        except NameError:
            pass

        try:
            del test_set
        except NameError:
            pass

        try:
            if trainer:
                del trainer.model
        except (NameError, AttributeError):
            pass
        release_memory()

    return model_name, accuracy, accuracy_to_time, res["score"]
