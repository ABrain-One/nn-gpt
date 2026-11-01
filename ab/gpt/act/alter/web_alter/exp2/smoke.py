"""Runtime smoke check of one generated network, mimicking Eval in seconds instead of a full epoch.

Same prm as Eval (Eval defaults updated with the original's prm, epoch 1), same dataset/transform loading
(nn-dataset load_dataset + get_in_shape), same required-function and hyperparameter-usage checks as
ab.gpt.util.eval.Eval (string-count mode, Eval's default for non-delta prefixes). Then: Net(...),
train_setup(prm), learn() on 2 real batches, forward on 1 batch, output shape check.
Run as a subprocess (generated code can crash or hang):
    python -m ab.gpt.act.alter.web_alter.exp2.smoke CODE_FILE DATAFRAME_DF
Prints one line "SMOKE_RESULT {json}" with ok / error / where / infra.
"""
import ast
import importlib.util
import json
import re
import sys
import traceback


def eval_prm(row):
    from ab.gpt.act.eval import Eval as E
    prm = E._default_eval_prm(
        lr=E.LR, batch=E.BATCH, dropout=E.DROPOUT, momentum=E.MOMENTUM, transform=E.TRANSFORM,
        stochastic_depth_prob=E.STOCHASTIC_DEPTH_PROB, norm_eps=E.NORM_EPS, norm_std=E.NORM_STD,
        tie_weights=E.TIE_WEIGHTS, dropout_aux=E.DROPOUT_AUX, attention_dropout=E.ATTENTION_DROPOUT,
        norm_momentum=E.NORM_MOMENTUM, score_thresh=E.SCORE_THRESH, nms_thresh=E.NMS_THRESH,
        iou_thresh=E.IOU_THRESH, detections_per_img=E.DETECTIONS_PER_IMG, topk_candidates=E.TOPK_CANDIDATES,
        neg_to_pos_ratio=E.NEG_TO_POS_RATIO, pretrained=E.PRETRAINED, patch_size=E.PATCH_SIZE)
    if isinstance(row.get("prm"), dict):
        prm.update(row["prm"])
    prm["epoch"] = 1
    if not isinstance(prm.get("transform"), str):
        prm["transform"] = E.TRANSFORM
    return prm


def static_checks(code):
    """Same checks (and messages) as ab.gpt.util.eval.Eval.evaluate before training."""
    sup = "supported_hyperparameters"
    for fn in (sup, "train_setup", "learn"):
        if not re.match(r"[\s\S]*\s+def\s" + re.escape(fn) + r"\(.*", code):
            raise Exception(f"The NN code lacks the required function '{fn}'.")
    keys = []
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.FunctionDef) and node.name == sup and isinstance(node.body[0], ast.Return):
            keys = ast.literal_eval(node.body[0].value)
    for k in [k for k in keys if k != "batch_size"]:
        if code.count('"' + k + '"') + code.count("'" + k + "'") < 2:
            raise Exception(f"The param '{k}' is not used in the code.")


def run(code_file, df_file, n_batches=2):
    import pandas as pd
    import torch
    from ab.nn.util.Loader import load_dataset
    from ab.nn.util.Util import get_in_shape, train_loader_f

    row = pd.read_pickle(df_file)
    prm = eval_prm(row)
    code = open(code_file).read()
    static_checks(code)
    out_shape, _, train_set, _ = load_dataset(row.get("task", "img-classification"), row.get("dataset", "cifar-10"),
                                              prm["transform"])
    in_shape = get_in_shape(train_set, 0)
    spec = importlib.util.spec_from_file_location("smoke_nn", code_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mod.Net(in_shape, out_shape, prm, device)
    model.to(device)
    model.train_setup(prm)
    batches = []
    for b in train_loader_f(train_set, int(prm["batch"]), 0):
        batches.append(b)
        if len(batches) >= n_batches:
            break
    model.train()
    model.learn(batches)
    model.eval()
    x = batches[0][0].to(device)
    with torch.no_grad():
        y = model(x)
    n_cls = int(out_shape[0]) if hasattr(out_shape, "__len__") else int(out_shape)
    if not hasattr(y, "shape") or tuple(y.shape) != (x.shape[0], n_cls):
        raise RuntimeError(f"forward() returned shape {tuple(getattr(y, 'shape', ()))}, expected {(x.shape[0], n_cls)}")


def main():
    code_file, df_file = sys.argv[1], sys.argv[2]
    try:
        run(code_file, df_file)
        res = {"ok": True}
    except BaseException as e:  # noqa: BLE001 - report everything, including SystemExit from generated code
        msg = f"{type(e).__name__}: {e}"
        frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename == code_file]
        where = f"line {frames[-1].lineno}: {frames[-1].line}" if frames else ""
        infra = "out of memory" in msg.lower() or "cuda error" in msg.lower()
        res = {"ok": False, "error": msg[:600], "where": where[:300], "infra": infra}
    print("SMOKE_RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
