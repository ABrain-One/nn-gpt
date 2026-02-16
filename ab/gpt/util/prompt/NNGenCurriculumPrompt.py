import json
import time
from typing import List, Dict

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from overrides import override
from transformers import PreTrainedTokenizerBase

import ab.nn.api as lemur
from ab.nn.api import JoinConf
from ab.gpt.util.prompt.Prompt import Prompt


class NNGenPrompt(Prompt):
    """
    Prompt generator for NN curriculum learning.

    Supports:
    - selection_mode = "wide"  (legacy pairwise / wide join)
    - selection_mode = "tall"  (SQL variable-N + Python packing)
    """

    def __init__(self, max_len: int, tokenizer: PreTrainedTokenizerBase, prompts_path: str):
        super().__init__(max_len, tokenizer)
        self.prompts_path = prompts_path

    # ------------------------------------------------------------------
    # Packing logic (THIS is the missing piece)
    # ------------------------------------------------------------------
    def _pack_k_models(
        self,
        rows: List[pd.Series],
        k: int,
    ) -> Dict[str, object]:
        """
        Convert k tall rows into a single packed dict:
        acc_1, hp_1, tr_1, nn_1, ..., acc_k, hp_k, tr_k, nn_k
        """
        packed = {}

        for i, row in enumerate(rows, start=1):
            packed[f"acc_{i}"] = row["accuracy"]
            packed[f"hp_{i}"] = row["prm"]
            packed[f"tr_{i}"] = row["transform_code"]
            packed[f"nn_{i}"] = row["nn_code"]

        # shared metadata (identical by construction)
        for key in ("dataset", "task", "metric", "epoch", "metric_code"):
            if key in rows[0]:
                packed[key] = rows[0][key]

        return packed

    # ------------------------------------------------------------------
    # SQL config builder
    # ------------------------------------------------------------------
    def _build_sql_conf(self, cfg: dict) -> JoinConf | None:
        n = int(cfg.get("num_joint_nns") or 1)
        if n < 2:
            return None

        return JoinConf(
            num_joint_nns=n,
            same_columns=tuple(cfg.get("keep_same") or ()),
            diff_columns=tuple(cfg.get("no_repeat") or ()),
            enhance_nn=cfg.get("improve"),
            similarity_mode=cfg.get("similarity_mode", "none"),
            similarity_band=cfg.get("similarity_band"),
            anchor_nn=cfg.get("anchor_nn"),
        )

    # ------------------------------------------------------------------
    @override
    def get_raw_dataset(self, only_best_accuracy, n_training_prompts=None) -> DataFrame:
        prompt_frames = []

        with open(self.prompts_path) as f:
            prompt_cfg = json.load(f)

        for key, cfg in prompt_cfg.items():
            print(f"[NNGenPrompt] Preparing key='{key}'", flush=True)

            df_out = DataFrame(columns=["instruction", "context", "response", "category", "text"])
            prompt_frames.append(df_out)

            selection_mode = cfg.get("selection_mode", "wide")
            k = int(cfg.get("num_joint_nns") or 1)

            sql_conf = self._build_sql_conf(cfg)

            t0 = time.time()
            data = lemur.data(
                only_best_accuracy=only_best_accuracy,
                task=cfg.get("task"),
                dataset=cfg.get("dataset"),
                metric=cfg.get("metric"),
                nn_prefixes=tuple(cfg.get("nn_prefixes") or ()),
                max_rows=n_training_prompts,
                sql=sql_conf,
            )
            print(f"[NNGenPrompt] fetched rows={len(data)} in {time.time()-t0:.1f}s")

            prompt_template = "\n".join(cfg["prompt"])
            output_template = "\n".join(cfg["output"])
            input_spec = cfg["input_list"]

            # --------------------------------------------------------------
            # WIDE MODE (legacy, trivial)
            # --------------------------------------------------------------
            if selection_mode == "wide":
                iterator = data.iterrows()

                for _, row in tqdm(iterator, total=len(data)):
                    para = {it["para"]: row[it["value"]] for it in input_spec}
                    inst = prompt_template.format(**para)
                    resp = output_template.format(**para)

                    text = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": inst},
                         {"role": "assistant", "content": resp}],
                        tokenize=False,
                    )
                    df_out.loc[len(df_out)] = [inst, "", resp, "", text]

            # --------------------------------------------------------------
            # TALL MODE (curriculum, PACKING REQUIRED)
            # --------------------------------------------------------------
            else:
                rows = list(data.itertuples(index=False))
                series_rows = [pd.Series(r._asdict()) for r in rows]

                for i in tqdm(range(0, len(series_rows) - k + 1, k)):
                    chunk = series_rows[i:i + k]
                    if len(chunk) < k:
                        break

                    packed = self._pack_k_models(chunk, k)

                    para = {}
                    for it in input_spec:
                        para[it["para"]] = packed[it["value"]]

                    inst = prompt_template.format(**para)
                    resp = output_template.format(**para)

                    text = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": inst},
                         {"role": "assistant", "content": resp}],
                        tokenize=False,
                    )
                    df_out.loc[len(df_out)] = [inst, "", resp, "", text]

        print("[NNGenPrompt] Prompt generation complete", flush=True)
        return pd.concat(prompt_frames, ignore_index=True)
