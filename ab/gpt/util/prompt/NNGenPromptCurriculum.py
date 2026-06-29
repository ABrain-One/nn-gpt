import json
import time
from typing import List, Dict, Optional

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from overrides import override
from transformers import PreTrainedTokenizerBase

import ab.nn.api as lemur
from ab.nn.api import JoinConf
from ab.gpt.util.prompt.Prompt import Prompt


class NNGenPrompt(Prompt):
    def __init__(self, max_len: int, tokenizer: PreTrainedTokenizerBase, prompts_path: str):
        super().__init__(max_len, tokenizer)
        self.prompts_path = prompts_path

    # FIXED — add cfg as optional third parameter
    @staticmethod
    def _pack_k_models(rows: List[pd.Series], k: int, cfg: Optional[dict] = None) -> Dict[str, object]:
        if len(rows) != k:
            raise ValueError(f"_pack_k_models expects exactly {k} rows, got {len(rows)}")

        packed = {}

        for i, row in enumerate(rows, start=1):
            nn_code = row.get("nn_code")
            if not isinstance(nn_code, str) or not nn_code.strip():
                raise ValueError(f"nn_code missing or empty for model at position {i}")

            prm = row.get("prm")
            if not isinstance(prm, dict):
                raise ValueError(f"prm must be dict at position {i}, got {type(prm)}")

            transform_code = row.get("transform_code")
            if not isinstance(transform_code, str) or not transform_code.strip():
                raise ValueError(f"transform_code missing or empty for model at position {i}")

            truncate = cfg.get("nn_code_truncate") if cfg else None
            nn_code_packed = (
                nn_code[:truncate] + "\n# ... [truncated]"
                if truncate and len(nn_code) > truncate
                else nn_code
            )

            packed[f"acc_{i}"] = row.get("accuracy")
            packed[f"hp_{i}"] = json.dumps(prm, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            packed[f"tr_{i}"] = transform_code
            packed[f"nn_{i}"] = nn_code_packed
            packed[f"name_{i}"] = row.get("nn")
            packed[f"jaccard_{i}"] = row.get("anchor_jaccard")

        for key in ("dataset", "task", "metric", "epoch"):
            val = rows[0].get(key)
            if val is not None:
                packed[key] = val

        packed["anchor_nn"] = rows[0].get("anchor_nn")
        return packed

    @staticmethod
    def _build_sql_conf(cfg: dict) -> JoinConf | None:
        n = int(cfg.get("num_joint_nns") or 1)
        if n < 2:
            return None

        anchor_strategy = cfg.get("anchor_strategy", "auto")
        anchor_nn = cfg.get("anchor_nn") if anchor_strategy == "fixed" else None

        return JoinConf(
            num_joint_nns=n,
            same_columns=tuple(cfg.get("keep_same") or ()),
            diff_columns=tuple(cfg.get("no_repeat") or ()),
            enhance_nn=cfg.get("improve"),
            task=cfg.get("task"),
            dataset=cfg.get("dataset"),
            metric=cfg.get("metric"),
            similarity_mode=cfg.get("similarity_mode", "none"),
            similarity_band=cfg.get("similarity_band"),
            anchor_nn=anchor_nn,
        )

    @override
    def get_raw_dataset(self, only_best_accuracy, n_training_prompts=None) -> DataFrame:
        prompt_frames = []

        with open(self.prompts_path) as f:
            prompt_cfg = json.load(f)

        for key, cfg in prompt_cfg.items():
            print(f"\n[NNGenPrompt] key={key}", flush=True)

            is_generation = cfg.get("is_generation", False)
            selection_mode = cfg.get("selection_mode", "wide")
            k = int(cfg.get("num_joint_nns") or 1)
            sql_conf = NNGenPrompt._build_sql_conf(cfg)

            # build both templates unconditionally — output_template is None only
            # when the output block is genuinely absent, not based on is_generation
            prompt_template = "\n".join(cfg["prompt"])
            output_block = cfg.get("output", [])
            output_template = "\n".join(output_block) if output_block else None

            print(f"[MODE] selection={selection_mode}, k={k}, "
                  f"generation={is_generation}, has_output={output_template is not None}")

            t0 = time.time()
            data = lemur.data(
                only_best_accuracy=only_best_accuracy,
                task=cfg.get("task"),
                dataset=cfg.get("dataset"),
                metric=cfg.get("metric"),
                nn_prefixes=tuple(cfg.get("nn_prefixes") or ()),
                max_rows=n_training_prompts * k *10,
                sql=sql_conf,
            )
            print(f"[DATA] rows={len(data)} fetched in {time.time() - t0:.2f}s")

            input_spec = cfg["input_list"]
            rows_out = []

            # ── WIDE MODE ────────────────────────────────────────────────────────
            if selection_mode == "wide":
                for _, row in tqdm(data.iterrows(), total=len(data)):
                    para = {}
                    for it in input_spec:
                        val_key = it["value"]
                        if val_key not in row.index:
                            raise KeyError(
                                f"[{key}] missing column '{val_key}' in wide row. "
                                f"Have={list(row.index)}"
                            )
                        para[it["para"]] = row[val_key]

                    inst = prompt_template.format(**para)

                    if is_generation:
                        rows_out.append({
                            "instruction": inst,
                            "context": "",
                            "response": "",
                            "category": "generation",
                            "text": inst,
                        })
                    else:
                        if not output_template:
                            raise ValueError(
                                f"[{key}] is_generation=false but 'output' block is empty. "
                                f"Add output entries to the config."
                            )
                        resp = output_template.format(**para)
                        text = self.tokenizer.apply_chat_template(
                            [{"role": "user", "content": inst},
                             {"role": "assistant", "content": resp}],
                            tokenize=False,
                        )
                        rows_out.append({
                            "instruction": inst,
                            "context": "",
                            "response": resp,
                            "category": "train",
                            "text": text,
                        })

            # ── TALL MODE (CURRICULUM) ────────────────────────────────────────────
            else:
                df = data.copy()

                if "anchor_nn" not in df.columns:
                    raise ValueError(
                        f"[{key}] tall mode requires anchor_nn column. "
                        f"Have={list(df.columns)}"
                    )

                # sort: best accuracy first, then best jaccard, stable tie-break on nn name
                sort_cols = ["anchor_nn"]
                ascending = [True]
                for col, asc in [("accuracy", False), ("anchor_jaccard", False), ("nn", True)]:
                    if col in df.columns:
                        sort_cols.append(col)
                        ascending.append(asc)

                df = df.sort_values(sort_cols, ascending=ascending)

                for anchor, g in tqdm(df.groupby("anchor_nn"),
                                      total=df["anchor_nn"].nunique()):
                    if len(g) < 2:
                        continue

                    # A = best (first after sort), B = weakest (last)

                    if "anchor_jaccard" in g.columns and len(g) >= k:
                        sorted_g = g.sort_values("anchor_jaccard", ascending=False).reset_index(drop=True)
                        num_chunks = len(sorted_g) // k

                        for chunk_id in range(num_chunks):
                            start = chunk_id * k
                            end = start + k
                            sub = sorted_g.iloc[start:end]

                            if len(sub) < k:
                                continue

                            chunk = [pd.Series(sub.iloc[j].to_dict()) for j in range(k)]
                            packed = NNGenPrompt._pack_k_models(chunk, k, cfg)

                            labels = "ABCDEF"
                            chunk_info = "  ".join(
                                f"{labels[i]}={chunk[i].get('nn', '?')} "
                                f"(j={chunk[i].get('anchor_jaccard', 0):.4f} "
                                f"acc={chunk[i].get('accuracy', '?')})"
                                for i in range(len(chunk))
                            )
                            print(f"[TALL] anchor={anchor} chunk={chunk_id} {chunk_info}")

                            para = {}
                            for it in input_spec:
                                src = it["value"]
                                if src not in packed:
                                    raise KeyError(
                                        f"[{key}] packed field '{src}' missing. Have={sorted(packed.keys())}"
                                    )
                                para[it["para"]] = packed[src]

                            inst = prompt_template.format(**para)

                            if is_generation:
                                rows_out.append({
                                    "instruction": inst,
                                    "context": "",
                                    "response": "",
                                    "category": "generation",
                                    "text": inst,
                                })
                            else:
                                if not output_template:
                                    raise ValueError(
                                        f"[{key}] is_generation=false but 'output' block is empty."
                                    )
                                resp = output_template.format(**para)
                                text = self.tokenizer.apply_chat_template(
                                    [
                                        {"role": "user", "content": inst},
                                        {"role": "assistant", "content": resp},
                                    ],
                                    tokenize=False,
                                )
                                rows_out.append({
                                    "instruction": inst,
                                    "context": "",
                                    "response": resp,
                                    "category": "train",
                                    "text": text,
                                })

            df_out = pd.DataFrame(
                rows_out,
                columns=["instruction", "context", "response", "category", "text"],
            )
            print(f"[OUT] rows={len(df_out)}")
            prompt_frames.append(df_out)

        out = (
            pd.concat(prompt_frames, ignore_index=True)
            if prompt_frames
            else pd.DataFrame(columns=["instruction", "context", "response", "category", "text"])
        )
        print(f"\n[FINAL DATASET] rows={len(out)}", flush=True)
        return out