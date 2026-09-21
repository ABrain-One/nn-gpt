import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import ab.nn.api as lemur
import pandas as pd
from overrides import override
from pandas import DataFrame
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from ab.gpt.util.prompt.Prompt import Prompt
from ab.gpt.util.data.lemur_enrichment import patch_join_nn_query, enrich_dataframe
from ab.gpt.util.Util import evaluate_delimited_formulas
from ab.gpt.util.Const import DEFAULT_DATASET, DEFAULT_NN_PREFIXES

# JoinConf is defined in ab/nn/util/db/Query.py and re-exported by ab.nn.api.
# Both previous files referenced the same frozen dataclass.
from ab.nn.util.db.Query import JoinConf


FRAME_COLUMNS = ['instruction', 'context', 'response', 'category', 'text']


def shuffle_data(df: DataFrame):
    return df.sample(frac=1).reset_index(drop=True)


def _empty_frame() -> DataFrame:
    return DataFrame(columns=FRAME_COLUMNS)


class NNGenPrompt(Prompt):
    """
    Assumes the existence of accuracies.json and folder-based dataset
    """

    # ---- category-label semantics for WIDE-mode rows -------------------
    # The callers depend on the difference, so it is a class-level switch.
    #
    #   False: wide rows get category "".
    #     Tune.py, Tune_prun.py, Tune_Onnx.py, GenerationPipeline.py and
    #     test_nngen.py never read the column, and this is what they have
    #     always received.
    #
    #   True : wide rows get
    #     "train". Tune_Curriculum.py filters on `category == "train"`
    #     (nn_tune) and `category == "generation"` (nn_gen); with "" those
    #     filters would select zero rows.
    #
    # Tall rows are always labelled "train"/"generation" in both classes.
    CURRICULUM_CATEGORIES = True

    def _category(self, is_generation: bool, tall: bool) -> str:
        if is_generation:
            return "generation"
        if tall or self.CURRICULUM_CATEGORIES:
            return "train"
        return ""

    def __init__(self, max_len: int, tokenizer: PreTrainedTokenizerBase, prompts_path, data_dir=None):
        super().__init__(max_len, tokenizer)
        self.prompts_path = prompts_path
        # When set, SFT trains on this on-disk chat corpus (data_dir/train.jsonl)
        # instead of querying LEMUR. Used by the iterative pipeline so each cycle's
        # fine-tuning uses its growing curated corpus, closing the feedback loop.
        self.data_dir = data_dir

    # ------------------------------------------------------------------
    # On-disk corpus path (unchanged from production)
    # ------------------------------------------------------------------
    def _raw_dataset_from_disk(self, n_training_prompts=None) -> DataFrame:
        """Build the SFT frame from a pipeline corpus of chat 'messages' rows
        (data_dir/train.jsonl), instead of LEMUR. Produces the exact same
        columns get_dataset()/preprocess_batch consume: the LEMUR path emits
        [instruction, context, response, category, text]; we mirror it."""
        train_path = Path(self.data_dir) / "train.jsonl"
        if not train_path.exists():
            raise FileNotFoundError(
                f"NNGenPrompt: data_dir set but corpus not found at {train_path}")
        rows = []
        with open(train_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if n_training_prompts:
            rows = rows[:n_training_prompts]
        frame = _empty_frame()
        for row in rows:
            messages = row.get("messages", [])
            user_txt = next(
                (m["content"] for m in messages if m["role"] == "user"), "")
            assistant_txt = next(
                (m["content"] for m in messages if m["role"] == "assistant"), "")
            # Full templated conversation (incl. assistant target) — same as the
            # LEMUR path, which builds text via _apply_chat_template(_build_messages(...)).
            text = self._apply_chat_template(messages, tokenize=False)
            frame.loc[len(frame)] = [user_txt, "", assistant_txt, "", text]
        print(f"[SFT] Loaded {len(frame)} training rows from {train_path} "
              f"(LEMUR query bypassed)", flush=True)
        return frame

    # ------------------------------------------------------------------
    # SQL config
    # ------------------------------------------------------------------
    @staticmethod
    def _build_sql_conf(cfg: dict, selection_mode: str) -> Optional[JoinConf]:
        """Build the JoinConf for one config key.

        For a legacy-shaped key (selection_mode="wide", similarity_mode absent
        or "none") this emits EXACTLY the four-field JoinConf the production
        file built, and nothing else. task/dataset/metric are deliberately NOT
        forwarded there: join_nn_query_sql_Var_num feeds them into
        build_stat_filters_sql, so passing them would silently add WHERE
        clauses to queries that do not have them today.
        """
        n = int(cfg.get("num_joint_nns") or 1)
        similarity_mode = cfg.get("similarity_mode", "none")

        if selection_mode == "tall" and similarity_mode != "anchor_band_db_minhash":
            raise ValueError(
                "selection_mode='tall' requires similarity_mode="
                "'anchor_band_db_minhash'. With similarity_mode='none' the "
                "query is routed to join_nn_query_sql_Var_num, which returns "
                "one row per model with no anchor_nn/anchor_jaccard columns "
                "and therefore cannot be chunked."
            )

        if n < 2:
            if similarity_mode != "none":
                raise ValueError(
                    f"similarity_mode='{similarity_mode}' requires "
                    f"num_joint_nns >= 2, got {n}."
                )
            # Legacy: use_join = num_joint_nns >= 2 -> sql=None
            return None

        # ---- legacy wide shape: preserve the historical call verbatim ----
        if selection_mode == "wide" and similarity_mode == "none":
            return JoinConf(
                num_joint_nns=n,
                same_columns=tuple(cfg.get('keep_same', [])),
                diff_columns=tuple(cfg.get('no_repeat', [])),
                enhance_nn=cfg.get('improve', False),
            )

        # ---- curriculum / similarity-aware shape ----
        anchor_strategy = cfg.get("anchor_strategy", "auto")
        anchor_nn = cfg.get("anchor_nn") if anchor_strategy == "fixed" else None

        kwargs = dict(
            num_joint_nns=n,
            same_columns=tuple(cfg.get("keep_same") or ()),
            diff_columns=tuple(cfg.get("no_repeat") or ()),
            enhance_nn=cfg.get("improve"),
            task=cfg.get("task"),
            dataset=cfg.get("dataset"),
            metric=cfg.get("metric"),
            similarity_mode=similarity_mode,
            similarity_band=cfg.get("similarity_band"),
            anchor_nn=anchor_nn,
        )
        # Optional knobs — only forwarded when present so JoinConf defaults hold.
        for opt in ("min_arch_jaccard", "max_arch_jaccard"):
            if cfg.get(opt) is not None:
                kwargs[opt] = float(cfg[opt])
        if cfg.get("overfetch_factor") is not None:
            kwargs["overfetch_factor"] = int(cfg["overfetch_factor"])

        return JoinConf(**kwargs)

    # ------------------------------------------------------------------
    # Tall-mode packing (from the curriculum file)
    # ------------------------------------------------------------------
    @staticmethod
    def _pack_k_models(rows: List[pd.Series], k: int, cfg: Optional[dict] = None) -> Dict[str, object]:
        """Flatten k per-model rows into a single flat dict of *_1 .. *_k keys.

        `nn_{i}` honours cfg['nn_code_truncate']; `nn_{i}_full` never does, so a
        delta-mode config can map the untruncated source into para and keep the
        production invariant that deltas are computed between FULL files.
        """
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
            packed[f"nn_{i}_full"] = nn_code
            packed[f"name_{i}"] = row.get("nn")
            packed[f"jaccard_{i}"] = row.get("anchor_jaccard")

        for key in ("dataset", "task", "metric", "epoch"):
            val = rows[0].get(key)
            if val is not None:
                packed[key] = val

        packed["anchor_nn"] = rows[0].get("anchor_nn")
        return packed

    # ------------------------------------------------------------------
    # Shared emit tail — used by BOTH modes
    # ------------------------------------------------------------------
    def _emit_row(
        self,
        *,
        key: str,
        para: dict,
        prompt_template: str,
        output_template: Optional[str],
        system_text: str,
        use_delta: bool,
        is_generation: bool,
        category: str,
        full_nn_code=None,
        full_addon_nn_code=None,
    ) -> list:
        """Turn a populated `para` dict into one [instruction, context,
        response, category, text] row. This is the exact production tail,
        lifted verbatim so wide-mode output is unchanged."""

        # ========== APPLY FORMULA EVALUATION TO PROMPT ==========
        inst = prompt_template.format(**para)
        inst = evaluate_delimited_formulas(inst, para)
        # ========================================================

        # Generation rows carry no target: no chat template, raw prompt as text.
        if is_generation:
            return [inst, "", "", category, inst]

        if output_template is None:
            raise ValueError(
                f"[{key}] is_generation=false but the 'output' block is empty. "
                f"Add output entries to the config."
            )

        # Compute delta if delta mode is enabled
        if use_delta and 'addon_nn_code' in para and 'nn_code' in para:
            try:
                from ab.gpt.util.nn.DeltaUtil import compute_delta
                baseline_code = full_nn_code if isinstance(full_nn_code, str) else para.get('nn_code', '')
                improved_code = full_addon_nn_code if isinstance(full_addon_nn_code, str) else para.get('addon_nn_code', '')

                if baseline_code and improved_code:
                    computed_delta = compute_delta(baseline_code, improved_code)
                    if not computed_delta:
                        computed_delta = ""
                else:
                    computed_delta = ""

                try:
                    response = output_template.format(**para)
                except KeyError:
                    response = output_template
                    for k, v in para.items():
                        response = response.replace(f'{{{k}}}', str(v))
                response = response.replace('{computed_delta}', computed_delta)
            except Exception as e:
                print(
                    f'[WARNING] Failed to compute delta for key {key}: {e}. Using regular output.', flush=True)
                try:
                    response = output_template.format(**para)
                except KeyError:
                    response = output_template
                    for k, v in para.items():
                        response = response.replace(f'{{{k}}}', str(v))
                response = response.replace('{computed_delta}', '')
        else:
            # Regular mode: use output template as-is
            response = output_template.format(**para)

        # ========== APPLY FORMULA EVALUATION TO RESPONSE ==========
        response = evaluate_delimited_formulas(response, para)
        # ==========================================================

        text = self._apply_chat_template(
            self._build_messages(
                inst, response, system_prompt=system_text or None),
            tokenize=False)

        return [inst, "", response, category, text]

    # ------------------------------------------------------------------
    # WIDE MODE — one LEMUR row per training sample (production path)
    # ------------------------------------------------------------------
    def _run_wide(self, key, key_dict, only_best_accuracy, n_training_prompts) -> DataFrame:
        dataframe = _empty_frame()

        prompt = '\n'.join(key_dict['prompt'])
        print('Preparing Data...', flush=True)

        num_joint_nns = key_dict.get('num_joint_nns') or 1
        # For JOIN queries, do NOT pass max_rows — the LIMIT applies before
        # the JOIN and causes an O(n²) correlated scan. Slice the result after.
        use_join = num_joint_nns >= 2
        system_text = '\n'.join(key_dict.get('system', []))
        is_generation = key_dict.get('is_generation', False)
        output_block = key_dict.get('output', [])
        output_template = '\n'.join(output_block) if output_block else None
        print(f'Preparing Data for key: {key}...', flush=True)

        # ========== SMALL SWITCH TO DETECT PRUNING CONFIG ==========
        # Check if this is a pruning task (key starts with 'pruning' or contains 'pruning')
        is_pruning = (
            key.lower().startswith('pruning') or
            'pruning' in key.lower()
        )

        print(f"[DEBUG] Key: {key}, is_pruning: {is_pruning}")

        if is_pruning:
            # Use prun table for pruning statistics
            data = lemur.prun_data(max_rows=n_training_prompts)
            # Filter only successful pruning experiments
            if 'status' in data.columns:
                data = data[data['status'] == 'success']
            print(
                f"[PRUN] Fetched {len(data)} records from PRUN table for key: {key}")
        else:
            # for classification tasks: Patch LEMUR's join query before the data call so that dataset_2
            # and its siblings appear in the result set.

            classification = use_join and key_dict.get('output_type') == 'classification'
            if classification:
                patch_join_nn_query()  # TODO: Generalize for all scenarios - SQL query implementation in the NN Dataset project
            data = lemur.data(
                only_best_accuracy=only_best_accuracy,
                task=key_dict.get('task'),
                dataset=key_dict.get('dataset', DEFAULT_DATASET),
                nn_prefixes=tuple(key_dict.get('nn_prefixes') or DEFAULT_NN_PREFIXES),
                max_rows=n_training_prompts,
                sql=NNGenPrompt._build_sql_conf(key_dict, "wide"),
            )
            # For classification tasks, enrich the DataFrame with normalised
            # accuracy and dataset-metadata columns needed for the prompt.
            if classification:
                enrich_dataframe(data)  # TODO: Generalize for all scenarios based on the formula implementation (see evaluate_delimited_formulas(..))
            print(f"[STAT] Fetched {len(data)} records from STAT table for key: {key}")
        # ==========================================================

        print('Data acquisition complete', flush=True)

        # Check if this is delta mode
        use_delta = key_dict.get('use_delta', False) or 'delta' in key.lower()

        for _, row in tqdm(data.iterrows(), total=n_training_prompts or len(data)):
            if n_training_prompts and len(dataframe) >= n_training_prompts:
                break

            para_dict = dict()
            for it in key_dict['input_list']:
                # Handle column name mapping gracefully
                db_column = it['value']
                try:
                    if db_column in row:
                        para_dict[it['para']] = row[db_column]
                    elif db_column == 'model_name' and 'nn' in row:
                        para_dict[it['para']] = row['nn']
                    elif db_column == 'nn' and 'model_name' in row:
                        para_dict[it['para']] = row['model_name']
                    else:
                        para_dict[it['para']] = row.get(
                            db_column, f"Missing: {db_column}")
                except Exception as e:
                    print(
                        f"[WARNING] Could not get column '{db_column}': {e}")
                    para_dict[it['para']] = None

            # Cap only the PROMPT-context code (baseline). Never truncate the
            # response fields (addon_nn_code / addon_transform_code) — they are
            # the training target; clipping them produces broken Python labels.
            # Over-long responses are dropped later by the max_new_tokens filter.
            # Stash the untouched codes first: the training-target delta
            # must be computed between the FULL baseline and FULL improved
            # files (its hunk lives inside train_setup, which the shrunk
            # prompt still shows verbatim), never between shrunk/capped
            # views.
            full_nn_code = para_dict.get('nn_code')
            full_addon_nn_code = para_dict.get('addon_nn_code')

            if key_dict.get('shrink_nn_code') and isinstance(para_dict.get('nn_code'), str):
                from ab.gpt.util.nn.DeltaUtil import shrink_nn_code_for_prompt
                para_dict['nn_code'] = shrink_nn_code_for_prompt(para_dict['nn_code'])

            nn_code_max_chars = key_dict.get('nn_code_max_chars')
            if nn_code_max_chars and 'nn_code' in para_dict and isinstance(para_dict['nn_code'], str):
                para_dict['nn_code'] = para_dict['nn_code'][:nn_code_max_chars]

            # Inject columns referenced in the output template but absent from input_list
            if key_dict.get('output_type') == 'classification' and output_template:
                for col in row.index:
                    if f'{{{col}}}' in output_template and col not in para_dict:
                        para_dict[col] = row[col]

            emitted = self._emit_row(
                key=key,
                para=para_dict,
                prompt_template=prompt,
                output_template=output_template,
                system_text=system_text,
                use_delta=use_delta,
                is_generation=is_generation,
                category=self._category(is_generation, tall=False),
                full_nn_code=full_nn_code,
                full_addon_nn_code=full_addon_nn_code,
            )

            # ========== PRINT FOR VERIFICATION (AFTER response EXISTS) ==========
            if len(dataframe) < 10:
                print(f"\n[EXAMPLE {len(dataframe)+1}]:")
                print(f"SYS: {system_text}")
                print(f"USER: {emitted[0][:1000]}...")
                print(f"OUTPUT: {emitted[2][:500]}...")
                print("-" * 50)
            # ================================================================

            dataframe.loc[len(dataframe)] = emitted

        del data
        return dataframe

    # ------------------------------------------------------------------
    # TALL MODE — k LEMUR rows per training sample (curriculum path)
    # ------------------------------------------------------------------
    def _run_tall(self, key, cfg, only_best_accuracy, n_training_prompts) -> DataFrame:
        k = int(cfg.get("num_joint_nns") or 1)
        is_generation = cfg.get("is_generation", False)
        sql_conf = NNGenPrompt._build_sql_conf(cfg, "tall")

        system_text = '\n'.join(cfg.get('system', []))
        use_delta = cfg.get('use_delta', False) or 'delta' in key.lower()

        # build both templates unconditionally — output_template is None only
        # when the output block is genuinely absent, not based on is_generation
        prompt_template = "\n".join(cfg["prompt"])
        output_block = cfg.get("output", [])
        output_template = "\n".join(output_block) if output_block else None

        print(f"[MODE] selection=tall, k={k}, "
              f"generation={is_generation}, has_output={output_template is not None}")

        t0 = time.time()
        # max_rows is ignored by join_nn_query_anchor_otf (see module docstring).
        # Volume is governed by JoinConf.overfetch_factor.
        data = lemur.data(
            only_best_accuracy=only_best_accuracy,
            task=cfg.get("task"),
            dataset=cfg.get("dataset"),
            metric=cfg.get("metric"),
            nn_prefixes=tuple(cfg.get("nn_prefixes") or ()),
            max_rows=None,
            sql=sql_conf,
        )
        print(f"[DATA] rows={len(data)} fetched in {time.time() - t0:.2f}s")

        input_spec = cfg["input_list"]
        rows_out = []

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

                    rows_out.append(self._emit_row(
                        key=key,
                        para=para,
                        prompt_template=prompt_template,
                        output_template=output_template,
                        system_text=system_text,
                        use_delta=use_delta,
                        is_generation=is_generation,
                        category=self._category(is_generation, tall=True),
                        # Untruncated sources for delta mode, when the config
                        # maps them into para under these names.
                        full_nn_code=para.get('nn_code'),
                        full_addon_nn_code=para.get('addon_nn_code'),
                    ))

        del data
        return DataFrame(rows_out, columns=FRAME_COLUMNS)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    @override
    def get_raw_dataset(self, only_best_accuracy, n_training_prompts=None) -> DataFrame:
        """
        :return:
            pandas.Dataframe object with columns described in nn_api.data()
        """
        # Iterative-pipeline path: train on the on-disk augmented corpus.
        if self.data_dir is not None:
            return self._raw_dataset_from_disk(n_training_prompts)

        prompt_lists = []

        # /workspace/nn-gpt/ab/gpt/conf/prompt/train/NN_gen.json
        with open(self.prompts_path) as prompt_file:
            prompt_dict = json.load(prompt_file)
        assert isinstance(prompt_dict, dict)

        for key in prompt_dict.keys():
            key_dict = prompt_dict[key]
            selection_mode = key_dict.get('selection_mode', 'wide')

            if selection_mode not in ('wide', 'tall'):
                raise ValueError(
                    f"[{key}] unknown selection_mode='{selection_mode}' "
                    f"(expected 'wide' or 'tall')"
                )

            print(f"\n[NNGenPrompt] key={key} selection_mode={selection_mode}", flush=True)

            if selection_mode == 'tall':
                frame = self._run_tall(key, key_dict, only_best_accuracy, n_training_prompts)
            else:
                frame = self._run_wide(key, key_dict, only_best_accuracy, n_training_prompts)

            print(f"[OUT] key={key} rows={len(frame)}")
            prompt_lists.append(frame)

        print('Prompts successfully generated', flush=True)

        out = (
            pd.concat(prompt_lists, ignore_index=True)
            if prompt_lists
            else _empty_frame()
        )
        print(f"\n[FINAL DATASET] rows={len(out)}", flush=True)
        return out