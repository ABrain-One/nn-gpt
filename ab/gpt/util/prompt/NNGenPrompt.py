import json
from pathlib import Path

import ab.nn.api as lemur
from overrides import override
import pandas as pd
from pandas import DataFrame
from transformers import PreTrainedTokenizerBase

from ab.gpt.util.prompt.Prompt import Prompt
from ab.gpt.util.lemur_enrichment import patch_join_nn_query, enrich_dataframe
from tqdm import tqdm

from ab.nn.util.db.Query import JoinConf
from ab.gpt.util.Util import evaluate_delimited_formulas
from ab.gpt.util.Const import DEFAULT_DATASET, DEFAULT_NN_PREFIXES


def shuffle_data(df: DataFrame):
    return df.sample(frac=1).reset_index(drop=True)


class NNGenPrompt(Prompt):
    """
    Assumes the existence of accuracies.json and folder-based dataset
    """

    def __init__(self, max_len: int, tokenizer: PreTrainedTokenizerBase, prompts_path, data_dir=None):
        super().__init__(max_len, tokenizer)
        self.prompts_path = prompts_path
        # When set, SFT trains on this on-disk chat corpus (data_dir/train.jsonl)
        # instead of querying LEMUR. Used by the iterative pipeline so each cycle's
        # fine-tuning uses its growing curated corpus, closing the feedback loop.
        self.data_dir = data_dir

    def get_config_seed(self):
        """
        Optional reproducibility seed read from the prompt config JSON, next to
        the other per-experiment values (task, dataset, min_accuracy, etc.), so
        each experiment config fully owns its own seed instead of only getting
        one indirectly from the LLM's own TrainingArguments default. Returns the
        first 'seed' found among the config's keys, or None if none is set --
        callers should fall back to their own default in that case.
        """
        if self.data_dir is not None:
            return None
        with open(self.prompts_path) as prompt_file:
            prompt_dict = json.load(prompt_file)
        for key_dict in prompt_dict.values():
            if 'seed' in key_dict and key_dict['seed'] is not None:
                return key_dict['seed']
        return None

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
        frame = DataFrame(
            columns=['instruction', 'context', 'response', 'category', 'text'])
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
            dataframe = DataFrame(
                columns=['instruction', 'context', 'response', 'category', 'text'])
            prompt_lists.append(dataframe)
            prompt = '\n'.join(prompt_dict[key]['prompt'])
            print('Preparing Data...', flush=True)
            key_dict = prompt_dict[key]
            num_joint_nns = key_dict.get('num_joint_nns') or 1

            # For JOIN queries, do NOT pass max_rows — the LIMIT applies before
            # the JOIN and causes an O(n²) correlated scan. Slice the result after.
            use_join = num_joint_nns >= 2
            system_text = '\n'.join(prompt_dict[key].get('system', []))
            prompt_template = '\n'.join(prompt_dict[key]['prompt'])
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
                    patch_join_nn_query() # # TODO: Generalize for all scenarios - SQL query implementation in the NN Dataset project

                # Accuracy-filtered training pools (added 2026-09-08 for the
                # professor's Experiment A/B/C plan). min_accuracy is applied as
                # a SQL WHERE condition inside lemur.data() (2026-09-18 fix, per
                # professor feedback on PR #237) rather than as a post-hoc
                # DataFrame filter, so it composes correctly with max_rows -- the
                # LIMIT now applies to the already-filtered set.
                min_accuracy = key_dict.get('min_accuracy')

                data = lemur.data(
                    only_best_accuracy=only_best_accuracy,
                    task=key_dict.get('task'),
                    dataset=key_dict.get('dataset', DEFAULT_DATASET),
                    epoch=key_dict.get('epoch'),
                    # Bug fix (2026-09-08): `key_dict.get('nn_prefixes') or DEFAULT_NN_PREFIXES`
                    # treated an explicit empty list (meaning "no prefix restriction",
                    # matching how DB_Read.data()'s own `if nn_prefixes:` check treats
                    # falsy values) the same as "key not set at all", silently narrowing
                    # every config with `"nn_prefixes": []` down to DEFAULT_NN_PREFIXES
                    # = ('ga-', 'GenFractalNet') only. Caught via validate_accfilter.py
                    # returning 13,797 rows instead of the real 48,988-row pool.
                    nn_prefixes=tuple(key_dict['nn_prefixes']) if 'nn_prefixes' in key_dict else DEFAULT_NN_PREFIXES,
                    min_accuracy=min_accuracy,
                    max_rows=n_training_prompts,
                    sql=None if not use_join else JoinConf(
                        num_joint_nns=num_joint_nns,
                        same_columns=tuple(key_dict.get('keep_same', [])),
                        diff_columns=tuple(key_dict.get('no_repeat', [])),
                        enhance_nn=key_dict.get('improve', False)
                    )
                )

                if min_accuracy is not None:
                    print(f"[ACCFILTER] min_accuracy={min_accuracy}: {len(data)} rows for key: {key} (filtered in SQL)")

                # For classification tasks, enrich the DataFrame with normalised
                # accuracy and dataset-metadata columns needed for the prompt.
                if classification:
                    enrich_dataframe(data)  # TODO: Generalize for all scenarios based on the formula implementation (see evaluate_delimited_formulas(..))
                print(f"[STAT] Fetched {len(data)} records from STAT table for key: {key}")
            # ==========================================================

            print('Data acquisition complete', flush=True)

            # Check if this is delta mode
            use_delta = key_dict.get(
                'use_delta', False) or 'delta' in key.lower()

            for _, row in tqdm(data.iterrows(), total=n_training_prompts or len(data)):
                if n_training_prompts and len(dataframe) >= n_training_prompts:
                    break

                para_dict = dict()
                for it in prompt_dict[key]['input_list']:
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

                nn_code_max_chars = key_dict.get('nn_code_max_chars')
                if nn_code_max_chars and 'nn_code' in para_dict and isinstance(para_dict['nn_code'], str):
                    para_dict['nn_code'] = para_dict['nn_code'][:nn_code_max_chars]

                # Inject columns referenced in the output template but absent from input_list
                if key_dict.get('output_type') == 'classification':
                    output_template = '\n'.join(key_dict['output'])
                    for col in row.index:
                        if f'{{{col}}}' in output_template and col not in para_dict:
                            para_dict[col] = row[col]

                # ========== APPLY FORMULA EVALUATION TO PROMPT ==========
                inst = prompt.format(**para_dict)
                inst = evaluate_delimited_formulas(inst, para_dict)
                # ========================================================

                # Compute delta if delta mode is enabled
                if use_delta and 'addon_nn_code' in para_dict and 'nn_code' in para_dict:
                    try:
                        from ab.gpt.util.DeltaUtil import compute_delta
                        baseline_code = para_dict.get('nn_code', '')
                        improved_code = para_dict.get('addon_nn_code', '')

                        if baseline_code and improved_code:
                            computed_delta = compute_delta(
                                baseline_code, improved_code)
                            if not computed_delta:
                                computed_delta = ""
                        else:
                            computed_delta = ""

                        output = '\n'.join(prompt_dict[key]['output'])
                        try:
                            response = output.format(**para_dict)
                        except KeyError:
                            response = output
                            for k, v in para_dict.items():
                                response = response.replace(f'{{{k}}}', str(v))
                        response = response.replace(
                            '{computed_delta}', computed_delta)
                    except Exception as e:
                        print(
                            f'[WARNING] Failed to compute delta for key {key}: {e}. Using regular output.', flush=True)
                        output = '\n'.join(prompt_dict[key]['output'])
                        try:
                            response = output.format(**para_dict)
                        except KeyError:
                            response = output
                            for k, v in para_dict.items():
                                response = response.replace(f'{{{k}}}', str(v))
                        response = response.replace('{computed_delta}', '')
                else:
                    # Regular mode: use output template as-is
                    output = '\n'.join(prompt_dict[key]['output'])
                    response = output.format(**para_dict)

                # ========== APPLY FORMULA EVALUATION TO RESPONSE ==========
                response = evaluate_delimited_formulas(response, para_dict)
                # ==========================================================

                # ========== PRINT FOR VERIFICATION (AFTER response EXISTS) ==========
                if len(dataframe) < 10:
                    print(f"\n[EXAMPLE {len(dataframe)+1}]:")
                    print(f"SYS: {system_text}")
                    print(f"USER: {inst[:1000]}...")
                    print(f"OUTPUT: {response[:500]}...")
                    print("-" * 50)
                # ================================================================

                text = self._apply_chat_template(
                    self._build_messages(
                        inst, response, system_prompt=system_text or None),
                    tokenize=False)

                # print(f"Prompt: {inst}", flush=True)
                # print(f"Output: {response}", flush=True)

                dataframe.loc[len(dataframe)] = [inst, "", response, "", text]

        print('Prompts successfully generated', flush=True)
        del data
        return pd.concat(prompt_lists, ignore_index=True)
