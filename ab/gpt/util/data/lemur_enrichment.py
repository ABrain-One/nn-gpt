from __future__ import annotations

import json
from pathlib import Path

import ab.nn.api as lemur

_DATASET_META_PATH = Path(__file__).parent.parent / 'conf' / 'dataset_meta.json'
_META_FIELDS = ['num_train_images', 'img_size', 'num_channels', 'num_classes']

# ─────────────
# SQL patch
# ─────────────────

def patch_join_nn_query() -> bool:
    """
    Returns True when the patch was applied, False when it was skipped (already
    patched or LEMUR internals changed).
    """
    if getattr(lemur, '_join_nn_query_patched', False):
        return False

    try:
        import ab.nn.util.db.Query as Q
        from ab.nn.util.db.Query import tmp_data, fill_hyper_prm

        def _fixed_join_nn_query(sql, limit_clause, cur):
            join_conditions = []
            for c in (sql.same_columns or []):
                join_conditions.append(f'd1.{c} = d2.{c}')
            for c in (sql.diff_columns or []):
                join_conditions.append(f'd1.{c} != d2.{c}')
            join_conditions.append('d1.id < d2.id')
            on_clause = ' AND '.join(join_conditions)

            cur.execute(f'''
                SELECT d1.*, d2.nn AS nn_2, d2.nn_code AS nn_code_2,
                    d2.accuracy AS accuracy_2, d2.dataset AS dataset_2,
                    d2.epoch AS epoch_2, d2.metric AS metric_2,
                    d2.prm_id AS prm_id_2
                FROM {tmp_data} d1
                JOIN {tmp_data} d2 ON {on_clause}
                {limit_clause}
            ''')
            return fill_hyper_prm(cur, sql.num_joint_nns)

        Q.join_nn_query = _fixed_join_nn_query

        try:
            import ab.nn.util.db.Read as _R
            _R.join_nn_query = _fixed_join_nn_query
        except Exception:
            pass  

        lemur._join_nn_query_patched = True
        print('[INFO] join_nn_query bug patched ✓')
        return True

    except Exception as exc:
        print(f'[WARNING] Could not patch join_nn_query '
              f'(LEMUR internal structure may have changed): {exc}')
        return False


# ─────
# DataFrame enrichment
# ─────

def _load_dataset_meta() -> dict:
    try:
        return json.loads(_DATASET_META_PATH.read_text())
    except Exception as exc:
        print(f'[WARNING] Could not load dataset_meta.json: {exc}')
        return {}


def _compute_best_per_dataset(epoch: int) -> dict:
    """Return {dataset_name: best_accuracy} at the given epoch."""
    try:
        df_all = lemur.data(only_best_accuracy=False, task='img-classification')
        return (
            df_all[df_all['epoch'] == epoch]
            .groupby('dataset')['accuracy']
            .max()
            .to_dict()
        )
    except Exception as exc:
        print(f'[WARNING] Could not compute normalisation baseline: {exc}')
        return {}

# New 

"""
adds normalised-accuracy and dataset-metadata columns to any LEMUR DataFrame.  Every column addition is guarded by an
   existence check, so that if the schema changes in LEMUR dataset, only and silently skips the affected column instead of throwing an error.

Add normalisation and metadata columns to a LEMUR DataFrame. The function returns The same *df* with the new columns added.

"""

def enrich_dataframe(df, *, epoch: int = 5, dataset_meta: dict | None = None,
                     meta_fields: list[str] | None = None):
   
    if dataset_meta is None:
        dataset_meta = _load_dataset_meta()
    if meta_fields is None:
        meta_fields = _META_FIELDS

    best_per_dataset = _compute_best_per_dataset(epoch)

    if 'dataset' in df.columns and 'accuracy' in df.columns:
        df['norm_acc'] = df.apply(
            lambda r: round(r['accuracy'] / best_per_dataset.get(r['dataset'], 1.0), 4),
            axis=1,
        )

    if 'dataset_2' in df.columns and 'accuracy_2' in df.columns:
        df['norm_acc_2'] = df.apply(
            lambda r: round(r['accuracy_2'] / best_per_dataset.get(r['dataset_2'], 1.0), 4),
            axis=1,
        )

    if 'norm_acc' in df.columns and 'norm_acc_2' in df.columns:
        df['better_dataset'] = df.apply(
            lambda r: r['dataset'] if r['norm_acc'] >= r['norm_acc_2'] else r['dataset_2'],
            axis=1,
        )

    if 'dataset' in df.columns:
        for field in meta_fields:
            df[f'{field}_1'] = df['dataset'].map(
                lambda d, f=field: dataset_meta.get(d, {}).get(f)
            )

    if 'dataset_2' in df.columns:
        for field in meta_fields:
            df[f'{field}_2'] = df['dataset_2'].map(
                lambda d, f=field: dataset_meta.get(d, {}).get(f)
            )

    return df
