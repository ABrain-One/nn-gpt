import sys
import ab.nn.api as lemur
import ab.gpt.TuneNNGen as TuneNNGen

def add_normalization_to_lemur(epoch=5):
    if getattr(lemur, '_normalization_patched', False):
        print("[INFO] Normalization patch already applied, skipping.")
        return

    # Patch join_nn_query to include dataset_2 in SQL SELECT
    import ab.nn.util.db.Query as Q
    from ab.nn.util.db.Query import tmp_data, fill_hyper_prm


    def _fixed_join_nn_query(sql, cur):
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
            LIMIT 1000
        ''')
        return fill_hyper_prm(cur, sql.num_joint_nns)

    # Q.join_nn_query = _fixed_join_nn_query
    # print("[INFO] join_nn_query bug patched ✓")
    Q.join_nn_query = _fixed_join_nn_query

    # Also patch in Read.py where it's imported directly
    import ab.nn.util.db.Read as _R
    _R.join_nn_query = _fixed_join_nn_query

    print("[INFO] join_nn_query bug patched ✓")
    df_all = lemur.data(only_best_accuracy=False, task='img-classification')
    best_per_dataset = (
        df_all[df_all['epoch'] == epoch]
        .groupby('dataset')['accuracy']
        .max()
        .to_dict()
    )
    print(f"[INFO] Normalization baseline (epoch {epoch}):")
    for ds, best in sorted(best_per_dataset.items()):
        print(f"  {ds:<20} best={best:.4f}")

    original_lemur_data = lemur.data

    def patched_lemur_data(*args, **kwargs):
        df = original_lemur_data(*args, **kwargs)
        if 'dataset' in df.columns and 'accuracy' in df.columns:
            df['norm_acc'] = df.apply(
                lambda r: round(r['accuracy'] / best_per_dataset.get(r['dataset'], 1.0), 4), axis=1)
        if 'dataset_2' in df.columns and 'accuracy_2' in df.columns:
            df['norm_acc_2'] = df.apply(
                lambda r: round(r['accuracy_2'] / best_per_dataset.get(r['dataset_2'], 1.0), 4), axis=1)
        if 'norm_acc' in df.columns and 'norm_acc_2' in df.columns:
            df['better_dataset'] = df.apply(
                lambda r: r['dataset'] if r['norm_acc'] >= r['norm_acc_2'] else r['dataset_2'], axis=1)
        return df

    # Preserve cache_clear from original so lemur.data.cache_clear() still works
    if hasattr(original_lemur_data, 'cache_clear'):
        patched_lemur_data.cache_clear = original_lemur_data.cache_clear

    lemur.data = patched_lemur_data
    lemur._normalization_patched = True
    print("[INFO] Normalization patch applied to lemur.data ✓")

def main(dry_run=False):
    add_normalization_to_lemur(epoch=5)
    TuneNNGen.main(
        llm_conf='ds_coder_1.3b_instruct.json',
        llm_tune_conf='NN_dataset_compare.json',
        nn_gen_conf='NN_dataset_compare.json',
        nn_gen_conf_id='dataset_comparison',
        max_new_tokens=64,
        max_prompts=3 if dry_run else None,
        onnx_run=True
    )

if __name__ == '__main__':
    if '--test' in sys.argv:
        add_normalization_to_lemur(epoch=5)
        df = lemur.data(only_best_accuracy=False, task='img-classification')
        print(df[['nn', 'dataset', 'accuracy', 'norm_acc']].head(5).to_string(index=False))
    elif '--dry-run' in sys.argv:
        main(dry_run=True)
    else:
        main()
