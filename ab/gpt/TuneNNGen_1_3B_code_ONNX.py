import ab.gpt.TuneNNGen as TuneNNGen

def add_normalization_to_lemur(epoch=5):
    """
    Wraps lemur.data() once to inject norm_acc, addon_norm_acc, better_dataset.
    To be called BEFORE the main function TuneNNGen.main().
    """
    # prevent double-patching (Guard)
    if getattr(lemur, '_normalization_patched', False):
        print("[INFO] Normalization patch already applied, skipping.")
        return

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
                lambda r: round(r['accuracy'] / best_per_dataset.get(r['dataset'], 1.0), 4),
                axis=1
            )

        if 'addon_dataset' in df.columns and 'addon_accuracy' in df.columns:
            df['addon_norm_acc'] = df.apply(
                lambda r: round(r['addon_accuracy'] / best_per_dataset.get(r['addon_dataset'], 1.0), 4),
                axis=1
            )

        if 'norm_acc' in df.columns and 'addon_norm_acc' in df.columns:
            df['better_dataset'] = df.apply(
                lambda r: r['dataset'] if r['norm_acc'] >= r['addon_norm_acc'] else r['addon_dataset'],
                axis=1
            )

        return df

    lemur.data = patched_lemur_data
    lemur._normalization_patched = True  # Guard flag
    print("[INFO] Normalization patch applied to lemur.data ✓")


def test_patch():
    """Validate normalization patch without running the LLM."""
    add_normalization_to_lemur(epoch=5)

    df = lemur.data(only_best_accuracy=False, task='img-classification')

    cols = ['nn', 'dataset', 'accuracy', 'norm_acc']
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"\n[ERROR] Missing columns: {missing}")
    else:
        print("\n[SUCCESS] Patch working correctly!")
        print(f"Total rows: {len(df)}")
        print(f"\nSample rows:")
        print(df[cols].head(5).to_string(index=False))

def main():
    add_normalization_to_lemur(epoch=5)

    TuneNNGen.main(
        llm_conf='ds_coder_1.3b_instruct.json',
        nn_gen_conf='NN_dataset_compare.json',
        nn_gen_conf_id='dataset_comparison',
        max_new_tokens=4 * 1024,
        max_prompts=3 if dry_run else None,
        onnx_run=True
    )


if __name__ == '__main__':
    if '--test' in sys.argv:
        test_patch()
    elif '--dry-run' in sys.argv:
        main(dry_run=True)
    else:
        main()
