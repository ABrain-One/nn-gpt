from pathlib import Path


def test_backbone_generation_passes_dataset_filters_to_lemur_data():
    source = Path("ab/gpt/util/Tune.py").read_text(encoding="utf-8")

    assert 'data_kwargs["nn_prefixes"] = sft_nn_prefixes' in source
    assert 'data_kwargs["dataset"] = sft_dataset' in source
    assert "lemur.data(**data_kwargs)" in source
