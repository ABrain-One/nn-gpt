from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_backbone_generation_passes_dataset_filters_to_lemur_data():
    source = (ROOT / "ab/gpt/util/tune/Tune.py").read_text(encoding="utf-8")

    assert 'data_kwargs["nn_prefixes"] = sft_nn_prefixes' in source
    assert 'data_kwargs["dataset"] = sft_dataset' in source
    assert "lemur.data(**data_kwargs)" in source


from _unittest_bridge import load_tests_from  # noqa: E402

load_tests = load_tests_from(globals())
