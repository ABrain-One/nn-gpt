"""The vLLM path must sample like model.generate(), which inherits unset fields from generation_config.json."""
import json

try:
    from ab.gpt.util.llm.VLLMSleep import inherited_sampling_params, read_generation_config
except ImportError as exc:
    from _unittest_bridge import skip_module

    skip_module(f"test_vllm_sampling_inherit needs ab.gpt (and so torch): {exc}")

QWEN_CODER = {"do_sample": True, "eos_token_id": [151645, 151643], "repetition_penalty": 1.1,
              "temperature": 0.7, "top_k": 20, "top_p": 0.8}


def test_repetition_penalty_is_inherited():
    kwargs, ignored = inherited_sampling_params(QWEN_CODER)
    assert kwargs == {"repetition_penalty": 1.1}
    assert ignored == {}


def test_explicit_override_wins_and_none_does_not():
    assert inherited_sampling_params(QWEN_CODER, {"repetition_penalty": 1.0})[0] == {"repetition_penalty": 1.0}
    assert inherited_sampling_params(QWEN_CODER, {"repetition_penalty": None})[0] == {"repetition_penalty": 1.1}


def test_neutral_values_and_hf_only_fields():
    kwargs, ignored = inherited_sampling_params({"repetition_penalty": 1.0, "min_p": 0.05,
                                                 "no_repeat_ngram_size": 3})
    assert kwargs == {"min_p": 0.05}
    assert ignored == {"no_repeat_ngram_size": 3}


def test_read_generation_config_from_folder(tmp_path):
    (tmp_path / "generation_config.json").write_text(json.dumps(QWEN_CODER))
    cfg, src = read_generation_config(None, str(tmp_path))
    assert cfg["repetition_penalty"] == 1.1 and src.endswith("generation_config.json")
    assert read_generation_config(str(tmp_path / "missing"))[0] == {}


from _unittest_bridge import load_tests_from  # noqa: E402

load_tests = load_tests_from(globals())
