"""The SFT corpus must reach the trainer as prompt/completion pairs.

Regression: get_dataset() pre-tokenized and dropped every text column, so
LoRA.train() saw only input_ids, logged "Dataset does not have 'text' field.
Using standard Trainer." and computed the loss over prompt AND response. The
prompt carries the baseline network and is several times longer than the
target, so most of each update taught the model to echo prompts.
"""
import importlib.util
import json
import sys
from pathlib import Path

from _unittest_bridge import load_tests_from, pytest  # stdlib stand-in without pytest

ROOT = Path(__file__).resolve().parents[1]

CHAT = ("<|im_start|>system\nYou are an architect.<|im_end|>\n"
        "<|im_start|>user\nBaseline <nn>class Net: pass</nn><|im_end|>\n"
        "<|im_start|>assistant\n{response}<|im_end|>\n")
RESPONSE = "<hp>{\"lr\": 0.01}</hp>\n<tr>t</tr>\n<nn>class Net: improved</nn>"


def _prompt_module():
    """Load Prompt.py in isolation.

    transformers is only needed for a type annotation here, so it is stubbed
    when absent; that keeps this test runnable without the training stack.
    """
    if "transformers" not in sys.modules:
        try:
            import transformers  # noqa: F401
        except ImportError:
            stub = type(sys)("transformers")
            stub.PreTrainedTokenizerBase = type("PreTrainedTokenizerBase", (), {})
            sys.modules["transformers"] = stub
    spec = importlib.util.spec_from_file_location(
        "_prompt_under_test", ROOT / "ab/gpt/util/prompt/Prompt.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_split_is_lossless_and_puts_the_answer_in_the_completion():
    P = _prompt_module().Prompt
    text = CHAT.format(response=RESPONSE)
    prompt, completion = P._split_prompt_completion(text, RESPONSE)

    assert prompt + completion == text          # nothing lost or shifted
    assert RESPONSE in completion
    assert RESPONSE not in prompt
    assert prompt.endswith("<|im_start|>assistant\n")   # answer starts after the header
    # the prompt is the bulk of the sequence -- which is why masking matters
    assert len(prompt) > len(completion) / 2


def test_split_refuses_unusable_rows():
    P = _prompt_module().Prompt
    assert P._split_prompt_completion("", "x") == (None, None)
    assert P._split_prompt_completion("text", "") == (None, None)
    assert P._split_prompt_completion("no answer here", "missing") == (None, None)
    # a response at position 0 leaves no prompt
    assert P._split_prompt_completion("abc", "abc") == (None, None)


def test_lora_train_branches_on_the_resulting_columns():
    # mirrors the dispatch at the top of LoRA.train()
    for columns, expect_sft, expect_masked in (
        ({"input_ids", "attention_mask"}, False, False),      # the old behaviour
        ({"prompt", "completion"}, True, True),               # after this patch
    ):
        has_text = "text" in columns
        has_pc = {"prompt", "completion"}.issubset(columns)
        assert (has_text or has_pc) is expect_sft
        assert has_pc is expect_masked


def test_hyperparameters_in_targets_are_json_not_python_repr():
    prm = {"batch": 64, "transform": "norm_32_flip", "lr": 0.002}
    rendered = json.dumps(prm, default=str)
    assert "'" not in rendered                      # repr() would use single quotes
    assert json.loads(rendered) == prm              # the extractor can parse it
    assert "'" in str(prm)                          # what was emitted before
    with pytest.raises(json.JSONDecodeError):
        json.loads(str(prm))


# --- end-to-end: filtering must stay aligned with the split ---------------

_CHAT = ("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
         "<|im_start|>assistant\n{resp}<|im_end|>\n")


class _Tok:
    def __call__(self, text=None, truncation=False, max_length=None, texts=None):
        texts = text if text is not None else texts
        if isinstance(texts, str):
            texts = [texts]
        ids = [list(range(min(len(t), max_length or 10 ** 9))) for t in texts]
        return {"input_ids": ids, "attention_mask": [[1] * len(i) for i in ids]}


def test_get_dataset_emits_aligned_prompt_completion_pairs():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("datasets")
    P = _prompt_module().Prompt

    class _Stub(P):
        def __init__(self, frame, max_len):
            self.max_len = max_len
            self.tokenizer = _Tok()
            self._frame = frame

        def get_raw_dataset(self, only_best_accuracy, max_prompts=None):
            return self._frame

    rows = []
    # row 1 is deliberately over the length limit and must be filtered out
    for i, pad in enumerate(["x" * 50, "y" * 5000, "z" * 60, "w" * 70]):
        resp = f'<hp>{{"lr": 0.0{i}}}</hp><nn>net{i}</nn>'
        rows.append({"instruction": f"u{i} {pad}", "context": "", "response": resp,
                     "category": "train", "text": _CHAT.format(user=f"u{i} {pad}", resp=resp)})
    frame = pd.DataFrame(rows, columns=["instruction", "context", "response", "category", "text"])

    ds = _Stub(frame, max_len=1000).get_dataset(max_new_tokens=4096, seed=1)

    assert set(ds.column_names) == {"prompt", "completion"}
    assert len(ds) == 3
    for row in ds:
        assert row["completion"].startswith("<hp>")
        # the filtered row's payload must not have leaked into a survivor
        assert "y" * 100 not in row["prompt"]
    assert not any('"lr": 0.01' in row["completion"] for row in ds)


def test_full_sequence_training_is_still_available():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("datasets")
    P = _prompt_module().Prompt

    class _Stub(P):
        def __init__(self, frame, max_len):
            self.max_len = max_len
            self.tokenizer = _Tok()
            self._frame = frame

        def get_raw_dataset(self, only_best_accuracy, max_prompts=None):
            return self._frame

    resp = '<hp>{"lr": 0.1}</hp><nn>net</nn>'
    frame = pd.DataFrame([{"instruction": "u", "context": "", "response": resp,
                           "category": "train", "text": _CHAT.format(user="u", resp=resp)}],
                         columns=["instruction", "context", "response", "category", "text"])
    ds = _Stub(frame, max_len=1000).get_dataset(max_new_tokens=4096, as_prompt_completion=False)
    assert "input_ids" in ds.column_names
    assert "prompt" not in ds.column_names


load_tests = load_tests_from(globals())
