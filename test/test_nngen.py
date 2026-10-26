"""NNGenPrompt turns LEMUR rows into training prompts.

NN_gen.json asks for pairs, and that query joins every row of the stat table
with a randomly ordered partner, which takes minutes. The default run feeds
the builder stub rows instead, so it checks the prompt, not the database.

NNGPT_LEMUR_E2E=1 runs the same check against the real query.
NNGPT_TEST_TOKENIZER=<model> adds the case that needs a tokenizer from the hub.
"""
import os
import unittest

try:
    import pandas as pd
    from ab.nn.util.Const import ab_root_path
    import ab.nn.api as lemur
    from ab.gpt.util.prompt.NNGenPrompt import NNGenPrompt
except ImportError as exc:
    from _unittest_bridge import skip_module

    skip_module(f"test_nngen needs ab.nn and ab.gpt: {exc}")

CONFIG = ab_root_path / 'ab/gpt/conf/prompt/train/NN_gen.json'
TOKENIZER = os.getenv("NNGPT_TEST_TOKENIZER")
E2E = os.getenv("NNGPT_LEMUR_E2E") == "1"

BASE = {
    'nn': 'NetA', 'nn_code': 'class Net: pass', 'accuracy': 0.6, 'epoch': 1,
    'task': 'img-classification', 'dataset': 'cifar-10', 'metric': 'acc',
    'prm_id': 1, 'prm': '{}', 'transform_code': 'def transform(n): return n',
}
PAIR = {**BASE, **{f"{k}_2": v for k, v in BASE.items()},
        'nn_2': 'NetB', 'nn_code_2': 'class Net: better', 'accuracy_2': 0.8}


class _CharTokenizer:
    """Enough of a tokenizer to template and measure a prompt."""

    def apply_chat_template(self, msgs, tokenize=False, **kwargs):
        text = "\n".join(m['role'] + ": " + m['content'] for m in msgs)
        return text if not tokenize else list(range(len(text)))

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text)))}


def _rows(**kwargs):
    return pd.DataFrame([PAIR])


class TestNNGenPrompt(unittest.TestCase):
    def setUp(self):
        self._saved = lemur.data
        lemur.data = _rows
        self.addCleanup(self._restore)

    def _restore(self):
        lemur.data = self._saved

    def _build(self):
        return NNGenPrompt(512, _CharTokenizer(), CONFIG).get_raw_dataset(
            only_best_accuracy=False, n_training_prompts=1)

    def test_a_pair_of_rows_becomes_a_training_prompt(self):
        frame = self._build()
        self.assertGreater(len(frame), 0, "no training prompt was built")
        self.assertIn("text", frame.columns)
        text = frame.iloc[0]['text']
        self.assertIn('class Net: pass', text, "the baseline is not in the prompt")

    @unittest.skipUnless(E2E, "set NNGPT_LEMUR_E2E=1 to query LEMUR")
    def test_the_real_query_feeds_the_builder(self):
        self._restore()
        try:
            rows = lemur.data(only_best_accuracy=False, max_rows=10, sql=None)
        except Exception as exc:
            self.skipTest(f"the LEMUR database is unavailable: {exc}")
        if len(rows) == 0:
            self.skipTest("the LEMUR database returned no rows")
        self.assertGreater(len(self._build()), 0, "no training prompt was built")

    @unittest.skipUnless(TOKENIZER, "set NNGPT_TEST_TOKENIZER to a model name")
    def test_prompts_are_built_with_the_real_tokenizer(self):
        from transformers import AutoTokenizer

        frame = NNGenPrompt(512, AutoTokenizer.from_pretrained(TOKENIZER),
                            CONFIG).get_raw_dataset(
            only_best_accuracy=False, n_training_prompts=1)
        self.assertGreater(len(frame), 0, "no training prompt was built")


if __name__ == '__main__':
    unittest.main()
