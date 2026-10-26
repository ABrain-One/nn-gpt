"""The pruning prompt is filled from the prun table.

Stubs replace ab.nn.api per test and are undone afterwards, so no database is
needed and nothing leaks into the tests that query LEMUR for real.
"""
import unittest

try:
    import pandas as pd
    from ab.nn.util.Const import ab_root_path
    import ab.nn.api as lemur
    from ab.gpt.util.prompt.NNGenPrompt import NNGenPrompt
except ImportError as exc:
    from _unittest_bridge import skip_module

    skip_module(f"test_prun needs ab.nn and ab.gpt: {exc}")

CONFIG = ab_root_path / 'ab/gpt/conf/prompt/test/NN_gen_prun.json'

ROW = {'nn': 'ModelA', 'model_name': 'ModelA', 'accuracy': 0.95,
       'duration': 0.05, 'pruning_ratio': 0.4, 'params_before': 1000,
       'params_after': 600, 'status': 'success'}


class MockTokenizer:
    def apply_chat_template(self, msgs, tokenize=False):
        return "\n".join([m['role'] + ": " + m['content'] for m in msgs])


def _prun_data(max_rows=None, **kwargs):
    return pd.DataFrame([ROW, dict(ROW, model_name='ModelB', status='failed')])


def _data(only_best_accuracy=False, max_rows=None, **kwargs):
    return pd.DataFrame([
        {'nn': 'ModelA', 'nn_code': 'codeA', 'transform_code': 'transA'}
    ])


class TestPrunPrompt(unittest.TestCase):
    def setUp(self):
        self._saved = (lemur.prun_data, lemur.data)
        lemur.prun_data, lemur.data = _prun_data, _data
        self.addCleanup(self._restore)

    def _restore(self):
        lemur.prun_data, lemur.data = self._saved

    def test_a_prompt_is_built_from_a_successful_pruning_row(self):
        frame = NNGenPrompt(1000, MockTokenizer(), CONFIG).get_raw_dataset(
            only_best_accuracy=True)

        self.assertEqual(1, len(frame), "the failed run was not filtered out")
        text = frame.iloc[0]['text']
        self.assertIn('ModelA', text)
        # accuracy / duration and params_after / params_before, evaluated
        self.assertIn('19.0000', text)
        self.assertIn('0.6000', text)
        self.assertNotIn('Missing:', text)


if __name__ == '__main__':
    unittest.main()
