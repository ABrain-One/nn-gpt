import json
import pandas as pd
from ab.nn.util.Const import ab_root_path
from overrides import override
from transformers import AutoTokenizer

from ab.gpt.util.prompt.NNGenPromptPrun import NNGenPromptPrun

# Mock tokenizer
class MockTokenizer:
    def apply_chat_template(self, msgs, tokenize=False):
        return "\n".join([m['role'] + ": " + m['content'] for m in msgs])

# Mock Data
import ab.nn.api as lemur
lemur.prun_data = lambda max_rows=None: pd.DataFrame([
    {'nn': 'ModelA', 'model_name': 'ModelA', 'accuracy': 0.95, 'duration': 0.05, 'status': 'success'}
])
lemur.data = lambda only_best_accuracy=False, max_rows=None: pd.DataFrame([
    {'nn': 'ModelA', 'nn_code': 'codeA', 'transform_code': 'transA'}
])

tokenizer = MockTokenizer()
generator = NNGenPromptPrun(1000, tokenizer, ab_root_path / 'ab/gpt/conf/prompt/train/NN_gen_train_efficiency_prun.json')

df = generator.get_raw_dataset(only_best_accuracy=True)
print("\nGenerated dataset length:", len(df))
print("\nFirst row text:\n")
print(df.iloc[0]['text'])

