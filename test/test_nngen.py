import sys
import os

from ab.nn.util.Const import ab_root_path

sys.path.insert(0, os.path.dirname(os.path.abspath('')))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath('')), '..', 'nn-dataset', 'nn-dataset'))

from ab.gpt.util.prompt.NNGenPrompt import NNGenPrompt
import ab.nn.api as lemur

data = lemur.data(only_best_accuracy=False, max_rows=10, sql=None)
print("lemur.data returned:", len(data))

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('open-r1/OlympicCoder-7B')

prompt = NNGenPrompt(512, tokenizer, ab_root_path / 'ab/gpt/conf/prompt/train/NN_gen.json')
df = prompt.get_raw_dataset(only_best_accuracy=False, n_training_prompts=10)
print("prompt.get_raw_dataset returned:", len(df))

