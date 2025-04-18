from ab.nn.util.Const import base_module, ab_root_path, out_dir
import json

new_nn_file = 'new_nn.py'
gpt = 'gpt'

gpt_dir = ab_root_path / base_module / gpt
conf_dir = gpt_dir / 'conf'
nngpt_dir = out_dir / 'nngpt'
acgpt_dir = out_dir / 'acgpt'
config_file = conf_dir / 'config.json'


with open(config_file) as f:
    base_llm = json.load(f)['base_model_name']

def model_dir(base):
    return base / 'llm'


def synth_dir(base):
    return base / 'synth_nn'


def tokenizer_dir(base):
    return base / 'tokenizer'


nngpt_model = model_dir(out_dir)
nngpt_upload = nngpt_model / 'upload'
llm_tokenizer_out = tokenizer_dir(nngpt_model)


def llm_dir(base, name):
    return model_dir(base) / name


def llm_tokenizer_dir(base, name):
    return tokenizer_dir(base) / name


def epoch_dir(epoch):
    return llm_dir(nngpt_dir, 'epoch') / f'A{epoch}'
