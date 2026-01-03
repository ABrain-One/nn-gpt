from ab.nn.util.Const import base_module, ab_root_path, out_dir

import json

new_nn_file = 'new_nn.py'
hp_file = 'hp.txt'
transformer_file = 'tr.py'
new_out_file = 'full_output.txt'

gpt = 'gpt'
gpt_dir = ab_root_path / base_module / gpt

conf_dir = gpt_dir / 'conf'
conf_prompt_dir = conf_dir / 'prompt'
conf_test_dir = conf_prompt_dir / 'test'
conf_train_dir = conf_prompt_dir / 'train'
conf_llm_dir = conf_dir / 'llm'

nngpt_dir = out_dir / 'nngpt'
acgpt_dir = out_dir / 'acgpt'
nnrag_dir = out_dir / 'rag'

new_dataset_dir = nngpt_dir / 'new_lemur'
new_lemur_nn_dir = new_dataset_dir / 'nn'
new_lemur_stat_dir = new_dataset_dir / 'train'

brute_dir = gpt_dir / 'brute'
ast_dir = brute_dir / 'ast'
ea_dir = brute_dir / 'ea'
fract_dir = brute_dir / 'fract'
trans_dir = brute_dir / 'trans'

config_file = conf_llm_dir / 'nngpt_ds_coder_1.3b_instruct.json'

with open(config_file) as f:
    base_llm = json.load(f)['base_model_name']


# Hugging Face cache directories
huggingface_cache = out_dir / 'llm'
huggingface_tokenizer_cache = out_dir / 'tokenizer'
default_huggingface_cache = huggingface_cache
default_huggingface_tokenizer_cache = huggingface_tokenizer_cache


def model_dir(base):
    return base / 'llm'

def synth_dir(base):
    return base / 'synth_nn'

def tokenizer_dir(base):
    return base / 'tokenizer'

nngpt_model = model_dir(out_dir)
nngpt_upload = nngpt_model / 'upload'
llm_tokenizer_out = tokenizer_dir(out_dir)

def llm_dir(base, name):
    return model_dir(base) / name

def llm_tokenizer_dir(base, name):
    return tokenizer_dir(base) / name

def epoch_dir(*args):
    e_dir = llm_dir(nngpt_dir, 'epoch')
    for d in args:
        e_dir = e_dir / f'A{d}'
    return e_dir
