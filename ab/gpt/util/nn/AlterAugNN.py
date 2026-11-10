"""Generate (augmentation schedule, network) pairs with an LLM.

Same shape as AlterNN.alter(): build prompts, generate, save every valid answer to
epoch_dir(e)/synth_nn/B<i>/ (new_nn.py, full_output.txt, original_<nn>.py, dataframe.df).

"""
import ast
import json
import random
import re
import shutil
from collections import Counter

import torch
from tqdm import tqdm

import ab.nn.api as nn_dataset
from ab.nn.util.Util import create_file
from ab.gpt.brute.trans.augment.ScheduleValidate import validate_schedule, ScheduleError
from ab.gpt.util.Const import conf_test_dir, epoch_dir, new_nn_file, synth_dir, new_out_file
from ab.gpt.util.llm.LLM import LLM
from ab.gpt.util.llm.VLLMClient import VLLMClient
from ab.gpt.util.Util import extract_code

# Same training protocol as AugEval, so generated candidates are comparable with the schedule

PROTOCOL = {'lr': 0.01, 'batch': 64, 'dropout': 0.2, 'momentum': 0.9, 'transform': 'norm_256'}

# The generated Net must carry the train_setup/learn 

REQUIRED_NN_PARTS = ('def supported_hyperparameters', 'def train_setup', 'def learn', 'batch_transform_fn')

_SCHEDULE_TAG = re.compile(r'<(tr|sched)>(.*?)</\1>', re.S)


def load_reference_pool(task, dataset, metric, nn_prefixes, pool_top, max_chars):
    """The best `pool_top` distinct LEMUR networks for this task/dataset whose code is short enough
    to put two of them in one prompt."""
    kw = {'nn_prefixes': tuple(nn_prefixes)} if nn_prefixes else {}
    df = nn_dataset.data(only_best_accuracy=True, task=task, dataset=dataset, metric=metric, **kw)
    df = df[df['nn_code'].notna() & (df['nn_code'].str.len() <= max_chars)]
    df = df.sort_values('accuracy', ascending=False).drop_duplicates('nn').drop_duplicates('nn_code')
    pool = df.head(pool_top).reset_index(drop=True)
    if len(pool) < 2:
        raise ValueError(f"Need at least 2 reference networks for {task}/{dataset}/{metric}"
                         f"{' with prefixes ' + str(nn_prefixes) if nn_prefixes else ''} "
                         f"(code <= {max_chars} chars); found {len(pool)}")
    return pool


def build_prompt(cfg, ref1, ref2):
    """input_list is read from reference network 1, addon_list from reference network 2."""
    para = {it['para']: ref1[it['value']] for it in cfg['input_list']}
    para.update({it['para']: ref2[it['value']] for it in cfg['addon_list']})
    return "\n".join(cfg['prompt']).format(**para)   # '{{' / '}}' in the template are literal braces


def extract_schedule_block(out):
    """The last <tr>..</tr> (or <sched>..</sched>) block before the <nn> block."""
    head = out.split('<nn>')[0] if '<nn>' in out else out
    blocks = _SCHEDULE_TAG.findall(head)
    return blocks[-1][1].strip() if blocks else None


def parse_schedule(text):
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        try:
            raw = ast.literal_eval(text)            
        except (ValueError, SyntaxError):
            raise ScheduleError('not a list')
    schedule, _ = validate_schedule(raw, policy='reject')
    return schedule


def parse_answer(out, ref_codes):
    """Return (schedule, nn_code, error)."""
    if not out:
        return None, None, 'empty answer'
    block = extract_schedule_block(out)
    if not block:
        return None, None, 'no <tr> block'
    try:
        schedule = parse_schedule(block)
    except ScheduleError as e:
        return None, None, f'invalid schedule: {e}'
    nn_code = extract_code(out)
    if not nn_code:
        return None, None, 'no <nn> code'
    missing = [p for p in REQUIRED_NN_PARTS if p not in nn_code]
    if missing:
        return None, None, f'network lacks {missing}'
    if not re.search(r"def supported_hyperparameters\(\).*?'augment'", nn_code, re.S):
        return None, None, "supported_hyperparameters does not declare 'augment'"
    if nn_code.strip() in ref_codes:
        return None, None, 'network copies a reference'
    return schedule, nn_code, None


def save_candidate(out_path, idx, out, schedule, nn_code, ref1, ref2, task, dataset, metric):
    model_dir = synth_dir(out_path) / f'B{idx}'
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / new_nn_file).write_text(nn_code)
    create_file(model_dir, new_out_file, out)
    (model_dir / 'sched.json').write_text(json.dumps(schedule))
    (model_dir / f"original_{ref1['nn']}.py").write_text(ref1['nn_code'])
    (model_dir / 'references.json').write_text(json.dumps({
        'reference_1': {'nn': ref1['nn'], 'accuracy': float(ref1['accuracy'])},
        'reference_2': {'nn': ref2['nn'], 'accuracy': float(ref2['accuracy'])}}, indent=2))
    row = ref1.copy()
    # NNEval reads task/dataset/metric from the row; without them it falls back to cifar-10.
    row['task'], row['dataset'], row['metric'] = task, dataset, metric
    prm = dict(row['prm']) if isinstance(row.get('prm'), dict) else {}
    prm.update(PROTOCOL)                   # not the reference network's own lr/batch/dropout
    prm['augment'] = schedule              # NNEval merges dataframe.df's prm into the training prm
    row['prm'] = prm
    row.to_pickle(model_dir / 'dataframe.df')


def _generate(prompts, remote, model, tokenizer, temperature, top_k, max_new_tokens):
    """One answer (str or None) per (system, prompt) pair."""
    if remote is not None:
        outs = [None] * len(prompts)
        for i, out in remote.chat_iter(prompts):
            outs[i] = out
        return outs
    outs = []
    for system_text, prompt in tqdm(prompts, desc='Generate schedule+network'):
        msgs = [{'role': 'system', 'content': system_text}] if system_text else []
        msgs.append({'role': 'user', 'content': prompt})
        inputs = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt')
        inputs = (inputs.input_ids if hasattr(inputs, 'input_ids') else inputs).to(model.device)
        with torch.no_grad():
            model.eval()
            o = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                               top_k=top_k, top_p=0.95, num_return_sequences=1,
                               eos_token_id=tokenizer.eos_token_id)
        outs.append(tokenizer.decode(o[0][len(inputs[0]):], skip_special_tokens=True))
    return outs


def alter_aug(epochs, test_conf, llm_name, gguf_file=None, n_prompts=10, pool_top=30, nn_prefixes=None,
              dataset='imagenette', metric='acc', max_ref_chars=6000, temperature=0.8, top_k=100,
              max_new_tokens=8 * 1024, seed=None, **kwargs):
    with open(conf_test_dir / test_conf) as f:
        prompt_dict = json.load(f)
    assert isinstance(prompt_dict, dict)

    rng = random.Random(seed)

    vllm_url = kwargs.get('vllm_url')
    remote = model = tokenizer = None
    if vllm_url:
        remote = VLLMClient(vllm_url, model=kwargs.get('vllm_model'), temperature=temperature, top_k=top_k,
                            top_p=0.95, max_tokens=kwargs.get('vllm_max_tokens'),
                            workers=kwargs.get('vllm_workers', 4))
        print(f"Using remote LLM {remote.model} at {remote.base_url}")
    else:
        loader = LLM(llm_name, gguf_file=gguf_file, load_in_4bit=kwargs.get('load_in_4bit', False))
        model, tokenizer = loader.get_model(), loader.get_tokenizer()
        print("Load Model Complete, Start Loop...")

    shutil.rmtree(epoch_dir(), ignore_errors=True)
    for epoch in range(epochs):
        out_path = epoch_dir(epoch)

        jobs = []                                   # ((system, prompt), ref1, ref2, task)
        for key, cfg in prompt_dict.items():
            pool = load_reference_pool(cfg['task'], dataset, metric, nn_prefixes, pool_top, max_ref_chars)
            system_text = "\n".join(cfg.get('system', []))
            for _ in range(n_prompts):
                a, b = rng.sample(range(len(pool)), 2)
                ref1, ref2 = pool.iloc[a], pool.iloc[b]
                jobs.append(((system_text, build_prompt(cfg, ref1, ref2)), ref1, ref2, cfg['task']))

        outs = _generate([j[0] for j in jobs], remote, model, tokenizer, temperature, top_k, max_new_tokens)

        saved, rejected = 0, Counter()
        for out, (_, ref1, ref2, task) in zip(outs, jobs):
            refs = {ref1['nn_code'].strip(), ref2['nn_code'].strip()}
            schedule, nn_code, err = parse_answer(out, refs)
            if err:
                rejected[err.split(':')[0]] += 1
                print(f"[INFO] Response invalid: {err}")
                continue
            save_candidate(out_path, saved, out, schedule, nn_code, ref1, ref2, task, dataset, metric)
            saved += 1
        print(f"[alter_aug] epoch {epoch}: {len(jobs)} prompts, {saved} saved, rejected {dict(rejected)}")