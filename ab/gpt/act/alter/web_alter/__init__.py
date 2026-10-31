"""Alter with web knowledge -- own alter script for the M.Sc. thesis (Syed Faizan Shah).

Differences from ab.gpt.util.nn.AlterNN.alter():
  * explicit network list (CSV with column 'nn'); dataset fixed via the prompt config
  * LLM backend switch: local HF model (default) or a vLLM OpenAI-compatible server (--vllm_url)
  * optional {extra_info} per network or family (--extra_info JSON); without it = baseline
  * k samples per network with fixed seeds and fixed supporting models -> paired conditions
  * own output folder (never deletes earlier runs), results.jsonl and run_meta.json
"""
import argparse, ast, json, random, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from ab.gpt.util.Const import conf_test_dir, new_nn_file, new_out_file
from ab.gpt.util.nn.AlterNN import format_prompt_with_supporting_models
from ab.gpt.util.Util import extract_code

REPO = Path(__file__).resolve().parent  # git finds the repo root from any subfolder


def family(nn):
    return nn.split('-')[0]


def load_rows(set_csv, task, dataset):
    import ab.nn.api as api
    names = pd.read_csv(set_csv)['nn'].tolist()
    d = api.data(only_best_accuracy=True, task=task, dataset=dataset).drop_duplicates('nn')
    missing = sorted(set(names) - set(d.nn))
    if missing:
        sys.exit(f'[web_alter] not found for {task}/{dataset}: {missing}')
    d = d.set_index('nn', drop=False).loc[names]
    return [r for _, r in d.iterrows()]


def build_jobs(rows, samples, seed, extra):
    rng = random.Random(seed)  # depends only on rows/samples/seed -> identical across conditions
    jobs = []
    for row in rows:
        others = [r for r in rows if r['nn'] != row['nn']]
        for k in range(samples):
            jobs.append(dict(row=row, sup=rng.choice(others), k=k, seed=seed + k,
                             extra=extra.get(row['nn']) or extra.get(family(row['nn'])) or ''))
    return jobs


def make_messages(conf, job):
    template = ''.join(line + '\n' for line in conf['prompt'])
    para = {it['para']: job['row'][it['value']] for it in conf['input_list']}
    sm = [{it['para']: job['sup'][it['value']] for it in conf['addon_list']}]
    para['supporting_models'] = sm
    para.update(sm[0])
    para['extra_info'] = (conf.get('extra_info_header', '') + job['extra'].strip() + '\n') if job['extra'] else ''
    prompt = format_prompt_with_supporting_models(template, para, sm)
    if '{nn_code}' in prompt or '{extra_info}' in prompt:
        raise ValueError('prompt placeholders were not filled -- check the config')
    system_text = '\n'.join(conf.get('system', []))
    return ([{'role': 'system', 'content': system_text}] if system_text else []) + \
           [{'role': 'user', 'content': prompt}]


class VLLMBackend:
    def __init__(self, url, model, a):
        import httpx
        self.client, self.a = httpx.Client(timeout=a.timeout), a
        base = url.rstrip('/')
        self.url = base + '/chat/completions'
        self.model = model or self.client.get(base + '/models').json()['data'][0]['id']
        self.name = f'vllm:{self.model}@{base}'

    def generate(self, msgs, seed):
        a, err = self.a, None
        body = dict(model=self.model, messages=msgs, max_tokens=a.max_new_tokens, temperature=a.temperature,
                    top_p=a.top_p, top_k=a.top_k, seed=seed)
        for attempt in range(3):
            try:
                r = self.client.post(self.url, json=body)
                r.raise_for_status()
                j = r.json()
                c, u = j['choices'][0], j.get('usage', {})
                return c['message']['content'], c.get('finish_reason'), u.get('prompt_tokens'), u.get('completion_tokens')
            except Exception as e:
                err = e
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f'vLLM request failed after 3 attempts: {err}')


class LocalBackend:
    def __init__(self, llm_name, a):
        import torch
        from ab.gpt.util.llm.LLM import LLM
        loader = LLM(llm_name)
        self.model, self.tok, self.torch, self.a = loader.get_model(), loader.get_tokenizer(), torch, a
        self.model.eval()
        self.name = f'local:{llm_name}'

    def generate(self, msgs, seed):
        a = self.a
        text = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        if a.suppress_thinking:  # close the auto-opened <think> block (reasoning models such as OlympicCoder)
            text = text.rstrip() + ('</think>\n\n' if text.rstrip().endswith('<think>') else '<think></think>\n\n')
        ids = self.tok(text, return_tensors='pt', add_special_tokens=False).to(self.model.device)
        self.torch.manual_seed(seed)
        with self.torch.no_grad():
            o = self.model.generate(**ids, max_new_tokens=a.max_new_tokens, do_sample=True, temperature=a.temperature,
                                    top_k=a.top_k, top_p=a.top_p, eos_token_id=self.tok.eos_token_id)
        new = o[0][ids.input_ids.shape[1]:]
        finish = 'length' if len(new) >= a.max_new_tokens else 'stop'
        return self.tok.decode(new, skip_special_tokens=True), finish, int(ids.input_ids.shape[1]), int(len(new))


def run_job(backend, conf, job, out_dir, idx):
    row = job['row']
    rec = dict(B=idx, nn=row['nn'], fam=family(row['nn']), sample=job['k'], seed=job['seed'],
               supporting=job['sup']['nn'], extra_info=bool(job['extra']))
    t0 = time.time()
    try:
        msgs = make_messages(conf, job)
        out, finish, pt, ct = backend.generate(msgs, job['seed'])
    except Exception as e:
        rec.update(error=str(e)[:300], seconds=round(time.time() - t0, 1))
        return rec
    code = extract_code(out)
    parses = False
    if code:
        try:
            ast.parse(code)
            parses = True
        except SyntaxError:
            pass
    d = out_dir / ('synth_nn' if code else 'failed') / f'B{idx}'  # synth_nn holds only folders with code, like alter()
    d.mkdir(parents=True, exist_ok=True)
    (d / new_out_file).write_text(out)
    (d / 'prompt.json').write_text(json.dumps(msgs, indent=1))
    (d / f"original_{row['nn']}.py").write_text(row['nn_code'])
    row.to_pickle(d / 'dataframe.df')
    if code:
        (d / new_nn_file).write_text(code)
    rec.update(finish=finish, prompt_tokens=pt, completion_tokens=ct, code=bool(code), parses=parses,
               seconds=round(time.time() - t0, 1))
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--set', required=True, help="CSV with column 'nn' (the original networks)")
    p.add_argument('--conf', default='NN_alter_web.json', help='prompt config in ab/gpt/conf/prompt/test/')
    p.add_argument('--key', default=None, help='config key (default: first key)')
    p.add_argument('--samples', type=int, default=5, help='generations per network')
    p.add_argument('--seed', type=int, default=1000)
    p.add_argument('--extra_info', default=None, help='JSON {nn or family: text}; omit for the baseline condition')
    p.add_argument('--llm', default='open-r1/OlympicCoder-7B', help='local HF model (used when --vllm_url is not set)')
    p.add_argument('--vllm_url', default=None, help='e.g. http://132.187.14.67:30004/v1 -> use vLLM instead of local')
    p.add_argument('--vllm_model', default=None, help='served model id (default: first model the server lists)')
    p.add_argument('--workers', type=int, default=4, help='parallel requests (vLLM only)')
    p.add_argument('--suppress_thinking', action='store_true', help='local reasoning models only')
    p.add_argument('--temperature', type=float, default=0.6)
    p.add_argument('--top_k', type=int, default=50)
    p.add_argument('--top_p', type=float, default=0.95)
    p.add_argument('--max_new_tokens', type=int, default=4096)
    p.add_argument('--timeout', type=float, default=600)
    p.add_argument('--limit', type=int, default=None, help='only the first N jobs (testing)')
    p.add_argument('--out', default=str(Path.home() / 'web_alter_runs'))
    p.add_argument('--name', default=None, help='run folder name (default: timestamp_condition)')
    a = p.parse_args()

    all_conf = json.load(open(conf_test_dir / a.conf))
    key = a.key or next(iter(all_conf))
    conf = all_conf[key]
    extra = json.load(open(a.extra_info)) if a.extra_info else {}
    cond = 'web' if a.extra_info else 'base'
    out_dir = Path(a.out) / (a.name or f"{datetime.now():%Y%m%d-%H%M}_{cond}")
    if out_dir.exists():
        sys.exit(f'[web_alter] {out_dir} exists -- choose another --name')

    rows = load_rows(a.set, conf['task'], conf.get('dataset', 'cifar-10'))
    jobs = build_jobs(rows, a.samples, a.seed, extra)[:a.limit]
    backend = VLLMBackend(a.vllm_url, a.vllm_model, a) if a.vllm_url else LocalBackend(a.llm, a)

    out_dir.mkdir(parents=True)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO, text=True).strip()
    except Exception:
        commit = 'unknown'
    meta = dict(vars(a), condition=cond, conf_key=key, dataset=conf.get('dataset', 'cifar-10'), backend=backend.name,
                nn_gpt_commit=commit, nn_dataset=version('nn-dataset'), networks=len(rows), jobs=len(jobs),
                with_extra_info=sum(bool(j['extra']) for j in jobs), started=datetime.now().isoformat(timespec='seconds'))
    (out_dir / 'run_meta.json').write_text(json.dumps(meta, indent=1))
    print(f'[web_alter] {cond}: {len(jobs)} jobs, {len(rows)} networks, backend {backend.name} -> {out_dir}')

    workers = a.workers if a.vllm_url else 1
    n_code = n_parse = n_err = 0
    with open(out_dir / 'results.jsonl', 'w') as log, ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(run_job, backend, conf, j, out_dir, i) for i, j in enumerate(jobs)]
        for done, f in enumerate(as_completed(futs), 1):
            rec = f.result()
            log.write(json.dumps(rec) + '\n')
            log.flush()
            n_err += 'error' in rec
            n_code += bool(rec.get('code'))
            n_parse += bool(rec.get('parses'))
            print(f"[{done}/{len(jobs)}] B{rec['B']} {rec['nn'][:30]:30s} "
                  f"{rec.get('error') or 'finish=%s code=%s parses=%s %ss' % (rec['finish'], rec['code'], rec['parses'], rec['seconds'])}")
    print(f'[web_alter] done: {len(jobs)} jobs | errors {n_err} | code found {n_code} | parses {n_parse} -> {out_dir}')


if __name__ == '__main__':
    main()
