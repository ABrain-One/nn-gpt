"""Minimal client for an OpenAI-compatible vLLM server (chat completions).

Lets pipelines such as NNAlter use a remotely served LLM (e.g. a 70B model)
instead of loading a local one.
"""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from tqdm import tqdm


class VLLMClient:
    def __init__(self, base_url, model=None, temperature=0.6, top_k=50, top_p=0.95,
                 max_tokens=None, workers=4, timeout=600, retries=3):
        """
        base_url:   server address, e.g. 'http://132.187.14.67:30004/v1' ('/v1' is added if missing)
        model:      served model id; default = first model the server lists
        max_tokens: None lets the server use the remaining context length
        workers:    number of parallel requests
        """
        self.base_url = base_url.rstrip('/')
        if not self.base_url.endswith('/v1'):
            self.base_url += '/v1'
        self.http = httpx.Client(timeout=timeout)
        self.model = model or self.http.get(f'{self.base_url}/models').json()['data'][0]['id']
        self.params = dict(temperature=temperature, top_k=top_k, top_p=top_p)
        if max_tokens:
            self.params['max_tokens'] = max_tokens
        self.workers, self.retries = max(1, int(workers)), retries

    def chat(self, system_text, user_text, seed=None):
        """One chat completion; returns the generated text."""
        msgs = [{'role': 'system', 'content': system_text}] if system_text else []
        msgs.append({'role': 'user', 'content': user_text})
        body = dict(model=self.model, messages=msgs, **self.params)
        if seed is not None:
            body['seed'] = seed
        err = None
        for attempt in range(self.retries):
            r = None
            try:
                r = self.http.post(f'{self.base_url}/chat/completions', json=body)
            except httpx.HTTPError as e:  # connection problems: retry
                err = e
            if r is not None:
                if r.status_code == 200:
                    return r.json()['choices'][0]['message']['content']
                err = f'HTTP {r.status_code}: {r.text[:300]}'
                if r.status_code < 500:  # e.g. prompt longer than the served context: retrying will not help
                    break
            time.sleep(5 * (attempt + 1))
        raise RuntimeError(f'vLLM request failed: {err}')

    def chat_iter(self, pairs, desc='Generate Codes (vLLM)'):
        """Run (system_text, user_text) pairs in parallel; yield (index, text or None) as they complete."""
        def one(pair):
            try:
                return self.chat(*pair)
            except Exception as e:
                print(f'[VLLM] {e}')
                return None

        with ThreadPoolExecutor(self.workers) as ex:
            futures = {ex.submit(one, p): i for i, p in enumerate(pairs)}
            for f in tqdm(as_completed(futures), total=len(futures), desc=desc):
                yield futures[f], f.result()
