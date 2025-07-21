from __future__ import annotations
import ast, hashlib, json, random, re, shutil
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ab.rag.retriever import Retriever, BLOCKS_100
from ..util.Const import conf_test_dir, epoch_dir, synth_dir, new_out_file
from ..util.Util import extract_code
from ab.nn.util.Util import create_file

MAX_TRIES = 3
LLM_MAX_TOKENS = 3500
random.seed(42)


blocks_dir = Path(__file__).resolve().parents[3] / "blocks"
blocks_dir.mkdir(exist_ok=True)  

retriever = Retriever() 


def _load_block(name: str) -> str | None:
    """
    Return bundled code for <name>.

    • If blocks/<name>.py exists, read & return it.
    • Otherwise call Retriever.get_block(), save it under blocks/, then return.
    • If Retriever fails, return None.
    """
    local_path = blocks_dir / f"{name}.py"
    if local_path.is_file():
        print(f"[CACHE] {name}")
        return local_path.read_text()

    print(f"[FETCH] {name}")
    code = retriever.get_block(name)
    if code:
        local_path.write_text(code)
    return code


def _checksum(src: str) -> str:
    return hashlib.sha1(src.encode()).hexdigest()[:8]

def alter(epochs: int, test_conf: str, llm_name: str) -> None:
    with open(conf_test_dir / test_conf) as f:
        template = json.load(f)["single_block_model"]

    print("Loading LLM …")
    tok = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        llm_name, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).cuda()
    print("Ready ✔")

    shutil.rmtree(epoch_dir(), ignore_errors=True)

    for ep in range(epochs):
        queue: List[str] = BLOCKS_100.copy()
        base_out = epoch_dir(ep)
        idx = 0

        while queue:
            block_name = queue.pop(random.randrange(len(queue)))
            block_code = _load_block(block_name)
            if block_code is None:
                continue

            m_cls = re.search(r'^\s*class\s+(\w+)', block_code, flags=re.MULTILINE)
            block_cls = m_cls.group(1) if m_cls else block_name
            m_sig = re.search(r'^\s*def __init__\([^\)]*\)', block_code, flags=re.MULTILINE)
            init_sig = m_sig.group(0) if m_sig else "def __init__(self):"

            prompt_base = "\n".join(template["prompt"]) \
                .replace("{block_name}", block_name) \
                .replace("{block_class}", block_cls) \
                .replace("{init_signature}", init_sig)

            for attempt in range(MAX_TRIES):
                prompt = prompt_base if attempt == 0 else prompt_base + f"\n# retry {attempt}"
                inp = tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(model.device)

                gen = model.generate(
                    inp,
                    max_new_tokens=LLM_MAX_TOKENS,
                    temperature=0.6,
                    top_k=50,
                    top_p=0.95,
                    do_sample=True,
                    eos_token_id=tok.eos_token_id,
                )

                raw = tok.decode(gen[0][len(inp[0]):], skip_special_tokens=True)
                wrapper_code = extract_code(raw)

                if wrapper_code:
                    cls_defs = re.findall(r'^\s*class\s+(\w+)', wrapper_code, flags=re.MULTILINE)
                    if len(cls_defs) == 1 and wrapper_code.count(block_cls) == 1:
                        break
            else:
                print(f"[WARN] LLM failed for {block_name} – skipped.")
                continue

            full_code = block_code.rstrip() + "\n\n" + wrapper_code.lstrip()
            try:
                ast.parse(full_code)
            except SyntaxError as e:
                print(f"[WARN] syntax error in {block_name}: {e} – skipped.")
                continue

            out_dir = synth_dir(base_out) / f"B{idx}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"rag-{_checksum(full_code)}.py"
            out_file.write_text(full_code)
            create_file(out_dir, new_out_file, raw)

            print(f"[INFO] ✔ {block_name} → {out_file}")
            idx += 1
