from __future__ import annotations
import difflib, hashlib, importlib.util, json, logging, random, re, shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import torch
import torch.fx as fx
from transformers import AutoModelForCausalLM, AutoTokenizer

from ab.gpt.util.Const import conf_test_dir, epoch_dir, new_out_file, synth_dir
from ab.nn.util.Util import create_file
from ab.rag.retriever import Retriever

# ───────────────────────── logger
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ───────────────────────── constants
BLOCKS_DIR       = Path("blocks")
DEFAULT_CFG      = "NN_synthesis_rag.json"
DEFAULT_MODEL    = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MAX_NEW_TOKENS   = 1500
TEMPERATURE      = 0.7
TOP_P            = 0.9
DEFAULT_VARIANTS = 50

PROMPT_TMPL = """
Below are three PyTorch building blocks. ***Design a *single* image‑classification model*** that **uses all three** blocks in a novel order, inserts adapters or skip‑connections if needed, and ends with global average pooling + `nn.Linear`.

* You may add lightweight helpers but **do NOT paste external code verbatim** beyond the three blocks.
* The file must define **`class Model(nn.Module)`**.
* Return *exactly* one python file inside ```python fences – nothing else.

{blocks}
"""

# ───────────────────────── helpers
def _ensure(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _load_block(name: str) -> str:
    return (BLOCKS_DIR / f"{name}.py").read_text()

def _sample_blocks(k: int = 3) -> Dict[str, str]:
    names = random.sample([p.stem for p in BLOCKS_DIR.glob("*.py")], k)
    return {n: _load_block(n) for n in names}

def _build_prompt(srcs: Dict[str, str]) -> str:
    joined = "\n".join(f"```python\n{s}\n```" for s in srcs.values())
    return PROMPT_TMPL.format(blocks=joined)

def _call_llm(prompt: str,
              tok: AutoTokenizer,
              model: AutoModelForCausalLM) -> str:
    inp = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inp,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=True,
        eos_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True)

def _extract_py(txt: str) -> str | None:
    m = re.search(r"```python(.*?)```", txt, re.S)
    return m.group(1).strip() if m else None

@dataclass
class Novelty:
    ok: bool
    reason: str = ""

def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def _novel(code: str, parts: List[str]) -> Novelty:
    for n in parts:
        if _similar(code, _load_block(n)) > 0.7:
            return Novelty(False, f"≥70% overlap with {n}")
    try:
        spec = importlib.util.spec_from_loader("cand", loader=None)
        mod  = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        exec(code, mod.__dict__)
        gm = fx.symbolic_trace(mod.Model())
        if len(gm.graph.nodes) < 20:
            return Novelty(False, "graph too small")
    except Exception as e:
        return Novelty(False, f"trace fail: {e}")
    return Novelty(True)
def alter(epochs: int,
          test_conf : str | None = None,
          model_name: str | None = None) -> None:

    cfg = json.loads((conf_test_dir / (test_conf or DEFAULT_CFG)).read_text())

    if not BLOCKS_DIR.exists():
        log.info("Populating ./blocks …")
        Retriever().dump_all_blocks(BLOCKS_DIR)

    # ─── single tokenizer & model ───
    mid   = model_name or DEFAULT_MODEL
    log.info("Loading model once: %s", mid)
    tok   = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
              mid, trust_remote_code=True,
              torch_dtype=torch.bfloat16).cuda()

    shutil.rmtree(epoch_dir(), ignore_errors=True)

    for ep in range(epochs):
        root = synth_dir(epoch_dir(ep)); _ensure(root)
        for sid, section in cfg.items():
            variants, kept, tries = section.get("variants", DEFAULT_VARIANTS), 0, 0
            while kept < variants and tries < variants * 3:
                parts  = _sample_blocks()
                prompt = _build_prompt(parts)
                code   = _extract_py(_call_llm(prompt, tok, model))
                tries += 1
                if not code:                     continue
                nov = _novel(code, list(parts.keys()))
                if not nov.ok:                   continue
                ck = hashlib.sha256(code.encode()).hexdigest()[:8]
                bdir = root / f"{sid}_{kept:03d}"; _ensure(bdir)
                (bdir / f"rag-{ck}.py").write_text(code)
                create_file(bdir, new_out_file, prompt)
                (bdir / "meta.json").write_text(
                    json.dumps({"parts": list(parts.keys())}))
                log.info("[%s] saved rag-%s.py", sid, ck)
                kept += 1

__all__ = ["alter"]
