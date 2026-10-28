"""Block and architecture extraction for knowledge-augmented nn-gpt prompts.

Two extractors:
- ast_blocks(code): deterministic count of the torch.nn layers the code creates.
- llm_profile(code): asks the cluster LLM for the architecture type and key
  building blocks, as strict JSON. Results are cached by a hash of the code.
"""
import ast
import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import httpx

LLM_URL = os.environ.get("KNOW_LLM_URL", "http://132.187.14.67:30004/v1/chat/completions")
LLM_MODEL = os.environ.get("KNOW_LLM_MODEL", "unsloth/Llama-3.3-70B-Instruct-FP8-Dynamic")
CACHE_DIR = Path(os.environ.get("KNOW_CACHE_DIR", Path.home() / ".cache" / "nn_knowledge" / "profiles"))
MAX_CODE_CHARS = 12000

SYSTEM_PROMPT = """You analyse PyTorch neural network code.
Return exactly ONE JSON object and nothing else, with these keys:
  "architecture_type": short name of the architecture family, e.g. "ResNet-style residual CNN",
  "key_blocks": list of up to 8 short descriptions of the main building blocks,
  "notes": one short sentence on anything structurally notable.
Base every statement only on the given code."""


def ast_blocks(code: str) -> dict:
    """Count layers the code builds: nn.X constructors, F.x functional calls, and
    custom layer classes defined in the same file (marked with *)."""
    tree = ast.parse(code)
    classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    local = set(classes) - {"Net"}
    layers = Counter()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            owner = ast.unparse(func.value)
            if owner in ("nn", "torch.nn"):
                layers[func.attr] += 1                  # nn.Conv2d(...)
            elif owner in ("F", "torch.nn.functional"):
                layers["F." + func.attr] += 1           # F.relu(...), F.conv2d(...)
        elif isinstance(func, ast.Name) and func.id in local:
            layers[func.id + "*"] += 1                  # custom layer defined in this file
    return {"layers": dict(layers.most_common()), "classes": classes}


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        return json.loads(match.group(0)) if match else {}


def llm_profile(code: str, layers: dict | None = None, timeout: float = 120.0) -> dict:
    """Ask the cluster LLM for the architecture type and key blocks (cached)."""
    key = hashlib.sha256(code.encode("utf-8")).hexdigest()
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())

    hint = f"\nLayers found by static analysis: {json.dumps(layers)}" if layers else ""
    user = f"Code:\n```python\n{code[:MAX_CODE_CHARS]}\n```{hint}"
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": 400,
    }
    response = httpx.post(LLM_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    profile = _parse_json(response.json()["choices"][0]["message"]["content"])

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, ensure_ascii=False))
    return profile


if __name__ == "__main__":
    import sys
    for file in sys.argv[1:]:
        code = Path(file).read_text(errors="replace")
        start = time.monotonic()
        blocks = ast_blocks(code)
        profile = llm_profile(code, blocks["layers"])
        print(f"\n=== {Path(file).name}  ({time.monotonic() - start:.1f}s)")
        print("AST layers :", blocks["layers"])
        print("LLM type   :", profile.get("architecture_type"))
        print("LLM blocks :", profile.get("key_blocks"))
        print("LLM notes  :", profile.get("notes"))
