"""Retrieve block-level improvement suggestions from the gist service.

For each distinct block set (from ab.gpt.util.knowledge.blocks.ast_blocks),
build a search query and a question from the blocks the network CONTAINS only
(never from blocks we expect it to lack), call the gist service in question
mode, and save the results as JSON.
"""
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx

from ab.gpt.util.knowledge.blocks import ast_blocks

GIST_URL = os.environ.get("KNOW_GIST_URL", "http://127.0.0.1:8999/gist/")
NOT_BLOCKS = {"Sequential", "ModuleList", "ModuleDict", "Identity", "Parameter",
              "CrossEntropyLoss", "MSELoss", "NLLLoss"}


def block_signature(code: str) -> tuple:
    """Deterministic key: sorted layer names the network builds (no F.* calls)."""
    layers = ast_blocks(code)["layers"]
    return tuple(sorted(k for k in layers if k not in NOT_BLOCKS and not k.startswith("F.")))


def build_request(signature: tuple, task: str = "CIFAR-10 image classification") -> dict:
    standard = [b for b in signature if not b.endswith("*")]
    custom = [b.rstrip("*") for b in signature if b.endswith("*")]
    keywords = " ".join(custom[:2] + ["CNN", "CIFAR-10", "improve", "accuracy"])
    described = ", ".join(standard + custom) or "unknown layers"
    question = (f"A convolutional network for {task} is built from these blocks: {described}. "
                "Which 2 to 4 blocks or architectural changes would most likely increase its "
                "accuracy? Rank them, most effective first.")
    return {"t": keywords, "question": question, "n": 5, "reply_word_limit": 80}


def retrieve(request: dict, timeout: float = 300.0) -> str:
    response = httpx.post(GIST_URL, json=request, timeout=timeout)
    response.raise_for_status()
    return response.json().get("reply", "")


if __name__ == "__main__":
    files, out = sys.argv[1:-1], Path(sys.argv[-1])
    groups = defaultdict(list)
    for f in files:
        groups[block_signature(Path(f).read_text(errors="replace"))].append(Path(f).name)

    print(f"{len(files)} networks -> {len(groups)} distinct block sets")
    results = []
    for i, (sig, names) in enumerate(sorted(groups.items(), key=lambda g: -len(g[1])), 1):
        request = build_request(sig)
        start = time.monotonic()
        try:
            reply = retrieve(request)
        except Exception as e:  # keep going; record the failure
            reply = f"ERROR: {type(e).__name__}: {e}"
        seconds = round(time.monotonic() - start, 1)
        results.append({"signature": list(sig), "networks": names, "request": request,
                        "reply": reply, "seconds": seconds})
        print(f"\n[{i}/{len(groups)}] {len(names)} nets | {seconds}s | {', '.join(sig)}")
        print("  ->", reply)
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(results)} results to {out}")
