"""Build --extra_info for web_alter.py: one gist question per network of the experiment set.

Differences from the pilot (retrieve.py): the search names the architecture family (the web has
papers about published families), and the question lists the blocks the network already uses and
asks only for architectural changes it does not have yet (no training or augmentation tips).

Usage:  python -m ab.gpt.util.knowledge.build_extra_info SET_CSV OUT_DIR
Writes OUT_DIR/extra_info.json ({nn: suggestion}) and OUT_DIR/extra_info_details.jsonl.
Resumable: networks already in extra_info.json are skipped, failed ones are retried on the next run.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

from ab.gpt.util.knowledge.retrieve import block_signature, retrieve


def build_request_v2(family: str, signature: tuple, task: str = "CIFAR-10 image classification") -> dict:
    blocks = ", ".join(b.rstrip("*") for b in signature) or "unknown layers"
    question = (f"A {family} network for {task} already uses these blocks: {blocks}. "
                "Which 2 to 4 architectural changes (blocks to add, replace or restructure) would most likely "
                "increase its accuracy? Do not suggest blocks it already uses, and do not suggest training, "
                "optimizer or data-augmentation changes. Rank them, most effective first.")
    return {"t": f"{family} architecture improve accuracy CIFAR-10", "question": question,
            "n": 5, "reply_word_limit": 80}


def main():
    set_csv, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    extra_file, log_file = out_dir / "extra_info.json", out_dir / "extra_info_details.jsonl"
    extra = json.loads(extra_file.read_text()) if extra_file.exists() else {}

    import ab.nn.api as api
    names = pd.read_csv(set_csv)["nn"].tolist()
    d = api.data(only_best_accuracy=True, task="img-classification", dataset="cifar-10")
    d = d.drop_duplicates("nn").set_index("nn")
    todo = [n for n in names if n not in extra]
    print(f"{len(names)} networks, {len(extra)} already done, {len(todo)} to ask", flush=True)

    for i, name in enumerate(todo, 1):
        family = name.split("-")[0]
        sig = block_signature(d.loc[name, "nn_code"])
        req = build_request_v2(family, sig)
        t0 = time.monotonic()
        try:
            reply, err = retrieve(req).strip(), None
        except Exception as e:  # record and continue; rerun to retry
            reply, err = "", f"{type(e).__name__}: {e}"
        sec = round(time.monotonic() - t0, 1)
        with open(log_file, "a") as f:
            f.write(json.dumps({"nn": name, "family": family, "signature": list(sig), "request": req,
                                "reply": reply, "error": err, "seconds": sec}, ensure_ascii=False) + "\n")
        if reply and not err:
            extra[name] = reply
            extra_file.write_text(json.dumps(extra, indent=1, ensure_ascii=False))
        print(f"[{i}/{len(todo)}] {name[:30]:30s} {sec:6.1f}s  {err or reply}", flush=True)

    missing = [n for n in names if n not in extra]
    print(f"done: {len(extra)}/{len(names)} networks have suggestions"
          + (f"; missing (rerun to retry): {missing}" if missing else ""), flush=True)


if __name__ == "__main__":
    main()
