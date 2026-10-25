#!/usr/bin/env python3
"""One-time WL signatures of the LEMUR corpus, to seed the structural archive.

Uses the same corpus as the lexical filter (ab.nn.api.data(task, dataset,
unique_nn=True)) and the same tracing settings as the novelty gate, so the
structural archive and the lexical index cover the same reference set.
Pretrained-weight downloads are blocked (the WL graph does not depend on weights);
entries that need them simply fail to trace. Output is appended incrementally and
already-processed entries are skipped, so a cancelled job can be resubmitted.

  python -m ab.gpt.act.kto.build_struct_corpus --out struct_corpus/cifar10_wl.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_CFG = {}


def _block_weight_downloads() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    def _blocked(*_a, **_k):
        raise RuntimeError("weight download blocked")
    try:
        import torch.hub as hub
        hub.load_state_dict_from_url = _blocked
    except Exception:  # noqa: BLE001
        pass
    try:
        from torchvision.models._api import WeightsEnum
        WeightsEnum.get_state_dict = _blocked  # type: ignore[assignment]
    except Exception:  # noqa: BLE001
        pass


def _init(cfg: dict) -> None:
    _CFG.update(cfg)
    _block_weight_downloads()
    import torch
    torch.set_num_threads(1)


class _Timeout(BaseException):
    """BaseException so the tracer's per-size `except Exception` cannot swallow it."""


def _timeout(*_a):
    raise _Timeout()


def _trace(item):
    key, code = item
    from ab.gpt.act.kto.structural_archive import graph_signature_for_file
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "net.py"
        f.write_text(code, encoding="utf-8")
        signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(_CFG["timeout"])
        try:
            sig, status = graph_signature_for_file(
                f, _CFG["sizes"], _CFG["in_ch"], _CFG["n_cls"], _CFG["granularity"], _CFG["wl_rounds"])
        except _Timeout:
            sig, status = None, "timeout"
        except KeyboardInterrupt:
            raise
        except BaseException as e:  # noqa: BLE001 — generated code may call exit()
            sig, status = None, f"raise:{type(e).__name__}"
        finally:
            signal.alarm(0)
    rec = {"c": key, "status": status}
    if sig:
        rec.update(sig)
    return rec


def trace_many(items, cfg: dict, workers: int, ctx=None, on_result=None, log=print) -> dict:
    """Signatures for (key, code) items: pooled, retried after a worker crash, then one
    process per remaining item so a crashing entry cannot take others down."""
    out = {}

    def _take(rec):
        out[rec["c"]] = rec
        if on_result:
            on_result(rec)

    todo = list(items)
    for _ in range(2):
        if not todo:
            break
        try:
            with ProcessPoolExecutor(max_workers=max(1, min(workers, len(todo))), mp_context=ctx,
                                     initializer=_init, initargs=(cfg,)) as ex:
                for fut in as_completed([ex.submit(_trace, it) for it in todo]):
                    try:
                        _take(fut.result())
                    except Exception:  # noqa: BLE001 — crashed worker; retried below
                        pass
        except Exception as e:  # noqa: BLE001
            log(f"tracing pool failed ({type(e).__name__}); retrying the rest")
        todo = [it for it in todo if it[0] not in out]
    for it in todo:
        try:
            with ProcessPoolExecutor(max_workers=1, mp_context=ctx, initializer=_init, initargs=(cfg,)) as ex:
                rec = ex.submit(_trace, it).result()
        except Exception:  # noqa: BLE001
            rec = {"c": it[0], "status": "worker_crash"}
        _take(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="WL signatures of the LEMUR corpus")
    ap.add_argument("--task", default="img-classification")
    ap.add_argument("--dataset", default="cifar-10")
    ap.add_argument("--sizes", default="32,64,96,224,256")
    ap.add_argument("--in_ch", type=int, default=3)
    ap.add_argument("--n_cls", type=int, default=10)
    ap.add_argument("--granularity", default="topology", choices=["topology", "typed"])
    ap.add_argument("--wl_rounds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--timeout", type=int, default=90, help="seconds per architecture")
    ap.add_argument("--out", default="struct_corpus/cifar10_wl.jsonl")
    args = ap.parse_args()

    from ab.nn.api import data
    df = data(task=args.task, dataset=(args.dataset or None), unique_nn=True)
    codes = [c for c in df["nn_code"].tolist() if isinstance(c, str) and c.strip()]
    keyed = {hashlib.blake2b(c.encode("utf-8"), digest_size=8).hexdigest(): c for c in codes}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["c"])
    todo = [(k, c) for k, c in keyed.items() if k not in done]
    print(f"corpus: {len(codes)} entries ({len(keyed)} distinct), {len(done)} done, {len(todo)} to trace", flush=True)

    cfg = dict(sizes=[int(s) for s in args.sizes.split(",") if s.strip()], in_ch=args.in_ch,
               n_cls=args.n_cls, granularity=args.granularity, wl_rounds=args.wl_rounds,
               timeout=args.timeout)
    settings = {k: cfg[k] for k in ("sizes", "in_ch", "n_cls", "granularity", "wl_rounds")}
    settings.update(task=args.task, dataset=args.dataset)
    meta_file = out.with_name(out.name + ".meta.json")
    if meta_file.exists():
        old = json.loads(meta_file.read_text(encoding="utf-8"))
        if {k: old.get(k) for k in settings} != settings:
            raise SystemExit(f"{out} was built with different settings ({meta_file}); use a new --out")
    # "complete" stays False until the end, so runs refuse a partially built corpus.
    meta_file.write_text(json.dumps({**settings, "complete": False}, indent=2), encoding="utf-8")
    progress = {"n": 0, "ok": 0}
    with open(out, "a", encoding="utf-8") as fh:
        def _write(rec):
            progress["n"] += 1
            progress["ok"] += bool(rec.get("h"))
            fh.write(json.dumps(rec) + "\n")
            if progress["n"] % 200 == 0:
                fh.flush()
                print(f"  {progress['n']}/{len(todo)} processed, {progress['ok']} traced", flush=True)
        trace_many(todo, cfg, args.workers, on_result=_write)

    hashes = set()
    total = traced = 0
    for line in out.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            total += 1
            if rec.get("h"):
                traced += 1
                hashes.add(rec["h"])
    meta_file.write_text(json.dumps({**settings, "complete": True, "entries": total, "traced": traced,
                                     "distinct_structures": len(hashes)}, indent=2), encoding="utf-8")
    print(f"done: {total} entries, {traced} traced ({100 * traced / max(1, total):.1f}%), "
          f"{len(hashes)} distinct WL structures -> {out}", flush=True)


if __name__ == "__main__":
    main()
