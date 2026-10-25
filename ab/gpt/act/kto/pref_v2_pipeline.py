#!/usr/bin/env python3
"""Pipeline v2: one self-improving loop shared by DPO, KTO and SFT updates.

All rules share generation, screening, evaluation, the preference buffer, the pair
sampler, the optimizer-step budget and the continual adapter; only the loss differs.

Generation g:
  1. sample M candidates from pi_{g-1} = frozen base + one LoRA adapter that is
     trained continually across generations (training reference = frozen base);
  2. screen in generation order against archives seeded with LEMUR and grown with
     every earlier candidate: MinHash-Jaccard >= s is a lexical duplicate, an
     already-archived WL hash a structural duplicate (filters only with
     --struct_archive, always logged), rarity = 1 - max WL-kernel cosine to the
     archive; duplicates skip evaluation;
  3. evaluate the rest (1 epoch on --dataset); good = non-duplicate with acc >= tau;
  4. select. binary: chosen = good, rejected = every other candidate. graded:
     chosen = good candidates with rarity >= the generation's median over good,
     rejected = every other candidate except non-duplicate bad ones rarer than
     their median. k LEMUR anchors join the chosen set;
  5. sample N (chosen, rejected) pairs uniformly from the last W generations, both
     sides under the same prompt. DPO trains on the N pairs, KTO on their 2N
     unpaired sides, SFT on the N chosen sides, with equal steps per epoch.
The "frozen" rule skips step 5: every generation samples the untrained model
(the sampling baseline the trained loop is compared against).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ab.gpt.act.kto.build_struct_corpus import trace_many
from ab.gpt.act.kto.kto_generator import build_prompt_messages
from ab.gpt.act.kto.kto_selfcontained_finetune import (
    SelfContainedKTOPipeline, _fenced, _read_jsonl, logger,
)
from ab.gpt.act.kto.structural_archive import StructuralArchive

RULES = ("dpo", "kto", "sft", "frozen")
SELECTIONS = ("binary", "graded")
# (per-device batch, grad accumulation): 8 pairs = 16 unpaired = 8 chosen per step,
# so every rule takes the same number of optimizer steps on the same N pairs.
BATCH = {"dpo": (1, 8), "kto": (2, 8), "sft": (1, 8)}


def _code_key(code: str) -> str:
    return hashlib.blake2b(code.encode("utf-8"), digest_size=8).hexdigest()


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _atomic_write(path: Path, text: str) -> None:
    """Temp file + rename: a job killed mid-write never leaves a truncated state file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def _write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    _atomic_write(path, "".join(json.dumps(r) + "\n" for r in records))


def _write_json(obj: Dict[str, Any], path: Path) -> None:
    _atomic_write(path, json.dumps(obj, indent=2, default=str))


class PrefV2Pipeline(SelfContainedKTOPipeline):

    def __init__(self, *, rule: str, struct_corpus: str, selection: str = "binary",
                 struct_archive: bool = False, anchors: int = 0, window: int = 5,
                 pairs_per_gen: int = 500, beta: float = 0.1,
                 anchor_params_limit: Optional[int] = None, gen_shards: int = 1,
                 trace_workers: int = 0, trace_timeout: int = 120,
                 graph_sizes: str = "32,64,96,224,256", graph_granularity: str = "topology",
                 wl_rounds: int = 3, **base_kwargs):
        if rule not in RULES or selection not in SELECTIONS:
            raise ValueError(f"rule must be in {RULES} and selection in {SELECTIONS}")
        base_kwargs["novelty_db"] = True
        base_kwargs["kto_beta"] = beta
        super().__init__(**base_kwargs)

        self.rule, self.selection, self.struct_archive = rule, selection, bool(struct_archive)
        self.n_anchors, self.window = int(anchors), int(window)
        self.pairs_per_gen, self.beta = int(pairs_per_gen), float(beta)
        self.gen_shards = max(1, int(gen_shards))
        self.trace_workers = int(trace_workers) or max(1, min(8, self._n_cpus() - 1))
        self.trace_cfg = dict(
            sizes=[int(s) for s in str(graph_sizes).split(",") if s.strip()], in_ch=3,
            n_cls={"cifar-10": 10, "cifar-100": 100}.get(self.dataset, 10),
            granularity=graph_granularity, wl_rounds=int(wl_rounds), timeout=int(trace_timeout))
        self.prompt_mode = os.environ.get("KTO_PROMPT_MODE", "enhanced").strip().lower()
        self._prompts: Dict[Optional[str], List[Dict[str, str]]] = {}
        self._candidates: Dict[int, Dict[str, Dict[str, Any]]] = {}
        self._screen_stats: Dict[int, Dict[str, int]] = {}
        self._train_failures = 0

        corpus = Path(struct_corpus)
        if not corpus.exists():
            raise FileNotFoundError(f"structural corpus not found: {corpus} "
                                    "(build it with ab.gpt.act.kto.build_struct_corpus)")
        self._check_corpus_meta(corpus)
        self.archive = StructuralArchive()
        self.lemur_hashes: set = set()
        corpus_params = self._load_corpus(corpus)

        self.anchor_codes: List[str] = []
        if self.n_anchors > 0:
            limit = self.params_limit if anchor_params_limit is None else int(anchor_params_limit)
            self.anchor_codes = self._anchor_pool(corpus_params, limit)

        self.buffer_file = self.output_dir / "pref_buffer.jsonl"
        self.buffer: List[Dict[str, Any]] = _read_jsonl(self.buffer_file)
        self._write_run_files(corpus)

        logger.info("=" * 80)
        logger.info(f"PIPELINE v2  rule={rule}  selection={selection}  struct_archive={self.struct_archive}  "
                    f"anchors={self.n_anchors}  beta={self.beta}")
        train_desc = (f"batch {BATCH[rule][0]}x{BATCH[rule][1]}; continual adapter, reference = frozen base"
                      if rule in BATCH else "no training (frozen generator)")
        logger.info(f"  dataset {self.dataset} (LEMUR {self.sim_db_dataset}), threshold {self.accuracy_threshold}; "
                    f"buffer: window {self.window} generations, {self.pairs_per_gen} pairs/generation; {train_desc}")
        logger.info(f"  archive: {self.archive.n_items} LEMUR structures ({len(self.lemur_hashes)} distinct hashes); "
                    f"lexical index {self.sim_index.size}; prompt mode {self.prompt_mode}")
        if self.n_anchors:
            logger.info(f"  anchor pool: {len(self.anchor_codes)} LEMUR architectures")
        logger.info(f"  generation shards {self.gen_shards}; tracing workers {self.trace_workers}")
        logger.info("=" * 80)

    # ── setup ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _n_cpus() -> int:
        try:
            return len(os.sched_getaffinity(0))
        except AttributeError:
            return os.cpu_count() or 2

    def _check_corpus_meta(self, corpus: Path) -> None:
        meta_file = corpus.with_name(corpus.name + ".meta.json")
        if not meta_file.exists():
            logger.warning(f"[v2] no {meta_file.name}; cannot verify corpus tracing settings")
            return
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if not meta.get("complete", True):
            raise RuntimeError(f"structural corpus {corpus} is incomplete: wait for the corpus job "
                               "to finish, or resubmit it to continue")
        want = {k: self.trace_cfg[k] for k in ("sizes", "in_ch", "n_cls", "granularity", "wl_rounds")}
        want["dataset"] = self.sim_db_dataset
        diff = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
        if diff:
            raise ValueError(f"structural corpus settings differ from this run (corpus, run): {diff}")

    def _load_corpus(self, corpus: Path) -> Dict[str, int]:
        params: Dict[str, int] = {}
        with open(corpus, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("h"):
                    self.archive.add(rec["h"], rec.get("f"))
                    self.lemur_hashes.add(rec["h"])
                    if rec.get("d"):
                        params[rec["c"]] = int(rec["d"].get("params", 0))
        return params

    def _anchor_pool(self, params: Dict[str, int], limit: int) -> List[str]:
        """Traced LEMUR architectures within the prompt's parameter budget."""
        from ab.nn.api import data
        df = data(task=self.sim_db_task, dataset=self.sim_db_dataset, unique_nn=True)
        pool, seen = [], set()
        for code in df["nn_code"].tolist():
            if not isinstance(code, str) or not code.strip():
                continue
            k = _code_key(code)
            if k in seen or k not in params or params[k] > limit:
                continue
            seen.add(k)
            pool.append(code)
        if not pool:
            raise RuntimeError(f"no traced LEMUR architecture has <= {limit} parameters")
        return pool

    def _write_run_files(self, corpus: Path) -> None:
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                    check=False).stdout.strip() or None
        except Exception:  # noqa: BLE001
            commit = None
        cfg = {
            "rule": self.rule, "selection": self.selection, "struct_archive": self.struct_archive,
            "anchors_per_generation": self.n_anchors, "anchor_pool": len(self.anchor_codes),
            "window": self.window, "pairs_per_generation": self.pairs_per_gen, "beta": self.beta,
            "dataset": self.dataset, "lemur_dataset": self.sim_db_dataset,
            "batch": BATCH.get(self.rule), "learning_rate": self.kto_learning_rate, "lora_r": self.kto_lora_r,
            "lora_alpha": self.kto_lora_alpha, "max_grad_norm": self.kto_max_grad_norm,
            "epochs": self.num_train_epochs, "max_prompt_length": self.kto_max_prompt_length,
            "max_completion_length": self.kto_max_completion_length,
            "cycles": self.cycles, "models_per_cycle": self.models_per_cycle,
            "accuracy_threshold": self.accuracy_threshold, "sim_threshold": self.sim_threshold,
            "temperature": self.temperature, "top_k": self.top_k, "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens, "gen_max_rejections": self.gen_max_rejections,
            "eval": {"transform": self.eval_transform, "lr": self.eval_lr, "momentum": self.eval_momentum,
                     "batch": self.eval_batch, "dropout": self.eval_dropout, "epochs": self.eval_train_epochs},
            "params_limit": self.params_limit, "trace": self.trace_cfg, "struct_corpus": str(corpus),
            "prompt_mode": self.prompt_mode, "seed": self.seed, "base_model": self.base_model,
            "git_commit": commit,
        }
        (self.output_dir / "v2_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        (self.output_dir / "prompts.json").write_text(
            json.dumps({"prompt_mode": self.prompt_mode, "training_prompt": self._prompt(None)}, indent=2),
            encoding="utf-8")

    def _prompt_key(self, hint: Optional[str]) -> Optional[str]:
        # The generator only puts the hint into the prompt in "enhanced" mode.
        return hint if self.prompt_mode == "enhanced" else None

    def _prompt(self, key: Optional[str]) -> List[Dict[str, str]]:
        if key not in self._prompts:
            self._prompts[key] = build_prompt_messages(
                dataset=self.dataset, params_limit=self.params_limit, diversity_hint=key)
        return self._prompts[key]

    def _seed(self, cycle: int, stream: int) -> int:
        return (self.seed * 1_000_003 + cycle) * 10 + stream

    # ── resume ─────────────────────────────────────────────────────────────────

    def _first_incomplete_cycle(self) -> int:
        for c in range(1, self.cycles + 1):
            try:
                json.loads((self.output_dir / f"cycle_{c}" / "metrics.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return c
        return self.cycles + 1

    def _restore(self, start: int) -> None:
        """Rebuild buffer, lexical index and structural archive as of generation `start`."""
        self.buffer = [it for it in self.buffer if it["gen"] < start]
        n_codes = n_feats = 0
        for c in range(1, start):
            cdir = self.output_dir / f"cycle_{c}"
            for rec in _read_jsonl(cdir / "generation_records.jsonl"):
                if not rec.get("ok"):
                    continue
                mdir = cdir / "nneval" / rec.get("model_id", "")
                code = _read(mdir / "new_nn.py") or _read(mdir / "new_nn.notnovel.py")
                if code:
                    self.sim_index.add(code)
                    n_codes += 1
            for r in _read_jsonl(cdir / "wl_features.jsonl"):
                self.archive.add(r.get("h"), r.get("f"))
                n_feats += 1
        if start > 1:
            logger.info(f"[v2] resumed at generation {start}: {len(self.buffer)} buffer items, "
                        f"{n_codes} codes re-indexed, {n_feats} structures re-archived")

    def run(self) -> Dict[str, Any]:
        """Unlike the base loop, an exception stops the run, so a resubmission resumes
        exactly at the failed generation instead of after it."""
        start = self.resume_from_cycle or self._first_incomplete_cycle()
        if start <= self.cycles:
            self._restore(start)
        for cycle in range(start, self.cycles + 1):
            self.cycle_results.append(self.run_cycle(cycle))
            self._save_aggregate()
        self._save_aggregate()
        logger.info(f"[v2] all {self.cycles} generations complete")
        return {"cycles": self.cycle_results}

    def _save_aggregate(self) -> None:
        cycles = []
        for c in range(1, self.cycles + 1):
            f = self.output_dir / f"cycle_{c}" / "metrics.json"
            try:
                cycles.append(json.loads(f.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        _write_json({"base_model": self.base_model, "accuracy_threshold": self.accuracy_threshold,
                            "rule": self.rule, "selection": self.selection,
                            "struct_archive": self.struct_archive, "anchors": self.n_anchors,
                            "beta": self.beta, "cycles": cycles},
                           self.output_dir / "all_cycles_results.json")

    def _prev_adapter(self, cycle: int) -> Optional[Path]:
        for c in range(cycle - 1, 0, -1):
            if (self._checkpoint_dir(c) / "adapter_config.json").exists():
                return self._checkpoint_dir(c)
        return None

    # ── 1. generation (optionally split across GPUs; per-candidate seeds make it split-invariant)

    @staticmethod
    def _visible_gpus() -> List[str]:
        env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if env:
            return [d.strip() for d in env.split(",") if d.strip()]
        try:
            import torch
            return [str(i) for i in range(torch.cuda.device_count())]
        except Exception:  # noqa: BLE001
            return []

    def _gen_cmd(self, cycle: int, nneval_dir: Path, records_file: Path, start: int, num: int,
                 adapter: Optional[Path]) -> List[str]:
        cmd = [sys.executable, "-u", "-m", "ab.gpt.act.kto.kto_generator",
               "--base_model", self.base_model, "--out_dir", str(nneval_dir),
               "--records_file", str(records_file), "--num_models", str(num),
               "--start_index", str(start), "--temperature", str(self.temperature),
               "--top_k", str(self.top_k), "--top_p", str(self.top_p),
               "--max_new_tokens", str(self.max_new_tokens),
               "--max_rejections", str(self.gen_max_rejections), "--dataset", self.dataset,
               "--params_limit", str(self.params_limit), "--context_length", str(self.context_length),
               "--seed", str(self.seed + cycle * 1000)]
        if adapter is not None:
            cmd += ["--adapter", str(adapter)]
        return cmd

    def generate_models(self, cycle: int) -> Tuple[Path, List[Dict[str, Any]]]:
        gpus = self._visible_gpus()
        if not gpus:
            return super().generate_models(cycle)
        shards = min(self.gen_shards, len(gpus), self.models_per_cycle)
        cdir = self._cycle_dir(cycle)
        nneval_dir, records_file = cdir / "nneval", cdir / "generation_records.jsonl"
        if records_file.exists() and nneval_dir.exists():
            logger.info(f"[cycle {cycle}] Reusing existing generations: {records_file}")
            return nneval_dir, _read_jsonl(records_file)
        if shards > 1:
            self._ensure_local_model(gpus[0])
        adapter = self._prev_adapter(cycle)
        logger.info(f"CYCLE {cycle}: GENERATION ({self.models_per_cycle} models, {shards} GPUs)  adapter={adapter}")
        per = math.ceil(self.models_per_cycle / shards)
        jobs = []
        for k in range(shards):
            start, num = k * per, min(per, self.models_per_cycle - k * per)
            if num > 0:
                f = cdir / f"generation_records.shard{k}.jsonl"
                jobs.append((self._gen_cmd(cycle, nneval_dir, f, start, num, adapter), f, gpus[k]))
        self._run_parallel(jobs, f"generation cycle {cycle}")
        records = sorted((r for _, f, _ in jobs for r in _read_jsonl(f)), key=lambda r: r.get("index", 0))
        _write_jsonl(records, records_file)
        logger.info(f"[cycle {cycle}] generation done: {sum(1 for r in records if r.get('ok'))}/{len(records)} extractable")
        return nneval_dir, records

    def _ensure_local_model(self, gpu: str) -> None:
        """The first load of a new base model saves a local copy (ab.gpt.util.llm.LLM); do that in
        one process so parallel generator shards never write the copy concurrently."""
        from ab.gpt.util.Const import llm_dir, llm_tokenizer_dir
        from ab.nn.util.Const import out_dir
        if (llm_dir(out_dir, self.base_model) / "config.json").exists() \
                and llm_tokenizer_dir(out_dir, self.base_model).exists():
            return
        logger.info(f"[v2] first use of {self.base_model}: creating its local copy before parallel generation")
        env = os.environ.copy()
        env["MKL_THREADING_LAYER"] = "GNU"
        env["CUDA_VISIBLE_DEVICES"] = gpu
        code = ("from ab.gpt.util.llm.LLM import LLM; from ab.gpt.util.llm.LLMUtil import quantization_config_4bit; "
                f"LLM({self.base_model!r}, quantization_config_4bit, context_length={self.context_length})")
        subprocess.run([sys.executable, "-c", code], env=env, check=True)

    def _run_parallel(self, jobs: List[Tuple[List[str], Path, str]], name: str) -> None:
        pending = list(range(len(jobs)))
        for attempt in range(1, self.max_retries + 1):
            procs = []
            for i in pending:
                cmd, _, gpu = jobs[i]
                env = os.environ.copy()
                env["MKL_THREADING_LAYER"] = "GNU"
                env["CUDA_VISIBLE_DEVICES"] = gpu
                logger.info(f"Running {name} [shard {i}, GPU {gpu}]: {' '.join(cmd)}")
                procs.append((i, subprocess.Popen(cmd, env=env)))
            pending = [i for i, p in procs if p.wait() != 0]
            if not pending:
                return
            logger.error(f"{name}: shard(s) {pending} failed (attempt {attempt}/{self.max_retries})")
            if attempt < self.max_retries:
                time.sleep(15.0 * attempt)
        raise subprocess.CalledProcessError(1, f"{name} shards {pending}")

    # ── 2. screening: lexical + structural novelty, rarity, descriptors ────────

    def _prefilter_novelty(self, cycle: int, nneval_dir: Path,
                           generation_records: List[Dict[str, Any]]) -> int:
        codes: Dict[str, str] = {}
        for rec in generation_records:
            if rec.get("ok"):
                mdir = nneval_dir / rec.get("model_id", "")
                code = _read(mdir / "new_nn.py") or _read(mdir / "new_nn.notnovel.py")
                if code:
                    codes[rec["model_id"]] = code
        # Generated code is executed only in spawned worker processes, never in-process.
        sigs = trace_many(list(codes.items()), self.trace_cfg, self.trace_workers,
                          ctx=get_context("spawn"), log=logger.warning)

        rows, feats, st = [], [], Counter()
        for rec in generation_records:
            mid = rec.get("model_id", "")
            row = {"model_id": mid, "hint": rec.get("diversity_hint"), "ok": bool(rec.get("ok"))}
            rows.append(row)
            rec["not_novel"] = False
            if mid not in codes:
                continue
            code = codes[mid]
            jac = self.sim_index.nearest_jaccard(code)
            self.sim_index.add(code)
            sig = sigs.get(mid, {})
            h, f = sig.get("h"), sig.get("f")
            rarity = self.archive.rarity(f) if h else None
            seen = self.archive.is_dup(h)
            if h:
                self.archive.add(h, f)
                feats.append({"model_id": mid, "h": h, "f": f})
            lex_dup = jac >= self.sim_threshold
            dup = lex_dup or (self.struct_archive and seen)
            row.update(jaccard=round(jac, 4), lex_dup=lex_dup, trace=sig.get("status"), wl_hash=h,
                       struct_seen=seen, struct_seen_lemur=bool(h) and h in self.lemur_hashes,
                       rarity=None if rarity is None else round(rarity, 6), desc=sig.get("d"), dup=dup)
            st.update(screened=1, traced=int(bool(h)), lex_dup=int(lex_dup), struct_seen=int(seen),
                      struct_seen_lemur=int(row["struct_seen_lemur"]), struct_only_dup=int(dup and not lex_dup),
                      dup=int(dup))
            rec["not_novel"] = dup
            nn_file, aside = nneval_dir / mid / "new_nn.py", nneval_dir / mid / "new_nn.notnovel.py"
            if dup and nn_file.exists():
                nn_file.replace(aside)
            elif not dup and aside.exists() and not nn_file.exists():
                aside.replace(nn_file)

        cdir = self._cycle_dir(cycle)
        _write_jsonl(rows, cdir / "candidates.jsonl")
        _write_jsonl(feats, cdir / "wl_features.jsonl")
        self._candidates[cycle] = {r["model_id"]: r for r in rows}
        self._screen_stats[cycle] = {k: int(st[k]) for k in (
            "screened", "traced", "lex_dup", "struct_seen", "struct_seen_lemur", "struct_only_dup", "dup")}
        logger.info(f"[cycle {cycle}] screening: {st['screened']} screened, {st['traced']} traced; "
                    f"lexical dup {st['lex_dup']}, WL hash seen {st['struct_seen']} "
                    f"(LEMUR {st['struct_seen_lemur']}), structural-only dup {st['struct_only_dup']}; "
                    f"{st['dup']} skip evaluation")
        return int(st["dup"])

    # ── 3-4. classification, selection, anchors, buffer ────────────────────────

    @staticmethod
    def _acc(ev: Dict[str, Any]) -> float:
        try:
            return float(ev.get("accuracy") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _rarity_median(items: List[Dict[str, Any]]) -> Optional[float]:
        r = [it["rarity"] for it in items if it.get("rarity") is not None]
        return statistics.median(r) if r else None

    @staticmethod
    def _rarity_mean(items: List[Dict[str, Any]]) -> Optional[float]:
        r = [it["rarity"] for it in items if it.get("rarity") is not None]
        return sum(r) / len(r) if r else None

    def _select(self, cycle: int, nneval_dir: Path, records: List[Dict[str, Any]],
                eval_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        rows = self._candidates.get(cycle, {})
        good, bad, accs = [], [], []
        for rec in records:
            mid = rec.get("model_id") or ""
            row = rows.setdefault(mid, {"model_id": mid})
            row["role"] = "none"
            item = {"gen": cycle, "id": mid, "src": "gen", "key": self._prompt_key(rec.get("diversity_hint")),
                    "rarity": row.get("rarity"), "dup": bool(rec.get("not_novel"))}
            mdir = nneval_dir / mid
            if not rec.get("ok"):
                raw = _read(Path(rec["raw_file"])).strip() if rec.get("raw_file") else ""
                if len(raw) < 40:
                    continue
                item.update(text=raw, reason="unparseable")
            elif item["dup"]:
                code = _read(mdir / "new_nn.notnovel.py")
                if not code:
                    continue
                item.update(text=_fenced(code), reason="duplicate")
            else:
                code = _read(mdir / "new_nn.py")
                if not code:
                    continue
                ev = eval_by_id.get(mid)
                if ev is None or not ev.get("success"):
                    item.update(text=_fenced(code), reason="not_evaluated" if ev is None else "failed")
                    if ev is not None:
                        row["eval_error"] = str(ev.get("error", ""))[:200]
                else:
                    acc = self._acc(ev)
                    accs.append(acc)
                    row["acc"] = item["acc"] = acc
                    item.update(text=_fenced(code),
                                reason="good" if acc >= self.accuracy_threshold else "low_accuracy")
            row["reason"] = item["reason"]
            (good if item["reason"] == "good" else bad).append(item)

        if self.selection == "binary":
            chosen, rejected = list(good), list(bad)
        else:
            m_good = self._rarity_median(good)
            chosen = [g for g in good if g.get("rarity") is not None and g["rarity"] >= m_good]
            nondup_bad = [b for b in bad if not b["dup"]]
            m_bad = self._rarity_median(nondup_bad)
            spared = {id(b) for b in nondup_bad if b.get("rarity") is not None and b["rarity"] > m_bad}
            rejected = [b for b in bad if id(b) not in spared]

        anchors = []
        if self.n_anchors and self.anchor_codes:
            rng = random.Random(self._seed(cycle, 1))
            for code in rng.sample(self.anchor_codes, min(self.n_anchors, len(self.anchor_codes))):
                anchors.append({"gen": cycle, "id": f"anchor_{_code_key(code)}", "src": "anchor", "key": None,
                                "rarity": None, "dup": False, "reason": "anchor", "text": _fenced(code)})

        for it in chosen + anchors:
            it["role"] = "chosen"
        for it in rejected:
            it["role"] = "rejected"
        for it in chosen + rejected:
            rows[it["id"]]["role"] = it["role"]
        self.buffer = [b for b in self.buffer if b["gen"] != cycle] + chosen + anchors + rejected
        _write_jsonl(self.buffer, self.buffer_file)
        _write_jsonl(list(rows.values()), self._cycle_dir(cycle) / "candidates.jsonl")

        reasons = Counter(it["reason"] for it in good + bad)
        passed = [a for a in accs if a >= self.accuracy_threshold]
        stats = {
            "new_desirable": len(good),
            "new_undesirable": len(bad),
            "not_novel_skipped": reasons["duplicate"],
            "undesirable_breakdown": {
                "non_novel": reasons["duplicate"], "low_accuracy": reasons["low_accuracy"],
                "runtime_error": reasons["failed"] + reasons["not_evaluated"],
                "unparseable": reasons["unparseable"], "non_compiling": 0,
            },
            "chosen_new": len(chosen), "rejected_new": len(rejected), "anchors": len(anchors),
            "evaluated_accuracies": len(accs),
            "best_accuracy": max(accs) if accs else 0.0,
            "avg_accuracy": (sum(passed) / len(passed)) if passed else 0.0,
            "avg_accuracy_all": (sum(accs) / len(accs)) if accs else 0.0,
            "effective_threshold": self.accuracy_threshold,
            "rarity_mean_good": self._rarity_mean(good), "rarity_mean_chosen": self._rarity_mean(chosen),
            "screening": self._screen_stats.get(cycle, {}),
        }
        logger.info(f"[cycle {cycle}] SELECTION ({self.selection}): good {len(good)} -> chosen {len(chosen)} "
                    f"+ {len(anchors)} anchors; bad {len(bad)} -> rejected {len(rejected)}; "
                    f"best {stats['best_accuracy']*100:.2f}%  avg(>=thr) {stats['avg_accuracy']*100:.2f}%")
        return stats

    # ── 5. pairs from the window, rule-specific training data ──────────────────

    def _sample_pairs(self, cycle: int, chosen: List[Dict[str, Any]],
                      rejected: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """N i.i.d. pairs: chosen uniform over the window, rejected uniform among the window's
        rejected under the same prompt (anchors take the prompt of their rejected partner)."""
        by_key: Dict[Optional[str], List[Dict[str, Any]]] = defaultdict(list)
        for r in rejected:
            by_key[r["key"]].append(r)
        eligible = [c for c in chosen if c["src"] == "anchor" or by_key.get(c["key"])]
        if not eligible or not rejected:
            return []
        rng = random.Random(self._seed(cycle, 2))
        pairs, tries = [], 0
        while len(pairs) < self.pairs_per_gen and tries < 20 * self.pairs_per_gen:
            tries += 1
            c = rng.choice(eligible)
            r = rng.choice(rejected if c["src"] == "anchor" else by_key[c["key"]])
            if c["text"] != r["text"]:
                pairs.append((c, r))
        return pairs

    def _build_training_data(self, cycle: int) -> Tuple[Optional[Path], Dict[str, Any]]:
        lo = cycle - self.window + 1
        win = [it for it in self.buffer if lo <= it["gen"] <= cycle]
        chosen = [it for it in win if it["role"] == "chosen"]
        rejected = [it for it in win if it["role"] == "rejected"]
        pairs = self._sample_pairs(cycle, chosen, rejected)

        cdir = self._cycle_dir(cycle)
        recs = [{"prompt_messages": self._prompt(r["key"]), "chosen": c["text"], "rejected": r["text"],
                 "_meta": {"cycle": cycle, "chosen_id": c["id"], "chosen_gen": c["gen"], "chosen_src": c["src"],
                           "rejected_id": r["id"], "rejected_gen": r["gen"], "rejected_reason": r["reason"]}}
                for c, r in pairs]
        pairs_file = cdir / "train_pairs.jsonl"
        _write_jsonl(recs, pairs_file)
        if self.rule == "dpo":
            data_file = pairs_file
        elif self.rule == "kto":
            data_file = cdir / "kto_train.jsonl"
            _write_jsonl([{"prompt_messages": p["prompt_messages"], "completion": p[side],
                           "label": side == "chosen", "_meta": p["_meta"]}
                          for p in recs for side in ("chosen", "rejected")], data_file)
        else:
            data_file = cdir / "sft_train.jsonl"
            _write_jsonl([{"prompt_messages": p["prompt_messages"], "completion": p["chosen"],
                           "_meta": p["_meta"]} for p in recs], data_file)

        stats = {
            "rule": self.rule, "window": [max(1, lo), cycle], "pairs": len(pairs),
            "window_chosen": len(chosen), "window_chosen_anchors": sum(1 for c in chosen if c["src"] == "anchor"),
            "window_rejected": len(rejected),
            "distinct_chosen_used": len({id(c) for c, _ in pairs}),
            "distinct_rejected_used": len({id(r) for _, r in pairs}),
            "train_file": str(data_file),
        }
        logger.info(f"[cycle {cycle}] TRAINING DATA: window gens {stats['window']}: {len(chosen)} chosen "
                    f"({stats['window_chosen_anchors']} anchors) x {len(rejected)} rejected -> {len(pairs)} pairs "
                    f"({stats['distinct_chosen_used']} distinct chosen) -> {data_file.name}")
        if len(pairs) < self.min_train_examples:
            logger.warning(f"[cycle {cycle}] only {len(pairs)} pairs; skipping training this generation")
            return None, stats
        return data_file, stats

    # ── 6. training (continual adapter, equal step budget) ─────────────────────

    def _train(self, cycle: int, data_file: Path) -> Dict[str, Any]:
        ckpt = self._checkpoint_dir(cycle)
        if (ckpt / "adapter_config.json").exists():
            logger.info(f"[cycle {cycle}] checkpoint already exists, skipping: {ckpt}")
            return {"success": True, "checkpoint_dir": str(ckpt), "skipped": True}
        prev = self._prev_adapter(cycle)
        bs, ga = BATCH[self.rule]
        py = [sys.executable, "-u", "-m"]
        if self.rule == "dpo":
            cmd = py + ["ab.gpt.act.tune.DPO", "--dpo_data_file", str(data_file),
                        "--dpo_checkpoint_dir", str(ckpt), "--dpo_beta", str(self.beta)]
        elif self.rule == "kto":
            cmd = py + ["ab.gpt.act.tune.KTO", "--kto_data_file", str(data_file),
                        "--kto_checkpoint_dir", str(ckpt), "--kto_beta", str(self.beta),
                        "--kto_desirable_weight", "1.0", "--kto_undesirable_weight", "1.0"]
        else:
            cmd = py + ["ab.gpt.act.kto.sft_train", "--sft_data_file", str(data_file),
                        "--sft_checkpoint_dir", str(ckpt)]
        cmd += ["--llm_conf", self.llm_conf, "--num_train_epochs", str(self.num_train_epochs),
                "--learning_rate", str(self.kto_learning_rate), "--r", str(self.kto_lora_r),
                "--lora_alpha", str(self.kto_lora_alpha), "--max_grad_norm", str(self.kto_max_grad_norm),
                "--max_prompt_length", str(self.kto_max_prompt_length),
                "--max_completion_length", str(self.kto_max_completion_length),
                "--per_device_train_batch_size", str(bs), "--gradient_accumulation_steps", str(ga)]
        if prev is not None:
            cmd += ["--peft", str(prev), "--continual_adapter"]
        logger.info(f"CYCLE {cycle}: {self.rule.upper()} TRAINING  "
                    f"({'continuing ' + str(prev) if prev else 'new adapter'})")
        t0 = time.time()
        try:
            self._run_subprocess(cmd, f"{self.rule.upper()} training cycle {cycle}")
        except subprocess.CalledProcessError as e:
            return {"success": False, "error": f"training_failed: exit_{e.returncode}",
                    "training_time_minutes": (time.time() - t0) / 60}
        minutes = (time.time() - t0) / 60
        if not (ckpt / "adapter_config.json").exists():
            return {"success": False, "error": "checkpoint_missing", "training_time_minutes": minutes}
        return {"success": True, "checkpoint_dir": str(ckpt), "training_time_minutes": minutes,
                "rule": self.rule, "beta": self.beta, "continued_from": str(prev) if prev else None}

    def run_cycle(self, cycle: int) -> Dict[str, Any]:
        t0 = time.time()
        logger.info("#" * 80)
        logger.info(f"# GENERATION {cycle} / {self.cycles}  (v2 {self.rule}, {self.selection})")
        logger.info("#" * 80)
        nneval_dir, records = self.generate_models(cycle)
        self._prefilter_novelty(cycle, nneval_dir, records)
        eval_by_id = self.evaluate_models(cycle, nneval_dir)
        self._free_gpu()
        sel_stats = self._select(cycle, nneval_dir, records, eval_by_id)
        if self.rule == "frozen":
            ds_stats = {"rule": "frozen"}
            train_stats = {"success": False, "skipped": True, "reason": "frozen"}
        else:
            data_file, ds_stats = self._build_training_data(cycle)
            train_stats = (self._train(cycle, data_file) if data_file is not None
                           else {"success": False, "skipped": True, "reason": "insufficient_pairs"})
        if train_stats.get("success") or train_stats.get("skipped"):
            self._train_failures = 0
        else:
            self._train_failures += 1
            if self._train_failures >= 2:
                raise RuntimeError(f"training failed in two consecutive generations ({train_stats.get('error')})")
        result = {
            "cycle": cycle, "timestamp": datetime.now().isoformat(),
            "generated": len(records), "extractable": sum(1 for r in records if r.get("ok")),
            "bucketing": sel_stats, "dataset": ds_stats, "training": train_stats,
            "cycle_time_minutes": (time.time() - t0) / 60,
        }
        _write_json(result, self._cycle_dir(cycle) / "metrics.json")   # written last: marks the generation done
        return result


def main() -> None:
    p = argparse.ArgumentParser(description="Pipeline v2 (DPO / KTO / SFT on one shared loop)")
    p.add_argument("--rule", choices=RULES, required=True)
    p.add_argument("--selection", choices=SELECTIONS, default="binary")
    p.add_argument("--struct_archive", action="store_true",
                   help="Structural duplicates (archived WL hash) also skip evaluation and are rejected")
    p.add_argument("--anchors", type=int, default=0, help="LEMUR anchors added as chosen per generation")
    p.add_argument("--window", type=int, default=5, help="Generations kept in the preference buffer")
    p.add_argument("--pairs_per_gen", type=int, default=500)
    p.add_argument("--beta", type=float, default=0.1, help="DPO/KTO beta (unused by SFT)")
    p.add_argument("--struct_corpus", required=True, help="build_struct_corpus output for the LEMUR corpus")
    p.add_argument("--anchor_params_limit", type=int, default=None,
                   help="Anchor parameter budget (default: the prompt's params limit)")
    p.add_argument("--gen_shards", type=int, default=1, help="Split generation across this many visible GPUs")
    p.add_argument("--trace_workers", type=int, default=0)
    p.add_argument("--trace_timeout", type=int, default=120)
    p.add_argument("--graph_sizes", default="32,64,96,224,256")
    p.add_argument("--graph_granularity", choices=["topology", "typed"], default="topology")
    p.add_argument("--wl_rounds", type=int, default=3)

    p.add_argument("--llm_conf", default="nngpt_unique_arch_rag.json")
    p.add_argument("--dataset", default="cifar-10", help="Benchmark: prompt, evaluation and tracing")
    p.add_argument("--cycles", type=int, default=25)
    p.add_argument("--models_per_cycle", type=int, default=100)
    p.add_argument("--accuracy_threshold", type=float, default=0.40)
    p.add_argument("--num_train_epochs", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--gen_max_rejections", type=int, default=3)
    p.add_argument("--sim_threshold", type=float, default=0.85)
    p.add_argument("--sim_db_task", default="img-classification")
    p.add_argument("--sim_db_dataset", default=None, help="LEMUR subset (default: --dataset)")
    p.add_argument("--eval_transform", default="norm_256_flip")
    p.add_argument("--eval_lr", type=float, default=0.01)
    p.add_argument("--eval_momentum", type=float, default=0.9)
    p.add_argument("--eval_batch", type=int, default=10)
    p.add_argument("--eval_dropout", type=float, default=0.2)
    p.add_argument("--eval_multi_gpu", action="store_true")
    p.add_argument("--output_subdir", default="pref_v2")
    p.add_argument("--resume_from_cycle", type=int, default=None,
                   help="Default: first generation without metrics.json")
    p.add_argument("--seed", type=int, default=43)
    a = p.parse_args()

    PrefV2Pipeline(
        rule=a.rule, selection=a.selection, struct_archive=a.struct_archive, anchors=a.anchors,
        window=a.window, pairs_per_gen=a.pairs_per_gen, beta=a.beta, struct_corpus=a.struct_corpus,
        anchor_params_limit=a.anchor_params_limit, gen_shards=a.gen_shards,
        trace_workers=a.trace_workers, trace_timeout=a.trace_timeout, graph_sizes=a.graph_sizes,
        graph_granularity=a.graph_granularity, wl_rounds=a.wl_rounds,
        llm_conf=a.llm_conf, dataset=a.dataset, cycles=a.cycles, models_per_cycle=a.models_per_cycle,
        accuracy_threshold=a.accuracy_threshold, num_train_epochs=a.num_train_epochs,
        temperature=a.temperature, top_k=a.top_k, top_p=a.top_p, max_new_tokens=a.max_new_tokens,
        gen_max_rejections=a.gen_max_rejections, sim_threshold=a.sim_threshold,
        sim_db_task=a.sim_db_task, sim_db_dataset=a.sim_db_dataset or a.dataset,
        eval_transform=a.eval_transform, eval_lr=a.eval_lr, eval_momentum=a.eval_momentum,
        eval_batch=a.eval_batch, eval_dropout=a.eval_dropout, eval_multi_gpu=a.eval_multi_gpu,
        output_subdir=a.output_subdir, resume_from_cycle=a.resume_from_cycle, seed=a.seed,
    ).run()


if __name__ == "__main__":
    main()
