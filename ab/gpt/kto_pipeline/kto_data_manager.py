#!/usr/bin/env python3
"""
KTO Training Data Manager for Iterative Fine-Tuning

Extends TrainingDataManager to maintain TWO training corpora in parallel:
  1. The standard SFT-format JSONL (train.jsonl) — needed for resume
     and inspection.  Inherited behaviour from TrainingDataManager.
  2. A KTO-format JSONL (kto_train.jsonl) — what KTOTrainer reads —
     consisting of all desirable examples (label=True) plus a balanced
     subset of accumulated undesirable examples (label=False).

Desirable / undesirable split for KTO
-------------------------------------
  Desirable   = original LEMUR-curated examples + every cycle's selected
                (above-threshold, novel) generated architectures.
  Undesirable = generated architectures that failed validation
                (compile / NameError / shape mismatch / ...) OR that
                trained but ended below `undesirable_floor_accuracy`.

Undesirable examples are persisted to `kto_undesirable_cache.jsonl`
across cycles, exactly the way NoveltyChecker persists seen models
to `seen_models.json`.

Balancing
---------
Default ratio is 1:1.  `balance()` caps undesirable at
`floor(len(desirable) * undesirable_ratio)`; if more are available we
keep the most-recent ones (later cycles produce sharper negatives).
Edge cases (no undesirables yet, extreme imbalance) are handled
gracefully and reported in the stats dict.
"""

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from ab.gpt.iterative_pipeline.training_data_manager import TrainingDataManager


class KTODataManager(TrainingDataManager):
    """KTO-aware data manager: tracks desirables + balanced undesirables."""

    def __init__(
        self,
        base_data_dir: str,
        undesirable_cache_file: Path,
        undesirable_ratio: float = 1.0,
        max_undesirable_per_cycle: int = 500,
        balance_seed: int = 43,
    ):
        """
        Args:
            base_data_dir: curation_output/chat_data (original SFT JSONL location)
            undesirable_cache_file: Where to persist the accumulated undesirables JSONL
            undesirable_ratio: target |undesirable| / |desirable| (default 1.0)
            max_undesirable_per_cycle: per-cycle cap on accepting new undesirables
            balance_seed: RNG seed for the shuffle (reproducible runs)
        """
        super().__init__(base_data_dir)
        self.undesirable_cache_file = Path(undesirable_cache_file)
        self.undesirable_ratio = undesirable_ratio
        self.max_undesirable_per_cycle = max_undesirable_per_cycle
        self._balance_seed = balance_seed

        # Accumulated undesirable examples across cycles
        self._accumulated_undesirable: List[Dict[str, Any]] = []
        if self.undesirable_cache_file.exists():
            self._load_undesirable_cache()

        # Build (once) the initial KTO file from the original SFT data so cycle 1
        # has a kto_train.jsonl to consume.  All original examples → label=True.
        self._initial_kto_file = self.base_data_dir / "kto_train.jsonl"
        if not self._initial_kto_file.exists():
            self._build_initial_kto_file()

    # ── public conversion helpers ────────────────────────────────────────────

    def _sft_to_kto(self, sft_example: Dict[str, Any], label: bool) -> Optional[Dict[str, Any]]:
        """
        Split an SFT chat example into KTO {prompt_messages, completion, label}.

        We store `prompt_messages` (list of {role, content}) rather than a
        pre-rendered prompt string so TuneNNGenKTO can apply the live
        tokenizer's chat template at training time, ensuring the prompt
        matches exactly what the generator sees.

        Returns None if the example is malformed (no user/assistant pair).
        """
        messages = sft_example.get("messages", [])
        if not isinstance(messages, list) or len(messages) < 2:
            return None

        system_msg = next((m for m in messages if m.get("role") == "system"), None)
        user_msg = next((m for m in messages if m.get("role") == "user"), None)
        assistant_msg = next((m for m in messages if m.get("role") == "assistant"), None)

        if user_msg is None or assistant_msg is None:
            return None

        prompt_messages = []
        if system_msg is not None:
            prompt_messages.append({"role": "system", "content": system_msg["content"]})
        prompt_messages.append({"role": "user", "content": user_msg["content"]})

        kto = {
            "prompt_messages": prompt_messages,
            "completion": assistant_msg.get("content", ""),
            "label": bool(label),
            "_meta": dict(sft_example.get("meta", {})),
        }
        return kto

    def convert_code_to_kto(
        self,
        code: str,
        model_id: str,
        metadata: Dict[str, Any],
        label: bool,
    ) -> Optional[Dict[str, Any]]:
        """
        Build a KTO example for a generated model.  Re-uses the parent's
        convert_code_to_chat_example so the prompt matches the SFT path
        exactly (same system / user template, same enhanced instructions),
        then splits the resulting chat example into KTO format.
        """
        sft = self.convert_code_to_chat_example(code, model_id, metadata)
        kto = self._sft_to_kto(sft, label=label)
        if kto is None:
            return None
        kto["_meta"]["model_id"] = model_id
        kto["_meta"]["cycle"] = metadata.get("cycle", 0)
        kto["_meta"]["first_epoch_accuracy"] = metadata.get("accuracy", None)
        kto["_meta"]["params"] = metadata.get("params", None)
        return kto

    # ── undesirable bookkeeping ──────────────────────────────────────────────

    def record_undesirable(
        self,
        model_id: str,
        failure_reason: str,
        code: Optional[str],
        metadata: Dict[str, Any],
    ) -> bool:
        """
        Add a failed / low-accuracy model to the accumulated undesirables.

        Returns True if recorded; False if skipped because there was no
        usable code to form a completion (the most common skip reason is
        an extraction failure — the LLM produced text we couldn't parse as
        Python).
        """
        if not code or not code.strip():
            return False

        kto = self.convert_code_to_kto(code, model_id, metadata, label=False)
        if kto is None:
            return False

        kto["_meta"]["failure_reason"] = failure_reason
        kto["_meta"]["accuracy"] = metadata.get("accuracy", None)
        self._accumulated_undesirable.append(kto)
        return True

    def _load_undesirable_cache(self):
        try:
            self._accumulated_undesirable = self._load_jsonl(self.undesirable_cache_file)
            print(f"[KTO] Loaded {len(self._accumulated_undesirable)} undesirable examples "
                  f"from {self.undesirable_cache_file}")
        except Exception as e:
            print(f"[KTO][WARN] Failed to load undesirable cache: {e}")
            self._accumulated_undesirable = []

    def save_undesirable_cache(self):
        self._save_jsonl(self._accumulated_undesirable, self.undesirable_cache_file)

    # ── balancing ────────────────────────────────────────────────────────────

    def balance(
        self,
        desirables: List[Dict[str, Any]],
        undesirables: List[Dict[str, Any]],
        ratio: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Cap undesirables so they don't drown the desirables (or vice versa).

        Returns (combined_list_shuffled, stats).
        """
        if ratio is None:
            ratio = self.undesirable_ratio

        D = len(desirables)
        Ua = len(undesirables)
        max_U = math.floor(D * ratio) if D > 0 else Ua

        if D == 0:
            selected_U: List[Dict[str, Any]] = []
            note = "no desirable examples — empty training round"
        elif Ua > max_U:
            # Cap: keep the most-recent (later cycles' negatives are
            # typically sharper / more on-distribution).
            selected_U = undesirables[-max_U:] if max_U > 0 else []
            note = f"capped from {Ua} to {len(selected_U)} (ratio={ratio})"
        elif Ua == 0:
            selected_U = []
            note = "no undesirables available — KTO falls back to desirables-only (SFT-like)"
        elif Ua < D * 0.1:
            selected_U = list(undesirables)
            note = f"WARNING: severe imbalance {Ua} undesirable << {D} desirable"
        else:
            selected_U = list(undesirables)
            note = "healthy"

        combined = list(desirables) + selected_U
        random.Random(self._balance_seed).shuffle(combined)

        stats = {
            "desirable_count": D,
            "undesirable_available": Ua,
            "undesirable_used": len(selected_U),
            "total": len(combined),
            "target_ratio": ratio,
            "actual_ratio": (len(selected_U) / D) if D > 0 else 0.0,
            "balance_note": note,
        }
        return combined, stats

    # ── core augmentation override ───────────────────────────────────────────

    def augment_training_data(
        self,
        new_examples: List[Dict[str, Any]],
        cycle: int,
        output_dir: Path,
    ) -> Dict[str, Any]:
        """
        Write BOTH the SFT-format cumulative JSONL (inherited) AND the
        balanced KTO-format JSONL that TuneNNGenKTO actually consumes.

        new_examples:  list of SFT chat examples (all label=True).
        cycle:         current cycle number.
        output_dir:    KTO pipeline root (e.g. out/kto_pipeline/...).
        """
        # 1. Parent writes train.jsonl — cumulative SFT examples, dev/test copies
        stats = super().augment_training_data(new_examples, cycle, output_dir)
        cycle_dir = Path(stats["output_dir"])

        # 2. Re-read the cumulative SFT JSONL and convert each entry to KTO desirable.
        sft_path = cycle_dir / "train.jsonl"
        all_sft = self._load_jsonl(sft_path)
        desirables: List[Dict[str, Any]] = []
        for sft_ex in all_sft:
            kto = self._sft_to_kto(sft_ex, label=True)
            if kto is not None:
                desirables.append(kto)

        # 3. Balance against the rolling undesirable cache.
        combined, balance_stats = self.balance(
            desirables, self._accumulated_undesirable, self.undesirable_ratio
        )

        # 4. Write kto_train.jsonl (with _meta retained for inspection — the
        #    TuneNNGenKTO loader strips _meta before building the Dataset).
        kto_file = cycle_dir / "kto_train.jsonl"
        self._save_jsonl(combined, kto_file)

        # 5. Persist undesirable cache (may have grown this cycle).
        self.save_undesirable_cache()

        # 6. Report
        print(f"[KTO] cycle {cycle}: wrote {kto_file}")
        print(f"[KTO] cycle {cycle}: desirable={balance_stats['desirable_count']}, "
              f"undesirable_used={balance_stats['undesirable_used']} / "
              f"available={balance_stats['undesirable_available']} "
              f"(target ratio={balance_stats['target_ratio']}, "
              f"actual={balance_stats['actual_ratio']:.2f}) — {balance_stats['balance_note']}")

        stats["kto"] = balance_stats
        stats["kto_file"] = str(kto_file)
        return stats

    # ── initial KTO file (cycle-1 input) ─────────────────────────────────────

    def _build_initial_kto_file(self):
        """
        On first init, convert every example in the original SFT train.jsonl
        to a KTO desirable.  This is the file that cycle-1 fine-tuning reads.
        """
        desirables: List[Dict[str, Any]] = []
        for sft_ex in self.original_train_data:
            kto = self._sft_to_kto(sft_ex, label=True)
            if kto is not None:
                desirables.append(kto)
        self._save_jsonl(desirables, self._initial_kto_file)
        print(f"[KTO] Initial kto_train.jsonl built at {self._initial_kto_file} "
              f"({len(desirables)} desirable examples; no undesirables yet)")
