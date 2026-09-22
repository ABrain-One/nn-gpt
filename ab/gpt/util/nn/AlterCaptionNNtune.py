import os
import ast
import json
import random
import re
import shutil
import traceback
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from string import Template

import torch
import ab.nn.api as nn_dataset
from tqdm import tqdm

from ab.nn.util.Util import create_file
from ab.gpt.util.Const import (
    conf_test_dir,
    epoch_dir,
    synth_dir,
    new_nn_file,
    new_out_file,
)
from ab.gpt.util.llm.LLM import LLM
from ab.gpt.util.Util import extract_code

# ============================================================
# Tunables (env-overridable, so you can tweak without editing)
# ============================================================
REPAIR_MAX_ATTEMPTS = int(os.getenv("NNGPT_REPAIR_TRIES", "4"))       # Syntax/API repair retries
GUARD_ATTEMPTS      = int(os.getenv("NNGPT_GUARD_TRIES", "4"))        # Guard retries
POST_GUARD_REPAIR   = int(os.getenv("NNGPT_POST_GUARD_REPAIR", "1"))  # LLM repair after guard fail
STRUCT_FIX_TRIES    = int(os.getenv("NNGPT_STRUCT_FIX_TRIES", "1"))   # Pre-guard “missing pieces” repair

# =============================
# Helpers
# =============================

def _sanitize_model_id(llm_name: str) -> str:
    safe = (llm_name or "").strip().strip("'\"")
    if "#" in safe:
        safe = safe.split("#", 1)[0]
    return safe.rstrip("/.-")

def _strip_think(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)

def _find_all_fenced_blocks(text: str) -> List[str]:
    blocks = []
    for m in re.finditer(r"```(?:python)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE):
        blocks.append(m.group(1).strip())
    return blocks

def _score_block(block: str) -> int:
    score = 0
    if "class Net" in block:
        score += 5
    if "def forward(" in block:
        score += 3
    if "supported_hyperparameters" in block:
        score += 2
    if "import torch" in block:
        score += 2
    score += min(len(block) // 1000, 5)
    return score

def _extract_best_code(full_text: str) -> str:
    text = _strip_think(full_text)
    blocks = _find_all_fenced_blocks(text)
    if blocks:
        return max(blocks, key=_score_block)
    legacy = extract_code(text)
    if legacy:
        return legacy
    if ("class Net" in text) or ("import torch" in text):
        return text
    return ""

def _quick_text_fixes(code: str) -> str:
    fixed = code
    fixed = fixed.replace("```python", "").replace("```", "")
    fixed = re.sub(r"\n(\s*)else:\s*(\n\s*#.*)?\n(?!\s)", r"\n\1else:\n\1    pass\n", fixed)

    fixed = re.sub(
        r"^\s*import\s+torch\.nn\.functional\s+as\s+F\s*$",
        "import torch.nn.functional as F",
        fixed,
        flags=re.MULTILINE,
    )
    fixed = re.sub(r"^\s*from\s+torch\s+import\s+nn\s*$", "import torch.nn as nn", fixed, flags=re.MULTILINE)
    if "import torch.nn as nn" not in fixed and "import torch" in fixed:
        fixed = "import torch.nn as nn\n" + fixed

    if re.search(r"\bmath\.", fixed) and "import math" not in fixed:
        fixed = "import math\n" + fixed

    # Normalize supported_hyperparameters
    if re.search(r"def\s+supported_hyperparameters\s*\(", fixed):
        fixed = re.sub(
            r"def\s+supported_hyperparameters\s*\([\s\S]*?\):[\s\S]*?(?=^(?:def|class|$))",
            "def supported_hyperparameters():\n    return {'lr','momentum'}\n\n",
            fixed,
            flags=re.MULTILINE,
        )
    else:
        fixed += "\n\ndef supported_hyperparameters():\n    return {'lr','momentum'}\n"

    fixed = re.sub(r"in_shape\[\s*0\s*\]", "in_shape[1]", fixed)
    fixed = re.sub(r"out_shape\[\s*1\s*\]", "out_shape[0]", fixed)
    fixed = fixed.replace("not_defaut", "is not None and")
    return fixed

def _balance_brackets(code: str) -> str:
    stack: List[str] = []
    openers = "([{"
    closers = ")]}"
    pair = {"(": ")", "[": "]", "{": "}"}
    for ch in code:
        if ch in openers:
            stack.append(ch)
        elif ch in closers:
            if stack and pair.get(stack[-1], None) == ch:
                stack.pop()
    if not stack:
        return code
    tail = "\n# --- auto-closed by AlterCaptionNN ---\n" + "".join(pair[o] for o in reversed(stack)) + "\n"
    return code + tail

def _syntax_ok(code: str) -> Tuple[bool, Optional[str]]:
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"{e.msg} (line {e.lineno})"
    except Exception as e:
        return False, str(e)

def _ensure_net_methods_ast(code: str) -> str:
    tree = ast.parse(code)
    class_info = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Net":
            class_info = node
            break
    if class_info is None:
        return code

    has_train_setup = any(isinstance(n, ast.FunctionDef) and n.name == "train_setup" for n in class_info.body)
    has_learn = any(isinstance(n, ast.FunctionDef) and n.name == "learn" for n in class_info.body)

    if has_train_setup and has_learn:
        return code

    lines = code.splitlines()
    indent = " " * (class_info.col_offset + 4)
    insert_at = (getattr(class_info, "end_lineno", None) or (class_info.body[-1].lineno if class_info.body else class_info.lineno)) - 1

    stubs: List[str] = []
    if not has_train_setup:
        stubs += [
            f"{indent}def train_setup(self, prm):",
            f"{indent}    self.to(self.device)",
            f"{indent}    self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)",
            f"{indent}    self.optimizer = torch.optim.AdamW(self.parameters(), lr=prm['lr'], betas=(prm.get('momentum', 0.9), 0.999))",
            "",
        ]
    if not has_learn:
        stubs += [
            f"{indent}def learn(self, train_data):",
            f"{indent}    self.train()",
            f"{indent}    for images, captions in train_data:",
            f"{indent}        images, captions = images.to(self.device), captions.to(self.device)",
            f"{indent}        logits = None",
            f"{indent}        if hasattr(self, 'forward'):",
            f"{indent}            out = self.forward(images, captions)",
            f"{indent}            logits = out[0] if isinstance(out, tuple) else out",
            f"{indent}        if logits is None:",
            f"{indent}            continue",
            f"{indent}        tgt = (captions[:,0,:] if captions.ndim==3 else captions)[:,1:]",
            f"{indent}        loss = self.criteria[0](logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))",
            f"{indent}        self.optimizer.zero_grad()",
            f"{indent}        loss.backward()",
            f"{indent}        nn.utils.clip_grad_norm_(self.parameters(), 3)",
            f"{indent}        self.optimizer.step()",
            "",
        ]
    lines[insert_at:insert_at] = stubs
    return "\n".join(lines)

def _struct_check(code: str) -> Tuple[bool, str]:
    """Check presence of class Net + required methods + supported_hyperparameters."""
    try:
        tree = ast.parse(code)
    except Exception as e:
        return False, f"AST parse failed: {e}"

    have_supported = False
    have_net = False
    have_init = False
    have_forward = False
    have_ts = False
    have_learn = False

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "supported_hyperparameters":
            have_supported = True
        if isinstance(node, ast.ClassDef) and node.name == "Net":
            have_net = True
            for n in node.body:
                if isinstance(n, ast.FunctionDef):
                    if n.name == "__init__":
                        have_init = True
                    elif n.name == "forward":
                        have_forward = True
                    elif n.name == "train_setup":
                        have_ts = True
                    elif n.name == "learn":
                        have_learn = True

    missing: List[str] = []
    if not have_net: missing.append("class Net")
    if have_net and not have_init: missing.append("Net.__init__")
    if have_net and not have_forward: missing.append("Net.forward")
    if have_net and not have_ts: missing.append("Net.train_setup")
    if have_net and not have_learn: missing.append("Net.learn")
    if not have_supported: missing.append("supported_hyperparameters")

    if missing:
        return False, "Missing: " + ", ".join(missing)
    return True, ""

# -------------------------------
# Integrated Guard (lightweight)
# -------------------------------
def _guard__validate_then_fix(path: Path, attempts: int = 3) -> None:
    """
    Minimal, safe guard that:
      - ensures code parses
      - ensures class Net and required methods exist
      - ensures supported_hyperparameters returns {'lr','momentum'}
      - writes fixes back to the same file
    Raises SystemExit with a message on persistent failure.
    """
    if not path.exists():
        raise SystemExit(f"Guard: file not found: {path}")

    for i in range(max(1, attempts)):
        code = path.read_text(encoding="utf-8")
        ok, err = _syntax_ok(code)
        if not ok:
            # cannot auto-fix syntax more here (outer loop handles LLM repair),
            # but we still try bracket balance once.
            code2 = _balance_brackets(code)
            ok2, err2 = _syntax_ok(code2)
            if ok2:
                path.write_text(code2, encoding="utf-8")
                code = code2
            else:
                # give up in guard; outer pipeline handles LLM fix cycles
                raise SystemExit(f"Guard: syntax invalid: {err2 or err}")

        # normalize supported_hyperparameters & add stubs if missing
        code_fixed = _quick_text_fixes(code)
        try:
            code_fixed = _ensure_net_methods_ast(code_fixed)
        except Exception:
            pass

        ok_struct, serr = _struct_check(code_fixed)
        if ok_struct:
            # final parse re-check before accept
            ok_final, ferr = _syntax_ok(code_fixed)
            if ok_final:
                path.write_text(code_fixed, encoding="utf-8")
                return
            else:
                # rewrite then continue attempts
                path.write_text(code_fixed, encoding="utf-8")
                continue
        else:
            # If missing structure, update code with our stubs/normalization and continue
            path.write_text(code_fixed, encoding="utf-8")
            if i == attempts - 1:
                raise SystemExit(f"Guard: {serr}")

# =============================
# LLM repair prompt
# =============================
REPAIR_SYSTEM = (
    "SYSTEM: Return exactly one fenced Python code block that compiles. "
    "Fix only syntax/API issues; do not redesign the model."
)

REPAIR_PROMPT_TPL = Template(
    "The following Python file failed to parse or is missing required API methods.\n\n"
    "Error:\n$error\n\n"
    "Original code:\n```python\n$code\n```\n\n"
    "Please return a corrected, runnable Python file with the SAME structure and API.\n\n"
    "Requirements:\n"
    "- Define class Net(nn.Module) with __init__, train_setup, learn, forward.\n"
    "- supported_hyperparameters() returns exactly {'lr','momentum'}.\n"
    "- Use only torch / torch.nn (no torchvision imports).\n"
    "- Keep teacher forcing and shape asserts in forward.\n"
    "- Return one fenced Python code block and nothing else.\n"
)

# =============================
# Main alter() entrypoint
# =============================
def alter(
    epochs: int,
    test_conf: str,
    llm_name: str,
    gguf_file: Optional[str] = None,
    only_nn: Optional[str] = None,
) -> None:
    # Load prompt configuration
    with open(conf_test_dir / test_conf, "r", encoding="utf-8") as f:
        prompt_dict = json.load(f)
    assert isinstance(prompt_dict, dict)

    # Load LLM
    safe_llm = _sanitize_model_id(llm_name)
    model_loader = LLM(safe_llm, gguf_file=gguf_file)
    model = model_loader.get_model()
    tokenizer = model_loader.get_tokenizer()

    # Ensure distinct PAD token & pass attention_mask
    if tokenizer.pad_token is None and tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        try:
            model.resize_token_embeddings(len(tokenizer))
        except Exception:
            pass

    print("Load Model Complete, Start Loop...")

    # Reset epoch output root
    shutil.rmtree(epoch_dir(), ignore_errors=True)

    # Perf tweaks
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    for ep in range(epochs):
        base_out = epoch_dir(ep)
        prompts = []  # list[(prompt_text, cap_row, chosen_addons_df)]

        for key in prompt_dict.keys():
            prompt_tpl = "\n".join(prompt_dict[key]["prompt"]) + "\n"

            # Randomly sample captioning anchors (1 per NN)
            cap_df = (
                nn_dataset.data(only_best_accuracy=True, task=prompt_dict[key]["task"])
                .groupby(by="nn")
                .sample(n=1, random_state=random.randrange(1_000_000))
            )

            if only_nn:
                cap_df = cap_df[cap_df["nn"] == only_nn]
                if len(cap_df) == 0:
                    print(f"[WARN] No rows found for nn='{only_nn}' under task='{prompt_dict[key]['task']}'. Skipping.")
                    continue

            # Classification inspirations: use full pool (not only-best)
            addon_df = None
            if prompt_dict[key].get("addon_task"):
                addon_df = nn_dataset.data(task=prompt_dict[key]["addon_task"])

            addon_entries = prompt_dict[key].get("addon_list", [])
            K = len(addon_entries)

            for _, cap_row in cap_df.iterrows():
                para_dict: Dict[str, str] = {}
                for it in prompt_dict[key]["input_list"]:
                    para_dict[it["para"]] = cap_row[it["value"]]

                chosen_addons = None
                if addon_df is not None and len(addon_df) > 0 and K > 0:
                    filtered = addon_df.loc[addon_df.nn != cap_row["nn"]] if len(addon_df) > 0 else addon_df
                    base_pool = filtered if len(filtered) > 0 else addon_df
                    n_pick = min(K, len(base_pool))
                    chosen_addons = (
                        base_pool.sample(n=n_pick, replace=False, random_state=random.randrange(1_000_000))
                        .sample(frac=1.0, replace=False, random_state=random.randrange(1_000_000))
                        .reset_index(drop=True)
                    )
                    for i, entry in enumerate(addon_entries):
                        if chosen_addons is not None and i < chosen_addons.shape[0]:
                            para_dict[entry["para"]] = chosen_addons.iloc[i][entry["value"]]
                        else:
                            para_dict[entry["para"]] = ""
                else:
                    for entry in addon_entries:
                        para_dict[entry["para"]] = ""

                the_prompt = prompt_tpl.format(**para_dict)
                prompts.append((the_prompt, cap_row, chosen_addons))

        # Generate code per prompt
        B_index = 0
        for _, (the_prompt, cap_row, chosen_addons) in tqdm(
            enumerate(prompts), total=len(prompts), desc="Generate Codes"
        ):
            model_dir = synth_dir(base_out) / f"B{B_index}"
            code_file = model_dir / new_nn_file        # canonical good file (NNEval scans this)
            code_any_file = model_dir / "new_nn.py"    # always saved best-attempt
            broken_file = model_dir / "new_nn_broken.py"
            df_file = model_dir / "dataframe.df"
            orig_caption_py = model_dir / "original_caption.py"
            prompt_txt = model_dir / "prompt_used.txt"

            # Tokenize with chat template (silence the torch.tensor warning)
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": the_prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            )
            input_ids = ids.to(model.device) if torch.is_tensor(ids) else torch.tensor(ids).to(model.device)

            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            attention_mask = (input_ids != tokenizer.pad_token_id).long()

            gen_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=12288,
                do_sample=True,
                temperature=0.9,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.08,
                num_beams=1,
                num_return_sequences=1,
                use_cache=True,
            )

            with torch.inference_mode():
                outputs = model.generate(**gen_kwargs)

            prompt_len = int(input_ids.shape[-1])
            out_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
            print("Response Available!")

            # Save raw LLM text and prompt always
            model_dir.mkdir(parents=True, exist_ok=True)
            create_file(model_dir, new_out_file, out_text)
            create_file(model_dir, prompt_txt.name, the_prompt)

            # --- Extraction pipeline
            raw_code = _extract_best_code(out_text)
            create_file(model_dir, "00_raw_extracted.py", raw_code or "# empty")

            if not raw_code:
                print("[INFO]Response invalid: no code found")

            # First aid fixes
            code = _quick_text_fixes(raw_code or "")
            create_file(model_dir, "10_quick_fixed.py", code or "# empty")

            # Bracket balancer
            code = _balance_brackets(code)
            create_file(model_dir, "15_balanced.py", code)

            # Parse check + AST stubs
            ok, err = _syntax_ok(code)
            if ok:
                try:
                    code = _ensure_net_methods_ast(code)
                except Exception:
                    pass
                create_file(model_dir, "18_ast_stubbed.py", code)
                ok, err = _syntax_ok(code)

            # LLM repair loop (up to REPAIR_MAX_ATTEMPTS)
            tries = 0
            while not ok and tries < REPAIR_MAX_ATTEMPTS:
                print(f"[REPAIR] Fixing: {err}")
                repair_input = REPAIR_PROMPT_TPL.substitute(error=err, code=code)
                rep_ids = tokenizer.apply_chat_template(
                    [{"role": "system", "content": REPAIR_SYSTEM},
                     {"role": "user", "content": repair_input}],
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(model.device)
                rep_attn = (rep_ids != (tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)).long()

                with torch.inference_mode():
                    rep_out = model.generate(
                        input_ids=rep_ids,
                        attention_mask=rep_attn,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=32*1024,
                        temperature=0.2,
                        top_p=0.9,
                        do_sample=True,
                        use_cache=True,
                    )

                rep_text = tokenizer.decode(rep_out[0][int(rep_ids.shape[-1]):], skip_special_tokens=True)
                rep_code = _extract_best_code(rep_text) or code
                rep_code = _quick_text_fixes(rep_code)
                rep_code = _balance_brackets(rep_code)

                try:
                    rep_code = _ensure_net_methods_ast(rep_code)
                except Exception:
                    pass

                code = rep_code
                create_file(model_dir, f"2{tries}_repaired.py", code)
                ok, err = _syntax_ok(code)
                tries += 1

            # Always save the best attempt as new_nn.py
            with open(code_any_file, "w", encoding="utf-8") as f:
                f.write(code)

            # Save dataset row & original caption code for traceability
            try:
                cap_row.to_pickle(df_file)
            except Exception as e:
                print(f"[WARN] Failed to pickle caption row: {e}")
            try:
                with open(orig_caption_py, "w", encoding="utf-8") as f:
                    f.write(str(cap_row.get("nn_code", "")))
            except Exception as e:
                print(f"[WARN] Failed to save original caption code: {e}")

            # Save chosen add-ons (classification inspirations)
            if chosen_addons is not None and len(chosen_addons) > 0:
                addons_dir = model_dir / "addons"
                addons_dir.mkdir(parents=True, exist_ok=True)
                for j in range(chosen_addons.shape[0]):
                    addon_row = chosen_addons.iloc[j]
                    try:
                        with open(addons_dir / f"addon_{j+1}.py", "w", encoding="utf-8") as f:
                            f.write(str(addon_row.get("nn_code", "")))
                    except Exception as e:
                        print(f"[WARN] Failed to save add-on #{j+1}: {e}")

            # If final code parses, do structure check & guard
            if ok:
                # Pre-guard structure check: if missing pieces, do STRUCT_FIX_TRIES repairs
                s_ok, s_err = _struct_check(code)
                s_tries = 0
                while not s_ok and s_tries < STRUCT_FIX_TRIES:
                    print(f"[STRUCT] Missing parts: {s_err}")
                    repair_input = REPAIR_PROMPT_TPL.substitute(error=s_err, code=code)
                    rep_ids = tokenizer.apply_chat_template(
                        [{"role": "system", "content": REPAIR_SYSTEM},
                         {"role": "user", "content": repair_input}],
                        add_generation_prompt=True,
                        return_tensors="pt",
                    ).to(model.device)
                    rep_attn = (rep_ids != (tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)).long()
                    with torch.inference_mode():
                        rep_out = model.generate(
                            input_ids=rep_ids,
                            attention_mask=rep_attn,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            max_new_tokens=32*1024,
                            temperature=0.2,
                            top_p=0.9,
                            do_sample=True,
                            use_cache=True,
                        )
                    rep_text = tokenizer.decode(rep_out[0][int(rep_ids.shape[-1]):], skip_special_tokens=True)
                    rep_code = _extract_best_code(rep_text) or code
                    rep_code = _quick_text_fixes(rep_code)
                    rep_code = _balance_brackets(rep_code)
                    try:
                        rep_code = _ensure_net_methods_ast(rep_code)
                    except Exception:
                        pass
                    code = rep_code
                    code_any_file.write_text(code, encoding="utf-8")
                    create_file(model_dir, f"3{s_tries}_struct_fixed.py", code)
                    s_ok, s_err = _struct_check(code)
                    s_tries += 1

                # Write candidate to canonical path before guard
                code_file.parent.mkdir(parents=True, exist_ok=True)
                code_file.write_text(code, encoding="utf-8")

                # Guard validation / fixing
                try:
                    _guard__validate_then_fix(code_file, attempts=GUARD_ATTEMPTS)
                    print(f"[INFO]Saved runnable & guarded code to: {code_file}")
                except SystemExit as ge:
                    guard_err = str(ge)
                    create_file(model_dir, "guard_error.txt", guard_err)
                    print(f"[GUARD] First pass failed: {guard_err}")

                    if POST_GUARD_REPAIR > 0:
                        try:
                            # post-guard LLM repair using guard error as hint
                            repair_input = REPAIR_PROMPT_TPL.substitute(error=guard_err, code=code_file.read_text(encoding="utf-8"))
                            rep_ids = tokenizer.apply_chat_template(
                                [{"role": "system", "content": REPAIR_SYSTEM},
                                 {"role": "user", "content": repair_input}],
                                add_generation_prompt=True,
                                return_tensors="pt",
                            ).to(model.device)
                            rep_attn = (rep_ids != (tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)).long()

                            with torch.inference_mode():
                                rep_out = model.generate(
                                    input_ids=rep_ids,
                                    attention_mask=rep_attn,
                                    pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    max_new_tokens=32*1024,
                                    temperature=0.2,
                                    top_p=0.9,
                                    do_sample=True,
                                    use_cache=True,
                                )

                            rep_text = tokenizer.decode(rep_out[0][int(rep_ids.shape[-1]):], skip_special_tokens=True)
                            rep_code = _extract_best_code(rep_text) or code
                            rep_code = _quick_text_fixes(rep_code)
                            rep_code = _balance_brackets(rep_code)
                            try:
                                rep_code = _ensure_net_methods_ast(rep_code)
                            except Exception:
                                pass

                            code = rep_code
                            code_file.write_text(code, encoding="utf-8")
                            _guard__validate_then_fix(code_file, attempts=GUARD_ATTEMPTS)
                            print(f"[INFO]Saved runnable & guarded code to: {code_file} (after post-guard repair)")
                        except Exception as e2:
                            create_file(model_dir, "post_guard_error.txt", "".join(traceback.format_exception_only(type(e2), e2)))
                            print(f"[DROP] Guard validation failed again after post-guard repair: {e2}")
            else:
                # Also save the final error & keep a broken copy
                create_file(model_dir, "repair_error.txt", str(err) if err else "unknown parse error")
                broken_file.write_text(code, encoding="utf-8")
                print(f"[SAVED_BROKEN] Could not finalize runnable code: {err}. Saved: {broken_file}")

            B_index += 1
