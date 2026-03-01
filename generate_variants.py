# generate_variants.py
import re
from pathlib import Path
import sys

# ---------- Custom Loss injection (only for NGL variants) ----------
NGL_CODE = """\
import torch
from torch import nn

class NGL(nn.Module):
    def __init__(self):
        super(NGL, self).__init__()

    def forward(self, x, target):
        target = torch.nn.functional.one_hot(target, num_classes=x.size(1))
        x = torch.softmax(x, dim=-1)
        loss = torch.mean(torch.exp(2.4092 - x - x*target) - torch.cos(torch.cos(torch.sin(x))))
        return loss
"""

# ---------- Losses / optims you want ----------
LOSS_SPECS = {
    "CrossEntropyLoss": ("nn.CrossEntropyLoss()", {"needs_log_softmax": False}),
    "MultiMarginLoss": ("nn.MultiMarginLoss()", {"needs_log_softmax": False}),
    "NLLLoss": ("nn.NLLLoss()", {"needs_log_softmax": True}),
    "KLDivLoss": ("nn.KLDivLoss(reduction='batchmean')", {"needs_log_softmax": True}),
    "NGL": ("NGL()", {"needs_log_softmax": False}),
}

OPTIM_SPECS = {
    "SGD": {
        "code": "torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm.get('momentum', 0.9))",
        "needs_momentum": True,
    },
    "Adam": {
        "code": "torch.optim.Adam(self.parameters(), lr=prm['lr'])",
        "needs_momentum": False,
    },
    "AdamW": {
        "code": "torch.optim.AdamW(self.parameters(), lr=prm['lr'])",
        "needs_momentum": False,
    },
    "RMSprop": {
        "code": "torch.optim.RMSprop(self.parameters(), lr=prm['lr'])",
        "needs_momentum": False,
    },
    "Adagrad": {
        "code": "torch.optim.Adagrad(self.parameters(), lr=prm['lr'])",
        "needs_momentum": False,
    },
    "Adadelta": {
        "code": "torch.optim.Adadelta(self.parameters(), lr=prm['lr'])",
        "needs_momentum": False,
    },
}

# ---------- Regex helpers ----------
TRAIN_SETUP_DEF_RE = re.compile(r"^\s*def\s+train_setup\s*\(\s*self\s*,", re.M)
DEF_LINE_RE = re.compile(r"^\s*def\s+\w+\s*\(", re.M)
SUPPORTED_HYPERPARAMS_RE = re.compile(
    r"^\s*def\s+supported_hyperparameters\s*\(\s*\)\s*(?:->.*?)?:\s*$",
    re.M,
)

LOSS_ASSIGN_RE = re.compile(
    r"^\s*self\.(?P<attr>[A-Za-z_]\w*)\s*=\s*(?P<ctor>(?:nn\.)?\w*Loss\s*\(|NGL\s*\()",
    re.M,
)

CRITERIA_TUPLE_RE = re.compile(
    r"^\s*self\.criteria\s*=\s*\(\s*(?:nn\.)?\w*Loss\s*\(",
    re.M,
)


# ---------- File/block extraction ----------
def get_train_setup_block(src: str):
    m = TRAIN_SETUP_DEF_RE.search(src)
    if not m:
        return None
    start = m.start()
    m2 = DEF_LINE_RE.search(src, m.end())
    end = m2.start() if m2 else len(src)
    return start, end, src[start:end]


def get_supported_hyperparameters_block(src: str):
    """Extract the entire supported_hyperparameters() function."""
    m = SUPPORTED_HYPERPARAMS_RE.search(src)
    if not m:
        return None
    start = m.start()
    m2 = re.search(r"^\s*(?:def|class)\s+\w+", src[m.end():], re.M)
    end = m.start() + m.end() + m2.start() if m2 else len(src)
    return start, end, src[start:end]


# ---------- Loss attribute detection ----------
def detect_loss_attr_any(train_setup_block: str) -> str | None:
    if re.search(r"\bself\.criteria\s*=", train_setup_block):
        return "criteria"
    if re.search(r"\bself\.loss_fn\s*=", train_setup_block):
        return "loss_fn"
    if re.search(r"\bself\.lossfn\s*=", train_setup_block):
        return "lossfn"
    if re.search(r"\bself\.criterion\s*=", train_setup_block):
        return "criterion"

    m = LOSS_ASSIGN_RE.search(train_setup_block)
    return m.group("attr") if m else None


# ---------- Argument-preserving loss replacement ----------
def _extract_call_args(text: str, call_start_idx: int) -> tuple[str, int]:
    if call_start_idx < 0 or call_start_idx >= len(text) or text[call_start_idx] != "(":
        raise ValueError("call_start_idx must point to '('")
    depth = 0
    i = call_start_idx
    while i < len(text):
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return text[call_start_idx + 1: i], i + 1
        i += 1
    raise ValueError("Unbalanced parentheses while parsing call args")


def _ctor_name_without_parens(ctor: str) -> str:
    s = ctor.strip()
    if "(" in s:
        return s[: s.find("(")].strip()
    return s


def replace_loss_line(block: str, loss_attr: str, new_loss_ctor: str) -> str:
    lines = block.splitlines(True)
    out = []
    i = 0

    new_name = _ctor_name_without_parens(new_loss_ctor)

    while i < len(lines):
        line = lines[i]
        if re.match(rf"^\s*self\.{re.escape(loss_attr)}\s*=", line):
            indent = re.match(r"^(\s*)", line).group(1)

            assign_text = line
            par = assign_text.count("(") - assign_text.count(")")
            i += 1
            while i < len(lines) and par > 0:
                assign_text += lines[i]
                par += lines[i].count("(") - lines[i].count(")")
                i += 1

            args_str = ""
            m_call = re.search(r"(?:nn\.)?\w*Loss\s*\(|NGL\s*\(", assign_text)
            if m_call:
                lparen_idx = assign_text.find("(", m_call.start())
                try:
                    args_str, _ = _extract_call_args(assign_text, lparen_idx)
                except Exception:
                    args_str = ""

            if loss_attr == "criteria":
                out.append(f"{indent}self.criteria = ({new_name}({args_str}).to(self.device),)\n")
            else:
                out.append(f"{indent}self.{loss_attr} = {new_name}({args_str}).to(self.device)\n")
            continue

        out.append(line)
        i += 1

    return "".join(out)


# ---------- Optimizer replacement ----------
def replace_optimizer_line(block: str, new_optim_expr: str) -> str:
    lines = block.splitlines(True)
    out = []
    i = 0
    while i < len(lines):
        if re.match(r"^\s*self\.optimizer\s*=", lines[i]):
            indent = re.match(r"^(\s*)", lines[i]).group(1)
            buf = lines[i]
            par = buf.count("(") - buf.count(")")
            i += 1
            while i < len(lines) and par > 0:
                buf += lines[i]
                par += lines[i].count("(") - lines[i].count(")")
                i += 1
            out.append(f"{indent}self.optimizer = {new_optim_expr}\n")
            continue
        out.append(lines[i])
        i += 1
    return "".join(out)


# ---------- supported_hyperparameters replacement ----------
def replace_supported_hyperparameters(src: str, needs_momentum: bool) -> str:
    block_info = get_supported_hyperparameters_block(src)
    if not block_info:
        return src

    start, end, block = block_info

    lines = block.splitlines(True)
    new_lines = []
    for line in lines:
        if re.match(r"^\s*return\s+", line):
            m = re.search(r"return\s+(\{[^}]*\})", line)
            if m:
                set_content = m.group(1)
                inner = set_content.strip("{}").strip()
                items = [item.strip().strip("'\"") for item in inner.split(",") if item.strip()]

                if needs_momentum:
                    if "momentum" not in items:
                        items.append("momentum")
                else:
                    items = [item for item in items if item != "momentum"]

                indent = re.match(r"^(\s*)", line).group(1)
                items_str = ", ".join(f"'{item}'" for item in items)
                new_lines.append(f"{indent}return {{{items_str}}}\n")
                continue

        new_lines.append(line)

    new_block = "".join(new_lines)
    return src[:start] + new_block + src[end:]


# ---------- NGL injection ----------
def ensure_ngl_injected(src: str) -> str:
    if "class NGL(nn.Module)" in src:
        return src

    lines = src.splitlines(True)
    insert_at = 0
    for idx, line in enumerate(lines):
        if re.match(r"^\s*(import|from)\s+", line):
            insert_at = idx + 1
    lines.insert(insert_at, "\n" + NGL_CODE + "\n")
    return "".join(lines)


# ---------- Variant creation ----------
def make_variant(src: str, loss_name: str, optim_name: str):
    block_info = get_train_setup_block(src)
    if not block_info:
        return None, "No train_setup() found"
    start, end, block = block_info

    loss_attr = detect_loss_attr_any(block)
    if not loss_attr:
        return None, "No loss attribute assignment found in train_setup()"

    loss_ctor, _meta = LOSS_SPECS[loss_name]
    optim_spec = OPTIM_SPECS[optim_name]
    optim_expr = optim_spec["code"]
    needs_momentum = optim_spec["needs_momentum"]

    new_block = block
    new_block = replace_loss_line(new_block, loss_attr, loss_ctor)
    new_block = replace_optimizer_line(new_block, optim_expr)

    new_src = src[:start] + new_block + src[end:]

    if loss_name == "NGL":
        new_src = ensure_ngl_injected(new_src)

    new_src = replace_supported_hyperparameters(new_src, needs_momentum)

    return new_src, None


# ---------- Main ----------
def main():
    base_dir = Path("./base_nn")

    # Use the standard nn-gpt directory structure
    out_base = Path("./out/nngpt/llm/epoch/A0/synth_nn")
    out_base.mkdir(parents=True, exist_ok=True)

    # Ensure current directory is in Python path
    cwd = Path.cwd()
    if str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))

    model_files = [p for p in base_dir.rglob("*.py") if p.name != "__init__.py"]

    if not model_files:
        print(f"WARNING: No model files found in {base_dir.resolve()}")
        print("Make sure your base models are in the ./base_nn directory")
        return

    # Generate variants into separate B* folders
    variant_counter = 0
    skipped_counter = 0

    for p in model_files:
        src = p.read_text(encoding="utf-8")
        base_model_name = p.stem

        for loss_name in LOSS_SPECS:
            for optim_name in OPTIM_SPECS:
                new_src, err = make_variant(src, loss_name, optim_name)
                if err:
                    print(f"Skipping {base_model_name} with {loss_name}/{optim_name}: {err}")
                    skipped_counter += 1
                    continue

                # Create B* directory
                variant_dir = out_base / f"B{variant_counter}"
                variant_dir.mkdir(parents=True, exist_ok=True)

                # Copy original base model
                original_filename = f"original_{base_model_name}.py"
                original_path = variant_dir / original_filename
                original_path.write_text(src, encoding="utf-8")

                # Write new_nn.py (the variant)
                model_path = variant_dir / "new_nn.py"
                model_path.write_text(new_src, encoding="utf-8")

                variant_counter += 1

    print(f"\n{'=' * 60}")
    print(f"Done. Base models read from: {base_dir.resolve()}")
    print(f"Generated {variant_counter} variants in: {out_base.resolve()}")
    if skipped_counter > 0:
        print(f"Skipped {skipped_counter} variants due to errors")
    print(f"\nTo evaluate, run:")
    print(f"  python ab/gpt/NNEval.py --nn_alter_epochs 1 --nn_train_epochs 5 --cycle 1")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
