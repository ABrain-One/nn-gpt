"""Deterministic loss/optimizer substitution for LEMUR neural networks.

This is the rule-based counterpart to ``AlterNN``: instead of an LLM rewriting a
network, a regex transform swaps the loss function and optimizer inside a model's
``train_setup()`` / ``learn()`` while keeping everything else intact.

The module is pure string-in / string-out so it can be reused from any pipeline.
``ab/gpt/NNVariants.py`` is the thin CLI wrapper that pulls model source from the
LEMUR dataset and writes the resulting variants into the standard synth_nn layout.

Public API:
    LOSS_SPECS, OPTIM_SPECS          - the substitution catalogues
    make_variant(src, loss, optim)   - transform one model source string
    iter_variants(src, losses, opts) - yield every (loss, optim, new_src, err)
"""
import re

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
# Each entry: (constructor, meta). Meta keys:
#   allowed_optimizers - restrict this loss to a specific set of optimizers
LOSS_SPECS = {
    "CrossEntropyLoss": ("nn.CrossEntropyLoss()", {}),
    "NGL": ("NGL()", {"allowed_optimizers": ("Adam", "AdamW")}),
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
LEARN_DEF_RE = re.compile(r"^\s*def\s+learn\s*\(\s*self\s*,", re.M)
DEF_LINE_RE = re.compile(r"^\s*def\s+\w+\s*\(", re.M)
SUPPORTED_HYPERPARAMS_RE = re.compile(
    r"^\s*def\s+supported_hyperparameters\s*\(\s*\)\s*(?:->.*?)?:\s*$",
    re.M,
)

LOSS_ASSIGN_RE = re.compile(
    r"^\s*self\.(?P<attr>[A-Za-z_]\w*)\s*=\s*(?P<ctor>(?:nn\.)?\w*Loss\s*\(|NGL\s*\()",
    re.M,
)


# ---------- File/block extraction ----------
def _get_block(src: str, def_re: re.Pattern):
    m = def_re.search(src)
    if not m:
        return None
    start = m.start()
    m2 = DEF_LINE_RE.search(src, m.end())
    end = m2.start() if m2 else len(src)
    return start, end, src[start:end]


def get_train_setup_block(src: str):
    return _get_block(src, TRAIN_SETUP_DEF_RE)


def get_learn_block(src: str):
    return _get_block(src, LEARN_DEF_RE)


def get_supported_hyperparameters_block(src: str):
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


def detect_use_tuple(src: str, loss_attr: str) -> bool:
    """Return True if learn() indexes into the loss attr with [0], meaning it expects a tuple."""
    return bool(re.search(rf"\bself\.{re.escape(loss_attr)}\[0\]\s*\(", src))


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


def replace_loss_line(block: str, loss_attr: str, new_loss_ctor: str, use_tuple: bool = False) -> str:
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

            # NGL takes no constructor args — never carry over originals
            args_str = ""
            if new_name != "NGL":
                m_call = re.search(r"(?:nn\.)?\w*Loss\s*\(|NGL\s*\(", assign_text)
                if m_call:
                    lparen_idx = assign_text.find("(", m_call.start())
                    try:
                        args_str, _ = _extract_call_args(assign_text, lparen_idx)
                    except Exception:
                        args_str = ""

            if use_tuple:
                out.append(f"{indent}self.{loss_attr} = ({new_name}({args_str}).to(self.device),)\n")
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


# ---------- Spatial output fix in learn() ----------
def fix_learn_spatial_outputs(src: str) -> str:
    """Inject a global-average-pool guard after 'var = self(inputs)' in learn().

    Segmentation models return (B, C, H, W); classification datasets provide (B,)
    labels.  Pooling collapses the spatial dims so the loss receives (B, C).
    The guard is a no-op for models that already return 2-D logits.
    """
    block_info = get_learn_block(src)
    if not block_info:
        return src
    start, end, block = block_info

    # Match:  <indent><var> = self(<single identifier>)
    # and insert the pooling guard on the next line.
    new_block = re.sub(
        r"(^(\s*)(\w+)\s*=\s*self\(\w+\)\n)",
        r"\1\2if \3.dim() == 4:\n\2    \3 = \3.mean(dim=(2, 3))\n",
        block,
        flags=re.M,
    )

    if new_block == block:
        return src
    return src[:start] + new_block + src[end:]


# ---------- supported_hyperparameters replacement ----------
def _get_used_prm_keys(src: str) -> set[str]:
    """Return all param keys actually accessed as prm['key'] or prm.get('key'...) in source."""
    keys: set[str] = set()
    keys.update(re.findall(r"""prm\[['"](\w+)['"]\]""", src))
    keys.update(re.findall(r"""prm\.get\(['"](\w+)['"]""", src))
    return keys


def replace_supported_hyperparameters(src: str, needs_momentum: bool) -> str:
    block_info = get_supported_hyperparameters_block(src)
    if not block_info:
        return src

    start, end, block = block_info
    used_keys = _get_used_prm_keys(src)

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

                # Remove params that are declared but never referenced in the code
                items = [
                    item for item in items
                    if item in used_keys or item == "lr"  # 'lr' is always required
                ]

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
    """Return ``(new_source, None)`` or ``(None, error_message)``.

    Substitutes the loss function ``loss_name`` and optimizer ``optim_name`` into
    ``src`` (a full model source string) and returns the rewritten source.
    """
    if loss_name not in LOSS_SPECS:
        return None, f"Unknown loss '{loss_name}' (choices: {', '.join(LOSS_SPECS)})"
    if optim_name not in OPTIM_SPECS:
        return None, f"Unknown optimizer '{optim_name}' (choices: {', '.join(OPTIM_SPECS)})"

    allowed_optimizers = LOSS_SPECS[loss_name][1].get("allowed_optimizers")
    if allowed_optimizers and optim_name not in allowed_optimizers:
        return None, (f"{loss_name} is restricted to optimizers "
                      f"{', '.join(allowed_optimizers)}; got '{optim_name}'")

    block_info = get_train_setup_block(src)
    if not block_info:
        return None, "No train_setup() found"
    start, end, block = block_info

    loss_attr = detect_loss_attr_any(block)
    if not loss_attr:
        return None, "No loss attribute assignment found in train_setup()"

    loss_ctor, _ = LOSS_SPECS[loss_name]
    optim_spec = OPTIM_SPECS[optim_name]
    optim_expr = optim_spec["code"]
    needs_momentum = optim_spec["needs_momentum"]

    # Detect whether learn() uses criteria[0](...) — keep tuple if so
    use_tuple = detect_use_tuple(src, loss_attr)

    new_block = block
    new_block = replace_loss_line(new_block, loss_attr, loss_ctor, use_tuple)
    new_block = replace_optimizer_line(new_block, optim_expr)

    new_src = src[:start] + new_block + src[end:]

    if loss_name == "NGL":
        new_src = ensure_ngl_injected(new_src)

    # Inject spatial-output guard into learn() so segmentation models work
    # on classification datasets (4-D output → global avg pool → 2-D logits)
    new_src = fix_learn_spatial_outputs(new_src)

    # Strip hyperparams that are declared but never used in the generated code
    new_src = replace_supported_hyperparameters(new_src, needs_momentum)

    return new_src, None


def iter_variants(src: str, losses=None, optimizers=None):
    """Yield ``(loss_name, optim_name, new_source, error)`` for the loss×optim grid.

    ``losses``/``optimizers`` default to the full catalogue. ``error`` is None on
    success, otherwise ``new_source`` is None and ``error`` explains the skip.
    """
    losses = list(losses) if losses else list(LOSS_SPECS)
    optimizers = list(optimizers) if optimizers else list(OPTIM_SPECS)
    for loss_name in losses:
        for optim_name in optimizers:
            new_src, err = make_variant(src, loss_name, optim_name)
            yield loss_name, optim_name, new_src, err
