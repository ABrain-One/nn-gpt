import os
import re
import shutil
from pathlib import Path
import ast


def exists(path):
    return path is not None and os.path.exists(str(path))


def read_py_file_as_string(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def extract_delta(txt):
    """
    Extract a unified diff from an LLM response.

    Supports:
    - fenced diff/patch blocks
    - XML-style <delta>...</delta>
    - raw unified diff text
    """
    if not txt:
        return None

    tagged = extract_str(txt, "<delta>", "</delta>")
    if tagged and "---" in tagged and "+++" in tagged:
        return tagged.strip()

    blocks = re.findall(r"```(?:diff|patch)?\n(.*?)\n```", txt, re.DOTALL)
    for block in blocks:
        if "---" in block and "+++" in block:
            return block.strip()

    match = re.search(r"(--- .*?\n\+\+\+ .*?\n@@ .*?@@[\s\S]*)", txt)
    if match:
        return match.group(1).strip()

    return None


def extract_str(res, start_sep, end_sep):
    if res is None:
        return None
    try:
        start_index = res.index(start_sep) + len(start_sep)
        end_index = res.index(end_sep, start_index)
        return res[start_index:end_index].strip()
    except (ValueError, IndexError):
        return None


def assemble_nn_code(llm_code):
    """
    Assembles the complete neural network code using the provided base template and LLM-generated bridge code.
    """
    llm_code = llm_code.replace("torch.gelu", "torch.nn.functional.gelu")
    
    # Extract only the class content if it's wrapped in markers
    if "# === LLM-GENERATED BRIDGE ===" in llm_code:
        idx = llm_code.find("class CrossModalBridge")
        if idx != -1:
            end_idx = llm_code.find("# === END LLM CODE ===", idx)
            if end_idx != -1:
                llm_code = llm_code[idx:end_idx]


def improve_code(code):
    """
    Clean common LLM hallucinations in generated bridge/head code.

    This function is intentionally limited to local code sanitation. The stable
    model interface is provided by assemble_nn_code().
    """
    if not code:
        return ""

    # Remove markdown fences if accidentally passed through.
    code = code.replace("```python", "").replace("```", "")

    # Remove pure explanatory comments while keeping Python code.
    code = re.sub(r"^\s*#.*?$", "", code, flags=re.MULTILINE)

    # Common hallucinated/non-code words.
    garbage = [
        "Gedanken",
        "spatial location encoding",
        "token survival probabilities",
        "time steps",
        "return x0 tensor after processing",
    ]
    for g in garbage:
        code = code.replace(g, "x")

    # Standardize common malformed keyword names.
    code = re.sub(r"\bnum\s+attends\s*=", "num_heads=", code)
    code = re.sub(r"\bn\s+attends\s*=", "nhead=", code)
    code = re.sub(r"\bdim\s+feedforward\s*=", "dim_feedforward=", code)
    code = code.replace("dmodel=", "d_model=")
    code = code.replace("dim_FEEDFORWARD=", "dim_feedforward=")
    code = code.replace("nn.TransformerLayer", "nn.TransformerEncoderLayer")
    code = code.replace("nn.nn.LayerNorm", "nn.LayerNorm")

    # Fix malformed MultiheadAttention arguments (LLM often messes these up)
    if "nn.MultiheadAttention" in code:
        # Standardize MultiheadAttention(embed_dim, num_heads)
        code = re.sub(r"nn\.MultiheadAttention\s*\((?:embed_dim\s*=\s*)?(\d+|768),\s*(?:num_heads\s*=\s*)?(\d+|8)\)", r"nn.MultiheadAttention(\1, \2)", code)
        # Fix missing commas or keyword hallucinations like embeddinghead=...
        code = re.sub(r"nn\.MultiheadAttention\s+(?:embeddinghead|embed_dim)\s*=\s*.*?\)", "nn.MultiheadAttention(768, 8)", code)
        code = re.sub(r"nn\.MultiheadAttention\((?:embeddinghead|embed_dim)\s*=\s*.*?\)", "nn.MultiheadAttention(768, 8)", code)
        code = code.replace("nnMultiheadSelfAttention", "nn.MultiheadAttention")
        code = code.replace("nn.MultiheadSelfAttention", "nn.MultiheadAttention")
    
    # Avoid invalid literal assignments such as "768 = prm.get(...)".
    code = re.sub(
        r"^\s*\d+\s*=\s*prm\.get\(.*?\)\s*$",
        "",
        code,
        flags=re.MULTILINE,
    )

    # Make unsafe B-dependent reshapes less likely to crash.
    code = re.sub(r"\.view\(B,\s*.*?\)", r".view(B, -1)", code)
    code = re.sub(r"\.reshape\(B,\s*.*?\)", r".reshape(B, -1)", code)

    # Fix simple local assignments from prm.get into safe typed values.
    def wrap_assignment(m):
        var_name = m.group(1)
        value = m.group(2).strip().rstrip(",")

        if any(x in var_name.lower() for x in ["dim", "size", "num", "heads", "layers"]):
            return f"{var_name} = int({value})"
        if "dropout" in var_name.lower():
            return f"{var_name} = float({value})"
        return f"{var_name} = {value}"

    code = re.sub(
        r"(\b[a-zA-Z_][a-zA-Z0-9_]*\b)\s*=\s*(prm\.get\(.*?\)|[0-9.]+|None)",
        wrap_assignment,
        code,
    )

    # Fix missing self. for local nn layers in __init__ where possible.
    # If the LLM wrote "proj = nn.Linear(...)" instead of "self.proj = nn.Linear(...)"
    layer_names = re.findall(r"self\.([a-zA-Z0-9_]+)\s*=\s*nn\.", code)
    code = re.sub(
        r"^\s{8}([a-z][a-zA-Z0-9_]*)\s*=\s*nn\.",
        r"        self.\1 = nn.",
        code,
        flags=re.MULTILINE,
    )
    for name in layer_names:
        code = re.sub(
            r"(?<!self\.)(?<![a-zA-Z0-9_])" + re.escape(name) + r"\(",
            r"self." + name + "(",
            code,
        )

    # Hallucination Repair: Define missing common projection matrices if used but not defined.
    if "self.projection_wq" in code and "self.projection_wq =" not in code:
        code = code.replace("def __init__(self, prm):", "def __init__(self, prm):\n        super().__init__()\n        self.projection_wq = nn.Linear(768, 768)\n        self.projection_wk = nn.Linear(768, 768)\n        self.projection_wv = nn.Linear(768, 768)")

    # Strip any remaining XML-style tags that LLM might have put inside the code.
    code = re.sub(r"<(hp|delta|nn|nn_head)>.*?</\1>", "", code, flags=re.DOTALL)
    code = re.sub(r"<(hp|delta|nn|nn_head)>", "", code)
    code = re.sub(r"</(hp|delta|nn|nn_head)>", "", code)

    # Replace undefined custom modules with safe built-ins when LLM invents them.
    undefined_module_replacements = {
        "ResidualBottleneck": "HallucinatedModule",
        "FlashAttentionLayer": "HallucinatedModule",
        "AdapterBlock": "HallucinatedModule",
        "CustomAttention": "HallucinatedModule",
        "CrossModalFusion": "HallucinatedModule",
        "GatingFusion": "HallucinatedModule",
        "TransformerCell": "HallucinatedModule",
        "CrossModalAttention": "HallucinatedModule",
        "FeedForward": "nn.Linear",
        "PositionWiseFeedForward": "nn.Linear",
        "Bottleneck": "HallucinatedModule",
        "TransformerBlock": "HallucinatedModule",
        "GatedFusion": "HallucinatedModule",
        "TransformerAdapter": "HallucinatedModule",
        "LinearProjection": "nn.Linear",
    }
    for old, new in undefined_module_replacements.items():
        if old in code and f"class {old}" not in code:
            # Fix malformed init calls to these hallucinated modules (e.g. GatingFusion(c visual = 768))
            code = re.sub(re.escape(old) + r"\(.*?\)", new + "()", code)
            code = code.replace(old, new)

    # Fix common malformed method definitions like "def self.xxxx(self):"
    code = re.sub(r"def\s+self\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", r"def \1(", code)
    
    # Final cleanup for common malformed torch calls
    code = code.replace("nn.nn.", "nn.")
    
    # Fix hallucinated layer name spacers (e.g. self.layer scooter_name = nn.Linear)
    code = re.sub(r"self\.([a-zA-Z_][a-zA-Z0-9_]*)\s+[a-zA-Z_][a-zA-Z0-9_]*\s*=", r"self.\1 =", code)

    # Fix expression-level hallucinations like (self.f fused) or (self.layer name)
    code = re.sub(r"\(self\.([a-zA-Z_][a-zA-Z0-9_]*)\s+([a-zA-Z_][a-zA-Z0-9_]*)\)", r"(self.\1_\2)", code)
    code = re.sub(r"self\.([a-zA-Z_][a-zA-Z0-9_]*)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+([+\-*/])", r"self.\1_\2 \3", code)
    code = code.replace("self.adapter_expansion", "4")

    # Ensure basic safe definitions for common hallucinated modules.
    safe_defs = [
        "class HallucinatedModule(nn.Module):\n    def __init__(self, *args, **kwargs): super().__init__()\n    def forward(self, x, *args, **kwargs): return x\n",
        "if 'MultiHeadAttention' in code and 'class MultiHeadAttention' not in code:\n"
        "    code = 'class MultiHeadAttention(nn.Module):\\n    def __init__(self, d=768, h=8, drp=0.1):\\n        super().__init__()\\n        self.mha = nn.MultiheadAttention(d, h, dropout=drp, batch_first=True)\\n    def forward(self, x, *args, **kwargs):\\n        return self.mha(x, x, x)[0]\\n\\n' + code",
    ]
    
    # We already handled replacements above.
    
    if "HallucinatedModule" in code and "class HallucinatedModule" not in code:
        code = "class HallucinatedModule(nn.Module):\n    def __init__(self, *args, **kwargs): super().__init__()\n    def forward(self, x, *args, **kwargs): return x\n\n" + code

    if "MultiHeadAttention" in code and "class MultiHeadAttention" not in code:
        mha_def = (
            "class MultiHeadAttention(nn.Module):\n"
            "    def __init__(self, d=768, h=8, drp=0.1):\n"
            "        super().__init__()\n"
            "        self.mha = nn.MultiheadAttention(d, h, dropout=drp, batch_first=True)\n"
            "    def forward(self, x, *args, **kwargs):\n"
            "        return self.mha(x, x, x)[0]\n\n"
        )
        code = mha_def + code

    # --- ADVANCED AUTO-RESOLVE: Missing Attribute Injector ---
    # Find all self.XXXX calls in forward()
    if "def forward" in code:
        forward_part = code.split("def forward")[1]
        calls = re.findall(r"self\.([a-zA-Z_][a-zA-Z0-9_]*)\(", forward_part)
        # Find all self.XXXX initializations in __init__
        init_part = code.split("def forward")[0]
        inits = re.findall(r"self\.([a-zA-Z_][a-zA-Z0-9_]*)\s*=", init_part)
        
        missing = [c for c in set(calls) if c not in set(inits) and c not in ["super", "forward"]]
        if missing:
            injection = ""
            for m in missing:
                # Default to Linear if it looks like a layer call
                injection += f"\n        self.{m} = nn.Linear(768, 768)"
            
            # Clean all existing super init variants (very aggressive)
            code = re.sub(r"[ \t]*super\(.*?\)\.__init__\(.*?\)[ \t]*", "", code)
            
            # Inject single clean init at start
            code = code.replace("def __init__(self, prm):", "def __init__(self, prm):\n        super().__init__()" + injection)

    # Ensure bridge class exists. If the LLM produced only inner layers, wrap fallback.
    if "class CrossModalBridge" not in code:
        code = (
            "class CrossModalBridge(nn.Module):\n"
            "    def __init__(self, prm):\n"
            "        super().__init__()\n"
            "        self.proj = nn.Sequential(\n"
            "            nn.Linear(768, 768),\n"
            "            nn.LayerNorm(768),\n"
            "            nn.GELU(),\n"
            "        )\n"
            "    def forward(self, x, captions=None):\n"
            "        return self.proj(x)\n"
        )

    # Final fix for common double self (occurs during aggressive regex replacements)
    code = code.replace("self.self.", "self.")
    
    return code


def extract_cross_modal_bridge_only(code):
    """
    Extract only class CrossModalBridge from generated or previously assembled code.

    This prevents LLM/full-model output from overriding stable classes such as:
    - FrozenBlip2Encoder
    - CaptionDecoder
    - Net
    """
    if not code:
        return ""

    code = improve_code(code)

    try:
        tree = ast.parse(code)
        lines = code.splitlines()

        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "CrossModalBridge":
                start = node.lineno - 1
                end = getattr(node, "end_lineno", None)
                if end is None:
                    # Fallback for older Python, scan until next top-level class/function.
                    end = len(lines)
                    for i in range(start + 1, len(lines)):
                        line = lines[i]
                        if line.startswith("class ") or line.startswith("def "):
                            end = i
                            break
                return "\n".join(lines[start:end]).strip()

    except Exception:
        pass

    # Regex fallback.
    m = re.search(
        r"(class\s+CrossModalBridge\s*\([\s\S]*?)(?=^class\s+|^def\s+|\Z)",
        code,
        flags=re.MULTILINE,
    )
    if m:
        return m.group(1).strip()

    # Safe fallback bridge.
    return (
        "class CrossModalBridge(nn.Module):\n"
        "    def __init__(self, prm):\n"
        "        super().__init__()\n"
        "        self.proj = nn.Sequential(\n"
        "            nn.Linear(768, 768),\n"
        "            nn.LayerNorm(768),\n"
        "            nn.GELU(),\n"
        "        )\n"
        "    def forward(self, x, captions=None):\n"
        "        return self.proj(x)\n"
    )

def extract_code(txt):
    """
    Extract Python code from LLM response.

    Preferred:
    - <nn_head>...</nn_head>
    - <nn>...</nn>
    - fenced python blocks
    """
    if not txt:
        return None

    tagged = extract_str(txt, "<nn_head>", "</nn_head>") or extract_str(txt, "<nn>", "</nn>")
    if tagged:
        return improve_code(tagged.strip())

    blocks = re.findall(r"```(?:python)?\n(.*?)\n```", txt, re.DOTALL)
    for block in blocks:
        if "class CrossModalBridge" in block:
            return improve_code(block.strip())

    if blocks:
        return improve_code(blocks[0].strip())

    # Last-resort extraction if response contains raw class code.
    idx = txt.find("class CrossModalBridge")
    if idx != -1:
        return improve_code(txt[idx:].strip())

    return None


def verify_nn_code(nn_dir, nn_file):
    """
    Basic generated-model verification:
    - syntactically valid Python
    - required pipeline interface exists
    """
    try:
        with open(nn_file, "r", encoding="utf-8") as f:
            code = f.read()

        ast.parse(code)

        required = [
            "def supported_hyperparameters",
            "class FrozenBlip2Encoder",
            "class CrossModalBridge",
            "class CaptionDecoder",
            "class Net",
            "def train_setup",
            "def learn",
            "def forward",
        ]

        ok = all(x in code for x in required)
        if not ok:
            missing = [x for x in required if x not in code]
            with open(Path(nn_dir) / "error_code_verification.txt", "w+") as f:
                f.write(f"Missing required components: {missing}")
            return False

        return True

    except Exception as e:
        with open(Path(nn_dir) / "error_code_verification.txt", "w+") as f:
            f.write(f"Verification failed: {e}")
        return False


def copy_to_lemur(gen_nn_dir, name, task, dataset, metric):
    from ab.gpt.util.Const import new_lemur_nn_dir, new_lemur_stat_dir, new_nn_file

    Path(new_lemur_nn_dir).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(gen_nn_dir / new_nn_file, new_lemur_nn_dir / f"{name}.py")

    dr_nm = new_lemur_stat_dir / f"{task}_{dataset}_{metric}_{name}"
    Path(dr_nm).mkdir(parents=True, exist_ok=True)

    for f_nm in [f for f in os.listdir(gen_nn_dir) if re.match(r"[0-9]+\.json", f)]:
        shutil.copyfile(gen_nn_dir / f_nm, dr_nm / f_nm)


# ── Safe bridge pool ─────────────────────────────────────────────────────────
# Each variant: input (B,32,768) → output (B,32,768).
# Updated to avoid residuals where domain shift is large (SOTA findings).
_SAFE_BRIDGES = [
    # 0 – MLP Projection (No Residual) - Recommended for BLIP-2 -> GPT-2
    "class CrossModalBridge(nn.Module):\n"
    "    def __init__(self, prm):\n"
    "        super().__init__()\n"
    "        d = 768\n"
    "        drop = float(safe_prm(prm, 'dropout', 0.1))\n"
    "        self.mapping = nn.Sequential(\n"
    "            nn.Linear(d, 4 * d),\n"
    "            nn.GELU(),\n"
    "            nn.Dropout(drop),\n"
    "            nn.Linear(4 * d, d),\n"
    "            nn.Dropout(drop)\n"
    "        )\n"
    "    def forward(self, x, captions=None):\n"
    "        return self.mapping(x)\n",
    # 1 – Gated Projection (No Residual)
    "class CrossModalBridge(nn.Module):\n"
    "    def __init__(self, prm):\n"
    "        super().__init__()\n"
    "        d = int(safe_prm(prm, 'ff_dim', 512))\n"
    "        self.norm = nn.LayerNorm(768)\n"
    "        self.fc = nn.Linear(768, d)\n"
    "        self.gate = nn.Linear(768, d)\n"
    "        self.out = nn.Linear(d, 768)\n"
    "    def forward(self, x, captions=None):\n"
    "        r = self.norm(x)\n"
    "        return self.out(self.fc(r) * torch.sigmoid(self.gate(r)))\n",
    # 2 – MHA Transformer Block (No Residual on the core projection)
    "class CrossModalBridge(nn.Module):\n"
    "    def __init__(self, prm):\n"
    "        super().__init__()\n"
    "        h = int(safe_prm(prm, 'num_heads', 8))\n"
    "        self.norm = nn.LayerNorm(768)\n"
    "        self.attn = nn.MultiheadAttention(768, h, batch_first=True)\n"
    "        self.proj = nn.Linear(768, 768)\n"
    "    def forward(self, x, captions=None):\n"
    "        r = self.norm(x)\n"
    "        a, _ = self.attn(r, r, r)\n"
    "        return self.proj(a)\n",
    # 3 – Bottleneck Projection
    "class CrossModalBridge(nn.Module):\n"
    "    def __init__(self, prm):\n"
    "        super().__init__()\n"
    "        bn = int(safe_prm(prm, 'decoder_dim', 256))\n"
    "        self.norm = nn.LayerNorm(768)\n"
    "        self.down = nn.Linear(768, bn)\n"
    "        self.up = nn.Linear(bn, 768)\n"
    "    def forward(self, x, captions=None):\n"
    "        return self.up(torch.gelu(self.down(self.norm(x))))\n",
    # 4 – Stacked MLP
    "class CrossModalBridge(nn.Module):\n"
    "    def __init__(self, prm):\n"
    "        super().__init__()\n"
    "        nl = int(safe_prm(prm, 'num_layers', 2))\n"
    "        layers = []\n"
    "        for _ in range(nl):\n"
    "            layers += [nn.LayerNorm(768), nn.Linear(768, 768), nn.GELU()]\n"
    "        self.mixer = nn.Sequential(*layers)\n"
    "    def forward(self, x, captions=None):\n"
    "        return self.mixer(x)\n",
]

# Patterns that guarantee a runtime shape error – if found, reject the bridge.
_BAD_BRIDGE_PATTERNS = [
    r'\.view\s*\(\s*[Bb]\s*,\s*-1',           # view(B,-1)  → collapses seq dim
    r'\.reshape\s*\(\s*[Bb]\s*,\s*-1',         # reshape(B,-1)
    r'\.flatten\s*\(',                           # flatten()
    r'nn\.LayerNorm\s*\(\s*\[',                 # LayerNorm([list]) wrong
    r'\.view\s*\(\s*[Bb]\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+',  # view to 4D
    r'nn\.Conv2d\s*\(',                          # Conv2d on seq
    r'nn\.AdaptiveAvgPool2d\s*\(',               # pool on seq
    r'self\s+[a-zA-Z_]',                          # 'self prv' space error
    r't_preprojector_input_size',                 # undefined attribute
    r'prm\.get\s*\(\s*["\'][^"\']+["\']\s*\)(?!\s*[,)])',  # prm.get without default
    r'\.permute\s*\(',                           # permute often breaks (B,32,768)
    r'\.transpose\s*\(',
]


def _is_bridge_valid(code: str) -> tuple:
    """Return (True, 'OK') or (False, reason_string)."""
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f'SyntaxError L{e.lineno}: {e.msg}'
    for pat in _BAD_BRIDGE_PATTERNS:
        if re.search(pat, code):
            return False, f'Forbidden pattern: {pat[:40]}'
    if 'def forward' not in code:
        return False, 'No forward() method'
    fwd = code.split('def forward')[1]
    if 'return' not in fwd:
        return False, 'forward() has no return'
    return True, 'OK'




def assemble_nn_code(nn_head_code, prm=None, device="cuda", q_former_hidden=768):
    """
    Build a stable Blip2Fast-style image-captioning model around an
    LLM-generated CrossModalBridge.

    Stable controlled skeleton:
    - FrozenBlip2Encoder
    - CaptionDecoder with GPT2-small
    - Net with train_setup / learn / forward

    LLM-controlled part:
    - CrossModalBridge only

    Contract:
    - CrossModalBridge input:  (B, 32, 768)
    - CrossModalBridge output: (B, 32, 768)

    This is captioning-specific and should be used only by the Blip2Fast /
    image-captioning NN-GPT generation path.
    """
    import random

    # 1) Extract only CrossModalBridge from LLM output.
    try:
        nn_head_code = extract_cross_modal_bridge_only(nn_head_code or _SAFE_BRIDGES[0])
    except Exception:
        nn_head_code = nn_head_code or _SAFE_BRIDGES[0]

    # 2) Make prm access null-safe.
    nn_head_code = nn_head_code.replace("prm.get(", "safe_prm(prm, ")

    # 2.5) Repair MultiheadAttention single-argument calls (e.g. self.slf_attn(x) -> self.slf_attn(x, x, x))
    import re
    nn_head_code = re.sub(
        r'self\.([a-zA-Z0-9_]*(?:attn|attention)[a-zA-Z0-9_]*)\s*\(\s*([a-zA-Z0-9_]+)\s*\)',
        r'self.\1(\2, \2, \2)',
        nn_head_code
    )

    # 2.6) Repair CrossModalBridge forward signature if it only accepts a single argument
    nn_head_code = re.sub(
        r'def\s+forward\s*\(\s*self\s*,\s*([a-zA-Z0-9_]+)\s*(?::\s*[a-zA-Z0-9_\.\[\]\s]+)?\)\s*:',
        r'def forward(self, \1, captions=None, *args, **kwargs):',
        nn_head_code
    )

    # 3) Validate bridge syntax and obvious shape-killing patterns.
    try:
        valid, reason = _is_bridge_valid(nn_head_code)
    except Exception as e:
        valid, reason = False, str(e)

    if not valid:
        chosen = random.choice(_SAFE_BRIDGES)
        print(f"  [REPAIR] LLM bridge rejected ({reason}). Using safe fallback bridge.")
        nn_head_code = chosen

    # 4) Repair a common unsafe init pattern in one safe bridge variant.
    if "nn.init.zeros_(self.mixer[-2].weight)" in nn_head_code:
        safe_init = (
            "for layer in self.mixer:\n"
            "            if hasattr(layer, 'weight'):\n"
            "                nn.init.zeros_(layer.weight)\n"
            "                if hasattr(layer, 'bias') and layer.bias is not None:\n"
            "                    nn.init.zeros_(layer.bias)"
        )
        nn_head_code = nn_head_code.replace("nn.init.zeros_(self.mixer[-2].weight)", safe_init)
        nn_head_code = nn_head_code.replace("nn.init.zeros_(self.mixer[-2].bias)", "")

    lines = [
        "import os",
        "import torch",
        "import torch.nn as nn",
        "import torch.nn.functional as F",
        "from transformers import Blip2Model, GPT2LMHeadModel, GPT2Config, GPT2Tokenizer",
        "",
        "os.environ['TOKENIZERS_PARALLELISM'] = 'false'",
        "",
        "",
        "def supported_hyperparameters():",
        "    return {'lr', 'batch', 'dropout', 'num_layers', 'num_heads', 'ff_dim', 'decoder_dim', 'load_in_4bit', 'freeze_gpt2'}",
        "",
        "",
        "def safe_prm(prm, key, default=None):",
        "    if prm is None:",
        "        prm = {}",
        "    if default is None:",
        "        lk = key.lower()",
        "        if any(x in lk for x in ['dim', 'size', 'hidden']):",
        "            default = 768",
        "        elif any(x in lk for x in ['head', 'layer', 'num']):",
        "            default = 8",
        "        elif 'dropout' in lk:",
        "            default = 0.1",
        "        elif 'lr' in lk:",
        "            default = 1e-4",
        "        else:",
        "            default = 0",
        "    val = prm.get(key, default) if isinstance(prm, dict) else default",
        "    if val is None:",
        "        return default",
        "    try:",
        "        if isinstance(default, bool):",
        "            return bool(val)",
        "        if isinstance(default, int):",
        "            return int(val)",
        "        if isinstance(default, float):",
        "            return float(val)",
        "    except Exception:",
        "        pass",
        "    return val",
        "",
        "",
        "class FrozenBlip2Encoder(nn.Module):",
        "    def __init__(self, device, load_in_4bit=True):",
        "        super().__init__()",
        "        self.device = device",
        "        self.hidden_size = 768",
        "        self.blip2 = None",
        "        model_id = 'Salesforce/blip2-opt-2.7b'",
        "",
        "        # Lazy-compatible design:",
        "        # If cached_blip2 transform is used, input is already (B,32,768),",
        "        # so this class will not need to run BLIP2 forward.",
        "        try:",
        "            self.blip2 = Blip2Model.from_pretrained(",
        "                model_id,",
        "                torch_dtype=torch.float16,",
        "                load_in_4bit=bool(load_in_4bit),",
        "                device_map={'': device},",
        "            )",
        "        except TypeError:",
        "            self.blip2 = Blip2Model.from_pretrained(",
        "                model_id,",
        "                torch_dtype=torch.float16,",
        "                low_cpu_mem_usage=True,",
        "                device_map={'': device},",
        "            )",
        "        except Exception as e:",
        "            # Cached feature mode can still work without loading BLIP2 here.",
        "            print(f'[FrozenBlip2Encoder WARN] BLIP2 load skipped/failed: {e}')",
        "            self.blip2 = None",
        "",
        "        if self.blip2 is not None:",
        "            for param in self.blip2.parameters():",
        "                param.requires_grad = False",
        "            self.blip2.eval()",
        "            try:",
        "                self.hidden_size = self.blip2.config.qformer_config.hidden_size",
        "            except Exception:",
        "                self.hidden_size = 768",
        "",
        "        self.eval()",
        "",
        "    def train(self, mode=True):",
        "        super().train(False)",
        "        if self.blip2 is not None:",
        "            self.blip2.eval()",
        "        return self",
        "",
        "    def forward(self, pixel_values):",
        "        # Fast cached feature path: cached_blip2 gives (B,32,768).",
        "        if torch.is_tensor(pixel_values):",
        "            if pixel_values.dim() == 3 and pixel_values.size(-1) == self.hidden_size:",
        "                return pixel_values.to(self.device).float()",
        "            if pixel_values.dim() == 2 and pixel_values.size(-1) == self.hidden_size:",
        "                return pixel_values.to(self.device).float().unsqueeze(1)",
        "",
        "        if self.blip2 is None:",
        "            raise RuntimeError('BLIP2 encoder is not loaded and input is not cached features (B,T,768). Use --transform cached_blip2 or fix BLIP2 loading.')",
        "",
        "        self.blip2.eval()",
        "        with torch.no_grad():",
        "            outputs = self.blip2.get_qformer_features(pixel_values=pixel_values.to(self.device))",
        "        if torch.is_tensor(outputs):",
        "            return outputs.float()",
        "        return outputs.last_hidden_state.float()",
        "",
        "",
        "# === LLM-GENERATED BRIDGE ===",
        nn_head_code,
        "# === END LLM-GENERATED BRIDGE ===",
        "",
        "",
        "class CaptionDecoder(nn.Module):",
        "    def __init__(self, q_former_hidden, device, prm):",
        "        super().__init__()",
        "        self.device = device",
        "        self.prm = prm if isinstance(prm, dict) else {}",
        "        gpt2_id = 'gpt2'",
        "",
        "        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_id)",
        "        self.tokenizer.pad_token = self.tokenizer.eos_token",
        "",
        "        config = GPT2Config.from_pretrained(gpt2_id)",
        "        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_id, config=config).to(device)",
        "        self.gpt2_hidden = int(config.n_embd)",
        "",
        "        freeze_gpt2 = bool(self.prm.get('freeze_gpt2', False))",
        "        if freeze_gpt2:",
        "            for p in self.gpt2.parameters():",
        "                p.requires_grad = False",
        "",
        "        self.bridge = CrossModalBridge(self.prm).to(device)",
        "",
        "    def _normalize_visual_embeds(self, visual_embeds, batch_size):",
        "        if not torch.is_tensor(visual_embeds):",
        "            raise TypeError('CrossModalBridge must return a torch.Tensor')",
        "",
        "        visual_embeds = visual_embeds.to(self.device).float()",
        "",
        "        if visual_embeds.dim() == 4:",
        "            visual_embeds = visual_embeds.reshape(visual_embeds.size(0), -1, visual_embeds.size(-1))",
        "        elif visual_embeds.dim() == 2:",
        "            visual_embeds = visual_embeds.unsqueeze(1)",
        "        elif visual_embeds.dim() > 4:",
        "            visual_embeds = visual_embeds.reshape(visual_embeds.size(0), -1, visual_embeds.size(-1))",
        "",
        "        if visual_embeds.dim() != 3:",
        "            visual_embeds = visual_embeds.reshape(batch_size, -1, visual_embeds.size(-1))",
        "",
        "        # Fix accidental (T,B,H) output.",
        "        if visual_embeds.size(0) != batch_size:",
        "            if visual_embeds.size(1) == batch_size:",
        "                visual_embeds = visual_embeds.transpose(0, 1).contiguous()",
        "            else:",
        "                visual_embeds = visual_embeds.reshape(batch_size, -1, visual_embeds.size(-1))",
        "",
        "        # Keep prefix length under control. cached_blip2 normally uses 32 tokens.",
        "        if visual_embeds.size(1) > 32:",
        "            visual_embeds = visual_embeds[:, :32, :]",
        "",
        "        hidden = visual_embeds.size(-1)",
        "        if hidden > self.gpt2_hidden:",
        "            visual_embeds = visual_embeds[..., :self.gpt2_hidden]",
        "        elif hidden < self.gpt2_hidden:",
        "            visual_embeds = F.pad(visual_embeds, (0, self.gpt2_hidden - hidden))",
        "",
        "        return visual_embeds",
        "",
        "    def forward(self, visual_features, caption_ids=None):",
        "        batch_size = visual_features.size(0)",
        "        visual_features = visual_features.to(self.device).float()",
        "        visual_embeds = self.bridge(visual_features, caption_ids)",
        "        visual_embeds = self._normalize_visual_embeds(visual_embeds, batch_size)",
        "",
        "        if caption_ids is not None:",
        "            if caption_ids.dim() == 3:",
        "                caption_ids = caption_ids[:, 0, :]",
        "            caption_ids = caption_ids.long().to(self.device)",
        "            caption_ids = caption_ids.clamp(min=0, max=self.gpt2.config.vocab_size - 1)",
        "",
        "            start_tokens = torch.full(",
        "                (batch_size, 1),",
        "                self.tokenizer.eos_token_id,",
        "                dtype=torch.long,",
        "                device=self.device,",
        "            )",
        "            caption_ids = torch.cat([start_tokens, caption_ids], dim=1)",
        "",
        "            text_embeds = self.gpt2.transformer.wte(caption_ids)",
        "            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)",
        "",
        "            ignore_labels = torch.full(",
        "                (batch_size, visual_embeds.shape[1] + 1),",
        "                -100,",
        "                dtype=torch.long,",
        "                device=self.device,",
        "            )",
        "            labels = torch.cat([ignore_labels, caption_ids[:, 1:]], dim=1)",
        "",
        "            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id",
        "            visual_mask = torch.ones((batch_size, visual_embeds.shape[1]), dtype=torch.long, device=self.device)",
        "            text_mask = (caption_ids != pad_id).long().to(self.device)",
        "            attention_mask = torch.cat([visual_mask, text_mask], dim=1)",
        "            return self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels).loss",
        "",
        "        # Inference: SOTA-safe greedy generation with per-sequence EOS mask.",
        "        start_token = torch.full(",
        "            (batch_size, 1),",
        "            self.tokenizer.eos_token_id,",
        "            dtype=torch.long,",
        "            device=self.device,",
        "        )",
        "        start_embed = self.gpt2.transformer.wte(start_token)",
        "        outputs_embeds = torch.cat([visual_embeds, start_embed], dim=1)",
        "",
        "        generated = []",
        "        past_key_values = None",
        "        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)",
        "",
        "        for _ in range(40):",
        "            out = self.gpt2(",
        "                inputs_embeds=outputs_embeds,",
        "                past_key_values=past_key_values,",
        "                use_cache=True,",
        "            )",
        "            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)",
        "",
        "            # Once a sample is finished, keep forcing EOS for that sample.",
        "            next_token[finished] = self.tokenizer.eos_token_id",
        "            generated.append(next_token)",
        "",
        "            finished |= (next_token.squeeze(-1) == self.tokenizer.eos_token_id)",
        "            if finished.all():",
        "                break",
        "",
        "            past_key_values = out.past_key_values",
        "            outputs_embeds = self.gpt2.transformer.wte(next_token)",
        "",
        "        if not generated:",
        "            return torch.empty((batch_size, 0), dtype=torch.long, device=self.device)",
        "        return torch.cat(generated, dim=1)",
        "",
        "",
        "class Net(nn.Module):",
        "    def __init__(self, in_shape, out_shape, prm, device):",
        "        super().__init__()",
        "        self.device = device",
        "        self.prm = prm if isinstance(prm, dict) else {}",
        "        self.vocab_size = int(out_shape[0]) if out_shape and len(out_shape) > 0 else 50257",
        "        print(f'!!! [MODEL SETUP] Hyperparameters: {self.prm}')",
        "",
        "        load_in_4bit = bool(self.prm.get('load_in_4bit', True))",
        "        self.encoder = FrozenBlip2Encoder(device, load_in_4bit=load_in_4bit)",
        "        self.decoder = CaptionDecoder(self.encoder.hidden_size, device, self.prm)",
        "",
        "        # Generic Train.py sometimes expects criterion. Captioning uses decoder loss.",
        "        self.criterion = lambda o, l: torch.tensor(0.0, device=self.device, requires_grad=True)",
        "        self.idx2word = None",
        "        self.optimizer = None",
        "        self._print_param_stats()",
        "",
        "    def _print_param_stats(self):",
        "        total = sum(p.numel() for p in self.parameters())",
        "        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)",
        "        pct = (100.0 * trainable / total) if total else 0.0",
        "        print(f'   Total params: {total:,} | Trainable: {trainable:,} ({pct:.1f}%)')",
        "",
        "    def _ensure_vocab(self):",
        "        if self.idx2word is not None:",
        "            return",
        "        try:",
        "            from ab.nn.loader.coco_.Caption import GLOBAL_CAPTION_VOCAB",
        "            self.idx2word = GLOBAL_CAPTION_VOCAB.get('idx2word', {})",
        "        except Exception:",
        "            self.idx2word = {}",
        "",
        "    def forward(self, pixel_values, captions=None):",
        "        self.encoder.eval()",
        "        visual_features = self.encoder(pixel_values)",
        "        if captions is not None:",
        "            self._ensure_vocab()",
        "            if captions.dim() == 3:",
        "                captions = captions[:, 0, :]",
        "            return self.decoder(visual_features, captions)",
        "        return self.decoder(visual_features, None)",
        "",
        "    def train_setup(self, prm):",
        "        if isinstance(prm, dict):",
        "            self.prm.update(prm)",
        "        trainable_params = [p for p in self.decoder.parameters() if p.requires_grad]",
        "        lr = float(self.prm.get('lr', 1e-4))",
        "        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr)",
        "",
        "    def learn(self, train_data):",
        "        self.encoder.eval()",
        "        self.decoder.train()",
        "        self._ensure_vocab()",
        "        if self.optimizer is None:",
        "            self.train_setup(self.prm)",
        "",
        "        total_loss = 0.0",
        "        n = 0",
        "",
        "        for images, captions in train_data:",
        "            if isinstance(images, list):",
        "                images = torch.stack(images)",
        "            if isinstance(captions, list):",
        "                captions = torch.stack(captions)",
        "",
        "            images = images.to(self.device)",
        "            captions = captions.to(self.device)",
        "            if captions.dim() == 3:",
        "                captions = captions[:, 0, :]",
        "",
        "            self.optimizer.zero_grad(set_to_none=True)",
        "            loss = self.forward(images, captions)",
        "            if not torch.is_tensor(loss):",
        "                loss = torch.tensor(float(loss), device=self.device, requires_grad=True)",
        "",
        "            loss.backward()",
        "            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.0)",
        "            self.optimizer.step()",
        "",
        "            total_loss += float(loss.detach().item())",
        "            n += 1",
        "",
        "        return 0.0, total_loss / max(n, 1)",
    ]

    return "\n".join(lines)
