import ast
import re
import textwrap

available_backbones = ['convnext_tiny', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_v2_s', 'googlenet', 'inception_v3', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_1_6gf', 'regnet_x_3_2gf', 'regnet_y_400mf', 'regnet_y_800mf', 'resnet18', 'resnet34', 'resnet50', 'resnext50_32x4d', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'squeezenet1_0', 'squeezenet1_1', 'swin_t', 'swin_v2_t']

available_patterns = [
    'Parallel_Triple', 
    'Backbone_A_to_Fractal', 
    'Backbone_B_to_Fractal', 
    'Dual_Backbone_Fuse_Then_Fractal',
    'Fractal_Then_Dual_Backbone',
    'Split_Stem_Parallel_Fuse',
    'Fractal_to_DualBackbone',
    'B_to_Fractal_plus_A',
    'A_to_Fractal_plus_B',
    'A_to_Fractal_to_B'
]

legacy_patterns = tuple(available_patterns)

open_discovery_goal_profiles = (
    {
        "name": "StemProjectCascade",
        "tags": ("stem", "project", "multi_stage"),
        "brief": "Build a real learned stem before branching. Each major branch should pass through an explicit project or bridge module before a later-stage fusion. Avoid a single terminal concat of raw branch outputs.",
        "module_hints": ("self.stem", "self.project_a", "self.project_b", "self.bridge", "self.fuse"),
        "realization": "Start from a visible `self.stem`, then keep explicit `self.project_*` or `self.bridge` modules on the branches before a later fuse. The fuse should happen after at least two visible stages, not as a single raw concat at classifier input.",
    },
    {
        "name": "DeepFractalProject",
        "tags": ("fractal_deep", "project", "multi_stage"),
        "brief": "Create one deeper path with at least two fractal-like stages or a fractal stage plus a projector before fusion. The deep path should be structurally different from a one-shot fractal branch.",
        "module_hints": ("self.project", "self.bridge", "self.fractal_stage1", "self.fractal_stage2", "self.fuse"),
        "realization": "Make one branch visibly deeper with `self.fractal_stage1` plus `self.fractal_stage2`, or a fractal stage followed by `self.project` or `self.bridge`, before the final fuse. Keep the projector or bridge in the actual forward graph.",
    },
    {
        "name": "SplitStemWideFuse",
        "tags": ("stem", "wide_fuse", "multi_stage"),
        "brief": "Use a shared stem to feed asymmetric branches, then perform a staged wide fusion. One branch may stay lightweight while another becomes deeper, but they should not meet only once at the classifier input.",
        "module_hints": ("self.stem", "self.branch_a", "self.branch_b", "self.project", "self.fuse"),
        "realization": "Use `self.stem` first, then split into asymmetric branches, then perform a visible staged or wide fuse. Do not wait until the classifier input to combine the branches only once.",
    },
    {
        "name": "ProjectReuseMixer",
        "tags": ("project", "branch_reuse", "multi_stage"),
        "brief": "Let one branch condition or align another through a project, bridge, adapter, or mixer block before the final classifier. Use at least two graph merges or one explicit reuse stage plus a final fuse.",
        "module_hints": ("self.project", "self.bridge", "self.adapter", "self.mixer", "self.fuse"),
        "realization": "Keep an explicit `self.project` or `self.bridge` path that conditions or reuses another branch through `self.adapter` or `self.mixer`, then finish with a later fuse. The reuse stage must be visible in `forward`.",
    },
    {
        "name": "StemProjectWide",
        "tags": ("stem", "project", "wide_fuse"),
        "brief": "Expose a visible stem and explicit projection modules, then use a wider fusion with three or more incoming branch features. The projected branches should remain visible in the computation graph.",
        "module_hints": ("self.stem", "self.project_a", "self.project_b", "self.project_c", "self.fuse"),
        "realization": "Build a visible `self.stem`, keep explicit projection modules, and make the final fuse visibly wide with three or more incoming projected features.",
    },
)

goal_profile_target_patterns = {
    "StemProjectCascade": "Stem_Project_Cascade",
    "DeepFractalProject": "Deep_Fractal_Project",
    "SplitStemWideFuse": "Split_Stem_Wide_Fuse",
    "ProjectReuseMixer": "Project_Reuse_Mixer",
    "StemProjectWide": "Stem_Project_Wide_Fuse",
}


def goal_profile_target_pattern(profile_or_name) -> str:
    if isinstance(profile_or_name, dict):
        profile_name = str(profile_or_name.get("name") or "").strip()
    else:
        profile_name = str(profile_or_name or "").strip()
    if not profile_name:
        return "Dual_Backbone_Custom"
    return goal_profile_target_patterns.get(profile_name, profile_name)


def goal_tag_parser_cues(tags) -> str:
    cues = []
    for tag in tuple(tags or ()):
        if tag == "stem":
            cues.append("- `stem`: apply a visible stem module before any branch split; do not let both backbones read the raw input directly.")
        elif tag == "project":
            cues.append("- `project`: include an explicit project or bridge module call in `forward`, not just the final classifier.")
        elif tag == "multi_stage":
            cues.append("- `multi_stage`: show at least two visible stages before the final classifier or final fuse.")
        elif tag == "wide_fuse":
            cues.append("- `wide_fuse`: make one fuse consume three or more incoming feature tensors.")
        elif tag == "fractal_deep":
            cues.append("- `fractal_deep`: use at least two visible fractal-stage calls in one branch, or one fractal call plus enough extra depth to avoid a shallow one-shot branch.")
        elif tag == "branch_reuse":
            cues.append("- `branch_reuse`: let one branch explicitly feed or condition another through a reuse, adapter, mixer, or extra merge stage.")
    return "\n".join(cues) if cues else "- No extra parser-visible cues."

skeleton_code = """import torch
import torch.nn as nn
import numpy as np
import gc
import torchvision
from torch.nn import MaxPool2d
from torch.amp import autocast, GradScaler

# ==========================================
# 1. FIXED INFRASTRUCTURE (DO NOT MODIFY)
# ==========================================
class TorchVision(nn.Module):
    def __init__(self, model: str, weights: str = "DEFAULT", unwrap: bool = True, truncate: int = 1, in_channels: int = 3):
        super().__init__()
        self.model_name = str(model)
        self.adapter = nn.Conv2d(in_channels, 3, kernel_size=1) if in_channels != 3 else nn.Identity()
        kwargs = {"aux_logits": False} if "inception" in model.lower() else {}
        try:
            if hasattr(torchvision.models, "get_model"):
                self.m = torchvision.models.get_model(model, weights=weights, **kwargs)
            else:
                self.m = torchvision.models.__dict__[model](pretrained=bool(weights), **kwargs)
        except:
            if hasattr(torchvision.models, "get_model"):
                self.m = torchvision.models.get_model(model, weights=weights)
            else:
                self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        
        if unwrap:
            layers = []
            for name, module in self.m.named_children():
                if "aux" in name.lower(): continue
                layers.append(module)
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
        else:
            self.m.head = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        return self.m(self.adapter(x))

def adaptive_pool_flatten(x):
    if x.ndim == 4: return torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
    if x.ndim == 3: return x.mean(dim=1)
    return x.flatten(1) if x.ndim > 2 else x

def autocast_ctx(enabled=True):
    return autocast("cuda", enabled=enabled)
def make_scaler(enabled=True):
    return GradScaler("cuda", enabled=enabled)

def supported_hyperparameters():
    return { 'lr', 'dropout', 'momentum' }

# ==========================================
# 2. DYNAMIC COMPONENTS (TO BE IMPLEMENTED)
# ==========================================

def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):

class FractalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob):
        super().__init__()
        self.num_columns = int(num_columns)
        depth = 2 ** max(self.num_columns - 1, 0)
        blocks = []
        for i in range(depth):
            level = nn.ModuleList()
            for j in range(self.num_columns):
                if (i + 1) % (2 ** j) == 0:
                    in_ch_ij = in_channels if (i + 1 == 2 ** j) else out_channels
                    level.append(drop_conv3x3_block(in_ch_ij, out_channels, dropout_prob=dropout_prob))
            blocks.append(level)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        outs = [x] * self.num_columns
        for level_block in self.blocks:
            outs_i = [blk(inp) for blk, inp in zip(level_block, outs)]
            joined = torch.stack(outs_i, dim=0).mean(dim=0)
            outs[:len(level_block)] = [joined] * len(level_block)
        return outs[0]

class FractalUnit(nn.Module):
    def __init__(self, in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob):
        super().__init__()
        self.block = FractalBlock(in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.block(x))

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        

    def infer_dimensions_dynamically(self, num_classes):
        self.to(self.device)
        was_training = self.training
        self.eval()
        classifier = getattr(self, "classifier", None)
        with torch.no_grad():
            c, h, w = self._input_spec
            dummy = torch.zeros(1, c, h, w).to(self.device)
            self.classifier = nn.Identity()
            output_feat = self.forward(dummy, is_probing=True)
            output_feat = adaptive_pool_flatten(output_feat)
            if output_feat.dim() != 2:
                output_feat = output_feat.flatten(1)
            dim_fused = output_feat.shape[1]
        self.classifier = nn.Linear(dim_fused, num_classes).to(self.device)
        if classifier is not None and isinstance(classifier, nn.Module):
            del classifier
        self.train(was_training)

    @staticmethod
    def _norm4d(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4: return x
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            return x.reshape(B * T, C, H, W)
        raise ValueError(f"Expected 4D/5D input, got {tuple(x.shape)}")

    def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:
        
    def train_setup(self, prm):
        self.to(self.device)
        self.freeze_backbones = bool(prm.get('freeze_backbones', getattr(self, 'freeze_backbones', True)))
        for backbone in (self.backbone_a, self.backbone_b):
            for param in backbone.parameters():
                param.requires_grad = not self.freeze_backbones
            if self.freeze_backbones:
                backbone.eval()
            else:
                backbone.train()
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        trainable_params = [param for param in self.parameters() if param.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters remain after applying freeze_backbones")
        self.optimizer = torch.optim.SGD(trainable_params, lr=prm['lr'], momentum=prm['momentum'])
        self._scaler = make_scaler(enabled=self.use_amp)

    def learn(self, train_data):
        self.train()
        if self.freeze_backbones:
            self.backbone_a.eval()
            self.backbone_b.eval()
        else:
            self.backbone_a.train()
            self.backbone_b.train()
        scaler = self._scaler
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        train_iter = iter(train_data)
        try:
            for batch_idx, (inputs, labels) in enumerate(train_iter):
                inputs = inputs.to(self.device).float()
                labels = labels.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                with autocast_ctx(enabled=self.use_amp):
                    outputs = self(inputs)
                    loss = self.criterion(outputs, labels)
                if not torch.isfinite(loss): continue
                batch_size = int(labels.size(0))
                total_loss += float(loss.detach().item()) * batch_size
                total_correct += int((outputs.detach().argmax(dim=1) == labels).sum().item())
                total_examples += batch_size
                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.parameters(), 3.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 3.0)
                    self.optimizer.step()
        finally:
            if hasattr(train_iter, 'shutdown'): train_iter.shutdown()
            del train_iter
            gc.collect()
        if total_examples == 0:
            return 0.0, None
        train_accuracy = total_correct / total_examples
        train_loss = total_loss / total_examples
        return train_accuracy, train_loss
"""

prompt_template="""
### Role & Context
You are a Senior AI Architect. Your task is to implement a **specific** model instance based on a strict skeleton to achieve an accuracy of {accuracy}. 

### Task Overview
Complete the three missing components. **DO NOT** write generic code. You must implement the architecture using the target design pattern provided.

[CODE SKELETON START]
{skeleton_code}
[CODE SKELETON END]

### Technical Specifications (MANDATORY REQUIREMENTS)

1. **Target Pattern: `{target_pattern}`**
   - YOU MUST explicitly set `self.pattern = '{target_pattern}'` inside `__init__`.
   - YOU MUST implement the logic for this specific pattern throughout the code.
   - **CRITICAL REQUIREMENT**: DO NOT just blindly copy the standard Parallel_Triple structure. You MUST be highly creative and design a truly unique structural flow in `forward`. Vary your module usage and connection topology!

2. **Component: `drop_conv3x3_block`**
   - Implement starting with `def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):`.
   - Return an `nn.Sequential` block (Conv2d -> BatchNorm2d -> Activation -> Dropout2d).

3. **Component: `Net.__init__`**
   - Implement starting with `def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:`.
   - **MANDATORY**: `self.pattern = '{target_pattern}'`
   - **Backbone Selection**: Choose EXACTLY TWO models from [{available_backbones}].
   - **Initialization**: 
     - Initialize `self.backbone_a` and `self.backbone_b` using `TorchVision(model='...', in_channels=...)`.
     - Initialize `self.features` (1-2 `FractalUnit` layers).
     - Store the image input spec on `self._input_spec` so the provided dynamic sizing helper can probe the classifier dimension.
     - Finish `__init__` by using the existing `infer_dimensions_dynamically` helper once with the class count from `out_shape`.
   - **Example Implementation Fragment**:
     ```python
     self.pattern = '{target_pattern}'
     self.backbone_a = TorchVision(model='resnet18', in_channels=in_shape[1]).to(device)
     ...
     ```

4. **Component: `Net.forward`**
   - Implement starting with `def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:`.
   - **Flow Control**: Implement the data flow for `{target_pattern}`. Use `adaptive_pool_flatten` for module outputs before fusion.
   - **Fusion Patterns Logic Blueprint**:
     * `Parallel_Triple`: `Result = Concat(backbone_a(x), backbone_b(x), features(x))`
     * `Backbone_A_to_Fractal`: `Result = features(backbone_a(x))` (Sequential flow)
     * `Split_Stem_Parallel_Fuse`: `stem_out = STEM(x); Result = Concat(backbone_a(stem_out), backbone_b(stem_out))`
   - **CRITICAL - NO GHOSTING**: You MUST use ALL defined components in the `forward` pass.
   - **CRITICAL RESTRICTION**: You MUST build the computational graph directly without using ANY `if self.pattern == ...` control flow or dynamic loops (like `getattr`/`hasattr`) inside `forward`.
   - **PARAM REMINDER**: Always pass `in_channels=...` when creating `TorchVision` models.

### Output Requirement (STRICT)
Output ONLY the implementation within the XML tags. Each tag MUST contain the complete function/method definition (signature and body). No markdown, no conversation.

<block>
# Full drop_conv3x3_block implementation
</block>
<init>
# Full __init__ implementation
</init>
<forward>
# Full forward implementation
</forward>
"""

open_discovery_skeleton_code = skeleton_code

compact_backbone_rl_prompt_template = """
### Role & Goal
You are a Senior AI Architect. Produce one trainable dual-backbone image-classification architecture that improves accuracy.

### Task Context
- Optimization Track: {goal_name}
- Optimization Target Tags: {target_tags}
- Design Brief: {design_brief}
- Seed Accuracy: `{accuracy}` (context only)

[CODE SKELETON START]
{skeleton_code}
[CODE SKELETON END]

### Basic Requirements
1. Output ONLY `<block>`, `<init>`, `<forward>`. No markdown, no explanation, no extra text.
2. Implement only `drop_conv3x3_block`, `Net.__init__`, and `Net.forward`.
3. Use EXACTLY two backbones named `self.backbone_a` and `self.backbone_b` from [{available_backbones}].
4. In `__init__`, set `self.pattern`, `self.device`, `self.use_amp`, and `self._input_spec = tuple(in_shape[1:])`, then call the existing `self.infer_dimensions_dynamically(...)` helper once with the class count after the modules used by `forward` are defined.
5. Treat the fixed infrastructure as read-only. Do not rewrite helper APIs or add replacement dimension-inference helpers.
6. Keep `forward` as a direct computation graph. Do not use `if self.pattern`, extra `import` lines, extra classes, or dynamic wrapper logic.
7. Use `adaptive_pool_flatten(...)` before concatenating or classifying branch outputs, and return classifier logits.
8. Do not reference undefined names such as `dropout_prob`, `in_channels`, or `features`.
9. Prioritize a runnable, trainable graph over novelty. Simple structure is acceptable if it trains cleanly. If you define `self.classifier`, still keep `self._input_spec` and the required dynamic sizing helper call.

### Output Requirement (STRICT)
Read the optimization feedback below, then write the final XML answer. The final answer must begin with `<block>` and end with `</forward>`.

<block>
{block_signature}
    ...
</block>
<init>
{init_signature}
    ...
</init>
<forward>
{forward_signature}
    ...
</forward>
"""

open_discovery_prompt_template = compact_backbone_rl_prompt_template
open_discovery_rl_prompt_template = compact_backbone_rl_prompt_template


def extract_target_pattern_from_code(code_str):
    try:
        tree = ast.parse(code_str)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == 'pattern'
                    and isinstance(target.value, ast.Name)
                    and target.value.id == 'self'
                ):
                    value = node.value
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        return value.value
    except Exception:
        pass
    match = re.search(r"self\.pattern\s*=\s*['\"]([^'\"]+)['\"]", code_str or "")
    return match.group(1) if match else None


def _extract_xml_tag(text, tag):
    if not isinstance(text, str):
        return None
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.IGNORECASE | re.DOTALL)
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def _strip_code_fence(code):
    if not isinstance(code, str):
        return None
    cleaned = code.strip()
    cleaned = re.sub(r"^```(?:python)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return textwrap.dedent(cleaned).strip()


def assemble_backbone_xml_completion(completion):
    block_code = _strip_code_fence(_extract_xml_tag(completion, 'block'))
    init_code = _strip_code_fence(_extract_xml_tag(completion, 'init'))
    forward_code = _strip_code_fence(_extract_xml_tag(completion, 'forward'))
    if not (block_code and init_code and forward_code):
        return None

    code = skeleton_code
    replacements = (
        (
            "def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):",
            block_code,
        ),
        (
            "    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:",
            textwrap.indent(init_code, "    "),
        ),
        (
            "    def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:",
            textwrap.indent(forward_code, "    "),
        ),
    )
    for signature, replacement in replacements:
        if signature not in code:
            raise ValueError(f"Backbone skeleton signature not found: {signature}")
        code = code.replace(signature, replacement, 1)
    return code

def parse_nn_code(code_str):
    try:
        tree = ast.parse(code_str)
        lines = code_str.splitlines()

        block_code = None
        init_code = None
        forward_code = None

        def get_source(node):
            return ast.get_source_segment(code_str, node)

        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == 'drop_conv3x3_block':
                block_code = get_source(node)

            elif isinstance(node, ast.ClassDef) and node.name == 'Net':
                for sub_node in node.body:
                    if isinstance(sub_node, ast.FunctionDef):
                        if sub_node.name == '__init__':
                            init_code = get_source(sub_node)
                        elif sub_node.name == 'forward':
                            forward_code = get_source(sub_node)

        def clean_code(c):
            return textwrap.dedent(c).strip() if c else None

        return clean_code(block_code), clean_code(init_code), clean_code(forward_code)

    except Exception as e:
        print(f"AST Parsing Failed: {e}")
        print(f"Code snippet: {code_str[:100]}...")
        return None, None, None
