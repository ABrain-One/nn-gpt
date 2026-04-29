import json
import shutil
import itertools
import random
import torchvision
from pathlib import Path
import torch
import gc

from ab.gpt.util.Const import epoch_dir, new_nn_file, synth_dir, fract_dir

FORWARD_PATTERNS = {
    "Parallel_Triple": """
    def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:
        x_f = adaptive_pool_flatten(self.features(x))
        x_a = adaptive_pool_flatten(self.backbone_a(x))
        x_b = adaptive_pool_flatten(self.backbone_b(x))
        fused = torch.cat([x_f, x_a, x_b], dim=1)
        if is_probing: return fused
        return self.classifier(fused)
    """,
    # "Residual_Bypass": """
    # def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:
    #     x = self._norm4d(x).to(self.device)
    #     identity = adaptive_pool_flatten(self.backbone_a(x))
    #     x_f = adaptive_pool_flatten(self.features(x))
    #
    #     mid = identity + x_f if identity.shape == x_f.shape else torch.cat([identity, x_f], dim=1)
    #
    #     fused = adaptive_pool_flatten(self.backbone_b(mid)) if mid.ndim == 4 else mid
    #     if is_probing: return fused
    #     return self.classifier(fused)
    # """,
    # "Fractal_First_Parallel": """
    # def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:
    #     x = self._norm4d(x).to(self.device)
    #     x_f = self.features(x)
    #     f_a = adaptive_pool_flatten(self.backbone_a(x_f))
    #     f_b = adaptive_pool_flatten(self.backbone_b(x_f))
    #     fused = torch.cat([f_a, f_b], dim=1)
    #     if is_probing: return fused
    #     return self.classifier(fused)
    # """,
    # "Backbone_A_First_Parallel": """
    # def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:
    #     x = self._norm4d(x).to(self.device)
    #     x_a = self.backbone_a(x)
    #     f_f = adaptive_pool_flatten(self.features(x_a))
    #     f_b = adaptive_pool_flatten(self.backbone_b(x_a))
    #     fused = torch.cat([f_f, f_b], dim=1)
    #     if is_probing: return fused
    #     return self.classifier(fused)
    # """,
    # "Sequential_Backbones_to_Fractal": """
    # def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:
    #     x = self._norm4d(x).to(self.device)
    #     x = self.backbone_a(x)
    #     x = self.backbone_b(x)
    #     x = self.features(x)
    #     fused = adaptive_pool_flatten(x)
    #     if is_probing: return fused
    #     return self.classifier(fused)
    # """,

    # "Sequential_Fractal_to_Backbones": """
    # def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:
    #     x = self._norm4d(x).to(self.device)
    #     x = self.features(x)
    #     x = self.backbone_a(x)
    #     x = self.backbone_b(x)
    #     fused = adaptive_pool_flatten(x)
    #     if is_probing: return fused
    #     return self.classifier(fused)
    # """,
#     # "Ensemble_Backbones_to_Fractal": """
#     # def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:
#     #     x = self._norm4d(x).to(self.device)
#     #     f_a = adaptive_pool_flatten(self.backbone_a(x))
#     #     f_b = adaptive_pool_flatten(self.backbone_b(x))
#     #     mid = torch.cat([f_a, f_b], dim=1)
#     #     mid_4d = mid.unsqueeze(-1).unsqueeze(-1)
#     #     mid_img = torch.nn.functional.interpolate(mid_4d, size=(14,14), mode='nearest')
        
#     #     fused = adaptive_pool_flatten(self.features(mid_img))
#     #     if is_probing: return fused
#     #     return self.classifier(fused)
#     # """,
# 
#     # "Split_A_Parallel_BF": """
#     # def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:
#     #     x = self._norm4d(x).to(self.device)
        
#     #     f_a = adaptive_pool_flatten(self.backbone_a(x))
        
#     #     x_bf = self.backbone_b(x)
#     #     if x_bf.dim() == 2:
#     #         x_bf = x_bf.unsqueeze(-1).unsqueeze(-1)
#     #     if x_bf.shape[-1] < 14:
#     #         x_bf = torch.nn.functional.interpolate(x_bf, size=(14,14), mode='nearest')
            
#     #     f_bf = adaptive_pool_flatten(self.features(x_bf))
#     #     fused = torch.cat([f_a, f_bf], dim=1)
#     #     if is_probing: return fused
#     #     return self.classifier(fused)
#     # """,

    # "Split_Fractal_Parallel_AB": """
    # def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:
    #     x = self._norm4d(x).to(self.device)
    #
    #     f_f = adaptive_pool_flatten(self.features(x))
    #
    #     x_ab = self.backbone_a(x)
    #     x_ab = self.backbone_b(x_ab)
    #     f_ab = adaptive_pool_flatten(x_ab)
    #     fused = torch.cat([f_f, f_ab], dim=1)
    #     if is_probing: return fused
    #     return self.classifier(fused)
    # """
}

DIVERSE_FORWARD_PATTERNS = {
    "A_to_Fractal_plus_B": """
    def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:
        x = self._norm4d(x).to(self.device)
        x_a_map = self.backbone_a(x)
        x_a_img = self._feature_to_input_image(x_a_map, "a")
        x_f = adaptive_pool_flatten(self.features(x_a_img))
        x_b = adaptive_pool_flatten(self.backbone_b(x))
        x_a = adaptive_pool_flatten(x_a_map)
        fused = torch.cat([x_a, x_f, x_b], dim=1)
        if is_probing: return fused
        return self.classifier(fused)
    """,
    "B_to_Fractal_plus_A": """
    def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:
        x = self._norm4d(x).to(self.device)
        x_b_map = self.backbone_b(x)
        x_b_img = self._feature_to_input_image(x_b_map, "b")
        x_f = adaptive_pool_flatten(self.features(x_b_img))
        x_a = adaptive_pool_flatten(self.backbone_a(x))
        x_b = adaptive_pool_flatten(x_b_map)
        fused = torch.cat([x_b, x_f, x_a], dim=1)
        if is_probing: return fused
        return self.classifier(fused)
    """,
    "Fractal_to_DualBackbone": """
    def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:
        x = self._norm4d(x).to(self.device)
        x_f_map = self.features(x)
        x_f_img = self._feature_to_input_image(x_f_map, "f")
        x_a = adaptive_pool_flatten(self.backbone_a(x_f_img))
        x_b = adaptive_pool_flatten(self.backbone_b(x_f_img))
        x_f = adaptive_pool_flatten(x_f_map)
        fused = torch.cat([x_f, x_a, x_b], dim=1)
        if is_probing: return fused
        return self.classifier(fused)
    """,
    "A_to_Fractal_to_B": """
    def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:
        x = self._norm4d(x).to(self.device)
        x_a_map = self.backbone_a(x)
        x_a_img = self._feature_to_input_image(x_a_map, "a")
        x_f_map = self.features(x_a_img)
        x_f_img = self._feature_to_input_image(x_f_map, "f")
        x_b = adaptive_pool_flatten(self.backbone_b(x_f_img))
        x_a = adaptive_pool_flatten(x_a_map)
        x_f = adaptive_pool_flatten(x_f_map)
        fused = torch.cat([x_a, x_f, x_b], dim=1)
        if is_probing: return fused
        return self.classifier(fused)
    """,
}

DIVERSE_FORWARD_HELPER = """
    def _feature_to_input_image(self, x: torch.Tensor, adapter_name: str) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        elif x.dim() == 3:
            x = x.mean(dim=1).unsqueeze(-1).unsqueeze(-1)
        elif x.dim() != 4:
            x = x.flatten(1).unsqueeze(-1).unsqueeze(-1)

        c_in, h, w = self._input_spec
        in_channels = int(x.shape[1])
        key = f"{adapter_name}_{in_channels}"
        if not hasattr(self, "_input_adapters"):
            self._input_adapters = nn.ModuleDict()
        if key not in self._input_adapters:
            self._input_adapters[key] = nn.Conv2d(in_channels, c_in, kernel_size=1).to(self.device)

        x = self._input_adapters[key](x)
        if x.shape[-2:] != (h, w):
            x = torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return x
"""

CHANNEL_LOGIC = {
    "Serial_Cascade": lambda img, a, b, f: (img, a, f),
    "Residual_Bypass": lambda img, a, b, f: (img, img, a + f),
    "Fractal_First_Parallel": lambda img, a, b, f: (f, img, f),
    "Backbone_A_First_Parallel": lambda img, a, b, f: (img, a, a),
    "Sequential_Backbones_to_Fractal": lambda img, a, b, f: (img, b, a),
    "Sequential_Fractal_to_Backbones": lambda img, a, b, f: (f, img, a),
    "Ensemble_Backbones_to_Fractal": lambda img, a, b, f: (img, a + b, img),
    "Split_A_Parallel_BF": lambda img, a, b, f: (img, b, img),
    "Split_Fractal_Parallel_AB": lambda img, a, b, f: (img, img, a)
}

CHANNEL_CACHE = {}

def probe_model_output_channels(model_name):
    if model_name in CHANNEL_CACHE:
        return CHANNEL_CACHE[model_name]
    try:
        if hasattr(torchvision.models, "get_model"):
            m = torchvision.models.get_model(model_name, weights=None)
        else:
            m = torchvision.models.__dict__[model_name](pretrained=False)
        layers = list(m.children())
        if len(layers) > 1:
            feature_extractor = torch.nn.Sequential(*layers[:-1])
        else:
            feature_extractor = m
        feature_extractor.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = feature_extractor(dummy)
            if isinstance(out, (tuple, list)): out = out[0]
            c = out.shape[1]
        CHANNEL_CACHE[model_name] = c
        return c
    except:
        return 512

def filter_backbones_by_size(max_params_millions=50):
    print(f"Filtering backbones with < {max_params_millions}M parameters...")
    candidates = [name for name in dir(torchvision.models)
                  if not name.startswith("_")
                  and callable(getattr(torchvision.models, name))
                  and name[0].islower()
                  and "get_" not in name]

    safe_list = []
    for name in candidates:
        try:
            model = torchvision.models.get_model(name, weights=None)
            param_count = sum(p.numel() for p in model.parameters())
            if (param_count / 1e6) < max_params_millions:
                import time
                start = time.time()
                model.cuda()
                for i in range(5):
                    x = torch.randn(2, 3, 224, 224).cuda()
                    y = model(x)
                    y.mean().backward()
                elapsed = time.time() - start
                if elapsed < 0.2:
                    safe_list.append(name)
            del model
        except Exception as e:
            print(f"Failed to test {name}: {e}")
            continue
    gc.collect()
    return safe_list


def generate_conv_block():
    conv_first = 'nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias)'
    conv_mid = 'nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)'
    bn = 'nn.BatchNorm2d(out_channels)'
    acts = ['nn.ReLU(inplace=True)', 'nn.GELU()', 'nn.SiLU(inplace=True)']
    dropout = 'nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()'

    def create_sequence():
        seq = [conv_first]
        pool = [bn, *acts, dropout, conv_mid]
        seq.extend(random.choices(pool, k=random.randint(2, 4)))
        return seq

    def is_valid(seq):
        for i in range(len(seq) - 1):
            curr, nxt = seq[i], seq[i + 1]
            if any(a in curr for a in ['ReLU', 'GELU', 'SiLU']) and \
                    any(a in nxt for a in ['ReLU', 'GELU', 'SiLU']):
                return False
            if 'BatchNorm' in curr and 'BatchNorm' in nxt:
                return False
            if 'BatchNorm' in nxt and 'Conv2d' not in curr:
                return False
        return True

    for _ in range(10):
        candidate = create_sequence()
        if is_valid(candidate):
            return ",\n        ".join(candidate)

    return ",\n        ".join([conv_first, bn, 'nn.ReLU(inplace=True)'])


def _generate_patterns(epochs, patterns, max_variants, helper_code=""):
    print("Load Model Complete, Start Loop...")

    shutil.rmtree(epoch_dir(), ignore_errors=True)
    available_backbones = filter_backbones_by_size(max_params_millions=30)

    for bb in available_backbones:
        probe_model_output_channels(bb)

    for epoch in range(epochs):
        out_path = epoch_dir(epoch)
        template_content = (fract_dir / 'backbone' / "FractalFusion_template.py").read_text()

        counter = 0

        for pattern_name, forward_code in patterns.items():
            for i in range(max_variants):
                block_code = generate_conv_block()
                bb_a, bb_b = random.sample(available_backbones, 2)

                n = random.randint(1, 2)
                cols = random.randint(2, 3)

                val_img = 3
                val_a = probe_model_output_channels(bb_a)
                val_b = probe_model_output_channels(bb_b)
                val_f = 64 * (2**(n-1))

                ch_a_in, ch_f_in, ch_b_in = val_a, val_f, val_b

                model_dir = synth_dir(out_path) / f"B{counter}"
                model_dir.mkdir(parents=True, exist_ok=True)

                nn_code = (template_content
                           .replace("$$", block_code)
                           .replace("?FORWARD", helper_code + forward_code)
                           .replace("?PATTERN", pattern_name)
                           .replace("?CH_A_IN", str(ch_a_in))
                           .replace("?CH_F_IN", str(ch_f_in))
                           .replace("?CH_B_IN", str(ch_b_in))
                           .replace("?N", str(n))
                           .replace("?COLS", str(cols))
                           .replace("?bb_a", f'"{bb_a}"')
                           .replace("?bb_b", f'"{bb_b}"'))

                (model_dir / new_nn_file).write_text(nn_code)

                counter += 1
                if counter % 50 == 0:
                    print(f"Generated {counter} models total...")


def alter(epochs, test_conf, llm_name, gguf_file=None):
    _generate_patterns(epochs, FORWARD_PATTERNS, 50)


def alter_diverse(epochs, variants_per_pattern=20):
    _generate_patterns(epochs, DIVERSE_FORWARD_PATTERNS, variants_per_pattern, DIVERSE_FORWARD_HELPER)
