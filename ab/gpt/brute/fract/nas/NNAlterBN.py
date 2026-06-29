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
    # ── PATTERN 1: Every branch sees the raw image independently.
    # Best for: Diversity of features. No channel mismatch. Most stable.
    "Parallel_Triple": """
    def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:
        parts = [adaptive_pool_flatten(self.features(x))]
        for bb in self.backbones:
            parts.append(adaptive_pool_flatten(bb(x)))
        fused = torch.cat(parts, dim=1)
        if is_probing: return fused
        return self.classifier(fused)
    """,

    # ── PATTERN 2: Backbones run in parallel, their outputs are SUMMED (not concatenated).
    # Best for: Large nb counts (avoids classifier explosion). Ensemble-like averaging.
    # "Parallel_Residual_Sum": """
    # def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:
    #     f_fractal = adaptive_pool_flatten(self.features(x))
    #     if not hasattr(self, "bb_projs"):
    #         projs = []
    #         bb_outs_temp = [adaptive_pool_flatten(bb(x)) for bb in self.backbones]
    #         if bb_outs_temp:
    #             target_dim = max(out.shape[1] for out in bb_outs_temp)
    #             for i, out in enumerate(bb_outs_temp):
    #                 dim = out.shape[1]
    #                 if dim != target_dim:
    #                     proj = torch.nn.Linear(dim, target_dim).to(x.device)
    #                 else:
    #                     proj = torch.nn.Identity().to(x.device)
    #                 projs.append(proj)
    #                 self.add_module(f"bb_proj_{i}", proj)
    #         self.bb_projs = projs

    #     bb_outs = []
    #     for bb, proj in zip(self.backbones, self.bb_projs):
    #         bb_outs.append(proj(adaptive_pool_flatten(bb(x))))

    #     if bb_outs:
    #         bb_sum = torch.stack(bb_outs, dim=0).sum(dim=0)
    #         fused = torch.cat([f_fractal, bb_sum], dim=1)
    #     else:
    #         fused = f_fractal
    #     if is_probing: return fused
    #     return self.classifier(fused)
    # """,

    # ── PATTERN 3: Fractal CNN processes raw image, then each backbone refines in sequence.
    # Best for: Backbones acting as deep refinement heads on top of custom CNN features.
    # "Fractal_Then_Sequential_Backbones": """
    # def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:
    #     x = self.features(x)
    #     for bb in self.backbones:
    #         if x.dim() == 2:
    #             x = x.unsqueeze(-1).unsqueeze(-1)
    #         if x.shape[1] != 3:
    #             x = torch.nn.functional.adaptive_avg_pool2d(x, (x.shape[-2], x.shape[-1]))
    #             x = x[:, :3, :, :] if x.shape[1] >= 3 else x.expand(-1, 3, -1, -1)
    #         x = bb(x)
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
        # Mirror TorchVision wrapper exactly: skip aux children, then truncate=1
        layers = []
        for name, module in m.named_children():
            if "aux" in name.lower():
                continue
            layers.append(module)
        feature_extractor = torch.nn.Sequential(*(layers[:-1] if len(layers) > 1 else layers))
        feature_extractor.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = feature_extractor(dummy)
            if isinstance(out, (tuple, list)):
                out = out[0]
            # Mirror adaptive_pool_flatten output dimension:
            #   4-D (B, C, H, W) → adaptive_avg_pool2d → flatten → dim = C = shape[1]
            #   3-D (B, L, C)    → mean(dim=1)         → dim = C = shape[2]
            #   2-D (B, C)       → already flat         → dim = C = shape[1]
            if out.ndim == 3:
                c = out.shape[2]
            else:
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
                  and "get_" not in name
                  and "list_" not in name]
    
    print(f'Candidates: {candidates}')
    safe_list = []
    for name in candidates:
        try:
            model = torchvision.models.get_model(name, weights=None)
            param_count = sum(p.numel() for p in model.parameters())
            if (param_count / 1e6) < max_params_millions:
                import time
                model.cuda()
                with torch.no_grad():
                    model(torch.randn(2, 3, 224, 224).cuda())
                start = time.time()
                for i in range(5):
                    x = torch.randn(2, 3, 224, 224).cuda()
                    y = model(x)
                    y.mean().backward()
                elapsed = time.time() - start
                if elapsed < 0.5:
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


def alter(epochs, test_conf, llm_name, gguf_file=None, max_variants=50, num_backbones=None, clean=True, model_prefix=""):
    print("Load Model Complete, Start Loop...")

    if clean:
        shutil.rmtree(epoch_dir(), ignore_errors=True)
    available_backbones = filter_backbones_by_size(max_params_millions=10)

    for bb in available_backbones:
        probe_model_output_channels(bb)

    for epoch in range(epochs):
        out_path = epoch_dir(epoch)
        template_content = (fract_dir / 'nas' / "FractalFusion_template.py").read_text()

        counter = 0

        for pattern_name, forward_code in FORWARD_PATTERNS.items():
            for i in range(max_variants):
                block_code = generate_conv_block()
                nb = num_backbones if num_backbones is not None else random.randint(1, 2)
                bbs = random.sample(available_backbones, min(nb, len(available_backbones)))
                backbone_names_str = ", ".join(f'"{bb}"' for bb in bbs)

                n = random.randint(1, 2)
                cols = random.randint(2, 3)

                model_name = f"{model_prefix}B{counter}" if model_prefix else f"B{counter}"
                model_dir = synth_dir(out_path) / model_name
                model_dir.mkdir(parents=True, exist_ok=True)

                nn_code = (template_content
                           .replace("$$", block_code)
                           .replace("?FORWARD", forward_code)
                           .replace("?PATTERN", pattern_name)
                           .replace("?BACKBONE_NAMES", backbone_names_str)
                           .replace("?N", str(n))
                           .replace("?COLS", str(cols)))

                (model_dir / new_nn_file).write_text(nn_code)

                # Calculate expected tensor shapes and DAG for LLM context
                fractal_out_channels = 64 * (2 ** (n - 1))
                shapes = {
                    "node_0_input": "[-1, 3, 224, 224]",
                    "node_1_fractal_features_pool": f"[-1, {fractal_out_channels}]"
                }
                
                nodes = [
                    {"id": "node_0", "op": "input", "shape": "[-1, 3, 224, 224]"},
                    {"id": "node_1", "op": "fractal_features_pool", "shape": f"[-1, {fractal_out_channels}]"}
                ]
                edges = [
                    ["node_0", "node_1"]
                ]
                
                concat_dim = fractal_out_channels
                for idx, bb_name in enumerate(bbs):
                    bb_ch = probe_model_output_channels(bb_name)
                    node_id = f"node_{idx+2}"
                    shapes[f"{node_id}_{bb_name}_pool"] = f"[-1, {bb_ch}]"
                    nodes.append({"id": node_id, "op": f"{bb_name}_pool", "shape": f"[-1, {bb_ch}]"})
                    edges.append(["node_0", node_id])
                    edges.append([node_id, "node_concat"])
                    concat_dim += bb_ch
                
                # Add fractal to concat edge
                edges.append(["node_1", "node_concat"])
                
                shapes["node_concat_fusion"] = f"[-1, {concat_dim}]"
                shapes["node_classifier_out"] = "[-1, num_classes]"
                
                nodes.append({"id": "node_concat", "op": "concat_fusion", "shape": f"[-1, {concat_dim}]"})
                nodes.append({"id": "node_classifier", "op": "classifier_out", "shape": "[-1, num_classes]"})
                edges.append(["node_concat", "node_classifier"])

                dag = {
                    "nodes": nodes,
                    "edges": edges
                }

                # Generate initial model description for LLM analysis
                description = {
                    "model_name": model_name,
                    "structure": {
                        "backbones": bbs,
                        "conv_block": [layer.strip() for layer in block_code.split(",\n        ")],
                        "fractal_N": n,
                        "fractal_cols": cols,
                        "fusion_pattern": pattern_name
                    },
                    "tensor_shape_trace": shapes,
                    "dag": dag
                }
                (model_dir / "model_description.json").write_text(
                    json.dumps(description, indent=4), encoding="utf-8"
                )

                counter += 1
                if counter % 50 == 0:
                    print(f"Generated {counter} models total...")

