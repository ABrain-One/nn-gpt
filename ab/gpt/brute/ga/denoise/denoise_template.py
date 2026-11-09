"""Skip-configurable denoise U-Net template.

``render_model(chromosome)`` returns a complete, standalone LEMUR ``Net`` source
string. The architecture is fully determined by a small gene dict that is baked
in as ``_GENES``; the surrounding parametric U-Net is fixed, so *every* genotype
renders a runnable, same-spatial-output, mobile-clean model. That guarantee is
what lets the GA smoke test isolate the search machinery (engine/crossover/
fitness) from model-construction bugs.

Skip-focused genes:
  - global_residual : bool   -> return inp + f(inp)  vs  f(inp)
  - fusion          : add|concat  -> encoder/decoder skip fusion mode
Structural genes (give crossover material):
  - depth : int   number of down/up stages
  - width : int   base channel count
  - act   : relu|gelu
"""

# The fixed parametric model. `{genes_json}` is the only substitution point.
_TEMPLATE = '''import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Genome for this individual (rendered by the denoise GA).
_GENES = {genes_json}


def supported_hyperparameters():
    return {{"lr"}}


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


def _act(name):
    return nn.GELU() if name == "gelu" else nn.ReLU(inplace=True)


class _Conv(nn.Module):
    """conv -> BatchNorm (mobile-GPU-delegate clean) -> activation."""
    def __init__(self, in_ch, out_ch, stride=1, act="gelu"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = _act(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class _Fuse(nn.Module):
    """Fuse a decoder feature (dec_ch) with an encoder skip (skip_ch)."""
    def __init__(self, dec_ch, skip_ch, mode):
        super().__init__()
        self.mode = mode
        if mode == "concat":
            self.reduce = nn.Conv2d(dec_ch + skip_ch, dec_ch, 1)
        else:  # add
            self.proj = nn.Conv2d(skip_ch, dec_ch, 1) if skip_ch != dec_ch else nn.Identity()

    def forward(self, dec, skip):
        if skip.shape[-2:] != dec.shape[-2:]:
            skip = F.interpolate(skip, size=dec.shape[-2:], mode="nearest")
        if self.mode == "concat":
            return self.reduce(torch.cat([dec, skip], dim=1))
        return dec + self.proj(skip)


class _DenoiseUNet(nn.Module):
    def __init__(self, genes, in_channels=3):
        super().__init__()
        depth = int(genes["depth"])
        width = int(genes["width"])
        act = genes["act"]
        fusion = genes["fusion"]
        self.global_residual = bool(genes["global_residual"])

        enc_ch = [width * (2 ** i) for i in range(depth)]

        # Encoder: stride-2 downsample stages.
        self.downs = nn.ModuleList()
        prev = in_channels
        for i in range(depth):
            self.downs.append(_Conv(prev, enc_ch[i], stride=2, act=act))
            prev = enc_ch[i]

        self.bottleneck = _Conv(enc_ch[-1], enc_ch[-1], stride=1, act=act)

        # Decoder: mirror upsampling with skip fusion.
        self.ups = nn.ModuleList()
        self.fuses = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        cur = enc_ch[-1]
        for k in range(depth):
            i = depth - 2 - k  # matching encoder skip index (-1 -> input)
            out_ch = enc_ch[i] if i >= 0 else width
            skip_ch = enc_ch[i] if i >= 0 else in_channels
            self.ups.append(nn.ConvTranspose2d(cur, out_ch, 3, stride=2,
                                               padding=1, output_padding=1))
            self.fuses.append(_Fuse(out_ch, skip_ch, fusion))
            self.dec_convs.append(_Conv(out_ch, out_ch, stride=1, act=act))
            cur = out_ch

        self.final = nn.Conv2d(cur, 3, 3, padding=1)

    def forward(self, inp):
        skips = []
        x = inp
        for down in self.downs:
            x = down(x)
            skips.append(x)
        x = self.bottleneck(x)

        depth = len(self.downs)
        for k in range(depth):
            i = depth - 2 - k
            skip = skips[i] if i >= 0 else inp
            x = self.ups[k](x)
            x = self.fuses[k](x, skip)
            x = self.dec_convs[k](x)

        out = self.final(x)
        if self.global_residual:
            out = inp + out
        return out


class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={{}}, device="cuda"):
        super().__init__()
        self.device = device
        self.net = _DenoiseUNet(_GENES, in_channels=3)
        self.criterion = CharbonnierLoss()
        self.train_setup(prm)
        self.to(self.device)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        t_max = prm.get("epoch_max", prm.get("epoch", 50))
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, t_max), eta_min=1e-5)

    def forward(self, x):
        return torch.clamp(self.net(x), 0.0, 1.0)

    def learn(self, train_data):
        self.train()
        for inputs, targets in train_data:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.forward(inputs), targets)
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()
'''


def render_model(chromosome):
    """Render a standalone Net source string from a gene dict."""
    genes = {
        "depth": chromosome["depth"],
        "width": chromosome["width"],
        "act": chromosome["act"],
        "fusion": chromosome["fusion"],
        "global_residual": chromosome["global_residual"],
    }
    # repr() -> a valid Python literal (True/False/'str'), not JSON.
    return _TEMPLATE.format(genes_json=repr(genes))
