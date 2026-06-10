from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F



def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var, mean = torch.var_mean(x, dim=1, keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class RestorationBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2, drop_prob: float = 0.):
        super().__init__()
        hidden = channels * expansion
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.sg = SimpleGate()
        
        # Simplified Channel Attention (SCA)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden // 2, hidden // 2, 1)
        )
        self.conv2 = nn.Conv2d(hidden // 2, channels, 1)
        
        self.norm2 = LayerNorm2d(channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            SimpleGate(),
            nn.Conv2d(hidden // 2, channels, 1),
        )
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.drop_path = DropPath(drop_prob) if drop_prob > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.dwconv(y)
        y = self.sg(y)
        y = y * self.sca(y)
        y = self.conv2(y)
        x = x + self.drop_path(self.beta * y)

        z = self.norm2(x)
        z = self.ffn(z)
        x = x + self.drop_path(self.gamma * z)
        return x


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, padding=1),
            nn.PixelUnshuffle(2),
            nn.Conv2d(channels * 8, channels * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, padding=1),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def _make_stage(channels: int, blocks: int, drop_probs) -> nn.Sequential:
    return nn.Sequential(*[RestorationBlock(channels, drop_prob=dp) for dp in drop_probs])


class PromptIR(nn.Module):
    def __init__(
        self,
        width: int = 64,
        enc_blocks: tuple[int, ...] | list[int] = (2, 2, 4, 8),
        bottleneck_blocks: int = 8,
        dec_blocks: tuple[int, ...] | list[int] = (8, 4, 2, 2),
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.intro = nn.Conv2d(3, width, 3, padding=1)
        
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.reduces = nn.ModuleList()

        # calculating drop path rates
        total_blocks = sum(enc_blocks) + bottleneck_blocks + sum(dec_blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        dpr_idx = 0

        c = width
        for num_blocks in enc_blocks:
            self.encoders.append(_make_stage(c, num_blocks, dpr[dpr_idx:dpr_idx+num_blocks]))
            dpr_idx += num_blocks
            self.downs.append(Downsample(c))
            c *= 2
        
        self.bottleneck = _make_stage(c, bottleneck_blocks, dpr[dpr_idx:dpr_idx+bottleneck_blocks])
        dpr_idx += bottleneck_blocks

        for num_blocks in dec_blocks:
            self.ups.append(Upsample(c))
            c //= 2
            self.reduces.append(nn.Conv2d(c * 2, c, 1))
            self.decoders.append(_make_stage(c, num_blocks, dpr[dpr_idx:dpr_idx+num_blocks]))
            dpr_idx += num_blocks

            
        self.outro = nn.Conv2d(width, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = self.intro(x)
        
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)
            
        x = self.bottleneck(x)
        
        for up, reduce, dec, skip in zip(self.ups, self.reduces, self.decoders, reversed(skips)):
            x = up(x)
            x = reduce(torch.cat([x, skip], dim=1))
            x = dec(x)
            
        return self.outro(x) + inp

def build_model(config: dict | None = None) -> PromptIR:
    config = config or {}
    return PromptIR(
        width=int(config.get("width", 64)),
        enc_blocks=tuple(config.get("enc_blocks", (2, 2, 4, 8))),
        bottleneck_blocks=int(config.get("bottleneck_blocks", 8)),
        dec_blocks=tuple(config.get("dec_blocks", (8, 4, 2, 2))),
        drop_path_rate=float(config.get("drop_path_rate", 0.0)),
    )
