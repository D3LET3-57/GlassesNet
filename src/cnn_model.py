"""
ResNet-style CNN built entirely from scratch with torch.nn.

No pretrained weights, no torchvision model zoo, no NumPy in the forward or
backward path. The architecture (stage count, widths, depth per stage) is
driven by the Config so the assignment's configurable-hyperparameter
requirement is satisfied.
"""
import torch
import torch.nn as nn

from src.config import Config


def get_activation(name: str, inplace: bool = False) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=inplace)
    if name == "silu":
        return nn.SiLU(inplace=inplace)
    if name == "gelu":
        return nn.GELU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")


class BasicBlock(nn.Module):
    """Two 3x3 convolutions with a residual connection."""

    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 activation: str = "relu"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = get_activation(activation, inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = get_activation(activation, inplace=True)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act2(out + identity)
        return out


class GlassesCNN(nn.Module):
    """Configurable ResNet-style binary classifier."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        act = cfg.activation

        self.stem = nn.Sequential(
            nn.Conv2d(3, cfg.stem_width, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(cfg.stem_width),
            get_activation(act, inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        stages = []
        in_ch = cfg.stem_width
        for s in range(cfg.num_hidden):
            out_ch = cfg.sizes[s]
            blocks = []
            for b in range(cfg.blocks_per_stage[s]):
                stride = 2 if (s > 0 and b == 0) else 1
                blocks.append(BasicBlock(in_ch, out_ch, stride, act))
                in_ch = out_ch
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(cfg.dropout),
            nn.Linear(in_ch, cfg.output_size),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.cfg.activation in ("relu", "leaky_relu", "silu"):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                            nonlinearity="relu")
                else:
                    nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        return self.head(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(cfg: Config) -> GlassesCNN:
    return GlassesCNN(cfg)
