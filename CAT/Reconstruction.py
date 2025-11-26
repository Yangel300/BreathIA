import torch
from torch import nn

class PostMLPReshape(nn.Module):
    def __init__(self, *target_shape: int):
        super().__init__()
        if len(target_shape) == 1 and isinstance(target_shape[0], (list, tuple)):
            target_shape = tuple(target_shape[0])
        self.target_shape = tuple(int(d) for d in target_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        F = x.shape[1]

        prod = 1
        for d in self.target_shape:
            prod *= d
        if F != prod:
            raise RuntimeError(
                f"PostMLPReshape: no puedo convertir de (B, {F}) a (B, {self.target_shape}); "
                f"se requieren {prod} features."
            )
        return x.reshape(B, *self.target_shape)


@dataclass
class Conv2DHeadConfig:
    in_channels: int          # = C que sale del PostMLPReshape
    out_channels: int = 16
    kernel_size: int = 3
    stride: int = 1
    padding: str | int = "same"   # "same" => padding automático
    use_bn: bool = True
    activation: str | None = "relu"  # "relu", "gelu", "silu" o None

class Conv2DHead(nn.Module):
    def __init__(self, cfg: Conv2DHeadConfig):
        super().__init__()
        if cfg.padding == "same":
            # para stride=1 y dilatación=1: SAME -> k//2
            pad = cfg.kernel_size // 2
        else:
            pad = int(cfg.padding)

        layers = []
        layers.append(
            nn.Conv2d(
                in_channels=cfg.in_channels,
                out_channels=cfg.out_channels,
                kernel_size=cfg.kernel_size,
                stride=cfg.stride,
                padding=pad,
                bias=not cfg.use_bn,
            )
        )
        if cfg.use_bn:
            layers.append(nn.BatchNorm2d(cfg.out_channels))

        if cfg.activation:
            act = cfg.activation.lower()
            if   act == "relu": layers.append(nn.ReLU(inplace=True))
            elif act == "gelu": layers.append(nn.GELU())
            elif act == "silu": layers.append(nn.SiLU(inplace=True))
            else: raise ValueError(f"Activación no soportada: {cfg.activation}")

        self.net = nn.Sequential(*layers)
        self.in_channels = cfg.in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise RuntimeError(f"Conv2DHead: se esperaba (B, C, H, W) y llegó {tuple(x.shape)}.")
        if x.shape[1] != self.in_channels:
            raise RuntimeError(
                f"Conv2DHead: C={x.shape[1]} no coincide con in_channels={self.in_channels}."
            )
        return self.net(x)