"""MobileNetV2 (Sandler et al., 2018) implemented from scratch and adapted for
CIFAR-10's 32x32 inputs.

Adaptation vs the ImageNet architecture (which downsamples input by 32x across
5 stride-2 points and would collapse a 32x32 image to <1px before the classifier):
  - stem conv: stride 2 -> 1 (drop one 2x downsample)
  - stage (t=6, c=24, n=2) stride: 2 -> 1 (drop a second 2x downsample)
This gives 8x total downsampling (32x32 -> 4x4 before global average pooling),
matching common CIFAR MobileNetV2 practice. All other stage strides are unchanged
from the original paper.
"""
import torch
import torch.nn as nn

# (expand_ratio t, out_channels c, num_blocks n, stride s)
CIFAR_INVERTED_RESIDUAL_CFG = [
    (1, 16, 1, 1),
    (6, 24, 2, 1),   # stride 2 -> 1 for CIFAR-10 (32x32 input)
    (6, 32, 3, 2),
    (6, 64, 4, 2),
    (6, 96, 3, 1),
    (6, 160, 3, 2),
    (6, 320, 1, 1),
]


def _make_divisible(value, divisor=8, min_value=None):
    """Round channel counts to the nearest multiple of `divisor`, never rounding
    down by more than 10% (matches the standard MobileNetV2 channel-rounding rule)."""
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


class ConvBNReLU6(nn.Sequential):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_c, out_c, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_c, momentum=0.1, eps=1e-5),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_c, out_c, stride, expand_ratio):
        super().__init__()
        assert stride in (1, 2)
        hidden_dim = int(round(in_c * expand_ratio))
        self.use_residual = stride == 1 and in_c == out_c

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU6(in_c, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNReLU6(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, out_c, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_c, momentum=0.1, eps=1e-5),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2CIFAR(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0, dropout=0.2,
                 inverted_residual_cfg=None):
        super().__init__()
        cfg = inverted_residual_cfg or CIFAR_INVERTED_RESIDUAL_CFG

        input_channel = _make_divisible(32 * width_mult)
        last_channel = _make_divisible(1280 * max(1.0, width_mult))

        features = [ConvBNReLU6(3, input_channel, kernel_size=3, stride=1)]

        for t, c, n, s in cfg:
            out_channel = _make_divisible(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, out_channel, stride, t))
                input_channel = out_channel

        features.append(ConvBNReLU6(input_channel, last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(last_channel, num_classes),
        )
        self.last_channel = last_channel
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_model(num_classes=10, width_mult=1.0, dropout=0.2):
    return MobileNetV2CIFAR(num_classes=num_classes, width_mult=width_mult, dropout=dropout)


if __name__ == "__main__":
    model = build_model()
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print("output shape:", out.shape)
    print("total params:", n_params, f"(~{n_params * 4 / 1024 / 1024:.2f} MB fp32)")
