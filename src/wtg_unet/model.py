from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """Two 3 x 3 convolution, batch-normalization, ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class WTGGate(nn.Module):
    """Width-aware topology gate for one encoder-decoder skip connection."""

    def __init__(
        self,
        skip_channels: int,
        decoder_channels: int,
        use_centerline: bool = True,
        use_radius: bool = True,
    ) -> None:
        super().__init__()
        self.use_centerline = use_centerline
        self.use_radius = use_radius
        middle = max(skip_channels // 2, 1)
        self.skip_projection = nn.Sequential(
            nn.Conv2d(skip_channels, middle, 1, bias=False),
            nn.BatchNorm2d(middle),
        )
        self.decoder_projection = nn.Sequential(
            nn.Conv2d(decoder_channels, middle, 1, bias=False),
            nn.BatchNorm2d(middle),
        )
        self.structure_projection = nn.Sequential(
            nn.Conv2d(2, middle, 1, bias=False),
            nn.BatchNorm2d(middle),
        )
        self.psi = nn.Conv2d(middle, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        skip: torch.Tensor,
        decoder: torch.Tensor,
        centerline: torch.Tensor,
        radius: torch.Tensor,
    ) -> torch.Tensor:
        if decoder.shape[-2:] != skip.shape[-2:]:
            decoder = F.interpolate(decoder, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if not self.use_centerline:
            centerline = torch.zeros_like(centerline)
        if not self.use_radius:
            radius = torch.zeros_like(radius)
        structure = torch.cat([centerline, radius], dim=1)
        if structure.shape[-2:] != skip.shape[-2:]:
            structure = F.interpolate(structure, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        attention = self.skip_projection(skip)
        attention = attention + self.decoder_projection(decoder)
        attention = attention + self.structure_projection(structure)
        attention = torch.sigmoid(self.psi(self.relu(attention)))
        return skip * attention


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_residual: bool,
        use_wtg: bool,
        wtg_use_centerline: bool,
        wtg_use_radius: bool,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.use_wtg = use_wtg
        self.gate = (
            WTGGate(
                skip_channels,
                out_channels,
                use_centerline=wtg_use_centerline,
                use_radius=wtg_use_radius,
            )
            if use_wtg
            else nn.Identity()
        )
        self.center_aux = nn.Conv2d(out_channels, 1, 1)
        self.radius_aux = nn.Conv2d(out_channels, 1, 1)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)
        self.residual = ResidualBlock(out_channels) if use_residual else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        centerline = torch.sigmoid(self.center_aux(x))
        radius = torch.sigmoid(self.radius_aux(x))
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if self.use_wtg:
            filtered_skip = self.gate(skip, x, centerline, radius)
        else:
            filtered_skip = skip
        x = self.conv(torch.cat([x, filtered_skip], dim=1))
        return self.residual(x)


class WTGUNet(nn.Module):
    """Four-level WTG-U-Net with mask, centerline and radius heads.

    The report describes structural predictions conditioning each WTG gate while
    the final structural heads are attached after the last decoder block. To
    make this dependency explicit in one forward pass, each decoder block has
    lightweight auxiliary centerline/radius projections for its gate; the
    public outputs are the final heads on the full-resolution decoder feature.
    """

    def __init__(
        self,
        in_channels: int = 1,
        encoder_channels: tuple[int, int, int, int] = (64, 128, 256, 512),
        bottleneck_channels: int = 1024,
        use_residual: bool = True,
        use_wtg: bool = True,
        wtg_use_centerline: bool = True,
        wtg_use_radius: bool = True,
    ) -> None:
        super().__init__()
        if len(encoder_channels) != 4:
            raise ValueError("WTGUNet requires four encoder channel values.")
        c1, c2, c3, c4 = encoder_channels
        self.encoder1 = ConvBlock(in_channels, c1)
        self.encoder2 = ConvBlock(c1, c2)
        self.encoder3 = ConvBlock(c2, c3)
        self.encoder4 = ConvBlock(c3, c4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c4, bottleneck_channels)
        self.bottleneck_residual = (
            ResidualBlock(bottleneck_channels) if use_residual else nn.Identity()
        )

        self.decoder4 = DecoderBlock(
            bottleneck_channels, c4, c4, use_residual, use_wtg,
            wtg_use_centerline, wtg_use_radius
        )
        self.decoder3 = DecoderBlock(
            c4, c3, c3, use_residual, use_wtg,
            wtg_use_centerline, wtg_use_radius
        )
        self.decoder2 = DecoderBlock(
            c3, c2, c2, use_residual, use_wtg,
            wtg_use_centerline, wtg_use_radius
        )
        self.decoder1 = DecoderBlock(
            c2, c1, c1, use_residual, use_wtg,
            wtg_use_centerline, wtg_use_radius
        )

        self.mask_head = nn.Conv2d(c1, 1, 1)
        self.centerline_head = nn.Conv2d(c1, 1, 1)
        self.radius_head = nn.Conv2d(c1, 1, 1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        e1 = self.encoder1(image)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        x = self.bottleneck(self.pool(e4))
        x = self.bottleneck_residual(x)

        x = self.decoder4(x, e4)
        x = self.decoder3(x, e3)
        x = self.decoder2(x, e2)
        x = self.decoder1(x, e1)

        mask_logits = self.mask_head(x)
        centerline_logits = self.centerline_head(x)
        radius_logits = self.radius_head(x)
        return {
            "mask_logits": mask_logits,
            "mask": torch.sigmoid(mask_logits),
            "centerline_logits": centerline_logits,
            "centerline": torch.sigmoid(centerline_logits),
            "radius_logits": radius_logits,
            "radius": torch.sigmoid(radius_logits),
        }


def build_model(config: dict[str, Any]) -> WTGUNet:
    model_cfg = config.get("model", {})
    channels = tuple(int(value) for value in model_cfg.get("encoder_channels", [64, 128, 256, 512]))
    return WTGUNet(
        in_channels=int(model_cfg.get("in_channels", 1)),
        encoder_channels=channels,
        bottleneck_channels=int(model_cfg.get("bottleneck_channels", 1024)),
        use_residual=bool(model_cfg.get("use_residual", True)),
        use_wtg=bool(model_cfg.get("use_wtg", True)),
        wtg_use_centerline=bool(model_cfg.get("wtg_use_centerline", True)),
        wtg_use_radius=bool(model_cfg.get("wtg_use_radius", True)),
    )
