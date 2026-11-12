"""
Baseline models - Phase I step 2 of the brief.

The brief mandates a U-Net/DeepLabV3+ baseline *before* the Transformer, and that
sequencing is methodologically important, not bureaucratic: without a baseline
trained under identical conditions, "the Transformer works better" is an
assertion. With one, it is a measurement. The single most valuable table in the
final report is:

    model          IoU    clDice   recall_visible   recall_occluded   APLS
    U-Net R34      ...      ...          ...              ...          ...
    DeepLabV3+     ...      ...          ...              ...          ...
    SegFormer-B3   ...      ...          ...              ...          ...

The `recall_occluded` column is where the Transformer should win decisively, and
if it does not, the project's central claim is unsupported and you want to know
that early - not on the final day.

All models expose the same interface as RoadSegFormer:
    forward(x) -> {"road_logits": (B,1,H,W), "occlusion_logits": (B,1,H,W)}
so train.py, inference.py and the metrics suite work unchanged across all of them.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import segmentation_models_pytorch as smp
    _HAS_SMP = True
except ImportError:  # smp is optional; the fallback U-Net below has no deps
    _HAS_SMP = False


class _AuxOcclusionHead(nn.Module):
    """Shared auxiliary head so every architecture can predict occlusion too."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, max(in_ch // 2, 16), 3, padding=1),
            nn.BatchNorm2d(max(in_ch // 2, 16)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(in_ch // 2, 16), 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class SMPBaseline(nn.Module):
    """Wrapper giving smp architectures the project's two-head interface."""

    def __init__(self, arch: str = "unet", encoder: str = "resnet34",
                 in_channels: int = 3, aux_occlusion_head: bool = True,
                 encoder_weights: str | None = "imagenet"):
        super().__init__()
        if not _HAS_SMP:
            raise ImportError(
                "segmentation-models-pytorch required for baselines; "
                "pip install segmentation-models-pytorch"
            )
        builder = {
            "unet": smp.Unet,
            "unetplusplus": smp.UnetPlusPlus,
            "deeplabv3plus": smp.DeepLabV3Plus,
            "fpn": smp.FPN,
        }[arch.lower()]

        self.aux_occlusion_head = aux_occlusion_head
        self.net = builder(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
            # aux_params gives us a second decoder output without a second decoder
            decoder_attention_type="scse" if arch.lower() == "unet" else None,
        )
        if aux_occlusion_head:
            # Tap the highest-resolution encoder stage for the occlusion head:
            # occluders are texture, so early features suit them better than
            # the deep semantic features used for roads.
            ch = self.net.encoder.out_channels[1]
            self.occ_head = _AuxOcclusionHead(ch)

    def forward(self, x) -> dict:
        h, w = x.shape[-2:]
        feats = self.net.encoder(x)
        decoded = self.net.decoder(feats) if not isinstance(feats, torch.Tensor) \
            else self.net.decoder(feats)
        logits = self.net.segmentation_head(decoded)
        logits = F.interpolate(logits, (h, w), mode="bilinear", align_corners=False)
        out = {"road_logits": logits}
        if self.aux_occlusion_head:
            occ = self.occ_head(feats[1])
            out["occlusion_logits"] = F.interpolate(occ, (h, w), mode="bilinear",
                                                    align_corners=False)
        return out


# ---------------------------------------------------------------------------
# Dependency-free U-Net, so a baseline always exists even without smp installed.
# ---------------------------------------------------------------------------
def _block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
        nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
        nn.ReLU(inplace=True),
    )


class SimpleUNet(nn.Module):
    """Plain U-Net. Reference implementation and CI smoke-test model - small
    enough to run a forward/backward pass on CPU in a unit test."""

    def __init__(self, in_channels: int = 3, base: int = 32,
                 aux_occlusion_head: bool = True):
        super().__init__()
        self.aux_occlusion_head = aux_occlusion_head
        chs = [base, base * 2, base * 4, base * 8]
        self.enc = nn.ModuleList()
        c_in = in_channels
        for c in chs:
            self.enc.append(_block(c_in, c))
            c_in = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _block(chs[-1], chs[-1] * 2)

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        c_in = chs[-1] * 2
        for c in reversed(chs):
            self.up.append(nn.ConvTranspose2d(c_in, c, 2, 2))
            self.dec.append(_block(c * 2, c))
            c_in = c
        self.head = nn.Conv2d(chs[0], 1, 1)
        if aux_occlusion_head:
            self.occ_head = _AuxOcclusionHead(chs[0])

    def forward(self, x) -> dict:
        skips = []
        for enc in self.enc:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, skip.shape[-2:], mode="bilinear",
                                  align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))
        out = {"road_logits": self.head(x)}
        if self.aux_occlusion_head:
            out["occlusion_logits"] = self.occ_head(x)
        return out


def build_model(name: str, in_channels: int = 3, aux_occlusion_head: bool = True,
                **kw) -> nn.Module:
    """Factory used by train.py so `--model` selects any architecture.

        unet_r34 | unetpp_r34 | deeplabv3plus_r50 | segformer_b0..b5 | simple_unet
    """
    n = name.lower()
    if n == "simple_unet":
        return SimpleUNet(in_channels, aux_occlusion_head=aux_occlusion_head, **kw)
    if n.startswith("segformer"):
        from .segformer import RoadSegFormer

        variant = n.split("_")[-1] if "_" in n else "b3"
        return RoadSegFormer(f"nvidia/mit-{variant}", in_channels, aux_occlusion_head)

    arch_map = {"unet": "unet", "unetpp": "unetplusplus",
                "deeplabv3plus": "deeplabv3plus", "fpn": "fpn"}
    parts = n.rsplit("_", 1)
    arch = arch_map.get(parts[0])
    if arch is None:
        raise ValueError(f"unknown model '{name}'")
    encoder = {"r18": "resnet18", "r34": "resnet34", "r50": "resnet50",
               "effb3": "efficientnet-b3"}.get(parts[1] if len(parts) > 1 else "r34",
                                               "resnet34")
    return SMPBaseline(arch, encoder, in_channels, aux_occlusion_head, **kw)


MODEL_ZOO = [
    "simple_unet", "unet_r34", "unet_r50", "unetpp_r34",
    "deeplabv3plus_r50", "segformer_b0", "segformer_b3", "segformer_b5",
]
