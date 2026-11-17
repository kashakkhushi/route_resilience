"""
Layer 3 - Transformer-based road segmentation.

SegFormer (MiT-B3 backbone) is the default. The reason a transformer matters
here rather than a plain CNN: self-attention gives every patch a receptive
field over the whole tile from the first block, so a road segment can be
completed from evidence 200 px away on the other side of a tree line.
A CNN would need many layers to build that context, and in practice it
learns local texture instead.

Two heads:
    road_logits      - the segmentation mask (main task)
    occlusion_logits - auxiliary head predicting "is this pixel occluded"

The auxiliary head is trained on the free labels produced by occlusion.py.
It costs almost nothing and gives two benefits: it forces the shared encoder
to represent occluders explicitly, and at inference it yields a per-pixel
confidence map that the graph healing stage uses to decide which gaps are
worth bridging.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation


# Fallback architecture table, used only when a checkpoint carries no saved
# config. Checkpoints written by scripts/kaggle_train.py always carry one.
_MIT_VARIANTS = {
    "b0": dict(hidden_sizes=[32, 64, 160, 256], depths=[2, 2, 2, 2], decoder_hidden_size=256),
    "b1": dict(hidden_sizes=[64, 128, 320, 512], depths=[2, 2, 2, 2], decoder_hidden_size=256),
    "b2": dict(hidden_sizes=[64, 128, 320, 512], depths=[3, 4, 6, 3], decoder_hidden_size=768),
    "b3": dict(hidden_sizes=[64, 128, 320, 512], depths=[3, 4, 18, 3], decoder_hidden_size=768),
    "b4": dict(hidden_sizes=[64, 128, 320, 512], depths=[3, 8, 27, 3], decoder_hidden_size=768),
    "b5": dict(hidden_sizes=[64, 128, 320, 512], depths=[3, 6, 40, 3], decoder_hidden_size=768),
}


class RoadSegFormer(nn.Module):
    def __init__(
        self,
        pretrained: str = "nvidia/mit-b3",
        in_channels: int = 3,
        aux_occlusion_head: bool = True,
        load_pretrained: bool = True,
        hf_config: dict | None = None,
    ):
        super().__init__()
        self.aux_occlusion_head = aux_occlusion_head

        if load_pretrained:
            # Training: start from the ImageNet-pretrained MiT backbone (needs network).
            self.net = SegformerForSemanticSegmentation.from_pretrained(
                pretrained, num_labels=1, ignore_mismatched_sizes=True
            )
        else:
            # Deployment / evaluation: rebuild the exact architecture with no
            # download. The checkpoint's state_dict supplies every weight, and
            # the config saved at training time guarantees the shapes match.
            if hf_config is not None:
                cfg = SegformerConfig.from_dict(dict(hf_config))
            else:
                variant = pretrained.rsplit("-", 1)[-1]
                cfg = SegformerConfig(num_labels=1, **_MIT_VARIANTS[variant])
            self.net = SegformerForSemanticSegmentation(cfg)

        # Multispectral input (e.g. RGB + NIR + NDVI): widen the patch embedding
        # and copy the pretrained RGB weights, averaging them into the new bands
        # so we keep the ImageNet initialization instead of starting from noise.
        if in_channels != 3:
            self._inflate_stem(in_channels)

        if aux_occlusion_head:
            self._fused = None
            self.net.decode_head.linear_fuse.register_forward_hook(self._capture_fused)
            hidden = self.net.config.decoder_hidden_size
            self.occ_head = nn.Sequential(
                nn.Conv2d(hidden, hidden // 2, 3, padding=1),
                nn.BatchNorm2d(hidden // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden // 2, 1, 1),
            )

    def _inflate_stem(self, in_channels: int) -> None:
        stem = self.net.segformer.encoder.patch_embeddings[0].proj
        old_w = stem.weight.data  # (out, 3, k, k)
        new = nn.Conv2d(
            in_channels,
            stem.out_channels,
            kernel_size=stem.kernel_size,
            stride=stem.stride,
            padding=stem.padding,
        )
        mean_w = old_w.mean(dim=1, keepdim=True)
        with torch.no_grad():
            new.weight[:, :3] = old_w[:, : min(3, in_channels)]
            if in_channels > 3:
                new.weight[:, 3:] = mean_w.repeat(1, in_channels - 3, 1, 1)
            new.bias.copy_(stem.bias) if stem.bias is not None else None
        self.net.segformer.encoder.patch_embeddings[0].proj = new

    def _capture_fused(self, _module, _inputs, output) -> None:
        self._fused = output

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h, w = x.shape[-2:]
        out = self.net(pixel_values=x)
        road = F.interpolate(out.logits, size=(h, w), mode="bilinear", align_corners=False)
        result = {"road_logits": road}

        if self.aux_occlusion_head:
            # The fused multi-scale decoder features are captured by a forward
            # hook on the decode head's fusion layer. Reading them this way
            # depends only on that layer existing, not on the names of the
            # head's internal projection modules, which differ between
            # transformers releases.
            occ = F.interpolate(self.occ_head(self._fused), size=(h, w),
                                mode="bilinear", align_corners=False)
            self._fused = None
            result["occlusion_logits"] = occ
        return result
