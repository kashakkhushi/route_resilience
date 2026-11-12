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


class RoadSegFormer(nn.Module):
    def __init__(
        self,
        pretrained: str = "nvidia/mit-b3",
        in_channels: int = 3,
        aux_occlusion_head: bool = True,
    ):
        super().__init__()
        self.aux_occlusion_head = aux_occlusion_head

        self.net = SegformerForSemanticSegmentation.from_pretrained(
            pretrained, num_labels=1, ignore_mismatched_sizes=True
        )

        # Multispectral input (e.g. RGB + NIR + NDVI): widen the patch embedding
        # and copy the pretrained RGB weights, averaging them into the new bands
        # so we keep the ImageNet initialization instead of starting from noise.
        if in_channels != 3:
            self._inflate_stem(in_channels)

        if aux_occlusion_head:
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

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h, w = x.shape[-2:]
        out = self.net(pixel_values=x, output_hidden_states=self.aux_occlusion_head)
        road = F.interpolate(out.logits, size=(h, w), mode="bilinear", align_corners=False)
        result = {"road_logits": road}

        if self.aux_occlusion_head:
            # Reuse the fused decoder features that feed the main classifier.
            feats = self.net.decode_head.linear_fuse.weight  # touch to keep graph explicit
            fused = self._fused_features(out.hidden_states)
            occ = F.interpolate(self.occ_head(fused), size=(h, w), mode="bilinear",
                                align_corners=False)
            result["occlusion_logits"] = occ
        return result

    def _fused_features(self, hidden_states) -> torch.Tensor:
        head = self.net.decode_head
        target = hidden_states[0].shape[2:]
        feats = []
        for state, mlp in zip(hidden_states, head.linear_c):
            b, c, hh, ww = state.shape
            f = mlp(state.flatten(2).transpose(1, 2)).permute(0, 2, 1).reshape(b, -1, hh, ww)
            feats.append(F.interpolate(f, size=target, mode="bilinear", align_corners=False))
        return head.linear_fuse(torch.cat(feats[::-1], dim=1))
