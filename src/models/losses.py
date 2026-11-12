"""
Losses.

The combination that matters for this project:

    L = w_bce * OcclusionWeightedBCE + w_dice * SoftDice + w_cldice * SoftCLDice

Why soft-clDice: Dice and IoU are pixel-counting metrics. A predicted road with
a 3-pixel break scores almost identically to an unbroken one, but the broken one
is useless for routing - it produces two disconnected graph components. clDice
compares *skeletons*, so a single break is penalized heavily. This is the loss
term that directly buys the "topology-preserving" claim in the pitch.

Why occlusion-weighted BCE: pixels the model is asked to infer through a tree
canopy are the hard positives. Upweighting them stops the network from taking
the easy route of only firing on visible asphalt.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    """Differentiable erosion = min-pool, done as -maxpool(-x) with a cross
    structuring element (separable 3x1 and 1x3)."""
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """Differentiable morphological skeleton (Shit et al., clDice, CVPR 2021).
    Iteratively peels layers and accumulates the residue that opening removes."""
    skel = F.relu(img - soft_open(img))
    for _ in range(iters):
        img = soft_erode(img)
        delta = F.relu(img - soft_open(img))
        skel = skel + F.relu(delta - skel * delta)
    return skel


class SoftCLDiceLoss(nn.Module):
    """Centerline Dice. Measures topology agreement rather than area agreement."""

    def __init__(self, iters: int = 10, smooth: float = 1.0):
        super().__init__()
        self.iters, self.smooth = iters, smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        skel_pred = soft_skeletonize(prob, self.iters)
        skel_true = soft_skeletonize(target, self.iters)

        # Topology precision: how much of the predicted centerline lies on true road.
        t_prec = ((skel_pred * target).sum((1, 2, 3)) + self.smooth) / (
            skel_pred.sum((1, 2, 3)) + self.smooth
        )
        # Topology sensitivity: how much of the true centerline is covered.
        t_sens = ((skel_true * prob).sum((1, 2, 3)) + self.smooth) / (
            skel_true.sum((1, 2, 3)) + self.smooth
        )
        cl_dice = 2.0 * t_prec * t_sens / (t_prec + t_sens)
        return 1.0 - cl_dice.mean()


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        inter = (prob * target).sum((1, 2, 3))
        denom = prob.sum((1, 2, 3)) + target.sum((1, 2, 3))
        return 1.0 - ((2 * inter + self.smooth) / (denom + self.smooth)).mean()


class OcclusionWeightedBCE(nn.Module):
    """BCE where occluded road pixels carry extra weight.

    pos_weight handles the ~1:20 road/background imbalance typical of
    DeepGlobe-style imagery; occ_weight handles the hard-positive problem.
    """

    def __init__(self, pos_weight: float = 8.0, occ_weight: float = 3.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.occ_weight = occ_weight

    def forward(self, logits, target, occlusion=None) -> torch.Tensor:
        w = torch.ones_like(target)
        w = torch.where(target > 0.5, w * self.pos_weight, w)
        if occlusion is not None:
            w = torch.where((target > 0.5) & (occlusion > 0.5), w * self.occ_weight, w)
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return (loss * w).sum() / w.sum().clamp(min=1.0)


class RoadLoss(nn.Module):
    """The full objective wired together, including the auxiliary occlusion head."""

    def __init__(self, w_bce=1.0, w_dice=1.0, w_cldice=0.5, w_occ=0.3,
                 cldice_warmup_epochs: int = 5):
        super().__init__()
        self.bce = OcclusionWeightedBCE()
        self.dice = SoftDiceLoss()
        self.cldice = SoftCLDiceLoss()
        self.w = dict(bce=w_bce, dice=w_dice, cldice=w_cldice, occ=w_occ)
        self.warmup = cldice_warmup_epochs

    def forward(self, outputs: dict, target: torch.Tensor,
                occlusion: torch.Tensor | None = None, epoch: int = 0) -> dict:
        logits = outputs["road_logits"]
        # clDice on a randomly-initialized model produces meaningless skeletons,
        # so ramp it in only once the mask is roughly correct.
        cl_scale = min(1.0, epoch / max(self.warmup, 1))

        parts = {
            "bce": self.w["bce"] * self.bce(logits, target, occlusion),
            "dice": self.w["dice"] * self.dice(logits, target),
            "cldice": self.w["cldice"] * cl_scale * self.cldice(logits, target),
        }
        if occlusion is not None and "occlusion_logits" in outputs:
            parts["occ"] = self.w["occ"] * F.binary_cross_entropy_with_logits(
                outputs["occlusion_logits"], occlusion
            )
        parts["total"] = sum(parts.values())
        return parts
