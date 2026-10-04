"""losses.py - định nghĩa các hàm loss: CE, Label Smoothing, Focal Loss, Class Weights, Mixup/CutMix.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross Entropy với Label Smoothing (ε).
    """

    def __init__(self, smoothing: float = 0.1, weight: torch.Tensor | None = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        n_classes = x.size(-1)

        if target.ndim == 1:
            # Hard labels -> convert sang smooth targets
            target_smooth = torch.full_like(logprobs, self.smoothing / (n_classes - 1))
            target_smooth.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        else:
            # Soft targets (Mixup / CutMix)
            target_smooth = target * (1.0 - self.smoothing) + self.smoothing / n_classes

        if self.weight is not None:
            logprobs = logprobs * self.weight.unsqueeze(0)

        loss = (-target_smooth * logprobs).sum(dim=-1).mean()
        return loss


class FocalLoss(nn.Module):
    """Focal Loss cho bài toán imbalanced classification.
    Khi gamma = 0, Focal Loss bằng đúng Cross Entropy.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 1:
            ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction="none")
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        else:
            # Cho soft labels
            p = F.softmax(input, dim=-1)
            pt = (p * target).sum(dim=-1)
            ce_loss = -(target * F.log_softmax(input, dim=-1)).sum(dim=-1)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def apply_mixup_cutmix(images: torch.Tensor, targets: torch.Tensor, alpha_mixup: float = 0.8,
                        alpha_cutmix: float = 1.0, prob: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """Áp dụng Mixup hoặc CutMix ngẫu nhiên lên batch ảnh và nhãn.
    """
    if torch.rand(1).item() > prob:
        if targets.ndim == 1:
            targets_onehot = F.one_hot(targets, num_classes=images.size(1)).float()
            return images, targets_onehot
        return images, targets

    batch_size = images.size(0)
    num_classes = 9
    if targets.ndim == 1:
        targets_onehot = F.one_hot(targets, num_classes=num_classes).float()
    else:
        targets_onehot = targets

    indices = torch.randperm(batch_size)
    shuffled_images = images[indices]
    shuffled_targets = targets_onehot[indices]

    use_cutmix = torch.rand(1).item() > 0.5

    if use_cutmix and alpha_cutmix > 0:
        lam = torch.distributions.Beta(alpha_cutmix, alpha_cutmix).sample().item()
        W, H = images.size(3), images.size(2)
        cut_rat = torch.sqrt(torch.tensor(1.0 - lam)).item()
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = torch.randint(0, W, (1,)).item()
        cy = torch.randint(0, H, (1,)).item()

        bbx1 = torch.clamp(cx - cut_w // 2, 0, W)
        bby1 = torch.clamp(cy - cut_h // 2, 0, H)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, W)
        bby2 = torch.clamp(cy + cut_h // 2, 0, H)

        images[:, :, bby1:bby2, bbx1:bbx2] = shuffled_images[:, :, bby1:bby2, bbx1:bbx2]
        lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        new_targets = lam * targets_onehot + (1.0 - lam) * shuffled_targets
    else:
        lam = torch.distributions.Beta(alpha_mixup, alpha_mixup).sample().item()
        images = lam * images + (1.0 - lam) * shuffled_images
        new_targets = lam * targets_onehot + (1.0 - lam) * shuffled_targets

    return images, new_targets


def build_loss(loss_name: str = "ce", gamma: float = 2.0, smoothing: float = 0.1,
               class_weights: torch.Tensor | None = None) -> nn.Module:
    """Factory tạo loss module tương ứng với cấu hình thí nghiệm.
    """
    if loss_name == "ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_name == "label_smoothing":
        return LabelSmoothingCrossEntropy(smoothing=smoothing, weight=class_weights)
    elif loss_name == "focal":
        return FocalLoss(gamma=gamma, weight=class_weights)
    else:
        raise ValueError(f"Không hỗ trợ loss_name '{loss_name}'")
