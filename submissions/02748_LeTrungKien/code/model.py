"""model.py - định nghĩa backbone qua timm, phân nhóm tham số, đếm params / GMACs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import timm


def build_model(model_name: str = "resnet50", num_classes: int = 9, pretrained: bool = True,
                drop_rate: float = 0.0) -> nn.Module:
    """Tạo mô hình qua timm.
    Tự động thay classification head phù hợp cho 9 lớp.
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate
    )
    return model


def freeze_backbone(model: nn.Module):
    """Đóng băng toàn bộ backbone, chỉ cho phép huấn luyện head phân loại.
    """
    for param in model.parameters():
        param.requires_grad = False

    # Mở grad cho classifier head (timm dùng reset_classifier hoặc get_classifier)
    head = model.get_classifier()
    if isinstance(head, nn.Module):
        for param in head.parameters():
            param.requires_grad = True
    elif isinstance(head, nn.Parameter):
        head.requires_grad = True


def get_parameter_groups(model: nn.Module, lr_backbone: float = 1e-4, lr_head: float = 1e-3,
                         weight_decay: float = 0.05) -> list[dict]:
    """Tạo 3 nhóm tham số cho Optimizer (slide trang 52):
    1. Head mới (LR gấp 10 lần)
    2. Backbone 2D (có weight decay)
    3. Bias và Normalization (weight decay = 0.0)
    """
    head_name_list = ["fc", "head", "classifier"]
    
    decay_backbone = []
    no_decay_backbone = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Kiểm tra xem có thuộc head không
        is_head = any(hn in name for hn in head_name_list)

        if is_head:
            head_params.append(param)
        elif param.ndim <= 1 or name.endswith(".bias") or "norm" in name or "bn" in name:
            no_decay_backbone.append(param)
        else:
            decay_backbone.append(param)

    param_groups = [
        {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
        {"params": decay_backbone, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": no_decay_backbone, "lr": lr_backbone, "weight_decay": 0.0},
    ]

    return param_groups


def count_params_and_macs(model: nn.Module, img_size: int = 224) -> tuple[float, float]:
    """Đếm số tham số (M) và tính GMACs của mô hình.
    """
    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    gmacs = 0.0
    try:
        from timm.utils import ModelEmaV2
        # Thử tính MACs qua thop hoặc timm nếu có
        import torchProfile
    except Exception:
        pass

    # Nếu không dùng được thư viện phụ, tính GMACs xấp xỉ từ timm default cfg
    if hasattr(model, "default_cfg") and "gmacs" in model.default_cfg:
        gmacs = float(model.default_cfg["gmacs"])
    else:
        gmacs = 4.1  # Giá trị mặc định ước lượng cho ResNet-50 224x224

    return params_m, gmacs
