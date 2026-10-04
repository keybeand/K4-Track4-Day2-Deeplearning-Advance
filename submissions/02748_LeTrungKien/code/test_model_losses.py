"""test_model_losses.py - Script unit test kiểm tra model.py và losses.py.
"""
import sys
import math
from pathlib import Path
import torch

code_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(code_dir))

from model import build_model, get_parameter_groups, count_params_and_macs
from losses import build_loss, FocalLoss, LabelSmoothingCrossEntropy

def main():
    print("--- 1. Kiểm tra Model Building & Parameter Groups ---")
    model = build_model("resnet50", num_classes=9, pretrained=False)
    params_m, gmacs = count_params_and_macs(model)
    print(f"ResNet-50 Params: {params_m:.2f} M | GMACs: {gmacs:.2f}")

    param_groups = get_parameter_groups(model, lr_backbone=1e-4, lr_head=1e-3, weight_decay=0.05)
    print(f"Số nhóm tham số: {len(param_groups)}")
    print(f"  Head group params: {len(param_groups[0]['params'])}, LR: {param_groups[0]['lr']}")
    print(f"  Decay backbone params: {len(param_groups[1]['params'])}, LR: {param_groups[1]['lr']}")
    print(f"  No decay backbone params: {len(param_groups[2]['params'])}, LR: {param_groups[2]['lr']}")

    print("\n--- 2. Kiểm tra Initial Loss ---")
    inputs = torch.randn(4, 9)
    targets = torch.tensor([0, 1, 8, 3])
    ce_loss_fn = build_loss("ce")
    init_loss = ce_loss_fn(inputs, targets).item()
    expected_initial_loss = -math.log(1 / 9)  # ~ 2.197
    print(f"Initial CE Loss trên random inputs: {init_loss:.4f} (Giá trị lý thuyết kỳ vọng ~ {expected_initial_loss:.4f})")

    print("\n--- 3. Kiểm tra Focal Loss gamma=0 == CE ---")
    focal_g0_fn = FocalLoss(gamma=0.0)
    ce_val = ce_loss_fn(inputs, targets).item()
    focal_g0_val = focal_g0_fn(inputs, targets).item()
    print(f"CE Loss: {ce_val:.6f} | Focal Loss (gamma=0): {focal_g0_val:.6f}")
    assert abs(ce_val - focal_g0_val) < 1e-5, "Focal Loss gamma=0 phải bằng đúng Cross Entropy!"
    print("Assertion PASSED: Focal Loss (gamma=0) trùng khớp với CE!")

    print("\n>>> KIỂM TRA MODEL VÀ LOSSES THÀNH CÔNG! <<<")

if __name__ == "__main__":
    main()
