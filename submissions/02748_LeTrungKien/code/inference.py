"""inference.py - TTA, Temperature Scaling (calibration ECE), Ensemble.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

code_dir = Path(__file__).resolve().parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from dataset import build_transforms, DeepWeedsDataset


def apply_tta_flip(model: nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
    """TTA Horizontal Flip (K=2 views).
    """
    model.eval()
    with torch.no_grad():
        out_orig = model(image_tensor)
        out_flip = model(torch.flip(image_tensor, dims=[-1]))
        prob_orig = torch.softmax(out_orig, dim=-1)
        prob_flip = torch.softmax(out_flip, dim=-1)
        return 0.5 * (prob_orig + prob_flip)


class TemperatureScaler(nn.Module):
    """Temperature Scaling để calibrate logits (slide trang 69).
    Khớp nhiệt độ T duy nhất trên tập Val.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, max_iter: int = 50):
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def eval_fn():
            optimizer.zero_grad()
            loss = criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_fn)
        print(f"Optimal Temperature T = {self.temperature.item():.4f}")


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Tính Expected Calibration Error (ECE) với 15 bins theo đúng định nghĩa.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return float(ece)
