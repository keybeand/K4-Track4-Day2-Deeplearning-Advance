"""train.py - Hàm huấn luyện một thí nghiệm (train loop, AMP, Cosine+Warmup, EMA, checkpoint, vẽ biểu đồ).
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score

code_dir = Path(__file__).resolve().parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from dataset import load_split, check_split, build_transforms, make_loader
from model import build_model, get_parameter_groups, freeze_backbone
from losses import build_loss, apply_mixup_cutmix

# Thêm eval.py vào path để lưu dự đoán chuẩn
repo_root = code_dir.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
try:
    from eval import save_predictions
except ImportError:
    save_predictions = None


@dataclass
class Config:
    exp_id: str = "T00_baseline"
    description: str = "Baseline ResNet-50 recipe T00"
    model_name: str = "resnet50"
    pretrained: bool = True
    freeze_backbone: bool = False
    
    # Preprocessing & Data
    img_size: int = 224
    aug: str = "basic"
    use_mixup_cutmix: bool = False
    sampler: str | None = None
    
    # Hyperparameters
    epochs: int = 10
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    seed: int = 42
    
    # Loss
    loss_name: str = "ce"
    gamma: float = 2.0
    smoothing: float = 0.1
    use_class_weights: bool = False
    
    # Directories
    labels_dir: str = ""
    images_dir: str = ""
    output_dir: str = ""
    curves_dir: str = ""
    predictions_dir: str = ""
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, config):
    model.train()
    total_loss = 0.0
    preds_all = []
    targets_all = []

    for images, targets, _ in loader:
        images = images.to(config.device)
        targets = targets.to(config.device)

        if config.use_mixup_cutmix:
            images, targets_soft = apply_mixup_cutmix(images, targets)
        else:
            targets_soft = targets

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=config.device if "cuda" in config.device else "cpu", enabled=("cuda" in config.device)):
            outputs = model(images)
            loss = criterion(outputs, targets_soft)

        if scaler is not None and "cuda" in config.device:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=-1).detach().cpu().numpy()
        preds_all.extend(preds)
        if targets.ndim > 1:
            targets_all.extend(targets.argmax(dim=-1).detach().cpu().numpy())
        else:
            targets_all.extend(targets.detach().cpu().numpy())

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(targets_all, preds_all, average="macro", zero_division=0)
    top1_acc = accuracy_score(targets_all, preds_all)

    return avg_loss, macro_f1, top1_acc


@torch.no_grad()
def evaluate(model, loader, criterion, config):
    model.eval()
    total_loss = 0.0
    preds_all = []
    targets_all = []
    probs_all = []
    filenames_all = []

    for images, targets, filenames in loader:
        images = images.to(config.device)
        targets = targets.to(config.device)

        with torch.amp.autocast(device_type=config.device if "cuda" in config.device else "cpu", enabled=("cuda" in config.device)):
            outputs = model(images)
            loss = criterion(outputs, targets)

        probs = torch.softmax(outputs, dim=-1).cpu().numpy()
        preds = outputs.argmax(dim=-1).cpu().numpy()

        total_loss += loss.item() * images.size(0)
        preds_all.extend(preds)
        targets_all.extend(targets.cpu().numpy())
        probs_all.append(probs)
        filenames_all.extend(filenames)

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(targets_all, preds_all, average="macro", zero_division=0)
    top1_acc = accuracy_score(targets_all, preds_all)
    probs_all = np.concatenate(probs_all, axis=0)

    return avg_loss, macro_f1, top1_acc, probs_all, np.array(targets_all), filenames_all


def run(config: Config):
    set_seed(config.seed)
    
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.curves_dir, exist_ok=True)
    os.makedirs(config.predictions_dir, exist_ok=True)

    # 1. Load data
    train_df, val_df, test_df = load_split(config.labels_dir, fold=0)
    
    class_weights = None
    if config.use_class_weights:
        counts = train_df["Label"].value_counts().sort_index().to_numpy()
        weights = 1.0 / counts
        weights = weights / weights.sum() * 9.0
        class_weights = torch.tensor(weights, dtype=torch.float32).to(config.device)

    train_transform = build_transforms(train=True, img_size=config.img_size, aug=config.aug)
    val_transform = build_transforms(train=False, img_size=config.img_size)

    train_loader = make_loader(train_df, config.images_dir, train_transform, config.batch_size, train=True, sampler=config.sampler)
    val_loader = make_loader(val_df, config.images_dir, val_transform, config.batch_size, train=False)

    # 2. Build model & Loss
    model = build_model(config.model_name, num_classes=9, pretrained=config.pretrained).to(config.device)
    if config.freeze_backbone:
        freeze_backbone(model)

    criterion = build_loss(config.loss_name, gamma=config.gamma, smoothing=config.smoothing, class_weights=class_weights)
    param_groups = get_parameter_groups(model, lr_backbone=config.lr_backbone, lr_head=config.lr_head, weight_decay=config.weight_decay)

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler('cuda') if ("cuda" in config.device) else None

    # 3. Training Loop
    history = {"train_loss": [], "train_f1": [], "val_loss": [], "val_f1": [], "val_acc": []}
    best_val_f1 = -1.0
    best_model_path = os.path.join(config.output_dir, f"{config.exp_id}_best.pth")

    start_time = time.time()
    for epoch in range(1, config.epochs + 1):
        t_loss, t_f1, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, config)
        v_loss, v_f1, v_acc, _, _, _ = evaluate(model, val_loader, criterion, config)

        history["train_loss"].append(t_loss)
        history["train_f1"].append(t_f1)
        history["val_loss"].append(v_loss)
        history["val_f1"].append(v_f1)
        history["val_acc"].append(v_acc)

        print(f"Epoch {epoch:02d}/{config.epochs:02d} - Train Loss: {t_loss:.4f} F1: {t_f1:.4f} | Val Loss: {v_loss:.4f} F1: {v_f1:.4f} Acc: {v_acc:.4f}")

        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            torch.save(model.state_dict(), best_model_path)

    total_time = time.time() - start_time
    print(f"[{config.exp_id}] Hoàn thành huấn luyện trong {total_time/60:.2f} phút. Best Val Macro-F1: {best_val_f1:.4f}")

    # 4. Plot Learning Curves
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:red")
    ax1.plot(range(1, config.epochs + 1), history["train_loss"], label="Train Loss", color="tab:red", linestyle="--")
    ax1.plot(range(1, config.epochs + 1), history["val_loss"], label="Val Loss", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Macro-F1 / Acc", color="tab:blue")
    ax2.plot(range(1, config.epochs + 1), history["val_f1"], label="Val Macro-F1", color="tab:blue")
    ax2.plot(range(1, config.epochs + 1), history["val_acc"], label="Val Acc", color="tab:cyan", linestyle=":")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    plt.title(f"Curve: {config.exp_id} ({config.model_name})")
    fig.tight_layout()
    curve_path = os.path.join(config.curves_dir, f"{config.exp_id}.png")
    plt.savefig(curve_path)
    plt.close()

    return history, best_val_f1
