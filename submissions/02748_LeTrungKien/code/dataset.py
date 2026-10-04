"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1.
Giao diện giữ nguyên để ghép nối với train.py, inference.py và eval.py.
"""
from __future__ import annotations

import random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

NUM_CLASSES = 9
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).
    Trả về ba DataFrame với các cột Filename, Label.
    """
    labels_path = Path(labels_dir)
    train_csv = labels_path / f"train_subset{fold}.csv"
    val_csv = labels_path / f"val_subset{fold}.csv"
    test_csv = labels_path / f"test_subset{fold}.csv"

    assert train_csv.exists(), f"Không tìm thấy file: {train_csv}"
    assert val_csv.exists(), f"Không tìm thấy file: {val_csv}"
    assert test_csv.exists(), f"Không tìm thấy file: {test_csv}"

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.
    """
    images_path = Path(images_dir)
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    n_total = n_train + n_val + n_test

    # 1. Kiểm tra hợp 3 tập bằng đúng 17.509 ảnh
    assert n_total == 17509, f"Tổng số ảnh phải bằng 17509, thực tế: {n_total}"

    # 2. Kiểm tra giao giữa các tập phải rỗng
    train_files = set(train_df["Filename"])
    val_files = set(val_df["Filename"])
    test_files = set(test_df["Filename"])

    train_val_overlap = train_files.intersection(val_files)
    train_test_overlap = train_files.intersection(test_files)
    val_test_overlap = val_files.intersection(test_files)

    assert len(train_val_overlap) == 0, f"Giao giữa Train và Val không rỗng! ({len(train_val_overlap)} ảnh)"
    assert len(train_test_overlap) == 0, f"Giao giữa Train và Test không rỗng! ({len(train_test_overlap)} ảnh)"
    assert len(val_test_overlap) == 0, f"Giao giữa Val và Test không rỗng! ({len(val_test_overlap)} ảnh)"

    # 3. Kiểm tra sự tồn tại của các file ảnh trên đĩa
    all_files = list(train_files | val_files | test_files)
    missing_files = [f for f in all_files if not (images_path / f).exists()]
    assert len(missing_files) == 0, f"Có {len(missing_files)} file ảnh không tồn tại trong {images_dir}"

    # 4. Thống kê theo lớp
    train_counts = train_df["Label"].value_counts().sort_index().to_dict()
    val_counts = val_df["Label"].value_counts().sort_index().to_dict()
    test_counts = test_df["Label"].value_counts().sort_index().to_dict()

    stats = {
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_total": n_total,
        "per_class": {
            "train": train_counts,
            "val": val_counts,
            "test": test_counts,
        },
        "overlap": {
            "train_val": len(train_val_overlap),
            "train_test": len(train_test_overlap),
            "val_test": len(val_test_overlap),
        },
    }

    print("=== DẠNG CHIA DỮ LIỆU ĐÃ KIỂM TRA THÀNH CÔNG (S1-S4) ===")
    print(f"Train: {n_train} ({n_train/n_total:.1%}), Val: {n_val} ({n_val/n_total:.1%}), Test: {n_test} ({n_test/n_total:.1%})")
    print(f"Giao giữa các tập: train∩val={len(train_val_overlap)}, train∩test={len(train_test_overlap)}, val∩test={len(val_test_overlap)}")
    print(f"Kiểm tra file đĩa: Tất cả {len(all_files)} file tồn tại đầy đủ.\n")

    return stats


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo torchvision transform dựa theo train và tham số aug.
    aug options: 'basic', 'color', 'trivial', 'randaug'
    """
    if train:
        if aug == "basic":
            return T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        elif aug == "color":
            return T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        elif aug == "trivial":
            return T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.TrivialAugmentWide(),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        elif aug == "randaug":
            return T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandAugment(num_ops=2, magnitude=9),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            raise ValueError(f"Không hỗ trợ mức aug '{aug}'")
    else:
        # Evaluation mode: CenterCrop & Normalization
        if img_size == 256:
            return T.Compose([
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            return T.Compose([
                T.Resize(256),
                T.CenterCrop(img_size),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ images_dir theo DataFrame (Filename, Label).
    Trả về (image_tensor, int(label), filename)
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        filename = row["Filename"]
        label = int(row["Label"])

        img_path = self.images_dir / filename
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, filename


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    """Tạo DataLoader cho DeepWeeds.
    """
    dataset = DeepWeedsDataset(df, images_dir, transform=transform)

    if train:
        if sampler == "balanced":
            class_counts = df["Label"].value_counts().sort_index().to_numpy()
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[df["Label"].values]
            sampler_obj = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler_obj,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                worker_init_fn=seed_worker,
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                worker_init_fn=seed_worker,
            )
    else:
        # Val / Test mode: Không shuffle, không drop_last
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

    return loader
