"""test_dataset.py - Script chạy kiểm tra dữ liệu và dataset.py trên máy local.
"""
import sys
from pathlib import Path

# Thêm đường dẫn thư mục code vào sys.path
code_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(code_dir))

from dataset import load_split, check_split, build_transforms, make_loader

def main():
    root_dir = code_dir.parent.parent.parent
    labels_dir = root_dir / "data" / "labels"
    images_dir = root_dir / "data" / "images"

    print("--- 1. Kiểm tra load_split & check_split ---")
    train_df, val_df, test_df = load_split(labels_dir, fold=0)
    stats = check_split(train_df, val_df, test_df, images_dir)

    print("--- 2. Kiểm tra DataLoader ---")
    train_transform = build_transforms(train=True, img_size=224, aug="basic")
    val_transform = build_transforms(train=False, img_size=224)

    train_loader = make_loader(train_df, images_dir, train_transform, batch_size=8, train=True, num_workers=0)
    val_loader = make_loader(val_df, images_dir, val_transform, batch_size=8, train=False, num_workers=0)

    images, labels, filenames = next(iter(train_loader))
    print(f"Train Batch - Shape: {images.shape}, Labels: {labels}, Filename đầu: {filenames[0]}")

    images_v, labels_v, filenames_v = next(iter(val_loader))
    print(f"Val Batch   - Shape: {images_v.shape}, Labels: {labels_v}, Filename đầu: {filenames_v[0]}")
    print("\n>>> KIỂM TRA DATASET VÀ SPLIT THÀNH CÔNG! <<<")

if __name__ == "__main__":
    main()
