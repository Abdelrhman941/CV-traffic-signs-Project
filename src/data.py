"""Data loading and preparation utilities."""

from pathlib import Path
from typing import Dict, Iterable, Tuple
import glob

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm

from .config import (
    BATCH_SIZE,
    IMAGE_SIZE,
    NUM_CLASSES,
    NUM_WORKERS,
    RANDOM_STATE,
    TEST_SIZE,
    TRAIN_DIR,
    TEST_DIR,
)
from .utils import IDX_TO_CLASS


def _iter_class_directories(train_dir: Path = TRAIN_DIR) -> Iterable[Path]:
    for class_idx in range(NUM_CLASSES):
        class_dir = train_dir / str(class_idx)
        if class_dir.is_dir():
            yield class_dir


def load_training_images(
    train_dir: Path = TRAIN_DIR, image_size: Tuple[int, int] = IMAGE_SIZE
) -> Tuple[torch.Tensor, torch.Tensor]:
    images, labels = [], []
    for class_dir in tqdm(_iter_class_directories(train_dir), desc="Loading data"):
        for img_file in glob.glob(str(class_dir / "*.*")):
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, image_size)
            images.append(img)
            labels.append(int(class_dir.name))

    images_np = np.array(images, dtype=np.uint8)
    labels_np = np.array(labels, dtype=np.int64)
    images_tensor = torch.from_numpy(images_np).permute(0, 3, 1, 2).float() / 255.0
    labels_tensor = torch.from_numpy(labels_np).long()
    return images_tensor, labels_tensor


def split_train_validation(
    images: torch.Tensor,
    labels: torch.Tensor,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    images_np = images.cpu().numpy()
    labels_np = labels.cpu().numpy()

    X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
        images_np,
        labels_np,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=labels_np,
    )

    X_train = torch.from_numpy(X_train_np).float()
    X_val = torch.from_numpy(X_val_np).float()
    y_train = torch.from_numpy(y_train_np).long()
    y_val = torch.from_numpy(y_val_np).long()
    return X_train, X_val, y_train, y_val


def build_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    train_dir: Path = TRAIN_DIR,
) -> Tuple[DataLoader, DataLoader]:
    images, labels = load_training_images(train_dir)
    X_train, X_val, y_train, y_val = split_train_validation(images, labels)
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader


def load_test_dataframe(test_dir: Path = TEST_DIR) -> pd.DataFrame:
    csv_path = test_dir.parent / "Test.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Test CSV was not found at {csv_path}")
    return pd.read_csv(csv_path)


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    return pd.read_csv(metadata_path)


def get_class_distribution(train_dir: Path = TRAIN_DIR) -> Dict[str, int]:
    distribution: Dict[str, int] = {}
    for class_dir in _iter_class_directories(train_dir):
        image_count = len(list(class_dir.glob("*.*")))
        class_name = IDX_TO_CLASS[int(class_dir.name)]
        distribution[class_name] = image_count
    return distribution


def default_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(IMAGE_SIZE),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
