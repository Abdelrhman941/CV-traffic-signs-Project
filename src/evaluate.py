"""Evaluation utilities."""

from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from tqdm import tqdm

from .config import NUM_CLASSES
from .utils import IDX_TO_CLASS


def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    class_names: Optional[Sequence[str]] = None,
    show_confusion_matrix: bool = True,
) -> Dict[str, float]:
    model.eval()
    val_loss = 0.0
    val_preds, val_targets = [], []

    accuracy_metric = MulticlassAccuracy(num_classes=NUM_CLASSES, average="weighted").to(device)
    precision_metric = MulticlassPrecision(num_classes=NUM_CLASSES, average="weighted").to(device)
    recall_metric = MulticlassRecall(num_classes=NUM_CLASSES, average="weighted").to(device)
    f1_metric = MulticlassF1Score(num_classes=NUM_CLASSES, average="weighted").to(device)

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
        for images, targets in progress_bar:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)

            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(targets.cpu().numpy())

            accuracy_metric.update(preds, targets)
            precision_metric.update(preds, targets)
            recall_metric.update(preds, targets)
            f1_metric.update(preds, targets)

    avg_val_loss = val_loss / len(data_loader)
    val_accuracy = accuracy_metric.compute().item()
    val_precision = precision_metric.compute().item()
    val_recall = recall_metric.compute().item()
    val_f1 = f1_metric.compute().item()

    metrics = {
        "loss": avg_val_loss,
        "accuracy": val_accuracy,
        "precision": val_precision,
        "recall": val_recall,
        "f1": val_f1,
    }

    print("--------------- Validation Performance ---------------")
    for key, value in metrics.items():
        print(f"{key.title():<12}: {value:.4f}")

    if class_names is None:
        class_names = [IDX_TO_CLASS[i] for i in range(len(IDX_TO_CLASS))]

    print("\n--------------- Classification Report ---------------")
    print(classification_report(val_targets, val_preds, target_names=class_names))

    if show_confusion_matrix:
        cm = confusion_matrix(val_targets, val_preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

    accuracy_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()

    return metrics
