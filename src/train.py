"""Training utilities."""

from typing import Dict, Optional, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    criterion: torch.nn.Module,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    epochs: int = 10,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, list]:
    history = {"loss": [], "accuracy": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            batch_acc = 100.0 * correct / total if total else 0
            progress_bar.set_postfix(batch_loss=loss.item(), acc=f"{batch_acc:.2f}%")

        if scheduler:
            scheduler.step()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total if total else 0
        history["loss"].append(epoch_loss)
        history["accuracy"].append(epoch_acc)
        print(f"Epoch [{epoch + 1}/{epochs}]\t | Loss: {epoch_loss:.4f}\t | Accuracy: {epoch_acc:.2f}%")

    return history
