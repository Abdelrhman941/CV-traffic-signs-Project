"""Entry point for training and evaluating the traffic sign recognition model."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src import config
from src import data as data_module
from src import evaluate as eval_module
from src import model as model_module
from src import train as train_module
from src import utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Traffic sign recognition pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    train_parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    train_parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    train_parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    train_parser.add_argument("--model-name", type=str, default="traffic_sign_model.pth")
    train_parser.add_argument("--skip-confusion", action="store_true", help="Skip confusion matrix plot")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate an existing checkpoint")
    eval_parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    eval_parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    eval_parser.add_argument("--show-confusion", action="store_true", help="Display confusion matrix")

    return parser.parse_args()


def run_training(args: argparse.Namespace) -> None:
    utils.set_seed(config.RANDOM_STATE)

    train_loader, val_loader = data_module.build_dataloaders(
        batch_size=args.batch_size, num_workers=config.NUM_WORKERS
    )

    model = model_module.build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA
    )

    history = train_module.train_model(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        criterion=criterion,
        scheduler=scheduler,
        epochs=args.epochs,
        device=config.DEVICE,
    )

    metrics = eval_module.evaluate_model(
        model=model,
        data_loader=val_loader,
        criterion=criterion,
        device=config.DEVICE,
        show_confusion_matrix=not args.skip_confusion,
    )

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.MODELS_DIR / args.model_name
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "metrics": metrics,
    }, checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")

    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    history_path = config.LOGS_DIR / "training_history.pt"
    torch.save(history, history_path)
    print(f"Training history saved to {history_path}")


def run_evaluation(args: argparse.Namespace) -> None:
    utils.set_seed(config.RANDOM_STATE)

    _, val_loader = data_module.build_dataloaders(
        batch_size=args.batch_size, num_workers=config.NUM_WORKERS
    )

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint}")

    state = torch.load(checkpoint, map_location=config.DEVICE)
    model = model_module.build_model()
    model.load_state_dict(state["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    eval_module.evaluate_model(
        model=model,
        data_loader=val_loader,
        criterion=criterion,
        device=config.DEVICE,
        show_confusion_matrix=args.show_confusion,
    )


def main() -> None:
    args = parse_args()
    if args.command == "train":
        run_training(args)
    elif args.command == "evaluate":
        run_evaluation(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
