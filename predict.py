"""CLI utility for running predictions with the traffic sign model."""

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from PIL import Image

from src import config
from src import model as model_module
from src.preprocessing import PreProcess
from src.utils import IDX_TO_CLASS, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict traffic sign classes")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(config.MODELS_DIR / "traffic_sign_model.pth"),
        help="Path to saved model checkpoint",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to a single image for prediction",
    )
    parser.add_argument(
        "--directory",
        type=str,
        help="Directory containing images for batch prediction",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Whether to convert images to grayscale during preprocessing",
    )
    return parser.parse_args()


def load_checkpoint(path: Path) -> torch.nn.Module:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    checkpoint = torch.load(path, map_location=config.DEVICE)
    model = model_module.build_model()

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(image_path: Path, to_grayscale: bool = False) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    preprocessor = PreProcess(size=config.IMAGE_SIZE, to_grayscale=to_grayscale)
    img_np = np.array(img)
    processed = preprocessor(img_np)
    if processed.ndim == 2:
        processed = np.expand_dims(processed, axis=-1)
    tensor = torch.tensor(processed).permute(2, 0, 1).unsqueeze(0).float()
    return tensor.to(config.DEVICE)


def predict_single(model: torch.nn.Module, image_path: Path, to_grayscale: bool) -> str:
    tensor = preprocess_image(image_path, to_grayscale)
    with torch.no_grad():
        output = model(tensor)
        prediction = int(output.argmax(dim=1).item())
    return IDX_TO_CLASS[prediction]


def iter_images(directory: Path) -> Iterable[Path]:
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"):
        yield from directory.glob(ext)


def predict_directory(model: torch.nn.Module, directory: Path, to_grayscale: bool) -> List[str]:
    predictions = []
    for image_path in iter_images(directory):
        label = predict_single(model, image_path, to_grayscale)
        predictions.append(f"{image_path.name}: {label}")
    return predictions


def main() -> None:
    args = parse_args()
    if not args.image and not args.directory:
        raise ValueError("You must provide --image or --directory")

    set_seed(config.RANDOM_STATE)
    model = load_checkpoint(Path(args.checkpoint))

    if args.image:
        label = predict_single(model, Path(args.image), args.grayscale)
        print(f"Prediction for {args.image}: {label}")

    if args.directory:
        directory = Path(args.directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        predictions = predict_directory(model, directory, args.grayscale)
        for item in predictions:
            print(item)


if __name__ == "__main__":
    main()
