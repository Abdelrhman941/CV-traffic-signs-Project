"""Project-wide configuration values and paths."""

from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
TRAIN_DIR = DATA_DIR / "Train"
TEST_DIR = DATA_DIR / "Test"
META_DIR = DATA_DIR / "Meta"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
OUTPUTS_DIR = BASE_DIR / "outputs"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

NUM_CLASSES = 43
IMAGE_SIZE = (30, 30)
RANDOM_STATE = 42
TEST_SIZE = 0.2
BATCH_SIZE = 32
NUM_WORKERS = 2
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
LR_STEP_SIZE = 10
LR_GAMMA = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
