"""Backend configuration settings."""

import os
from pathlib import Path
from enum import Enum


# Base Paths
BACKEND_DIR = Path(__file__).parent
PROJECT_DIR = BACKEND_DIR.parent
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"
UPLOAD_DIR = BACKEND_DIR / "uploads"
MODEL_PATH = NOTEBOOKS_DIR / "best_model.pth"

# Server settings
HOST = "0.0.0.0"
PORT = 8000
RELOAD = True
DEBUG = True

# CORS Settings
ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:3000",
    "*",  # Allow all for development
]

# API settings
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

# Model Configuration
NUM_CLASSES = 43
IMAGE_SIZE = (64, 64)
DEVICE = "cuda"  # Will be overridden at runtime if CUDA not available

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# Traffic Sign Classes (German Traffic Sign Recognition Benchmark)
class TrafficSignClass(Enum):
    SPEED_LIMIT_20 = "Speed limit (20km/h)"
    SPEED_LIMIT_30 = "Speed limit (30km/h)"
    SPEED_LIMIT_50 = "Speed limit (50km/h)"
    SPEED_LIMIT_60 = "Speed limit (60km/h)"
    SPEED_LIMIT_70 = "Speed limit (70km/h)"
    SPEED_LIMIT_80 = "Speed limit (80km/h)"
    END_SPEED_LIMIT_80 = "End of speed limit (80km/h)"
    SPEED_LIMIT_100 = "Speed limit (100km/h)"
    SPEED_LIMIT_120 = "Speed limit (120km/h)"
    NO_PASSING = "No passing"
    NO_PASSING_VEH_OVER_3_5_TONS = "No passing veh over 3.5 tons"
    RIGHT_OF_WAY_AT_INTERSECTION = "Right-of-way at intersection"
    PRIORITY_ROAD = "Priority road"
    YIELD = "Yield"
    STOP = "Stop"
    NO_VEHICLES = "No vehicles"
    VEH_OVER_3_5_TONS_PROHIBITED = "Veh > 3.5 tons prohibited"
    NO_ENTRY = "No entry"
    GENERAL_CAUTION = "General caution"
    DANGEROUS_CURVE_LEFT = "Dangerous curve left"
    DANGEROUS_CURVE_RIGHT = "Dangerous curve right"
    DOUBLE_CURVE = "Double curve"
    BUMPY_ROAD = "Bumpy road"
    SLIPPERY_ROAD = "Slippery road"
    ROAD_NARROWS_ON_THE_RIGHT = "Road narrows on the right"
    ROAD_WORK = "Road work"
    TRAFFIC_SIGNALS = "Traffic signals"
    PEDESTRIANS = "Pedestrians"
    CHILDREN_CROSSING = "Children crossing"
    BICYCLES_CROSSING = "Bicycles crossing"
    BEWARE_OF_ICE_SNOW = "Beware of ice/snow"
    WILD_ANIMALS_CROSSING = "Wild animals crossing"
    END_SPEED_PASSING_LIMITS = "End speed + passing limits"
    TURN_RIGHT_AHEAD = "Turn right ahead"
    TURN_LEFT_AHEAD = "Turn left ahead"
    AHEAD_ONLY = "Ahead only"
    GO_STRAIGHT_OR_RIGHT = "Go straight or right"
    GO_STRAIGHT_OR_LEFT = "Go straight or left"
    KEEP_RIGHT = "Keep right"
    KEEP_LEFT = "Keep left"
    ROUNDABOUT_MANDATORY = "Roundabout mandatory"
    END_NO_PASSING = "End of no passing"
    END_NO_PASSING_VEH_OVER_3_5_TONS = "End no passing veh > 3.5 tons"


# Class Mappings
IDX_TO_CLASS = {i: sign.value for i, sign in enumerate(TrafficSignClass)}
CLASS_TO_IDX = {sign.value: i for i, sign in enumerate(TrafficSignClass)}


def get_class_name(idx: int) -> str:
    """Get class name from index."""
    return IDX_TO_CLASS.get(idx, "Unknown")


def get_class_idx(name: str) -> int:
    """Get class index from name."""
    return CLASS_TO_IDX.get(name, -1)
