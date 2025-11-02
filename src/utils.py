"""Utility helpers shared across the project."""

from enum import Enum
from typing import Dict
import random
import numpy as np
import torch

from .config import RANDOM_STATE


class TrafficSignClass(Enum):
    """Enumeration of the 43 German Traffic Sign classes."""

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


IDX_TO_CLASS: Dict[int, str] = {i: sign.value for i, sign in enumerate(TrafficSignClass)}
CLASS_TO_IDX: Dict[str, int] = {sign.value: i for i, sign in enumerate(TrafficSignClass)}


def get_class_name(idx: int) -> str:
    """Return the human readable class name for a given index."""

    return IDX_TO_CLASS.get(idx, "Unknown")


def get_class_idx(name: str) -> int:
    """Return the index for a class name; -1 when unknown."""

    return CLASS_TO_IDX.get(name, -1)


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
