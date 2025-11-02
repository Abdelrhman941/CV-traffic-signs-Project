"""Image preprocessing primitives."""

from dataclasses import dataclass
from typing import Tuple
import cv2
import numpy as np


@dataclass
class PreProcess:
    """Flexible preprocessing pipeline for traffic sign images."""

    size: Tuple[int, int] = (64, 64)
    to_grayscale: bool = False
    normalize: str = "minmax"

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, self.size)

    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, (5, 5), 0)

    @staticmethod
    def _to_uint8(image: np.ndarray) -> tuple[np.ndarray, bool, tuple[float, float]]:
        was_float = np.issubdtype(image.dtype, np.floating)
        if not was_float:
            return image, False, (0.0, 1.0)

        min_val = float(np.min(image))
        max_val = float(np.max(image))
        if np.isclose(max_val - min_val, 0.0):
            scaled = np.zeros_like(image, dtype=np.uint8)
        else:
            normalized = (image - min_val) / (max_val - min_val)
            scaled = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
        return scaled, True, (min_val, max_val)

    @staticmethod
    def _restore_dtype(image: np.ndarray, was_float: bool, min_max: tuple[float, float]) -> np.ndarray:
        if not was_float:
            return image
        min_val, max_val = min_max
        if np.isclose(max_val - min_val, 0.0):
            return np.full_like(image, fill_value=min_val, dtype=np.float32)
        normalized = image.astype(np.float32) / 255.0
        return normalized * (max_val - min_val) + min_val

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        image_uint8, was_float, min_max = self._to_uint8(image)

        if len(image.shape) == 3:
            lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            lab = cv2.merge((cl, a, b))
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            result = clahe.apply(image_uint8)

        return self._restore_dtype(result, was_float, min_max)

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        if self.normalize == "minmax":
            return image.astype(np.float32) / 255.0
        if self.normalize == "standard":
            return (image.astype(np.float32) - np.mean(image)) / (np.std(image) + 1e-8)
        return image

    def brighten_image(self, image: np.ndarray, alpha: float = 1.0, beta: int = 50) -> np.ndarray:
        image_uint8, was_float, min_max = self._to_uint8(image)
        adjusted = cv2.convertScaleAbs(image_uint8, alpha=alpha, beta=beta)
        return self._restore_dtype(adjusted, was_float, min_max)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = self.resize_image(image)
        if self.to_grayscale:
            image = self.convert_to_grayscale(image)
        image = self.reduce_noise(image)
        image = self.enhance_contrast(image)
        image = self.normalize_image(image)
        return image


class ThresholdingMethods:
    """Collection of segmentation helpers."""

    @staticmethod
    def otsu_threshold(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresholded

    @staticmethod
    def adaptive_mean(image: np.ndarray, block_size: int = 15, c: int = 5) -> np.ndarray:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c
        )

    @staticmethod
    def chow_kaneko(image: np.ndarray, block_size: int = 15) -> np.ndarray:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        rows, cols = image.shape
        result = np.zeros_like(image)
        for i in range(0, rows, block_size):
            for j in range(0, cols, block_size):
                block = image[i : i + block_size, j : j + block_size]
                threshold = np.mean(block)
                result[i : i + block_size, j : j + block_size] = (block > threshold).astype(np.uint8) * 255
        return result

    @staticmethod
    def cheng_jin_kuo(image: np.ndarray, block_size: int = 15, k: float = 0.5) -> np.ndarray:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        rows, cols = image.shape
        result = np.zeros_like(image)
        for i in range(0, rows, block_size):
            for j in range(0, cols, block_size):
                block = image[i : i + block_size, j : j + block_size]
                local_mean = np.mean(block)
                local_std = np.std(block)
                threshold = local_mean - k * local_std
                result[i : i + block_size, j : j + block_size] = (block > threshold).astype(np.uint8) * 255
        return result
