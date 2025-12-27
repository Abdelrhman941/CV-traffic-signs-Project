import cv2
import numpy as np
import torch
from tqdm import tqdm


class PreProcess:
    """Image preprocessing class with all operations from notebook."""

    def __init__(self, size=(64, 64), to_grayscale=False, normalize="minmax", apply_threshold=True, threshold_method="otsu"):
        self.size = size
        self.to_grayscale = to_grayscale
        self.normalize_method = normalize
        self.apply_threshold = apply_threshold
        self.threshold_method = threshold_method

    def resize_image(self, image):
        return cv2.resize(image, self.size)

    def convert_to_grayscale(self, image):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def reduce_noise(self, image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    def enhance_contrast(self, image):
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            lab = cv2.merge((cl, a, b))
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return image

    def brighten_image(self, image, alpha=1.0, beta=50):
        gray = self.convert_to_grayscale(image)
        if np.mean(gray) < 100:
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return image

    def apply_thresholding(self, image):
        if len(image.shape) == 3:
            image = self.convert_to_grayscale(image)

        if self.threshold_method == "otsu":
            _, result = cv2.threshold(
                image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif self.threshold_method == "adaptive_mean":
            result = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5
            )
        elif self.threshold_method == "chow_kaneko":
            result = self._chow_kaneko_logic(image)
        elif self.threshold_method == "cheng_jin_kuo":
            result = self._cheng_jin_kuo_logic(image)
        else:
            result = image
        return result

    def _chow_kaneko_logic(self, image, block_size=15):
        rows, cols = image.shape
        result = np.zeros_like(image)
        for i in range(0, rows, block_size):
            for j in range(0, cols, block_size):
                block = image[i : i + block_size, j : j + block_size]
                if block.size == 0:
                    continue
                threshold = np.mean(block)
                result[i : i + block_size, j : j + block_size] = (
                    block > threshold
                ).astype(np.uint8) * 255
        return result

    def _cheng_jin_kuo_logic(self, image, block_size=15, k=0.5):
        rows, cols = image.shape
        result = np.zeros_like(image)
        for i in range(0, rows, block_size):
            for j in range(0, cols, block_size):
                block = image[i : i + block_size, j : j + block_size]
                if block.size == 0:
                    continue
                threshold = np.mean(block) - k * np.std(block)
                result[i : i + block_size, j : j + block_size] = (
                    block > threshold
                ).astype(np.uint8) * 255
        return result

    def canny_and_thickening(self, binary_img, low_threshold=50, high_threshold=150):
        edges = cv2.Canny(binary_img, low_threshold, high_threshold)
        kernel = np.ones((2, 2), np.uint8)
        image = cv2.dilate(edges, kernel, iterations=1)
        return image

    def normalize_image(self, image):
        if self.normalize_method == "minmax":
            return image.astype(np.float32) / 255.0
        return image

    def __call__(self, image):
        image = self.resize_image(image)
        image = self.brighten_image(image)
        image = self.enhance_contrast(image)
        image = self.reduce_noise(image)
        if self.apply_threshold:
            image = self.apply_thresholding(image)
        image = self.canny_and_thickening(image)
        image = self.normalize_image(image)
        return image


class ChainCodeExtractor:
    """Chain code feature extraction class from notebook."""

    chain_code_direction_8 = {
        (0, 1): 0,      # Right
        (-1, 1): 1,     # Upper-right
        (-1, 0): 2,     # Up
        (-1, -1): 3,    # Upper-left
        (0, -1): 4,     # Left
        (1, -1): 5,     # Lower-left
        (1, 0): 6,      # Down
        (1, 1): 7,      # Lower-right
    }

    @staticmethod
    def extract_chain_code(image, max_code_length=None):
        """Extract chain code from edge image."""
        binary_img = (image * 255).astype(np.uint8)

        contours, _ = cv2.findContours(
            binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        chain_codes = []

        for i in range(1, len(largest_contour)):
            prev, curr = largest_contour[i - 1][0], largest_contour[i][0]
            dx, dy = curr[0] - prev[0], curr[1] - prev[1]
            direction = ChainCodeExtractor.chain_code_direction_8.get((dx, dy))
            if direction is not None:
                chain_codes.append(direction)

        if max_code_length:
            if len(chain_codes) < max_code_length:
                chain_codes += [0] * (max_code_length - len(chain_codes))
            else:
                chain_codes = chain_codes[:max_code_length]

        return chain_codes, binary_img, largest_contour


def preprocess_for_model(image, preprocessor):
    """
    Preprocess image for model inference.
    Returns tensor ready for model input: (1, 1, H, W)
    """
    # Apply preprocessing
    processed_img = preprocessor(image)

    # Ensure single channel
    if processed_img.ndim == 3 and processed_img.shape[2] == 3:
        processed_img = cv2.cvtColor(
            (processed_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
        )
    elif processed_img.ndim == 3 and processed_img.shape[2] == 1:
        processed_img = processed_img.squeeze(axis=2)

    # Convert to tensor: (H, W) -> (1, 1, H, W)
    img_tensor = torch.from_numpy(processed_img).unsqueeze(0).unsqueeze(0).float()

    return img_tensor
