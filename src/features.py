"""Feature extraction utilities for traffic sign images."""

from typing import List, Optional, Tuple
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class ChainCodeExtractor:
    """Generate directional chain codes from image contours."""

    chain_code_direction_8 = {
        (0, 1): 0,
        (-1, 1): 1,
        (-1, 0): 2,
        (-1, -1): 3,
        (0, -1): 4,
        (1, -1): 5,
        (1, 0): 6,
        (1, 1): 7,
    }

    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    @staticmethod
    def extract_chain_code(
        image: np.ndarray, max_code_length: Optional[int] = None
    ) -> Optional[Tuple[List[int], np.ndarray, np.ndarray]]:
        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        sharpened = cv2.filter2D(gray_img, -1, ChainCodeExtractor.sharpening_kernel)
        _, binary_img = cv2.threshold(sharpened, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        chain_codes: List[int] = []

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

    @staticmethod
    def process_images(
        images: np.ndarray, num_samples_to_show: int = 10
    ) -> np.ndarray:
        features: List[List[int]] = []
        max_code_length = 0

        for img in tqdm(images, desc="Finding max chain code length"):
            result = ChainCodeExtractor.extract_chain_code(img)
            if result:
                chain_codes, _, _ = result
                max_code_length = max(max_code_length, len(chain_codes))

        for idx, img in enumerate(tqdm(images, desc="Extracting chain codes")):
            result = ChainCodeExtractor.extract_chain_code(img, max_code_length)
            if result:
                chain_codes, binary_img, contour = result
                features.append(chain_codes)

                if idx < num_samples_to_show:
                    plt.figure(figsize=(12, 4))
                    plt.subplot(1, 2, 1)
                    plt.imshow(img, cmap="gray")
                    plt.title(f"Original Image {idx + 1}")
                    plt.axis("off")
                    plt.subplot(1, 2, 2)
                    plt.imshow(binary_img, cmap="gray")
                    plt.plot(contour[:, 0, 0], contour[:, 0, 1], "r-", lw=2)
                    plt.title(f"Contour and Binary Image {idx + 1}")
                    plt.axis("off")
                    plt.show()
        return np.array(features)
