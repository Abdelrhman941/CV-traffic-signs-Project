"""FastAPI routes for traffic sign recognition."""

import base64
import io
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image

from Backend.models.schemas import (
    PreprocessRequest,
    SegmentRequest,
    ClassifyRequest,
    ClassifyResponse,
    ImageResponse,
    UploadResponse,
    PredictionResult,
)
from Backend.utils.helpers import PreProcess, preprocess_for_model
from Backend.config import IDX_TO_CLASS, IMAGE_SIZE
import torch

router = APIRouter(prefix="/api", tags=["api"])

# Global model variable (will be set from main.py)
model = None
device = None


def set_model(loaded_model, model_device):
    """Set the global model and device."""
    global model, device
    model = loaded_model
    device = model_device


def decode_base64_image(base64_str: str) -> np.ndarray:
    """Decode base64 string to numpy array (OpenCV format)."""
    try:
        # Remove data URL prefix if present
        if "base64," in base64_str:
            base64_str = base64_str.split("base64,")[1]

        # Decode base64
        img_data = base64.b64decode(base64_str)

        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(img_data))

        # Convert to numpy array (RGB)
        img_array = np.array(pil_image)

        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")


def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode numpy array to base64 string."""
    try:
        # Normalize image to 0-255 range
        if image.dtype == np.float32 or image.dtype == np.float64:
            # Float image (0-1 range), convert to 0-255
            image = (image * 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            # Other types, ensure uint8
            image = image.astype(np.uint8)

        # Handle grayscale vs color
        if len(image.shape) == 2:
            # Grayscale - convert to PIL
            pil_image = Image.fromarray(image, mode="L")
        elif len(image.shape) == 3:
            if image.shape[2] == 1:
                # Single channel (H, W, 1) -> squeeze to (H, W)
                pil_image = Image.fromarray(image.squeeze(), mode="L")
            elif image.shape[2] == 3:
                # Color - convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb, mode="RGB")
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
        else:
            raise ValueError(f"Invalid image dimensions: {image.shape}")

        # Convert to bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()

        # Encode to base64
        base64_str = base64.b64encode(img_bytes).decode()
        return f"data:image/png;base64,{base64_str}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error encoding image: {str(e)}")


@router.get("/health")
async def health_check():
    """Check API health status."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown",
    }


@router.post("/upload", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """Upload and validate an image file."""
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Encode to base64
        base64_image = encode_image_to_base64(image)

        return UploadResponse(
            success=True,
            original_image=base64_image
        )
    except Exception as e:
        return UploadResponse(
            success=False,
            error=str(e)
        )


@router.post("/preprocess", response_model=ImageResponse)
async def preprocess_image(request: PreprocessRequest):
    """Apply preprocessing to an image."""
    try:
        # Decode image
        image = decode_base64_image(request.image)

        # Create preprocessor with parameters
        preprocessor = PreProcess(
            size=(request.params.resize_width, request.params.resize_height),
            to_grayscale=request.params.to_grayscale,
            normalize="minmax",
            apply_threshold=True,
            threshold_method="otsu"
        )

        # Apply preprocessing
        processed = preprocessor(image)

        # Encode result
        processed_base64 = encode_image_to_base64(processed)

        return ImageResponse(
            success=True,
            processed_image=processed_base64
        )
    except Exception as e:
        return ImageResponse(
            success=False,
            error=str(e)
        )


@router.post("/segment", response_model=ImageResponse)
async def segment_image(request: SegmentRequest):
    """Apply segmentation/thresholding to an image."""
    try:
        # Decode image
        image = decode_base64_image(request.image)

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply segmentation method
        if request.params.method == "otsu":
            _, segmented = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        elif request.params.method == "adaptive_mean":
            segmented = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY, request.params.block_size, request.params.c
            )

        elif request.params.method == "chow_kaneko":
            segmented = _chow_kaneko_threshold(gray, request.params.block_size)

        elif request.params.method == "cheng_jin_kuo":
            segmented = _cheng_jin_kuo_threshold(gray, request.params.block_size, request.params.k)

        else:
            raise HTTPException(status_code=400, detail="Invalid segmentation method")

        # Encode result
        segmented_base64 = encode_image_to_base64(segmented)

        return ImageResponse(
            success=True,
            processed_image=segmented_base64
        )
    except Exception as e:
        return ImageResponse(
            success=False,
            error=str(e)
        )


@router.post("/classify", response_model=ClassifyResponse)
async def classify_image(request: ClassifyRequest):
    """Classify a traffic sign image."""
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        # Decode image
        image = decode_base64_image(request.image)

        # Preprocess for model
        preprocessor = PreProcess(size=IMAGE_SIZE, to_grayscale=False, normalize="minmax")
        img_tensor = preprocess_for_model(image, preprocessor)
        img_tensor = img_tensor.to(device)

        # Predict
        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)

            # Get top 5 predictions
            top5_probs, top5_indices = torch.topk(probabilities, 5, dim=1)

            # Main prediction
            main_idx = top5_indices[0][0].item()
            main_conf = top5_probs[0][0].item()

            # Prepare results
            main_prediction = PredictionResult(
                class_id=main_idx,
                class_name=IDX_TO_CLASS[main_idx],
                confidence=main_conf
            )

            # Top 5 predictions
            top5_predictions = [
                PredictionResult(
                    class_id=idx.item(),
                    class_name=IDX_TO_CLASS[idx.item()],
                    confidence=prob.item()
                )
                for prob, idx in zip(top5_probs[0], top5_indices[0])
            ]

        return ClassifyResponse(
            success=True,
            prediction=main_prediction,
            top5_predictions=top5_predictions
        )
    except Exception as e:
        return ClassifyResponse(
            success=False,
            error=str(e)
        )


# Helper functions for segmentation
def _chow_kaneko_threshold(image, block_size=15):
    """Chow-Kaneko adaptive thresholding."""
    rows, cols = image.shape
    result = np.zeros_like(image)
    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            block = image[i:i + block_size, j:j + block_size]
            if block.size == 0:
                continue
            threshold = np.mean(block)
            result[i:i + block_size, j:j + block_size] = (
                block > threshold
            ).astype(np.uint8) * 255
    return result


def _cheng_jin_kuo_threshold(image, block_size=15, k=0.5):
    """Cheng-Jin-Kuo adaptive thresholding."""
    rows, cols = image.shape
    result = np.zeros_like(image)
    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            block = image[i:i + block_size, j:j + block_size]
            if block.size == 0:
                continue
            threshold = np.mean(block) - k * np.std(block)
            result[i:i + block_size, j:j + block_size] = (
                block > threshold
            ).astype(np.uint8) * 255
    return result
