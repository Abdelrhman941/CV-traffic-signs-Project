"""FastAPI backend for Traffic Sign Recognition System."""

import io
import base64
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import PreProcess, ThresholdingMethods
from src.model import build_model
from src.config import DEVICE, IMAGE_SIZE
from src.utils import IDX_TO_CLASS

app = FastAPI(title="Traffic Sign Recognition API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None


def load_model():
    """Load the trained model."""
    global model
    if model is None:
        model_path = Path(__file__).parent.parent / "models" / "traffic_sign_model.pth"
        if model_path.exists():
            try:
                model = build_model()
                checkpoint = torch.load(model_path, map_location=DEVICE)
                model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
                model.eval()
                print(f"Model loaded from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                model = None
        else:
            print(f"Model not found at {model_path}")
    return model


def numpy_to_base64(image: np.ndarray) -> str:
    """Convert numpy array to base64 string."""
    if image.dtype != np.uint8:
        # Normalize to 0-255 range
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)

    # Convert to PIL Image
    if len(image.shape) == 2:  # Grayscale
        pil_img = Image.fromarray(image, mode='L')
    else:  # RGB
        pil_img = Image.fromarray(image, mode='RGB')

    # Convert to base64
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def base64_to_numpy(base64_str: str) -> np.ndarray:
    """Convert base64 string to numpy array."""
    # Remove data URL prefix if present
    if "base64," in base64_str:
        base64_str = base64_str.split("base64,")[1]

    img_bytes = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_bytes))
    return np.array(img)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Traffic Sign Recognition API", "status": "online"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    model_loaded = load_model() is not None
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "device": str(DEVICE)
    }


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload and return the original image."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(image)

        return JSONResponse({
            "success": True,
            "original_image": numpy_to_base64(img_array),
            "width": img_array.shape[1],
            "height": img_array.shape[0],
            "channels": img_array.shape[2] if len(img_array.shape) == 3 else 1
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.post("/api/preprocess")
async def preprocess_image(data: Dict[str, Any]):
    """Apply preprocessing to an image."""
    try:
        image_data = data.get("image")
        params = data.get("params", {})

        # Convert base64 to numpy
        img_array = base64_to_numpy(image_data)

        # Extract parameters
        resize_width = params.get("resize_width", 256)
        resize_height = params.get("resize_height", 256)
        noise_reduction = params.get("noise_reduction", 5)
        brightness_alpha = params.get("brightness_alpha", 1.0)
        brightness_beta = params.get("brightness_beta", 50)
        contrast = params.get("contrast", 2)
        to_grayscale = params.get("to_grayscale", True)

        # Apply preprocessing steps
        processed = img_array.copy()

        # Resize
        processed = cv2.resize(processed, (resize_width, resize_height))

        # Convert to grayscale if needed
        if to_grayscale and len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)

        # Noise reduction
        if noise_reduction > 0:
            kernel_size = max(1, noise_reduction) * 2 + 1  # Ensure odd number
            processed = cv2.GaussianBlur(processed, (kernel_size, kernel_size), 0)

        # Brightness adjustment
        if brightness_alpha != 1.0 or brightness_beta != 0:
            processed = cv2.convertScaleAbs(processed, alpha=brightness_alpha, beta=brightness_beta)

        # Contrast enhancement
        if contrast > 0:
            if len(processed.shape) == 3:
                lab = cv2.cvtColor(processed, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=float(contrast), tileGridSize=(8, 8))
                cl = clahe.apply(l)
                processed = cv2.merge((cl, a, b))
                processed = cv2.cvtColor(processed, cv2.COLOR_LAB2RGB)
            else:
                clahe = cv2.createCLAHE(clipLimit=float(contrast), tileGridSize=(8, 8))
                processed = clahe.apply(processed)

        return JSONResponse({
            "success": True,
            "processed_image": numpy_to_base64(processed),
            "params_applied": params
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in preprocessing: {str(e)}")


@app.post("/api/segment")
async def segment_image(data: Dict[str, Any]):
    """Apply segmentation to an image."""
    try:
        image_data = data.get("image")
        params = data.get("params", {})

        # Convert base64 to numpy
        img_array = base64_to_numpy(image_data)

        # Extract parameters
        method = params.get("method", "otsu")
        block_size = params.get("block_size", 15)
        c = params.get("c", 5)
        k = params.get("k", 0.5)

        # Ensure image is grayscale for segmentation
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array.copy()

        # Convert to uint8 if needed
        if gray.dtype != np.uint8:
            if gray.max() <= 1.0:
                gray = (gray * 255).astype(np.uint8)
            else:
                gray = np.clip(gray, 0, 255).astype(np.uint8)

        # Apply segmentation method
        if method == "otsu":
            segmented = ThresholdingMethods.otsu_threshold(gray)
        elif method == "adaptive_mean":
            # Ensure block_size is odd
            block_size = max(3, block_size)
            if block_size % 2 == 0:
                block_size += 1
            segmented = ThresholdingMethods.adaptive_mean(gray, block_size, c)
        elif method == "chow_kaneko":
            segmented = ThresholdingMethods.chow_kaneko(gray, block_size)
        elif method == "cheng_jin_kuo":
            segmented = ThresholdingMethods.cheng_jin_kuo(gray, block_size, k)
        else:
            raise ValueError(f"Unknown segmentation method: {method}")

        return JSONResponse({
            "success": True,
            "segmented_image": numpy_to_base64(segmented),
            "method_used": method,
            "params_applied": params
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in segmentation: {str(e)}")


@app.post("/api/extract_features")
async def extract_features(data: Dict[str, Any]):
    """Extract visual features from an image."""
    try:
        image_data = data.get("image")

        # Convert base64 to numpy
        img_array = base64_to_numpy(image_data)

        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array.copy()

        # Convert to uint8
        if gray.dtype != np.uint8:
            if gray.max() <= 1.0:
                gray = (gray * 255).astype(np.uint8)
            else:
                gray = np.clip(gray, 0, 255).astype(np.uint8)

        # Extract features
        features = {}

        # Edge detection
        edges = cv2.Canny(gray, 100, 200)
        features["edges"] = numpy_to_base64(edges)

        # Contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on a copy of the image
        if len(img_array.shape) == 2:
            contour_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            contour_img = img_array.copy()

        cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 2)
        features["contours_image"] = numpy_to_base64(contour_img)
        features["num_contours"] = len(contours)

        # Histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        features["histogram"] = hist.flatten().tolist()

        # Shape descriptors
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)

            features["area"] = float(area)
            features["perimeter"] = float(perimeter)

            if perimeter > 0:
                features["circularity"] = float(4 * np.pi * area / (perimeter * perimeter))

        return JSONResponse({
            "success": True,
            "features": features
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting features: {str(e)}")


@app.post("/api/classify")
async def classify_image(data: Dict[str, Any]):
    """Classify a traffic sign image."""
    try:
        image_data = data.get("image")

        # Load model if not already loaded
        model = load_model()
        if model is None:
            return JSONResponse({
                "success": False,
                "error": "Model not loaded. Please train the model first."
            })

        # Convert base64 to numpy
        img_array = base64_to_numpy(image_data)

        # Preprocess for model
        preprocessor = PreProcess(size=IMAGE_SIZE, to_grayscale=False, normalize="minmax")

        # Ensure RGB
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

        processed = preprocessor(img_array)

        # Convert to tensor
        if processed.ndim == 2:
            processed = np.expand_dims(processed, axis=-1)

        tensor = torch.tensor(processed).permute(2, 0, 1).unsqueeze(0).float()
        tensor = tensor.to(DEVICE)

        # Predict
        with torch.no_grad():
            output = model(tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            top5_probs, top5_indices = torch.topk(probabilities, 5)

            predicted_class = int(output.argmax(dim=1).item())
            confidence = float(probabilities[0, predicted_class].item())

        # Get top 5 predictions
        top5_predictions = []
        for prob, idx in zip(top5_probs[0], top5_indices[0]):
            top5_predictions.append({
                "class_id": int(idx.item()),
                "class_name": IDX_TO_CLASS[int(idx.item())],
                "confidence": float(prob.item())
            })

        return JSONResponse({
            "success": True,
            "prediction": {
                "class_id": predicted_class,
                "class_name": IDX_TO_CLASS[predicted_class],
                "confidence": confidence
            },
            "top5_predictions": top5_predictions
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in classification: {str(e)}")


@app.get("/api/classes")
async def get_classes():
    """Get all available traffic sign classes."""
    return JSONResponse({
        "success": True,
        "classes": [{"id": i, "name": name} for i, name in IDX_TO_CLASS.items()]
    })

if __name__ == "__main__":
    import uvicorn
    print("Starting Traffic Sign Recognition API")
    print("- API  : http://localhost:8000/api/")
    print("- Docs : http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")
