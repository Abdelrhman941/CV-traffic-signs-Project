from pydantic import BaseModel
from typing import Optional, List


class PreprocessParams(BaseModel):
    """Parameters for image preprocessing."""
    resize_width: int = 64
    resize_height: int = 64
    noise_reduction: int = 5
    brightness_alpha: float = 1.0
    brightness_beta: int = 50
    contrast: int = 2
    to_grayscale: bool = True


class SegmentParams(BaseModel):
    """Parameters for image segmentation."""
    method: str = "otsu"  # otsu, adaptive_mean, chow_kaneko, cheng_jin_kuo
    block_size: int = 15
    c: int = 5
    k: float = 0.5


class PreprocessRequest(BaseModel):
    """Request model for preprocessing endpoint."""
    image: str  # Base64 encoded image
    params: PreprocessParams


class SegmentRequest(BaseModel):
    """Request model for segmentation endpoint."""
    image: str  # Base64 encoded image
    params: SegmentParams


class ClassifyRequest(BaseModel):
    """Request model for classification endpoint."""
    image: str  # Base64 encoded image


class PredictionResult(BaseModel):
    """Single prediction result."""
    class_id: int
    class_name: str
    confidence: float


class ClassifyResponse(BaseModel):
    """Response model for classification endpoint."""
    success: bool
    prediction: Optional[PredictionResult] = None
    top5_predictions: Optional[List[PredictionResult]] = None
    error: Optional[str] = None


class ImageResponse(BaseModel):
    """Response model for image processing endpoints."""
    success: bool
    processed_image: Optional[str] = None  # Base64 encoded
    error: Optional[str] = None


class UploadResponse(BaseModel):
    """Response model for upload endpoint."""
    success: bool
    original_image: Optional[str] = None  # Base64 encoded
    error: Optional[str] = None


"""
UploadResponse        → original_image
PreprocessRequest     → Image + Params
ImageResponse         → processed_image
SegmentRequest        → Image + Params
ClassifyRequest       → Image
ClassifyResponse      → Prediction(s)

"""
