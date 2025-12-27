"""
FastAPI Backend for Traffic Sign Recognition System.
"""

import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uvicorn

from Backend.config import (
    HOST,
    PORT,
    RELOAD,
    ALLOWED_ORIGINS,
    MODEL_PATH,
    NUM_CLASSES,
    IMAGE_SIZE,
)
from Backend.api.routes import router, set_model


# Define CNN Model Architecture (must match training)
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64 -> 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 -> 16
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.classifier(x)


# Initialize FastAPI app
app = FastAPI(
    title="Traffic Sign Recognition API",
    description="AI-powered traffic sign detection and classification",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


# Serve static frontend files
frontend_path = Path(__file__).parent.parent / "Frontend"
if frontend_path.exists():
    app.mount(
        "/", StaticFiles(directory=str(frontend_path), html=True), name="frontend"
    )


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    print("=" * 60)
    print("🚦 Traffic Sign Recognition System - Starting...")
    print("=" * 60)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Device: {device}")

    # Load model
    print(f"📦 Loading model from: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        print(f"❌ ERROR: Model file not found at {MODEL_PATH}")
        print("   Please ensure best_model.pth exists in the notebooks folder")
        return

    try:
        model = CNNModel(NUM_CLASSES)
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        # Set model in routes
        set_model(model, device)

        print("✅ Model loaded successfully!")
        print(f"   Classes    : {NUM_CLASSES}")
        print(f"   Input size : {IMAGE_SIZE}")
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        return

    print("=" * 60)
    print(f"🌐 Server running at : http://localhost:{PORT}")
    print(f"📖 API docs at       : http://localhost:{PORT}/docs")
    print("=" * 60)


@app.get("/")
async def root():
    """Root endpoint - redirects to frontend."""
    return {
        "message": "Traffic Sign Recognition API",
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    uvicorn.run("Backend.main:app", host=HOST, port=PORT, reload=RELOAD)
