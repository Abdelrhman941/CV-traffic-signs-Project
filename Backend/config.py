"""Backend configuration settings."""

from pathlib import Path

# Paths
BACKEND_DIR = Path(__file__).parent
PROJECT_DIR = BACKEND_DIR.parent
MODELS_DIR = PROJECT_DIR / "models"
UPLOAD_DIR = BACKEND_DIR / "uploads"

# Server settings
HOST = "0.0.0.0"
PORT = 8080
RELOAD = True

# API settings
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
