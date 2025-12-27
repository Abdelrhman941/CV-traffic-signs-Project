# **🚦 Traffic Sign Recognition System**

Advanced AI-powered traffic sign detection and classification system using deep learning and computer vision.

<body>
    <div style = "
        width: 100%;
        border-radius: 100px;
        height: 20px;
        background: linear-gradient(to right,#B6AE9F,#C5C7BC,#DEDED1,#C5C7BC,#B6AE9F);">
    </div>
</body>

## **📋 Features**

- **43 Traffic Sign Classes** - Complete German Traffic Sign Recognition Benchmark (GTSRB) support
- **Advanced Preprocessing** - Brightness adjustment, contrast enhancement, noise reduction, and edge detection
- **Multiple Segmentation Methods** - Otsu, Adaptive Mean, Chow-Kaneko, and Cheng-Jin-Kuo thresholding
- **Deep Learning Classification** - Custom CNN architecture with 95%+ accuracy
- **Interactive Web Interface** - Real-time image processing pipeline visualization
- **RESTful API** - FastAPI backend with comprehensive endpoints

<body>
    <div style = "
        width: 100%;
        border-radius: 100px;
        height: 20px;
        background: linear-gradient(to right,#B6AE9F,#C5C7BC,#DEDED1,#C5C7BC,#B6AE9F);">
    </div>
</body>

## **Project Structure**

```
├── Data/                      # Dataset (GTSRB)
│
├── Backend/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py          # API endpoints
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py         # Pydantic models
│   ├── utils/
│   │   ├── __init__.py
│   │   └── helpers.py         # Preprocessing & utilities
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── main.py                # FastAPI application
│   └── .env                   # Environment variables
│
├── Frontend/
│   ├── index.html             # Web interface
│   ├── app.js                 # Frontend logic
│   └── styles.css             # Styling
│
├── notebooks/
│   ├── code.ipynb             # Training & experiments
│   └── best_model.pth         # Trained model weights
│
├── .gitignore                 # Git ignore file
├── run.sh                     # run script
└── requirements.txt           # Python dependencies
```

<body>
    <div style = "
        width: 100%;
        border-radius: 100px;
        height: 20px;
        background: linear-gradient(to right,#B6AE9F,#C5C7BC,#DEDED1,#C5C7BC,#B6AE9F);">
    </div>
</body>

## **Quick Start**

### Prerequisites

- Python 3.8+
- pip
- (Optional) CUDA-capable GPU for faster inference

### Installation & Running

```bash
# Make script executable
chmod +x run.sh

# Run the system
./run.sh
```

The script will:
1. ✅ Check Python installation
2. 📦 Create virtual environment (if needed)
3. 📥 Install all dependencies
4. 🔍 Verify model file exists
5. 🚀 Start backend server
6. 🌐 Open browser automatically

### Manual Installation

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run backend
cd Backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

<body>
    <div style = "
        width: 100%;
        border-radius: 100px;
        height: 20px;
        background: linear-gradient(to right,#B6AE9F,#C5C7BC,#DEDED1,#C5C7BC,#B6AE9F);">
    </div>
</body>

## **🌐 Access Points**

Once running, access the system at:

- **Frontend**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/api/health

<body>
    <div style = "
        width: 100%;
        border-radius: 100px;
        height: 20px;
        background: linear-gradient(to right,#B6AE9F,#C5C7BC,#DEDED1,#C5C7BC,#B6AE9F);">
    </div>
</body>

## **🎯 API Endpoints**

### Health Check
```http
GET /api/health
```

### Upload Image
```http
POST /api/upload
Content-Type: multipart/form-data

Body: file (image file)
```

### Preprocess Image
```http
POST /api/preprocess
Content-Type: application/json

{
  "image": "base64_encoded_image",
  "params": {
    "resize_width": 64,
    "resize_height": 64,
    "noise_reduction": 5,
    "brightness_alpha": 1.0,
    "brightness_beta": 50,
    "contrast": 2,
    "to_grayscale": true
  }
}
```

### Segment Image
```http
POST /api/segment
Content-Type: application/json

{
  "image": "base64_encoded_image",
  "params": {
    "method": "otsu",
    "block_size": 15,
    "c": 5,
    "k": 0.5
  }
}
```

### Classify Image
```http
POST /api/classify
Content-Type: application/json

{
  "image": "base64_encoded_image"
}
```

<body>
    <div style = "
        width: 100%;
        border-radius: 100px;
        height: 20px;
        background: linear-gradient(to right,#B6AE9F,#C5C7BC,#DEDED1,#C5C7BC,#B6AE9F);">
    </div>
</body>

## **🧠 Model Architecture**

### CNN Model
```
Input: (1, 64, 64) - Grayscale image

Features:
├── Conv2D(1→32)  + BatchNorm + ReLU + MaxPool
└── Conv2D(32→64) + BatchNorm + ReLU + MaxPool

Classifier:
├── Linear(64×16×16 → 128) + ReLU + Dropout(0.5)
└── Linear(128 → 43) - Output classes
```

### Training Details
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-5)
- **Loss**: CrossEntropyLoss with label smoothing (0.1)
- **Scheduler**: ReduceLROnPlateau
- **Dataset**: German Traffic Sign Recognition Benchmark (GTSRB)
- **Accuracy**: 95%+ on test set

<body>
    <div style = "
        width: 100%;
        border-radius: 100px;
        height: 20px;
        background: linear-gradient(to right,#B6AE9F,#C5C7BC,#DEDED1,#C5C7BC,#B6AE9F);">
    </div>
</body>

## **> Preprocessing Pipeline**

1. **Resize** - Standardize to 64×64 pixels
2. **Brightness Adjustment** - Enhance dark images
3. **Contrast Enhancement** - CLAHE (Contrast Limited Adaptive Histogram Equalization)
4. **Noise Reduction** - Gaussian blur (5×5)
5. **Thresholding** - Otsu's method (or alternative methods)
6. **Edge Detection** - Canny edge detection
7. **Edge Thickening** - Dilation for better visibility
8. **Normalization** - MinMax scaling to [0, 1]

<body>
    <div style = "
        width: 100%;
        border-radius: 100px;
        height: 20px;
        background: linear-gradient(to right,#B6AE9F,#C5C7BC,#DEDED1,#C5C7BC,#B6AE9F);">
    </div>
</body>

## **> Traffic Sign Classes**

The system recognizes 43 classes including:
- Speed limits (20-120 km/h)
- Prohibitory signs (No entry, No passing, Stop, Yield)
- Danger warnings (Curves, Slippery road, Road work)
- Mandatory signs (Turn directions, Keep right/left)
- And more...

<body>
    <div style = "
        width: 100%;
        border-radius: 100px;
        height: 20px;
        background: linear-gradient(to right,#B6AE9F,#C5C7BC,#DEDED1,#C5C7BC,#B6AE9F);">
    </div>
</body>

## **> Technology Stack**

### Backend
- **FastAPI** - Modern, fast web framework
- **PyTorch** - Deep learning framework
- **OpenCV** - Computer vision operations
- **Pydantic** - Data validation
- **Uvicorn** - ASGI server

### Frontend
- **HTML5/CSS3** - Modern web standards
- **Vanilla JavaScript** - No framework dependencies
- **Font Awesome** - Icon library

<body>
    <div style = "
        width: 100%;
        border-radius: 100px;
        height: 20px;
        background: linear-gradient(to right,#B6AE9F,#C5C7BC,#DEDED1,#C5C7BC,#B6AE9F);">
    </div>
</body>

## **🐛 Troubleshooting**

### - Model Not Found
```
❌ ERROR: Model file not found at notebooks/best_model.pth
```
**Solution**: Ensure `best_model.pth` exists in the `notebooks/` folder

### - Port Already in Use
```
⚠️ Port 8000 is in use
```
**Solution**: The script automatically kills existing processes. If issue persists, manually kill:
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

### - CUDA Not Available
The system automatically falls back to CPU if CUDA is not available.

<body>
    <div style = "
        width: 100%;
        height: 20px;
        background: linear-gradient(to right,#B6AE9F,#C5C7BC,#DEDED1,#C5C7BC,#B6AE9F);">
    </div>
</body>

> [!NOTE]
> Thank you for using the Traffic Sign Recognition System! happy driving!
