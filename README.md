# Traffic Sign Recognition System 🚦

> ![alt text](images/1.png)
AI-powered traffic sign classification system with PyTorch CNN, featuring both CLI tools and a modern web interface. Achieves 95%+ accuracy on German Traffic Sign Recognition Benchmark (GTSRB).

## 🌟 Key Features

- **Deep Learning**: Custom CNN with PyTorch for 43 traffic sign classes
- **Computer Vision**: Advanced preprocessing, segmentation & feature extraction
- **Modern Web UI**: Beautiful HTML/CSS/JS frontend + FastAPI backend
- **Old Streamlit GUI**: Simple prototype interface (deprecated)
- **CLI Tools**: Training, evaluation, and prediction scripts
- **GPU Support**: Automatic CUDA acceleration

## 📁 Project Structure

```
traffic-sign-recognition/
├── Backend/                  # FastAPI REST API server
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Backend configuration
│   └── main.py               # API endpoints & model inference
│
├── Frontend/                 # Modern web interface
│   ├── app.js                # Interactive controls & API calls
│   ├── index.html            # Main page with Font Awesome icons
│   └── styles.css            # Eye-friendly design & responsive layout
│
├── images/                   # Screenshot images for documentation
│   ├── 1.png                 # UI screenshot
│   ├── 2.png                 # UI screenshot
│   ├── 3.png                 # UI screenshot
│   ├── 4.png                 # UI screenshot
│   ├── 5.png                 # UI screenshot
│   ├── 6.png                 # UI screenshot
│   └── traffic-lights.png    # Logo/icon
│
├── models/                    # Saved model checkpoints
│   └── traffic_sign_model.pth # Trained CNN weights
│
├── notebooks/                # Jupyter experiments
│   └── code.ipynb            # Development notebook
│
├── src/                      # Core modules
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Paths & hyperparameters
│   ├── data.py               # Data loading & augmentation
│   ├── evaluate.py           # Model evaluation & metrics
│   ├── features.py           # Feature extraction methods
│   ├── model.py              # CNN architecture
│   ├── preprocessing.py      # Image preprocessing & thresholding
│   ├── train.py              # Training loop & optimizer
│   └── utils.py              # Helper functions
│
├── .gitignore                 # Git ignore patterns
├── README.md                  # Project documentation
├── main.py                    # CLI for training/evaluation
├── predict.py                 # CLI for predictions
├── requirements.txt           # Project dependencies
└── run.sh                     # Quick start script (Linux/Mac)
```

## 🚀 Quick Start

### Option 1: Automated (Recommended)

```bash
chmod +x run.sh && ./run.sh
```

Opens:
- Backend API: `http://localhost:8000` (+ docs at `/docs`)
- Frontend: Opens in default browser automatically

### Option 2: Manual Setup

**1. Install Dependencies**
```bash
pip install -r requirements.txt
```

**2. Start Backend**
```bash
cd Backend
python main.py
```

**3. Open Frontend**
Open `Frontend/index.html` in your browser

## 🎯 Usage

### Web Interface (Recommended)

1. **Upload Image**: Drag & drop traffic sign image
2. **Preprocess**: Adjust resize, denoise, brightness, contrast
3. **Segment**: Choose Otsu/Adaptive/Chow-Kaneko/Cheng-Jin-Kuo methods
4. **Classify**: Get prediction with confidence + top 5 results

### CLI Tools

**Train Model:**
```bash
python main.py train --epochs 30 --batch-size 32
```

**Evaluate:**
```bash
python main.py evaluate --checkpoint models/traffic_sign_model.pth
```

**Predict:**
```bash
# Single image
python predict.py --image path/to/sign.jpg

# Batch directory
python predict.py --directory path/to/images/
```

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/api/health` | GET | System health (GPU, model status) |
| `/api/upload` | POST | Upload image |
| `/api/preprocess` | POST | Image preprocessing |
| `/api/segment` | POST | Image segmentation |
| `/api/extract_features` | POST | Feature extraction |
| `/api/classify` | POST | Classify traffic sign |
| `/api/classes` | GET | List all 43 classes |

Full API documentation: `http://localhost:8000/docs`

## 📊 Traffic Sign Classes (43 Total)

- **Speed Limits**: 20, 30, 50, 60, 70, 80, 100, 120 km/h
- **Warnings**: Curves, pedestrians, children, animals, bumpy road, etc.
- **Mandatory**: Straight, turn right/left, roundabout, keep right/left
- **Prohibitions**: No entry, no passing, no vehicles, weight limits

---

## **📚 Dataset**

[German Traffic Sign Recognition Benchmark (GTSRB)](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- 50,000+ images
- 43 classes
- Various lighting/weather conditions
