#!/bin/bash

# Traffic Sign Recognition System - Quick Start Script
# Starts both Backend API and Frontend Server

echo "🚦 Traffic Sign Recognition System - Quick Start"
echo "================================================"
echo ""

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python found: $(python --version)"
echo ""

# Check if model file exists
if [ ! -f "models/traffic_sign_model.pth" ]; then
    echo "⚠️  Warning: Model file not found at models/traffic_sign_model.pth"
    echo "   Please ensure the trained model is in the correct location."
    echo ""
fi

# Move to backend directory
echo "📦 Checking backend setup..."
cd Backend || { echo "❌ Backend folder not found!"; exit 1; }

# Create virtual environment if missing
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo ""

# Ask user whether to install dependencies
read -p "❓  Do you want to install dependencies from requirements.txt? (y/n): " install_choice

if [[ "$install_choice" == "y" || "$install_choice" == "Y" ]]; then
    echo "Installing/updating dependencies..."
    pip install -q -r requirements.txt
    echo "✅ Dependencies installed."
else
    echo "⏩ Skipping dependency installation."
fi

python main.py &
BACKEND_PID=$!

# Wait a few seconds for backend startup
sleep 3

# Start frontend
cd ../Frontend || { echo "❌ Frontend folder not found!"; exit 1; }

python -m http.server 8080 &
FRONTEND_PID=$!

echo "================================================"
echo "🎉 Application is ready!"
echo ""
echo "- Open your browser and navigate to:"
echo "   http://localhost:8080"
echo ""
echo "- API Documentation:"
echo "   http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all servers"
echo "================================================"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ All servers stopped"
    exit 0
}

# Trap Ctrl+C
trap cleanup INT

# Wait for user to stop manually
wait
