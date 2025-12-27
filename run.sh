#!/bin/bash

# Traffic Sign Recognition System - Run Script
# This script starts both frontend and backend servers

echo "------------------------------------------------------------"
echo "🚦 Traffic Sign Recognition System"
echo "------------------------------------------------------------"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python
if ! command_exists python && ! command_exists python3; then
    echo -e "${RED}❌ Python is not installed!${NC}"
    exit 1
fi

PYTHON_CMD=$(command_exists python3 && echo "python3" || echo "python")
echo -e "${GREEN}✅ Python found: $PYTHON_CMD${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
    echo -e "${YELLOW}⚠️  No virtual environment found${NC}"
    echo -e "${BLUE}-? Creating virtual environment...${NC}"
    $PYTHON_CMD -m venv venv

    # Activate virtual environment
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
    fi

    echo -e "${BLUE}-> Installing dependencies...${NC}"
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo -e "${GREEN}✅ Virtual environment found${NC}"

    # Activate virtual environment
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    elif [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
    elif [ -f ".venv/Scripts/activate" ]; then
        source .venv/Scripts/activate
    fi
fi

# Check if model exists
if [ ! -f "notebooks/best_model.pth" ]; then
    echo -e "${RED}❌ Model file not found!${NC}"
    echo -e "${YELLOW}   Expected: notebooks/best_model.pth${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Model file found${NC}"

# Kill any existing servers on port 8000
echo -e "${BLUE}-> Checking for existing servers...${NC}"
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Port 8000 is in use, killing existing process...${NC}"
    kill -9 $(lsof -t -i:8000) 2>/dev/null
fi

# Start backend server
echo -e "${BLUE}-> Starting backend server...${NC}"
cd Backend
$PYTHON_CMD -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo -e "${BLUE}⏳ Waiting for backend to start...${NC}"
sleep 3

# Check if backend is running
if ! ps -p $BACKEND_PID > /dev/null; then
    echo -e "${RED}❌ Backend failed to start!${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Backend started successfully (PID: $BACKEND_PID)${NC}"

# Open browser
echo -e "${BLUE}- Opening browser...${NC}"
sleep 2

# Detect OS and open browser
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open http://localhost:8000 2>/dev/null
elif [[ "$OSTYPE" == "darwin"* ]]; then
    open http://localhost:8000
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    start http://localhost:8000
else
    echo -e "${YELLOW}⚠️  Could not detect OS. Please open http://localhost:8000 manually${NC}"
fi

echo ""
echo "------------------------------------------------------------"
echo -e "${GREEN}✅ System is running!${NC}"
echo "------------------------------------------------------------"
echo -e "- Frontend   : ${BLUE}http://localhost:8000${NC}"
echo -e "- API Docs   : ${BLUE}http://localhost:8000/docs${NC}"
echo -e "- API Health : ${BLUE}http://localhost:8000/api/health${NC}"
echo "------------------------------------------------------------"
echo -e "${YELLOW}Press Ctrl+C to stop all servers${NC}"
echo "------------------------------------------------------------"

# Wait for user interrupt
wait $BACKEND_PID
