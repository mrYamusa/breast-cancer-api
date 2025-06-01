#!/bin/bash

# Breast Cancer Classification API Server Startup Script

echo "🔬 Starting Breast Cancer Classification API Server..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Check if model files exist
if [ ! -f "deployment_model/breast_cancer_model.pth" ]; then
    echo "❌ Model file not found: deployment_model/breast_cancer_model.pth"
    echo "Please ensure your trained model files are in the deployment_model/ directory"
    exit 1
fi

if [ ! -f "deployment_model/model_config.json" ]; then
    echo "❌ Config file not found: deployment_model/model_config.json"
    echo "Please ensure your model config file is in the deployment_model/ directory"
    exit 1
fi

echo "✅ Model files found!"
echo "🚀 Starting FastAPI server..."
echo "📱 Open your browser to: http://localhost:8000"
echo "📚 API documentation: http://localhost:8000/docs"
echo ""

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload