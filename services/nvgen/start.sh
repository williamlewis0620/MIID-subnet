#!/bin/bash

# Name Variant Generation Service Startup Script

echo "🚀 Starting Name Variant Generation Service..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ Error: pip3 is not installed or not in PATH"
    exit 1
fi

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "📦 Installing dependencies..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to install dependencies"
        exit 1
    fi
    echo "✅ Dependencies installed successfully"
else
    echo "⚠️  Warning: requirements.txt not found, skipping dependency installation"
fi

# Create cache directory if it doesn't exist
mkdir -p cache
echo "📁 Cache directory ready"

# Start the service
echo "🌐 Starting service on http://0.0.0.0:8000"
echo "📖 API documentation available at http://localhost:8000/docs"
echo "🔧 Press Ctrl+C to stop the service"
echo ""

python3 main.py
