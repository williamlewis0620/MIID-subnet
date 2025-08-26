#!/bin/bash

# High-Performance Startup Script for Name Variant Generation Service
# This script starts the service with optimized settings for handling dozens of concurrent requests

set -e

echo "🚀 Starting High-Performance Name Variant Generation Service..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed or not in PATH"
    exit 1
fi

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "📦 Installing dependencies..."
    pip3 install -r requirements.txt
fi

# Create cache directory if it doesn't exist
if [ ! -d "cache" ]; then
    echo "📁 Creating cache directory..."
    mkdir -p cache
fi

# Load high-performance environment configuration
if [ -f "high_performance.env" ]; then
    echo "⚙️  Loading high-performance configuration..."
    export $(cat high_performance.env | grep -v '^#' | xargs)
fi

# Set default high-performance values if not set
export NVGEN_UVICORN_WORKERS=${NVGEN_UVICORN_WORKERS:-1}
export NVGEN_LIMIT_CONCURRENCY=${NVGEN_LIMIT_CONCURRENCY:-4000}
export NVGEN_LIMIT_MAX_REQUESTS=${NVGEN_LIMIT_MAX_REQUESTS:-100000}
export NVGEN_TIMEOUT_KEEP_ALIVE=${NVGEN_TIMEOUT_KEEP_ALIVE:-120}
export NVGEN_BACKLOG=${NVGEN_BACKLOG:-8192}

echo "📋 High-Performance Configuration:"
echo "   Workers: $NVGEN_UVICORN_WORKERS"
echo "   Max Concurrent Connections: $NVGEN_LIMIT_CONCURRENCY"
echo "   Max Requests per Worker: $NVGEN_LIMIT_MAX_REQUESTS"
echo "   Keep-alive Timeout: ${NVGEN_TIMEOUT_KEEP_ALIVE}s"
echo "   Connection Backlog: $NVGEN_BACKLOG"
echo "   RAM Usage Limit: 48% (60GB of 125GB)"
echo "   CPU Usage: 50% (16 threads of 32)"

echo ""
echo "🎯 Starting service with optimized settings for AMD Ryzen 9 5950X..."
echo "   This configuration uses ~50% of available resources"
echo "   Can handle thousands of concurrent requests"
echo ""

# Start the high-performance server
python3 start_high_performance.py
