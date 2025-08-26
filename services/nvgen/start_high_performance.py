#!/usr/bin/env python3
"""
High-Performance Startup Script for Name Variant Generation Service

This script starts the service with optimized settings for handling dozens of concurrent requests.
It uses multiple uvicorn workers and optimized connection limits.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    HOST, PORT, UVICORN_WORKERS, UVICORN_LIMIT_CONCURRENCY,
    UVICORN_LIMIT_MAX_REQUESTS, UVICORN_TIMEOUT_KEEP_ALIVE, UVICORN_BACKLOG
)

def start_high_performance_server():
    """Start the service with high-performance settings"""
    
    print("🚀 Starting High-Performance Name Variant Generation Service...")
    print(f"   Host: {HOST}")
    print(f"   Port: {PORT}")
    print(f"   Workers: {UVICORN_WORKERS}")
    print(f"   Max Concurrent Connections: {UVICORN_LIMIT_CONCURRENCY}")
    print(f"   Max Requests per Worker: {UVICORN_LIMIT_MAX_REQUESTS}")
    print(f"   Keep-alive Timeout: {UVICORN_TIMEOUT_KEEP_ALIVE}s")
    print(f"   Connection Backlog: {UVICORN_BACKLOG}")
    print("\n📋 High-Performance Configuration:")
    print("   - Multiple uvicorn workers for better concurrency")
    print("   - Increased connection limits")
    print("   - Optimized keep-alive settings")
    print("   - Enhanced connection backlog")
    print("\n" + "="*60)
    
    # Build uvicorn command with high-performance settings
    cmd = [
        sys.executable, "-m", "uvicorn",
        "main:app",
        "--host", HOST,
        "--port", str(PORT),
        "--workers", str(UVICORN_WORKERS),
        "--limit-concurrency", str(UVICORN_LIMIT_CONCURRENCY),
        "--limit-max-requests", str(UVICORN_LIMIT_MAX_REQUESTS),
        "--timeout-keep-alive", str(UVICORN_TIMEOUT_KEEP_ALIVE),
        "--backlog", str(UVICORN_BACKLOG),
        "--log-level", "info",
        "--access-log",
        "--loop", "asyncio"
    ]
    
    try:
        # Start the server
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 High-performance server stopped by user")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error starting high-performance server: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_high_performance_server()
