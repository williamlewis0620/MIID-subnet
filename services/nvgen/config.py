#!/usr/bin/env python3
"""
Configuration file for the Name Variant Generation Service
"""

import os
from pathlib import Path

# Service Configuration
HOST = os.getenv("NVGEN_HOST", "0.0.0.0")
PORT = int(os.getenv("NVGEN_PORT", "8001"))

# Timeout Configuration
INSTANCE_TIMEOUT = int(os.getenv("NVGEN_INSTANCE_TIMEOUT", "60"))  # 1 minute
WHOLE_POOL_TIMEOUT = int(os.getenv("NVGEN_WHOLE_POOL_TIMEOUT", "600"))  # 10 minutes
CONSUMED_TIMEOUT = int(os.getenv("NVGEN_CONSUMED_TIMEOUT", "1200"))  # 20 minutes

# Resource Management
MAX_RAM_PERCENT = float(os.getenv("NVGEN_MAX_RAM_PERCENT", "80.0"))
MAX_CONCURRENT_POOLS = int(os.getenv("NVGEN_MAX_CONCURRENT_POOLS", "3"))

# Cache Configuration
CACHE_DIR = Path(os.getenv("NVGEN_CACHE_DIR", "cache"))
CACHE_DIR.mkdir(exist_ok=True)

# Pool Generation Configuration
BUCKET_K = int(os.getenv("NVGEN_BUCKET_K", "15"))
ALPHABET = os.getenv("NVGEN_ALPHABET", "abcdefghijklmnopqrstuvwxyz")

# Logging Configuration
LOG_LEVEL = os.getenv("NVGEN_LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("NVGEN_LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Performance Configuration
WORKER_PROCESSES = int(os.getenv("NVGEN_WORKER_PROCESSES", "16"))  # 50% of 32 threads
WORKER_THREADS = int(os.getenv("NVGEN_WORKER_THREADS", "16"))     # 50% of 32 threads

# Parallel Processing Configuration
USE_PARALLEL_BFS = os.getenv("NVGEN_USE_PARALLEL_BFS", "false").lower() == "true"  # Enable parallel BFS
MAX_BFS_WORKERS = int(os.getenv("NVGEN_MAX_BFS_WORKERS", "8"))  # Max workers for BFS operations

# Uvicorn Server Configuration
UVICORN_WORKERS = int(os.getenv("NVGEN_UVICORN_WORKERS", "1"))  # 50% of 16 cores
UVICORN_LIMIT_CONCURRENCY = int(os.getenv("NVGEN_LIMIT_CONCURRENCY", "4000"))  # Optimized for high concurrency
UVICORN_LIMIT_MAX_REQUESTS = int(os.getenv("NVGEN_LIMIT_MAX_REQUESTS", "100000"))  # High request capacity
UVICORN_TIMEOUT_KEEP_ALIVE = int(os.getenv("NVGEN_TIMEOUT_KEEP_ALIVE", "120"))  # 2 minutes keep-alive
UVICORN_BACKLOG = int(os.getenv("NVGEN_BACKLOG", "8192"))  # Large connection backlog

# API Configuration
API_TITLE = "Name Variant Generation Service"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
A FastAPI-based service that provides name variants to miners' name variant pool requests.

## Features

- Phonetic-aware name variant generation using Soundex, Metaphone, and NYSIIS algorithms
- Queue-based pool generation with timeout management
- Multi-CPU pool generation with RAM monitoring
- File-based caching for each name
- Consumed variant tracking with automatic expiration
- Resource monitoring to prevent system overload
"""

# Validation
def validate_config():
    """Validate configuration values"""
    if INSTANCE_TIMEOUT <= 0:
        raise ValueError("INSTANCE_TIMEOUT must be positive")
    if WHOLE_POOL_TIMEOUT <= 0:
        raise ValueError("WHOLE_POOL_TIMEOUT must be positive")
    if CONSUMED_TIMEOUT <= 0:
        raise ValueError("CONSUMED_TIMEOUT must be positive")
    if MAX_RAM_PERCENT <= 0 or MAX_RAM_PERCENT > 100:
        raise ValueError("MAX_RAM_PERCENT must be between 0 and 100")
    if MAX_CONCURRENT_POOLS <= 0:
        raise ValueError("MAX_CONCURRENT_POOLS must be positive")
    if BUCKET_K <= 0:
        raise ValueError("BUCKET_K must be positive")
    if WORKER_PROCESSES <= 0:
        raise ValueError("WORKER_PROCESSES must be positive")
    if WORKER_THREADS <= 0:
        raise ValueError("WORKER_THREADS must be positive")

# Validate configuration on import
validate_config()
