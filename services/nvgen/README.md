# Name Variant Generation Service

A FastAPI-based service that provides name variants to miners' name variant pool requests. The service implements a sophisticated phonetic-aware expansion algorithm with caching, queue management, and resource monitoring.

## Features

- **Phonetic-aware name variant generation** using Soundex, Metaphone, and NYSIIS algorithms
- **Queue-based pool generation** with timeout management
- **Multi-CPU pool generation** with RAM monitoring
- **File-based caching** for each name
- **Consumed variant tracking** with automatic expiration
- **Resource monitoring** to prevent system overload

## Architecture

### Core Components

1. **Pool Generation Algorithm**: Phonetic-aware expansion with fixed radius
2. **Caching System**: File-based cache for each name with JSON storage
3. **Queue Management**: Background processing of whole pool generation requests
4. **Resource Monitoring**: RAM usage tracking to prevent system overload
5. **Variant Tracking**: Consumed variant management with 20-minute expiration

### Pool Structure

The service generates pools organized by:
- **Length Difference (ld)**: Difference in length between original and variant
- **Orthographic Level (o)**: 4 levels (0-3) based on string similarity
- **Phonetic Class (p)**: 8 classes (P0-P7) based on Soundex/Metaphone/NYSIIS matching

## API Endpoints

### GET /pool

**Endpoint:** `GET /pool?original_name={name}&timeout={seconds}`

**Description:** Provide name variants pool except for already consumed variations.

**Parameters:**
- `original_name` (required): The name to generate variants for
- `timeout` (optional): Custom timeout in seconds for instance pool generation (default: 60s)

**Logic:**
1. Check if whole pool is cached → return whole pool
2. Check if instance pool is cached → return instance pool  
3. If no cache exists and no ongoing generation → generate instance pool with specified timeout
4. If generation is ongoing → wait for completion

**Response:**
```json
{
  "name": "john",
  "pools": [...],
  "source": "whole_pool_cache" | "instance_pool_cache" | "instance_generated",
  "consumed_count": 5
}
```

**Examples:**
```bash
# Default timeout (60s)
curl "http://localhost:8000/pool?original_name=john"

# Custom timeout (30s)
curl "http://localhost:8000/pool?original_name=john&timeout=30"

# Custom timeout (120s)
curl "http://localhost:8000/pool?original_name=john&timeout=120"
```

### POST /pool?name={name}
Request whole pool generation for a name.

**Action:**
- Creates a cache file with instance_pool if it doesn't exist
- The pool generation worker will detect this and generate a whole pool

**Response:**
```json
{
  "message": "Whole pool generation requested for john",
  "name": "john",
  "status": "queued"
}
```

### POST /consumed?original_name={name}
Marks variants as consumed for 20 minutes.

**Request:**
```json
{
  "variants": ["joh", "jone", "jon"]
}
```

**Response:**
```json
{
  "message": "Marked 3 variants as consumed",
  "expires_at": "2024-01-15T10:30:00",
  "consumed_count": 8
}
```

### GET /status
Get service status and statistics.

**Response:**
```json
{
  "status": "running",
  "ram_usage_percent": 45.2,
  "consumed_variants": 150,
  "queue_size": 3,
  "currently_generating": ["alice", "bob"],
  "whole_pool_requests": 5,
  "cache_directory": "/path/to/cache"
}
```

### POST /stop-generation
Stop pool generation but keep service running.

**Response:**
```json
{
  "message": "Pool generation stopped. Service remains running for API requests.",
  "status": "generation_stopped"
}
```

### POST /shutdown
Safely stop the service.

**Response:**
```json
{
  "message": "Shutdown initiated. Service will stop after completing ongoing operations.",
  "status": "shutting_down"
}
```

## Cache File Structure

Each name has a single cache file (`{name}.txt`) that contains both instance and whole pools:

```json
{
  "name": "john",
  "instance_pool": [...],
  "whole_pool": [...],
  "instance_stats": {
    "max_r": 4,
    "r": 3,
    "ld_groups": 4,
    "variants_total": 150,
    "timeout_hit": false,
    "pool_stats": "good"
  },
  "whole_stats": {
    "max_r": 4,
    "r": 4,
    "ld_groups": 4,
    "variants_total": 300,
    "timeout_hit": false,
    "pool_stats": "good"
  },
  "instance_timestamp": 1705123456.789,
  "whole_timestamp": 1705123456.789,
  "timestamp": 1705123456.789
}
```

- **instance_pool**: Quick generation pool (60s timeout)
- **whole_pool**: Comprehensive generation pool (600s timeout)
- **instance_stats/whole_stats**: Generation statistics including:
  - **max_r**: Maximum radius attempted
  - **r**: Actual radius reached
  - **ld_groups**: Number of length difference groups
  - **variants_total**: Total number of variants generated
  - **timeout_hit**: Whether timeout was reached
  - **pool_stats**: Reason for stopping ("good", "timed_out", "ram_overhead")
- **instance_timestamp/whole_timestamp**: When each pool was generated

Both pools can coexist in the same file, and the service will use the most appropriate one based on availability.

## Pool Generation Status

The `pool_stats` field indicates why pool generation stopped:

- **"good"**: Generation completed normally without hitting any limits
- **"timed_out"**: Generation stopped due to timeout (60s for instance, 600s for whole)
- **"ram_overhead"**: Generation stopped due to RAM usage exceeding 80%
- **"terminated"**: Generation stopped due to termination signal (graceful shutdown)

### RAM Monitoring

The service continuously monitors RAM usage during pool generation:
- Checks RAM usage at the beginning of each BFS layer
- Checks RAM usage during variant generation
- Stops immediately if RAM usage exceeds 80% to prevent system overload
- Returns partial results with `pool_stats: "ram_overhead"`

## Signal Handling & Graceful Shutdown

The service implements comprehensive signal handling for safe termination:

### Supported Signals

- **Ctrl+C (SIGINT)**: Graceful shutdown - stops generation and shuts down service
- **Ctrl+Z (SIGTSTP)**: Suspend generation - stops pool generation but keeps service running
- **SIGTERM**: Graceful shutdown - same as Ctrl+C
- **SIGQUIT**: Graceful shutdown - same as Ctrl+C  
- **SIGHUP**: Graceful shutdown - same as Ctrl+C

### Graceful Shutdown Process

When a termination signal is received:

1. **Stop Generation**: Sets `stop_generation = True` to halt new pool generation
2. **Wait for Completion**: Allows ongoing operations to complete
3. **Signal Waiting Requests**: Notifies any waiting API requests
4. **Cleanup**: Performs final cleanup operations
5. **Exit**: Terminates the service cleanly

### Status During Shutdown

The `/status` endpoint shows shutdown status:

```json
{
  "status": "shutting_down",
  "shutdown_requested": true,
  "generation_stopped": true,
  "active_worker_processes": [12345, 12346],
  ...
}
```

### Worker Process Management

The service uses ProcessPoolExecutor for whole pool generation with safe termination:

- **Automatic Management**: ProcessPoolExecutor automatically handles worker process lifecycle
- **Safe Termination**: Worker processes check a termination flag periodically during generation
- **Graceful Shutdown**: Workers can terminate safely without signal handler issues
- **Resource Cleanup**: No manual process tracking required - executor handles everything

### Termination Mechanism

Worker processes use a flag-based termination system:

1. **Termination Flag**: Global `terminate_workers` flag signals all workers to stop
2. **Periodic Checks**: Workers check the flag during BFS layer generation
3. **Safe Exit**: Workers exit gracefully when flag is set, similar to timeout checks
4. **Pool Stats**: Termination reason is recorded as `"terminated"` in pool statistics

### Testing Signal Handling

Run the signal handling test:

```bash
python test_signals.py
```

This will test various termination signals and verify graceful shutdown.

## High-Performance Configuration

For handling thousands of concurrent requests, optimized for AMD Ryzen 9 5950X (16 cores, 32 threads, 125GB RAM):

### Quick Start (High-Performance)

```bash
# Option 1: Use the high-performance startup script
./start_high_performance.sh

# Option 2: Use environment variables
export NVGEN_UVICORN_WORKERS=8
export NVGEN_LIMIT_CONCURRENCY=4000
export NVGEN_LIMIT_MAX_REQUESTS=100000
python3 main.py

# Option 3: Use the high-performance Python script
python3 start_high_performance.py
```

### High-Performance Settings (Optimized for 50% Resource Usage)

The high-performance configuration includes:

- **Multiple Workers**: 8 uvicorn workers (50% of 16 cores)
- **Increased Limits**: 4000 concurrent connections, 100000 requests per worker
- **Optimized Timeouts**: 120s keep-alive timeout
- **Enhanced Backlog**: 8192 connection backlog
- **Resource Scaling**: 16 worker processes/threads (50% of 32 threads)
- **RAM Management**: 48% RAM limit (60GB of 125GB available)
- **Parallel BFS**: Multi-CPU BFS processing for faster variant generation

### Parallel Processing

The service now supports parallel BFS (Breadth-First Search) processing to distribute name variant generation across multiple CPUs:

#### Configuration
```bash
# Enable/disable parallel BFS processing
NVGEN_USE_PARALLEL_BFS=true

# Maximum workers for BFS operations
NVGEN_MAX_BFS_WORKERS=8
```

#### Performance Benefits
- **Multi-CPU Utilization**: Distributes BFS operations across available CPU cores
- **Faster Generation**: Parallel processing can significantly reduce generation time
- **Scalable**: Automatically adapts to available CPU resources
- **Configurable**: Can be enabled/disabled and tuned via environment variables

#### Testing Parallel Processing
```bash
# Test parallel vs sequential BFS performance
python3 test_parallel_bfs.py --name wilson

# Test with different name lengths
python3 test_parallel_bfs.py --all-lengths

# Custom timeout
python3 test_parallel_bfs.py --name christopher --timeout 120
```

### Resource Allocation

| Resource | Total Available | Service Usage | Percentage |
|----------|----------------|---------------|------------|
| CPU Cores | 16 | 8 | 50% |
| CPU Threads | 32 | 16 | 50% |
| RAM | 125GB | 60GB | 48% |
| Concurrent Connections | - | 4000 | - |
| Requests per Worker | - | 100000 | - |

## Configuration

Key configuration parameters in `main.py`:

```python
INSTANCE_TIMEOUT = 60      # 1 minute for instance pool generation
WHOLE_POOL_TIMEOUT = 600   # 10 minutes for whole pool generation
CONSUMED_TIMEOUT = 1200    # 20 minutes for consumed variant expiration
MAX_RAM_PERCENT = 80       # Maximum RAM usage before pausing generation
MAX_CONCURRENT_POOLS = 3   # Maximum concurrent pool generations
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the service:
```bash
python main.py
```

The service will start on `http://0.0.0.0:8000`

## Usage Examples

### Using curl

1. Get name variants:
```bash
curl "http://localhost:8000/pool?original_name=john"
```

2. Mark variants as consumed:
```bash
curl -X POST "http://localhost:8000/consumed?original_name=john" \
  -H "Content-Type: application/json" \
  -d '{"variants": ["joh", "jone"]}'
```

3. Check service status:
```bash
curl http://localhost:8000/status
```

### Using Python

```python
import requests

# Get variants
response = requests.get("http://localhost:8000/pool", params={"original_name": "john"})
variants = response.json()

# Mark as consumed
requests.post("http://localhost:8000/consumed", 
              params={"original_name": "john"},
              json={"variants": ["joh", "jone"]})
```

## Algorithm Details

### Phonetic Classes (P0-P7)

- **P0**: Soundex!=, Metaphone!=, NYSIIS!=
- **P1**: Soundex==, Metaphone!=, NYSIIS!=
- **P2**: Soundex!=, Metaphone==, NYSIIS!=
- **P3**: Soundex!=, Metaphone!=, NYSIIS==
- **P4**: Soundex!=, Metaphone==, NYSIIS==
- **P5**: Soundex==, Metaphone!=, NYSIIS==
- **P6**: Soundex==, Metaphone==, NYSIIS!=
- **P7**: Soundex==, Metaphone==, NYSIIS==

### Orthographic Levels

- **Level 0**: 70-100% similarity
- **Level 1**: 50-69% similarity
- **Level 2**: 20-49% similarity
- **Level 3**: 0-19% similarity

## Performance Considerations

- **RAM Monitoring**: Service automatically pauses generation when RAM usage exceeds 80%
- **Timeout Management**: Instance pools (1 min) vs whole pools (10 min)
- **Concurrent Processing**: Uses ProcessPoolExecutor for CPU-intensive work
- **Caching**: File-based cache reduces regeneration overhead
- **Queue Management**: Prevents overwhelming the system with multiple requests

## Error Handling

- Graceful handling of generation failures
- Automatic cleanup of expired consumed variants
- Resource monitoring to prevent system overload
- Comprehensive logging for debugging

## Development

The service is built with:
- **FastAPI**: Modern web framework for building APIs
- **Uvicorn**: ASGI server for running the application
- **psutil**: System and process utilities for resource monitoring
- **jellyfish**: Phonetic algorithms (Soundex, Metaphone, NYSIIS)
- **python-Levenshtein**: Fast string similarity calculations
