# Name Variant Generation Service - Implementation Summary

## Overview

This implementation provides a complete FastAPI-based service for generating name variants using a sophisticated phonetic-aware algorithm. The service is designed to handle multiple concurrent requests while managing system resources efficiently.

## Architecture Components

### 1. Core Service (`main.py`)
- **FastAPI Application**: RESTful API with automatic documentation
- **Queue Management**: Background processing of whole pool generation requests
- **Resource Monitoring**: RAM usage tracking to prevent system overload
- **Caching System**: File-based cache with JSON storage
- **Variant Tracking**: Consumed variant management with automatic expiration

### 2. Pool Generation Algorithm (`pool_generator.py`)
- **Phonetic Algorithms**: Soundex, Metaphone, and NYSIIS integration
- **Orthographic Similarity**: String similarity calculations
- **BFS Expansion**: Breadth-first search with timeout management
- **Top-K Selection**: Efficient variant selection per cell

### 3. Configuration Management (`config.py`)
- **Environment Variables**: Configurable via environment variables
- **Validation**: Configuration value validation
- **Performance Tuning**: Worker process and thread configuration

## Key Features Implemented

### ✅ Protocol Implementation
- **GET /{original_name}**: Provides name variants pool except consumed variations
- **POST /{original_name}/consumed**: Marks variants as consumed for 20 minutes

### ✅ Core Architecture Requirements
- **Queue-based Processing**: Multiple name whole pool generation requests queued and processed sequentially
- **Multi-CPU Distribution**: Uses ProcessPoolExecutor for CPU-intensive pool generation
- **RAM Monitoring**: Monitors RAM usage and stops generation if over 80%
- **File-based Caching**: Each name cached in `{original_name}.txt` files

### ✅ Timeout Management
- **Instance Pool Generation**: 60-second timeout for quick responses
- **Whole Pool Generation**: 600-second timeout for comprehensive results
- **Consumed Variant Expiration**: 20-minute timeout for consumed variants

### ✅ Resource Management
- **RAM Monitoring**: Real-time RAM usage tracking with psutil
- **Process Management**: Controlled worker processes and threads
- **Queue Management**: Prevents system overload with request queuing

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/pool?original_name={name}` | GET | Get name variants pool |
| `/consumed?original_name={name}` | POST | Mark variants as consumed |
| `/status` | GET | Service status and statistics |
| `/cache/{name}` | GET | Get raw cached pool data |

### Data Structure
The pool data is returned as a 3D array structure: `pools[ld][orth][phon]` where:
- `ld`: Length difference level (0 to max length difference)
- `orth`: Orthographic level (0-3)
- `phon`: Phonetic class (0-7)

Each cell contains an array of variant strings.

## File Structure

```
services/nvgen_service/
├── main.py                 # FastAPI application
├── pool_generator.py       # Core algorithm implementation
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── README.md             # Comprehensive documentation
├── test_service.py       # Test script
├── start.sh              # Startup script
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Deployment configuration
└── cache/                # Cache directory (created automatically)
```

## Configuration Options

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `NVGEN_HOST` | `0.0.0.0` | Service host |
| `NVGEN_PORT` | `8000` | Service port |
| `NVGEN_INSTANCE_TIMEOUT` | `60` | Instance pool timeout (seconds) |
| `NVGEN_WHOLE_POOL_TIMEOUT` | `600` | Whole pool timeout (seconds) |
| `NVGEN_CONSUMED_TIMEOUT` | `1200` | Consumed variant timeout (seconds) |
| `NVGEN_MAX_RAM_PERCENT` | `80.0` | Maximum RAM usage percentage |
| `NVGEN_WORKER_PROCESSES` | `4` | Number of worker processes |
| `NVGEN_WORKER_THREADS` | `2` | Number of worker threads |

## Pool Structure

The service generates pools organized by:

### Length Difference (ld)
- Difference in length between original name and variant

### Orthographic Level (o) - 4 levels
- **Level 0**: 70-100% similarity
- **Level 1**: 50-69% similarity  
- **Level 2**: 20-49% similarity
- **Level 3**: 0-19% similarity

### Phonetic Class (p) - 8 classes
- **P0**: Soundex!=, Metaphone!=, NYSIIS!=
- **P1**: Soundex==, Metaphone!=, NYSIIS!=
- **P2**: Soundex!=, Metaphone==, NYSIIS!=
- **P3**: Soundex!=, Metaphone!=, NYSIIS==
- **P4**: Soundex!=, Metaphone==, NYSIIS==
- **P5**: Soundex==, Metaphone!=, NYSIIS==
- **P6**: Soundex==, Metaphone==, NYSIIS!=
- **P7**: Soundex==, Metaphone==, NYSIIS==

## Performance Features

### Resource Management
- **RAM Monitoring**: Automatic pause when RAM usage exceeds 80%
- **Process Control**: Configurable worker processes and threads
- **Queue Management**: Sequential processing to prevent overload

### Caching Strategy
- **File-based Cache**: Persistent storage in JSON format
- **Memory Cache**: In-memory cache for fast access
- **Automatic Loading**: Cache restoration on service startup

### Concurrent Processing
- **Instance Pools**: Quick generation with ThreadPoolExecutor
- **Whole Pools**: CPU-intensive generation with ProcessPoolExecutor
- **Background Tasks**: Non-blocking pool generation

## Deployment Options

### 1. Direct Python Execution
```bash
./start.sh
```

### 2. Docker Container
```bash
docker build -t nvgen-service .
docker run -p 8000:8000 nvgen-service
```

### 3. Docker Compose
```bash
docker-compose up -d
```

## Testing

The service includes a comprehensive test script (`test_service.py`) that:
- Tests all API endpoints
- Verifies variant generation
- Tests consumed variant tracking
- Checks service status
- Validates caching functionality

## Monitoring and Debugging

### Status Endpoint
Provides real-time information about:
- Service status
- RAM usage
- Cache statistics
- Queue status
- Currently generating pools

### Logging
Comprehensive logging for:
- Pool generation progress
- Resource usage
- Error handling
- Performance metrics

## Security Considerations

- **Input Validation**: All inputs validated via Pydantic models
- **Resource Limits**: Configurable limits prevent DoS attacks
- **Error Handling**: Graceful error handling without information leakage
- **Timeout Protection**: Prevents hanging requests

## Scalability Features

- **Horizontal Scaling**: Stateless design allows multiple instances
- **Load Balancing**: Can be deployed behind load balancers
- **Resource Isolation**: Docker containerization for isolation
- **Configuration Management**: Environment-based configuration

## Future Enhancements

Potential improvements for production deployment:
- **Database Integration**: Replace file cache with database
- **Redis Caching**: Add Redis for distributed caching
- **Metrics Collection**: Prometheus/Grafana integration
- **Authentication**: API key or JWT authentication
- **Rate Limiting**: Request rate limiting
- **Health Checks**: Enhanced health check endpoints
