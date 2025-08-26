#!/usr/bin/env python3
"""
Name Variant Generation Service

This service provides name variants to miners' name variant pool requests.
- GET /{original_name}: Provides name variants pool except for already consumed variations
- POST /{list of variants}: Marks variants as consumed for 20 minutes

Architecture:
- Queue-based whole pool generation with timeout management
- Multi-CPU pool generation with RAM monitoring
- File-based caching for each name
- Concurrent instance pool generation with whole pool background generation
"""

import asyncio
import json
import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Callable
from uuid import uuid4

import psutil
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

# Import the existing pool generation algorithm
from pool_generator import expand_fixed_radius

# Import configuration
from config import (
    CACHE_DIR, INSTANCE_TIMEOUT, WHOLE_POOL_TIMEOUT, CONSUMED_TIMEOUT,
    MAX_RAM_PERCENT, MAX_CONCURRENT_POOLS, BUCKET_K, WORKER_PROCESSES,
    WORKER_THREADS, API_TITLE, API_VERSION, API_DESCRIPTION,
    HOST, PORT, UVICORN_WORKERS, UVICORN_LIMIT_CONCURRENCY, 
    UVICORN_LIMIT_MAX_REQUESTS, UVICORN_TIMEOUT_KEEP_ALIVE, UVICORN_BACKLOG,
    USE_PARALLEL_BFS, MAX_BFS_WORKERS
)

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION
)

# Global state
consumed_variants: Dict[str, float] = {}  # variant -> expiry timestamp
stop_generation: bool = False
is_generating_instance: Dict[str, bool] = {}  # name -> is currently generating instance pool
is_generating_whole: Dict[str, bool] = {}     # name -> is currently generating whole pool
pending_requests: Dict[str, asyncio.Event] = {}
shutdown_requested: bool = False
shutdown_event: Optional[asyncio.Event] = None
terminate_workers: bool = False  # Flag to signal worker processes to terminate

def signal_handler(signum, frame):
    """Handle termination signals gracefully"""
    global shutdown_requested, stop_generation, terminate_workers
    
    signal_name = {
        signal.SIGINT: "SIGINT (Ctrl+C)",
        signal.SIGTERM: "SIGTERM",
        signal.SIGTSTP: "SIGTSTP (Ctrl+Z)",
        signal.SIGQUIT: "SIGQUIT",
        signal.SIGHUP: "SIGHUP"
    }.get(signum, f"Signal {signum}")
    
    print(f"\n🛑 Received {signal_name}, initiating graceful shutdown...")
    
    # Set shutdown flags
    shutdown_requested = True
    stop_generation = True
    terminate_workers = True  # Signal worker processes to terminate
    
    # Note: ProcessPoolExecutor automatically handles worker process cleanup
    # when the context exits, so we don't need to manually terminate processes
    
    # Signal the shutdown event if it exists
    if shutdown_event and not shutdown_event.is_set():
        shutdown_event.set()
    
    # For SIGTSTP (Ctrl+Z), we might want to suspend instead of shutdown
    if signum == signal.SIGTSTP:
        print("⚠️  Received SIGTSTP (Ctrl+Z) - suspending generation but keeping service alive")
        stop_generation = True
        terminate_workers = True
        return
    
    print("🔄 Waiting for active operations to complete...")

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill command
    signal.signal(signal.SIGTSTP, signal_handler)  # Ctrl+Z
    signal.signal(signal.SIGQUIT, signal_handler)  # Ctrl+\
    signal.signal(signal.SIGHUP, signal_handler)   # Terminal hangup
    
    print("✅ Signal handlers configured for graceful shutdown")
    print("   Ctrl+C, Ctrl+Z, SIGTERM, SIGQUIT, SIGHUP will be handled gracefully")

def cleanup_on_shutdown():
    """Perform cleanup operations before shutdown"""
    print("\n🧹 Performing cleanup operations...")
    
    # Stop all generation processes
    global stop_generation
    stop_generation = True
    
    # Note: ProcessPoolExecutor automatically handles worker process cleanup
    # when the context exits, so we don't need to manually terminate processes
    
    # Wait for any ongoing generation to complete
    active_generations = [name for name, is_gen in is_generating_instance.items() if is_gen] + \
                         [name for name, is_gen in is_generating_whole.items() if is_gen]
    if active_generations:
        print(f"   Waiting for {len(active_generations)} active generations to complete...")
        print(f"   Active: {', '.join(active_generations)}")
    
    # Signal waiting requests
    for name, event in pending_requests.items():
        if not event.is_set():
            event.set()
    
    print("   ✅ Cleanup completed")

class ConsumedVariantsRequest(BaseModel):
    variants: List[str]

def get_ram_usage() -> float:
    """Get current RAM usage percentage"""
    return psutil.virtual_memory().percent

def is_ram_over_limit() -> bool:
    """Check if RAM usage is over the limit"""
    return get_ram_usage() > MAX_RAM_PERCENT

def get_cache_file(name: str) -> Path:
    """Get cache file path for a name"""
    return CACHE_DIR / f"{name}.txt"

def load_pool_from_cache(name: str) -> Optional[Dict]:
    """Load pool data from cache file"""
    cache_file = get_cache_file(name)
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    except Exception as e:
        print(f"Error loading cache for {name}: {e}")
        return None

def save_pool_to_cache(name: str, pools: Dict, stats: Dict, pool_type: str = "instance", timeout: Optional[float] = None):
    """Save pool data to cache file, appending to existing data if present"""
    cache_file = get_cache_file(name)
    try:
        # Load existing data if present
        existing_data = {}
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except Exception as e:
                print(f"Error loading existing cache for {name}: {e}")
                existing_data = {}
        
        # Update with new pool data
        data = {
            "name": name,
            "instance_pool": existing_data.get("instance_pool"),
            "whole_pool": existing_data.get("whole_pool"),
            "instance_stats": existing_data.get("instance_stats"),
            "whole_stats": existing_data.get("whole_stats"),
            "instance_timestamp": existing_data.get("instance_timestamp"),
            "whole_timestamp": existing_data.get("whole_timestamp"),
            "timestamp": time.time(),
            "timeout": timeout
        }
        
        # Set the new pool data
        if pool_type == "instance":
            data["instance_pool"] = pools
            data["instance_stats"] = stats
            data["instance_timestamp"] = time.time()
        elif pool_type == "whole":
            data["whole_pool"] = pools
            data["whole_stats"] = stats
            data["whole_timestamp"] = time.time()
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving cache for {name}: {e}")

def get_pool_type_from_cache(name: str) -> Optional[str]:
    """Get the type of pool cached for a name (deprecated - use specific functions)"""
    data = load_pool_from_cache(name)
    if data:
        if data.get("whole_pool"):
            return "whole"
        elif data.get("instance_pool"):
            return "instance"
    return None

def has_whole_pool_cache(name: str) -> bool:
    """Check if whole pool is cached for a name"""
    data = load_pool_from_cache(name)
    return data is not None and data.get("whole_pool") is not None

def has_instance_pool_cache(name: str) -> bool:
    """Check if instance pool is cached for a name"""
    data = load_pool_from_cache(name)
    return data is not None and data.get("instance_pool") is not None

def get_whole_pool_from_cache(name: str) -> Optional[Dict]:
    """Get whole pool data from cache"""
    data = load_pool_from_cache(name)
    if data and data.get("whole_pool"):
        return data["whole_pool"]
    return None

def get_instance_pool_from_cache(name: str) -> Optional[Dict]:
    """Get instance pool data from cache"""
    data = load_pool_from_cache(name)
    if data and data.get("instance_pool"):
        return data["instance_pool"]
    return None

def filter_consumed_variants(pools: List, consumed: Set[str]) -> List:
    """Filter out consumed variants from pools"""
    filtered_pools = []
    
    for ld_level in pools:
        ld_filtered = []
        for orth_level in ld_level:
            orth_filtered = []
            for variants in orth_level:
                # Handle different data types
                if isinstance(variants, str):
                    # Single string variant
                    variant_set = {variants} if variants not in consumed else set()
                elif isinstance(variants, list):
                    # List of variants
                    variant_set = set(variants)
                elif isinstance(variants, set):
                    # Set of variants
                    variant_set = variants
                else:
                    # Unknown type, skip
                    variant_set = set()
                
                filtered_variants = variant_set - consumed
                orth_filtered.append(filtered_variants)
            ld_filtered.append(orth_filtered)
        filtered_pools.append(ld_filtered)
    
    return filtered_pools

def cleanup_expired_consumed():
    """Clean up expired consumed variants"""
    current_time = time.time()
    expired = [variant for variant, expiry in consumed_variants.items() 
               if current_time > expiry]
    
    for variant in expired:
        del consumed_variants[variant]

def check_termination_flag() -> bool:
    """Check if worker processes should terminate"""
    return terminate_workers

def generate_pool_worker_wrapper(name: str, timeout_seconds: float) -> Tuple[List, Dict]:
    """Wrapper for pool generation worker that handles process termination"""
    import os
    
    # Get current process PID
    current_pid = os.getpid()
    
    try:
        # Call the actual worker function with termination check
        result = generate_pool_worker(name, timeout_seconds, check_termination_flag)
        print(f"Worker process {current_pid} completed successfully")
        return result
    except Exception as e:
        print(f"Error in worker process {current_pid}: {e}")
        return [], {"error": str(e)}

def generate_pool_worker(name: str, timeout_seconds: float, check_terminate: Optional[Callable] = None) -> Tuple[List, Dict]:
    """Worker function for pool generation (runs in separate process)"""
    try:
        # Check if we should stop generation
        if check_terminate and check_terminate():
            return [], {"error": "Generation stopped by user request"}
        
        pools, stats = expand_fixed_radius(
            name=name,
            timeout_seconds=timeout_seconds,
            bucket_k=BUCKET_K,
            check_terminate=check_terminate,
            use_parallel=USE_PARALLEL_BFS,  # Use configuration setting
            max_workers=MAX_BFS_WORKERS  # Use configured max workers
        )
        
        # Check again after generation
        if check_terminate and check_terminate():
            return [], {"error": "Generation stopped by user request"}
        
        # Convert sets to lists for JSON serialization
        serializable_pools = []
        for ld_level in pools:
            ld_serializable = []
            for orth_level in ld_level:
                orth_serializable = []
                for variants in orth_level:
                    orth_serializable.append(list(variants))
                ld_serializable.append(orth_serializable)
            serializable_pools.append(ld_serializable)
        
        return serializable_pools, stats
    except Exception as e:
        print(f"Error generating pool for {name}: {e}")
        return [], {"error": str(e)}

async def generate_whole_pool(name: str):
    """Generate whole pool with timeout and RAM monitoring"""
    if is_generating_whole.get(name, False):
        return
    
    is_generating_whole[name] = True
    
    try:
        # Check if we should stop generation
        if stop_generation:
            print(f"Generation stopped for {name}")
            return
        
        # Check RAM usage before starting
        if is_ram_over_limit():
            print(f"RAM usage too high ({get_ram_usage():.1f}%), skipping whole pool generation for {name}")
            return
        
        print(f"Starting whole pool generation for {name}")
        
        # Use ProcessPoolExecutor for CPU-intensive work
        with ProcessPoolExecutor(max_workers=min(os.cpu_count(), WORKER_PROCESSES)) as executor:
            loop = asyncio.get_event_loop()
            
            # Submit the task and get the future
            future = loop.run_in_executor(
                executor, 
                generate_pool_worker_wrapper, 
                name, 
                WHOLE_POOL_TIMEOUT
            )
            
            # Note: ProcessPoolExecutor automatically manages worker processes
            # The executor will handle process cleanup when the context exits
            # We don't need to manually track PIDs as the executor handles this
            
            # Wait for the result
            pools, stats = await future
        
        # Check if we should stop before saving
        if stop_generation:
            print(f"Generation stopped for {name} before saving")
            return
        
        if pools:
            # Save to cache
            save_pool_to_cache(name, pools, stats, "whole", WHOLE_POOL_TIMEOUT)
            
            print(f"Completed whole pool generation for {name}: {stats.get('variants_total', 0)} variants")
        # Print pool statistics in tabular format
        if pools and stats.get('variants_total', 0) > 0:
            try:
                from tabulate import tabulate
                
                # Get the maximum levels for iteration
                max_ld = len(pools) - 1 if pools else 0
                max_orth = 3  # Fixed at 4 levels (0-3)
                max_phon = 7  # Fixed at 8 levels (0-7)
                
                # Print tables for each Levenshtein distance level
                for ld in range(len(pools)):
                    if ld < len(pools) and any(any(len(variants) > 0 for variants in orth_level) for orth_level in pools[ld]):
                        print(f"\nname={name}, level={ld}")
                        print("     " + "-" * (9 * (max_phon + 1) + 4))
                        
                        # Data rows
                        table_data = []
                        for orth in range(4):  # Fixed 4 orthographic levels
                            row = [f"O{orth}"]
                            
                            for phon in range(8):  # Fixed 8 phonetic levels
                                count = 0
                                if ld < len(pools) and orth < len(pools[ld]) and phon < len(pools[ld][orth]):
                                    count = len(pools[ld][orth][phon])
                                row.append(str(count) if count > 0 else "")
                            
                            table_data.append(row)
                        
                        # Print table using tabulate
                        print(tabulate(table_data, headers=f"\nname={name}, level={ld}", tablefmt="grid", stralign="center"))
                        
            except ImportError:
                print("tabulate library not available, skipping table display")
            except Exception as e:
                print(f"Error displaying pool table: {e}")
        else:
            print(f"Whole pool generation failed for {name}")
            
    except Exception as e:
        print(f"Error in whole pool generation for {name}: {e}")
    finally:
        is_generating_whole[name] = False

async def generate_instance_pool(name: str, timeout: Optional[float] = None) -> Dict:
    """Generate instance pool with short timeout"""
    try:
        print(f"Generating instance pool for {name}")
        
        timeout = timeout if timeout is not None else INSTANCE_TIMEOUT
        # Check if we should stop generation
        if stop_generation:
            print(f"Generation stopped for {name}")
            return {}
        
        # Use ThreadPoolExecutor for quick instance generation
        with ThreadPoolExecutor(max_workers=WORKER_THREADS) as executor:
            loop = asyncio.get_event_loop()
            pools, stats = await loop.run_in_executor(
                executor,
                generate_pool_worker_wrapper,
                name,
                timeout
            )
        
        # Check if we should stop before saving
        if stop_generation:
            print(f"Generation stopped for {name} before saving")
            return {}
        
        if pools:
            # Save to cache
            save_pool_to_cache(name, pools, stats, "instance", timeout)
            
            try:
                print(json.dumps(stats, indent=2))
                from tabulate import tabulate
                
                # Get the maximum levels for iteration
                max_ld = len(pools) - 1 if pools else 0
                max_orth = 3  # Fixed at 4 levels (0-3)
                max_phon = 7  # Fixed at 8 levels (0-7)
                
                # Print tables for each Levenshtein distance level
                for ld in range(len(pools)):
                    if ld < len(pools) and any(any(len(variants) > 0 for variants in orth_level) for orth_level in pools[ld]):
                        print("     " + "-" * (9 * (max_phon + 1) + 4))
                        
                        # Data rows
                        table_data = []
                        for orth in range(4):  # Fixed 4 orthographic levels
                            row = [f"O{orth}"]
                            
                            for phon in range(8):  # Fixed 8 phonetic levels
                                count = 0
                                if ld < len(pools) and orth < len(pools[ld]) and phon < len(pools[ld][orth]):
                                    count = len(pools[ld][orth][phon])
                                row.append(str(count) if count > 0 else "")
                            
                            table_data.append(row)
                        
                        # Print table using tabulate
                        print(tabulate(table_data, headers=f"\nname={name}, level={ld}", tablefmt="grid", stralign="center"))
                        
            except ImportError:
                print("tabulate library not available, skipping table display")
            except Exception as e:
                print(f"Error displaying pool table: {e}")
        
        return pools
    except Exception as e:
        print(f"Error generating instance pool for {name}: {e}")
        return {}

async def pool_generation_worker():
    """Background worker that generates whole pools for names needing them"""
    print("Pool generation worker started")
    
    while not stop_generation and not shutdown_requested:
        try:
            # Check for shutdown event
            if shutdown_event and shutdown_event.is_set():
                print("🛑 Shutdown event received, stopping pool generation worker")
                break
            
            # Scan cache files for names needing whole pool generation
            for cache_file in CACHE_DIR.glob("*.txt"):
                if stop_generation or shutdown_requested:
                    break
                
                name = cache_file.stem  # Get filename without extension
                
                # Check if this name needs whole pool generation
                if has_instance_pool_cache(name) and not has_whole_pool_cache(name):
                    # Check if already generating whole pool
                    if is_generating_whole.get(name, False):
                        continue
                    
                    # Check RAM usage
                    if is_ram_over_limit():
                        print(f"RAM usage too high ({get_ram_usage():.1f}%), pausing whole pool generation")
                        await asyncio.sleep(30)  # Wait and try again
                        break
                    
                    # Generate whole pool
                    await generate_whole_pool(name)
            
            # Sleep before next scan (with shutdown check)
            for _ in range(10):  # 10 seconds total
                if stop_generation or shutdown_requested:
                    break
                await asyncio.sleep(1)
            
        except Exception as e:
            print(f"Error in pool generation worker: {e}")
            await asyncio.sleep(5)
    
    print("Pool generation worker stopped")

@app.on_event("startup")
async def startup_event():
    """Initialize background tasks and signal handlers"""
    global shutdown_event
    
    # Setup signal handlers for graceful shutdown
    setup_signal_handlers()
    
    # Create shutdown event
    shutdown_event = asyncio.Event()
    
    # asyncio.create_task(pool_generation_worker())
    
    # Create cache directory if it doesn't exist
    CACHE_DIR.mkdir(exist_ok=True)
    
    print("🚀 Service started successfully")
    print(f"   API available at: http://{HOST}:{PORT}")
    print(f"   Cache directory: {CACHE_DIR.absolute()}")
    print(f"   RAM limit: {MAX_RAM_PERCENT}%")
    print(f"   Instance timeout: {INSTANCE_TIMEOUT}s")
    print(f"   Whole pool timeout: {WHOLE_POOL_TIMEOUT}s")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on service shutdown"""
    print("\n🛑 Service shutdown initiated...")
    cleanup_on_shutdown()
    print("👋 Service shutdown completed")

@app.get("/status")
async def get_status():
    """Get service status and statistics"""
    cleanup_expired_consumed()
    
    # Count cache files
    cache_files = list(CACHE_DIR.glob("*.txt"))
    cached_names = len(cache_files)
    
    # Get currently generating names for both types
    generating_instance = [name for name, is_gen in is_generating_instance.items() if is_gen]
    generating_whole = [name for name, is_gen in is_generating_whole.items() if is_gen]
    
    return {
        "status": "running" if not shutdown_requested else "shutting_down",
        "ram_usage_percent": get_ram_usage(),
        "cached_names": cached_names,
        "consumed_variants": len(consumed_variants),
        "currently_generating_instance": generating_instance,
        "currently_generating_whole": generating_whole,
        "pending_requests": len(pending_requests),
        "cache_directory": str(CACHE_DIR.absolute()),
        "shutdown_requested": shutdown_requested,
        "generation_stopped": stop_generation
    }

@app.get("/pool")
async def get_name_variants(original_name: str, timeout: Optional[float] = None, background_tasks: BackgroundTasks = None):
    """
    GET /pool?original_name={name}&timeout={seconds}: Provide name variants pool except for already consumed variations
    
    Check if whole pool is cached first, then instance pool, then generate instance pool if needed.
    If timeout parameter is specified, use it for instance pool generation instead of default timeout.
    Handles concurrent requests by making other requests wait for the first one to complete.
    """
    # Clean up expired consumed variants
    cleanup_expired_consumed()
    
    # Get consumed variants for this name
    consumed = {variant for variant, expiry in consumed_variants.items() 
                if time.time() <= expiry}
    
    # Use specified timeout or default
    instance_timeout = timeout if timeout is not None else INSTANCE_TIMEOUT
    
    # Check if we have a whole pool cached
    if has_whole_pool_cache(original_name):
        cached_data = get_whole_pool_from_cache(original_name)
        if cached_data:
            # Filter out consumed variants
            filtered_pools = filter_consumed_variants(cached_data, consumed)
            
            return {
                "name": original_name,
                "pools": filtered_pools,
                "source": "whole_pool_cache",
                "consumed_count": len(consumed)
            }
    
    # Check if we have an instance pool cached
    if has_instance_pool_cache(original_name):
        cached_data = get_instance_pool_from_cache(original_name)
        if cached_data:
            # Filter out consumed variants
            filtered_pools = filter_consumed_variants(cached_data, consumed)
            
            return {
                "name": original_name,
                "pools": filtered_pools,
                "source": "instance_pool_cache",
                "consumed_count": len(consumed)
            }
    
    # No cached pool, handle concurrent requests
    if is_generating_instance.get(original_name, False):
        # Another request is already generating instance pool for this name, wait for it
        if original_name not in pending_requests:
            pending_requests[original_name] = asyncio.Event()
        
        # Wait for the generation to complete (with timeout)
        try:
            await asyncio.wait_for(pending_requests[original_name].wait(), timeout=instance_timeout)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Request timeout waiting for pool generation")
        
        # Check again after waiting
        if has_whole_pool_cache(original_name):
            cached_data = get_whole_pool_from_cache(original_name)
            if cached_data:
                filtered_pools = filter_consumed_variants(cached_data, consumed)
                return {
                    "name": original_name,
                    "pools": filtered_pools,
                    "source": "whole_pool_cache",
                    "consumed_count": len(consumed)
                }
        
        if has_instance_pool_cache(original_name):
            cached_data = get_instance_pool_from_cache(original_name)
            if cached_data:
                filtered_pools = filter_consumed_variants(cached_data, consumed)
                return {
                    "name": original_name,
                    "pools": filtered_pools,
                    "source": "instance_pool_cache",
                    "consumed_count": len(consumed)
                }
    
    # No cached pool and no ongoing instance generation, start generating
    is_generating_instance[original_name] = True
    
    try:
        # Create event for other requests to wait on
        if original_name not in pending_requests:
            pending_requests[original_name] = asyncio.Event()
        
        # Generate instance pool with specified timeout
        instance_pools = await generate_instance_pool(original_name, timeout=instance_timeout)

        if instance_pools:
            # Filter out consumed variants
            filtered_pools = filter_consumed_variants(instance_pools, consumed)
            
            # Signal waiting requests
            pending_requests[original_name].set()
            
            return {
                "name": original_name,
                "pools": filtered_pools,
                "source": "instance_generated",
                "consumed_count": len(consumed)
            }
        else:
            # Signal waiting requests even if failed
            pending_requests[original_name].set()
            raise HTTPException(status_code=500, detail="Failed to generate name variants")
    
    finally:
        # Clean up
        is_generating_instance[original_name] = False
        if original_name in pending_requests:
            del pending_requests[original_name]

@app.post("/consumed")
async def mark_variants_consumed(original_name: str, request: ConsumedVariantsRequest):
    """
    POST /consumed?original_name={name}: Mark variants as consumed for 20 minutes
    
    The list of variants which is already consumed by requester, so you need to flag 
    these variants not to send other pool requesters requesting the same name. 
    This flag will be released after 20 min.
    """
    current_time = time.time()
    expiry_time = current_time + CONSUMED_TIMEOUT
    
    # Mark variants as consumed
    for variant in request.variants:
        consumed_variants[variant] = expiry_time
    
    return {
        "message": f"Marked {len(request.variants)} variants as consumed",
        "expires_at": datetime.fromtimestamp(expiry_time).isoformat(),
        "consumed_count": len(consumed_variants)
    }

@app.post("/pool")
async def request_whole_pool(name: str):
    """
    POST /pool?name={name}: Request whole pool generation
    
    This creates a cache file with instance_pool if it doesn't exist,
    which will trigger the pool generation worker to generate a whole pool.
    """
    # Check if we already have a cache file
    if not has_whole_pool_cache(name):
        # Create a minimal cache file to trigger whole pool generation
        save_pool_to_cache(name, [], {"message": "placeholder"}, "instance", timeout = 0)
    
    return {
        "message": f"Whole pool generation requested for {name}",
        "name": name,
        "status": "queued"
    }

@app.post("/shutdown")
async def shutdown_service():
    """
    POST /shutdown: Safely stop the service
    
    This sets the stop flag and waits for ongoing operations to complete.
    """
    global stop_generation
    stop_generation = True
    
    return {
        "message": "Shutdown initiated. Service will stop after completing ongoing operations.",
        "status": "shutting_down"
    }

@app.post("/stop-generation")
async def stop_pool_generation():
    """
    POST /stop-generation: Stop pool generation but keep service running
    
    This stops new pool generation but keeps the service running for API requests.
    """
    global stop_generation
    stop_generation = True
    
    return {
        "message": "Pool generation stopped. Service remains running for API requests.",
        "status": "generation_stopped"
    }

if __name__ == "__main__":
    try:
        print("🚀 Starting Name Variant Generation Service...")
        print(f"   Host: {HOST}")
        print(f"   Port: {PORT}")
        print(f"   Cache directory: {CACHE_DIR.absolute()}")
        print(f"   RAM limit: {MAX_RAM_PERCENT}%")
        print(f"   Instance timeout: {INSTANCE_TIMEOUT}s")
        print(f"   Whole pool timeout: {WHOLE_POOL_TIMEOUT}s")
        print("\n📋 Signal handling enabled:")
        print("   Ctrl+C (SIGINT)  - Graceful shutdown")
        print("   Ctrl+Z (SIGTSTP) - Suspend generation")
        print("   SIGTERM          - Graceful shutdown")
        print("   SIGQUIT          - Graceful shutdown")
        print("   SIGHUP           - Graceful shutdown")
        print("\n" + "="*60)
        
        # Setup signal handlers before starting uvicorn
        setup_signal_handlers()
        
        # Start the server
        uvicorn.run(
            app, 
            host=HOST, 
            port=PORT,
            log_level="info",
            workers=UVICORN_WORKERS,  # Single worker process (for shared state)
            loop="asyncio",
            # Increase connection limits
            limit_concurrency=UVICORN_LIMIT_CONCURRENCY,  # Maximum concurrent connections
            limit_max_requests=UVICORN_LIMIT_MAX_REQUESTS,  # Maximum requests per worker
            timeout_keep_alive=UVICORN_TIMEOUT_KEEP_ALIVE,  # Keep-alive timeout
            access_log=True,
            # Performance optimizations
            backlog=UVICORN_BACKLOG,  # Connection backlog
        )
        
    except KeyboardInterrupt:
        print("\n🛑 KeyboardInterrupt received, shutting down gracefully...")
        cleanup_on_shutdown()
        print("👋 Service stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error starting service: {e}")
        cleanup_on_shutdown()
        sys.exit(1)
