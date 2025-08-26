#!/usr/bin/env python3

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
    CACHE_DIR, INSTANCE_TIMEOUT, WHOLE_POOL_TIMEOUT,
    MAX_RAM_PERCENT, BUCKET_K, WORKER_PROCESSES,
    WORKER_THREADS, USE_PARALLEL_BFS, MAX_BFS_WORKERS
)

# Global state
consumed_variants: Dict[str, float] = {}  # variant -> expiry timestamp
stop_generation: bool = False
is_generating_instance: Dict[str, bool] = {}  # name -> is currently generating instance pool
is_generating_whole: Dict[str, bool] = {}     # name -> is currently generating whole pool
pending_requests: Dict[str, asyncio.Event] = {}
shutdown_event: Optional[asyncio.Event] = None
terminate_workers: bool = False  # Flag to signal worker processes to terminate

def get_ram_usage() -> float:
    """Get current RAM usage percentage"""
    return psutil.virtual_memory().percent


def get_cache_file(name: str) -> Path:
    """Get cache file path for a name"""
    return CACHE_DIR / f"{name}.txt"


def is_ram_over_limit() -> bool:
    """Check if RAM usage is over the limit"""
    return get_ram_usage() > MAX_RAM_PERCENT


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

async def pool_generation_worker():
    """Background worker that generates whole pools for names needing them"""
    print("Pool generation worker started")
    
    while not stop_generation:
        try:
            # Scan cache files for names needing whole pool generation
            for cache_file in CACHE_DIR.glob("*.txt"):
                if stop_generation:
                    break
                
                name = cache_file.stem  # Get filename without extension
                
                if not has_whole_pool_cache(name):
                    if is_ram_over_limit():
                        print(f"RAM usage too high ({get_ram_usage():.1f}%), pausing whole pool generation")
                        await asyncio.sleep(30)  # Wait and try again
                        break
                    
                    # Generate whole pool
                    await generate_whole_pool(name)
            
            # Sleep before next scan (with shutdown check)
            for _ in range(10):  # 10 seconds total
                if stop_generation :
                    break
                await asyncio.sleep(1)
            
        except Exception as e:
            print(f"Error in pool generation worker: {e}")
            await asyncio.sleep(5)
    
    print("Pool generation worker stopped")


if __name__ == "__main__":
    asyncio.run(pool_generation_worker())