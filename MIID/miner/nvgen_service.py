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
from asyncio.tasks import Task
import json
import sys
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Callable
import hashlib
from contextlib import asynccontextmanager
import re

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

# Suppress gRPC warnings
os.environ['GRPC_PYTHON_LOG_LEVEL'] = 'error'
os.environ['ABSL_LOGGING_MIN_LEVEL'] = '2'

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

HOST = "0.0.0.0"
PORT = 8000




async def test():
    query_files = [
        # "/work/54/miners/pub54-2/net54_uid192/netuid54/miner/validator_59/run_2025-08-24-09-07/query.json",
        # "/work/54/miners/pub54-2/net54_uid192/netuid54/miner/validator_59/run_2025-08-24-11-03/query.json",
        # "/work/54/miners/pub54-2/net54_uid192/netuid54/miner/validator_59/run_2025-08-24-10-54/query.json",
        "/work/54/tasks_fvs/1.json",
        "/work/54/tasks_fvs/1.json",
        "/work/54/tasks_fvs/1.json",
        "/work/54/tasks_fvs/1.json",
        "/work/54/tasks_fvs/1.json",
        "/work/54/tasks_fvs/1.json",
        "/work/54/tasks_fvs/1.json",
        "/work/54/tasks_fvs/1.json",
        "/work/54/tasks_fvs/1.json",
        "/work/54/tasks_fvs/1.json",
        "/work/54/tasks_fvs/1.json",
        "/work/54/tasks_fvs/1.json",
        "/work/54/tasks_fvs/1.json",
        "/work/54/tasks_fvs/1.json",
        "/work/54/tasks_fvs/1.json",
        "/work/54/tasks_fvs/1.json",
        "/work/54/tasks_fvs/1.json",
        "/work/54/tasks_fvs/1.json",
        "/work/54/tasks_fvs/1.json",
        "/work/54/tasks_fvs/1.json",
    ]
    query_datas = []
    for i in range(len(query_files)):
        with open(query_files[i], 'r') as f:
            query_data = json.load(f)
        query_datas.append(query_data)
    tasks = []

    for i, query_data in enumerate(query_datas):
        task = TaskRequest(
            names=query_data['names'],
            query_template=query_data.get('query_template', None),
            query_params=query_data.get('query_params', None),
            timeout=700.0)
        tasks.append(solve_task(task))
    results = await asyncio.gather(*tasks)
    out = json.dumps(results, indent=4)
    with open("results.json", "w") as f:
        f.write(out)


# Global state
pending_requests: Dict[str, asyncio.Event] = {}
answer_candidate_cache = {}

# Query parsing system
query_queue = asyncio.Queue()
parsed_query_cache = {}
query_processing_events = {}
worker_task = None
worker_running = False

class QueryParseRequest(BaseModel):
    query_text: str
    max_retries: Optional[int] = 10

class QueryParseResponse(BaseModel):
    query_text: str
    parsed_params: Dict
    status: str

def make_query_cache_key(query_text: str) -> str:
    """Create a cache key for query text"""
    return hashlib.sha256(query_text.encode()).hexdigest()[:16]

async def query_parse_worker():
    """Background worker that processes query parsing requests"""
    global worker_running
    worker_running = True
    print("🔧 Query parse worker started")
    
    while worker_running:
        try:
            # Get request from queue with timeout
            request = await query_queue.get()
            
            query_text = request["query_text"]
            event = request["event"]
            cache_key = make_query_cache_key(query_text)
            
            print(f"📝 Processing query parse request: {cache_key[:8]}...")
            
            try:
                # Parse the query using Gemini API
                from MIID.miner.parse_query_gemini_safe import query_parser
                parsed_params = await query_parser(query_text, max_retries=request.get("max_retries", 10))
                
                # Cache the result
                parsed_query_cache[cache_key] = {
                    "query_text": query_text,
                    "parsed_params": parsed_params,
                    "timestamp": asyncio.get_event_loop().time()
                }
                
                print(f"✅ Query parsed successfully: {cache_key[:8]}...")
                
            except Exception as e:
                print(f"❌ Query parsing failed: {e}")
                # Cache error result to avoid repeated failures
                parsed_query_cache[cache_key] = {
                    "query_text": query_text,
                    "parsed_params": None,
                    "error": str(e),
                    "timestamp": asyncio.get_event_loop().time()
                }
            
            finally:
                # Signal completion
                event.set()
                query_queue.task_done()
                
        except asyncio.TimeoutError:
            # No requests in queue, continue loop
            continue
        except Exception as e:
            print(f"❌ Worker error: {e}")
            continue
    
    print("🔧 Query parse worker stopped")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager"""
    global worker_task
    
    # Startup
    print("🚀 Starting nvgen service...")
    worker_task = asyncio.create_task(query_parse_worker())
    
    # Start async task after service is ready
    async def startup_task():
        # Wait a bit for the service to be fully ready
        await asyncio.sleep(2)
        print("🎯 Running startup async task...")
        try:
            # await test()
            print("✅ Startup task completed successfully")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ Startup task failed: {e}")
    
    # Run startup task in background
    startup_task_handle = asyncio.create_task(startup_task())
    
    yield
    
    # Shutdown
    print("🛑 Shutting down nvgen service...")
    global worker_running
    worker_running = False
    
    # Cancel startup task if still running
    if not startup_task_handle.done():
        startup_task_handle.cancel()
        try:
            await startup_task_handle
        except asyncio.CancelledError:
            pass
    
    if worker_task:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass



app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan
)

def run_single_generation(args):
    """Helper function for multiprocessing that generates variations for a single name"""
    i, name, query_params, timeout, key = args
    print(f"Running for [{key}] {i+1}. {name}")
    from MIID.miner.generate_name_variations import generate_name_variations
    return generate_name_variations(
        original_name=name,
        query_params=query_params,
        timeout=timeout,
        key=key
    )

def make_key(names: List[str], query_template: str) -> str:
    import hashlib
    return hashlib.sha256(str(names).encode() + query_template.encode()).hexdigest()[:6]

async def get_or_parse_query(query_text: str, max_retries: int = 10) -> Dict:
    """Get parsed query from cache or queue for parsing"""
    cache_key = make_query_cache_key(query_text)
    
    # Check cache first
    if cache_key in parsed_query_cache:
        cached_result = parsed_query_cache[cache_key]
        if cached_result.get("parsed_params") is not None:
            print(f"📋 Cache hit for query: {cache_key[:8]}...")
            return cached_result["parsed_params"]
        elif cached_result.get("error"):
            print(f"❌ Cache hit with error for query: {cache_key[:8]}...")
            raise HTTPException(status_code=400, detail=f"Query parsing failed: {cached_result['error']}")
    
    # Check if already being processed
    if cache_key in query_processing_events:
        print(f"⏳ Query already being processed: {cache_key[:8]}...")
        event = query_processing_events[cache_key]
        await event.wait()
        
        # Check cache again after waiting
        if cache_key in parsed_query_cache:
            cached_result = parsed_query_cache[cache_key]
            if cached_result.get("parsed_params") is not None:
                return cached_result["parsed_params"]
            elif cached_result.get("error"):
                raise HTTPException(status_code=400, detail=f"Query parsing failed: {cached_result['error']}")
    
    # Create new processing event
    event = asyncio.Event()
    query_processing_events[cache_key] = event
    
    # Add to queue
    await query_queue.put({
        "query_text": query_text,
        "event": event,
        "max_retries": max_retries
    })
    
    print(f"📝 Queued query for parsing: {cache_key[:8]}...")
    
    # Wait for completion
    await event.wait()
    
    # Clean up event
    del query_processing_events[cache_key]
    
    # Return result
    if cache_key in parsed_query_cache:
        cached_result = parsed_query_cache[cache_key]
        if cached_result.get("parsed_params") is not None:
            return cached_result["parsed_params"]
        elif cached_result.get("error"):
            raise HTTPException(status_code=400, detail=f"Query parsing failed: {cached_result['error']}")
    
    raise HTTPException(status_code=500, detail="Query parsing failed")

@app.post("/parse_query")
async def parse_query_endpoint(request: QueryParseRequest) -> QueryParseResponse:
    """Parse a query template using the background worker"""
    try:
        parsed_params = await get_or_parse_query(request.query_text, request.max_retries)
        return QueryParseResponse(
            query_text=request.query_text,
            parsed_params=parsed_params,
            status="success"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/parse_query/{query_hash}")
async def get_parsed_query(query_hash: str) -> QueryParseResponse:
    """Get a parsed query from cache by its hash"""
    if query_hash in parsed_query_cache:
        cached_result = parsed_query_cache[query_hash]
        return QueryParseResponse(
            query_text=cached_result["query_text"],
            parsed_params=cached_result["parsed_params"],
            status="cached" if cached_result.get("parsed_params") else "error"
        )
    else:
        raise HTTPException(status_code=404, detail="Query not found in cache")

@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    total_queries = len(parsed_query_cache)
    successful_parses = sum(1 for result in parsed_query_cache.values() if result.get("parsed_params") is not None)
    failed_parses = sum(1 for result in parsed_query_cache.values() if result.get("error"))
    
    return {
        "total_queries": total_queries,
        "successful_parses": successful_parses,
        "failed_parses": failed_parses,
        "queue_size": query_queue.qsize(),
        "pending_processing": len(query_processing_events)
    }

async def calculate_answer_candidate(names: List[str], query_template: str, query_params: Optional[Dict] = None, timeout: Optional[float] = None) -> List[str]:
    """
    Calculate answer candidate from names and query template
    """
    try:
        if query_params is None:
            query_params = await get_or_parse_query(query_template, max_retries=1)
        if query_params is None:
            print(f"Failed to parse query: {query_template}")
            return []
    except Exception as e:
        print(f"Failed to parse query: {e}")
        return []
    
    task_args = []
    max_cpus = os.cpu_count()
    max_workers = max(1, min(8, len(names)))
    timeout_per_worker = int(timeout / max_workers)
    key = make_key(names, query_template)
    for i, name in enumerate(names):
        args = (i, name, query_params, timeout_per_worker, key)
        task_args.append(args)
    
    with ProcessPoolExecutor(max_workers = max_workers) as executor:
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(executor, run_single_generation, args) for args in task_args]
        try:
            results = await asyncio.gather(*tasks)
        except asyncio.TimeoutError:
            print(f"Timeout waiting for {len(tasks)} tasks to complete")
            return []
        return results

from MIID.miner.generate_name_variations import AnswerCandidate
class AnswerCandidateForNoisy:
    def __init__(self, answer_candidates: List[AnswerCandidate]):
        self.answer_candidates = answer_candidates
        self.serial = 0
        self.buckets_exact = {}
        self.answer_list = []

    def calc_reward_and_penalty(self, answers: List[Dict[str, List[str]]]):
        from types import SimpleNamespace
        responses = {}
        responses = [SimpleNamespace(
            variations=answer
        ) for answer in answers]
        # Calculate rule-based metadata
        rule_based = {
            "selected_rules": self.answer_candidates[0].query_params["selected_rules"],
            "rule_percentage": self.answer_candidates[0].query_params["rule_percentage"] * 100
        }
        import bittensor as bt
        debug_level = bt.logging.get_level()
        bt.logging.setLevel('CRITICAL')
        from MIID.validator.reward import get_name_variation_rewards
        _, metrics = get_name_variation_rewards(
            None,
            seed_names=list(answers[0].keys()), 
            responses=responses,
            uids=list(range(len(answers))),
            variation_count=self.answer_candidates[0].query_params["variation_count"],
            phonetic_similarity=self.answer_candidates[0].query_params["phonetic_similarity"],
            orthographic_similarity=self.answer_candidates[0].query_params["orthographic_similarity"],
            rule_based=rule_based,
        )
        # print (json.dumps(metrics, indent=4))
        penalty = False
        for metric in metrics:
            if 'collusion' in metric['penalties'] or 'duplication' in metric['penalties']:
                penalty = True
                break
        return metrics[-1], penalty
    
    def get_next_answer(self) -> Set[str]:
        COLLUSION_GROUP_SIZE_THRESHOLD = 1
        answer = {}
        metric = {}
        try_count = 100
        while try_count >= 0:
            self.serial += 1
            try_count -= 1
            from MIID.miner.kth_plan import kth_plan
            noisy_plan, noisy_count = kth_plan(len(self.answer_candidates), max(0, self.serial - COLLUSION_GROUP_SIZE_THRESHOLD) + 1)
            for i,cand in enumerate(self.answer_candidates):
                answer[cand.name] = cand.get_next_answer(noisy_plan[i])
            metric, penalty = self.calc_reward_and_penalty(self.answer_list + [answer])
            if not penalty:
                self.answer_list.append(answer)
                break
        return answer, metric


class TaskRequest(BaseModel):
    names: List[str]
    query_template: str
    query_params: Optional[Dict] = None
    timeout: Optional[float] = None

@app.post("/task")
async def solve_task(request: TaskRequest, background_tasks: BackgroundTasks = None):
    """
    POST /task: Solve task with name variations generation
    
    Solve task by generating name variants pool for each name in names, then solving the query template with the pool.
    If timeout parameter is specified, use it for instance pool generation instead of default timeout.
    Handles concurrent requests by making other requests wait for the first one to complete.
    """
    def clear_cache():
        if len(answer_candidate_cache) > 100:
            del answer_candidate_cache[list(answer_candidate_cache.keys())[0]]
    clear_cache()
    names = request.names
    query_template = request.query_template
    query_params = request.query_params
    timeout = request.timeout
    key = make_key(names, query_template)
    if key in pending_requests:
        print(f"Pending request hit for {key}")
        await asyncio.wait_for(pending_requests[key].wait(), timeout=timeout)
    if key in answer_candidate_cache:
        answer_candidate = answer_candidate_cache[key]
    else:
        print(f"Calculate answer candidate for {key} with timeout {timeout}")
        pending_requests[key] = asyncio.Event()
        try:
            answer_candidate = await calculate_answer_candidate(names, query_template, query_params, timeout)
            answer_candidate = AnswerCandidateForNoisy(answer_candidate)
            answer_candidate_cache[key] = answer_candidate
            pending_requests[key].set()
        except asyncio.TimeoutError:
            pending_requests[key].set()
            raise HTTPException(status_code=408, detail="Request timeout waiting for pool generation")
    
    answer, metric = answer_candidate.get_next_answer()
    print(f"Answer candidate: {answer_candidate.serial}")
    return {name: list(answer[name]) for name in answer}, metric, answer_candidate.answer_candidates[0].query_params

if __name__ == "__main__":
    import argparse
    port = argparse.ArgumentParser()
    port.add_argument("--port", type=int, default=PORT)
    args = port.parse_args()
    PORT = args.port
    print(f"Starting nvgen service on port {PORT}")
    try:
        import uvicorn
        # Start the server
        uvicorn.run(
            app, 
            host=HOST, 
            port=PORT,
            log_level="info",
            access_log=True,
            limit_concurrency=1000,
            limit_max_requests=1000,
        )
        
    except KeyboardInterrupt:
        import traceback
        traceback.print_exc()
        print("\n🛑 KeyboardInterrupt received, shutting down gracefully...")
        print("👋 Service stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error starting service: {e}")
        sys.exit(1)
