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

from fastapi import FastAPI, HTTPException, BackgroundTasks


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

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION
)

# Global state
pending_requests: Dict[str, asyncio.Event] = {}

answer_candidate_cache = {}

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


async def calculate_answer_candidate(names: List[str], query_template: str, timeout: Optional[float] = None) -> List[str]:
    """
    Calculate answer candidate from names and query template
    """
    from MIID.miner.parse_query import query_parser
    try:
        query_params = await query_parser(query_template, max_retries=1)
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
        self.gen_idx = 0
        self.buckets_exact = {}

    def calculate_reward(self, answer):
        from types import SimpleNamespace
        responses = {}
        responses = [SimpleNamespace(
            variations=answer
        )]
        # Calculate rule-based metadata
        rule_based = {
            "selected_rules": self.answer_candidates[0].query_params["selected_rules"],
            "rule_percentage": self.answer_candidates[0].query_params["rule_percentage"] * 100
        }
        import bittensor as bt
        debug_level = bt.logging.get_level()
        bt.logging.setLevel('CRITICAL')
        from MIID.miner.generate_name_variations import get_name_variation_rewards_exclude_phonetic
        scores, metrics = get_name_variation_rewards_exclude_phonetic(
            seed_names=list(answer.keys()), 
            responses=responses,
            uids=[0],
            variation_count=self.answer_candidates[0].query_params["variation_count"],
            phonetic_similarity=self.answer_candidates[0].query_params["phonetic_similarity"],
            orthographic_similarity=self.answer_candidates[0].query_params["orthographic_similarity"],
            rule_based=rule_based,
        )
        bt.logging.setLevel(debug_level)
        return scores[0], metrics[0]

    def get_next_answer(self) -> Set[str]:
        init_answer = {}
        for cand in self.answer_candidates:
            init_answer[cand.name] = cand.get_next_answer()
        reward = 0.0
        if self.gen_idx < 5:
            reward, metric = self.calculate_reward(init_answer)
            fmt15 = f"{reward:.15f}"
            if self.gen_idx == 0:
                self.buckets_exact[fmt15] = [self.gen_idx]
            else:
                self.buckets_exact[fmt15].append(self.gen_idx)
            self.gen_idx += 1
            return init_answer, metric
        try_count = 1000
        while try_count > 0:
            try_count -= 1
            answer = {}
            for cand in self.answer_candidates:
                answer[cand.name] = init_answer[cand.name].copy()
            mod_count = self.gen_idx // len(self.answer_candidates)
            cand_idx = self.gen_idx % len(self.answer_candidates)
            cand = self.answer_candidates[cand_idx]
            modifications = []
            attempts = 0
            for idx, var in enumerate(answer[cand.name]):
                replaced = var
                for v in var.split(" "):
                    if v in cand.replacement_for_noisy:
                        replaced = replaced.replace(v, cand.replacement_for_noisy[v])
                        attempts += 1
                        if attempts >= mod_count:
                            break
                if replaced != var:
                    modifications.append((var, replaced))
                if attempts >= mod_count:
                    break
            for var, replaced in modifications:
                answer[cand.name].remove(var)
                answer[cand.name].add(replaced)
            reward, metric = self.calculate_reward(answer)
            fmt15 = f"{reward:.15f}"
            if fmt15 not in self.buckets_exact or len(self.buckets_exact[fmt15]) < 5:
                if fmt15 not in self.buckets_exact:
                    self.buckets_exact[fmt15] = [self.gen_idx]
                else:
                    self.buckets_exact[fmt15].append(self.gen_idx)
                self.gen_idx += 1
                return answer, metric
            else:
                self.gen_idx += 1
                continue

from pydantic import BaseModel
class TaskRequest(BaseModel):
    names: List[str]
    query_template: str
    timeout: Optional[float] = None

@app.post("/task")
async def solve_task(request: TaskRequest, background_tasks: BackgroundTasks = None):
    """
    GET /task?names={names}&query_template={query_template}&timeout={seconds}: Solve task
    
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
            answer_candidate = await calculate_answer_candidate(names, query_template, timeout)
            answer_candidate = AnswerCandidateForNoisy(answer_candidate)
            answer_candidate_cache[key] = answer_candidate
            pending_requests[key].set()
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Request timeout waiting for pool generation")
    
    answer, metric = answer_candidate.get_next_answer()
    return {name: list(answer[name]) for name in answer}, metric, answer_candidate.answer_candidates[0].query_params


if __name__ == "__main__":
    try:
        import uvicorn
        # Start the server
        uvicorn.run(
            app, 
            host=HOST, 
            port=PORT,
            log_level="info",
            access_log=True,
        )
        
    except KeyboardInterrupt:
        print("\n🛑 KeyboardInterrupt received, shutting down gracefully...")
        print("👋 Service stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error starting service: {e}")
        sys.exit(1)
