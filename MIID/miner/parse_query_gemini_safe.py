"""
Safe Gemini Query Parser - Avoids gRPC conflicts with multiprocessing

This module provides a safe way to use Google's Gemini API without gRPC warnings
when used in multiprocessing environments.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional
import logging
import requests
import threading
from MIID.validator.rule_extractor import RULE_DESCRIPTIONS
logger = logging.getLogger(__name__)

# Suppress gRPC warnings
os.environ['GRPC_PYTHON_LOG_LEVEL'] = 'error'
os.environ['ABSL_LOGGING_MIN_LEVEL'] = '2'

# Thread-local storage for Gemini clients
_thread_local = threading.local()

def _get_api_keys() -> List[str]:
    """Get API keys from environment variables"""
    api_keys = []
    
    # Try to get API keys from environment variables
    env_key = os.getenv('GOOGLE_API_KEY')
    if env_key and env_key not in api_keys:
        api_keys.append(env_key)
    
    # Try to get multiple API keys from environment
    env_keys = os.getenv('GOOGLE_API_KEYS')
    if env_keys:
        env_key_list = [key.strip() for key in env_keys.split(',') if key.strip()]
        for key in env_key_list:
            if key not in api_keys:
                api_keys.append(key)
    
    # If no API keys found, use default
    if not api_keys:
        default_key = "AIzaSyCvo_Pjngu84CFtyeBaU4J3eR_tpbuaLOI" # my key
        # default_key = "AIzaSyB9UHlanXuy4AFnk7tNGwrNzR-5ZSHsMFI" # from gpt
        api_keys.append(default_key)
    
    return api_keys

def _get_gemini_client():
    """Get or create a thread-local Gemini client"""
    if not hasattr(_thread_local, 'gemini_client'):
        try:
            import google.generativeai as genai
            api_keys = _get_api_keys()
            if api_keys:
                genai.configure(api_key=api_keys[0])
                _thread_local.gemini_client = genai.GenerativeModel('gemini-1.5-flash')
            else:
                _thread_local.gemini_client = None
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini client: {e}")
            _thread_local.gemini_client = None
    
    return _thread_local.gemini_client
# -----------------------------
# LLM-BASED PARSING (Gemini)
# -----------------------------

def _extract_recent_mistake_examples(max_examples: int = 100) -> List[Dict[str, str]]:
    """Extract up to N recent (Template/Query, Expected/Answer) pairs from known logs.

    Supports files containing blocks with 'Template:' and 'Expected:' or with
    'Query:' and 'Answer:'. Returns newest examples first.
    """
    candidate_paths = [
        os.path.join(os.path.dirname(__file__), "errored_queries_list.log"),
    ]

    def parse_pairs(lines: List[str]) -> List[Dict[str, str]]:
        pairs: List[Dict[str, str]] = []
        capturing_template = False
        template_lines: List[str] = []
        for line in lines:
            stripped = line.strip()
            # Prefer explicit Template: if present
            if stripped.startswith("Template:"):
                capturing_template = True
                template_lines = [line.split("Template:", 1)[1]]
                continue
            # Fallback: Query:
            if not capturing_template and stripped.startswith("Query:"):
                capturing_template = True
                template_lines = [line.split("Query:", 1)[1]]
                continue
            if capturing_template:
                if stripped.startswith("Expected:") or stripped.startswith("Answer:"):
                    expected_str = (
                        line.split(":", 1)[1].strip()
                        if ":" in line
                        else stripped
                    )
                    raw_template = "".join(template_lines).strip()
                    raw_template = re.sub(r"\s+", " ", raw_template)
                    if len(raw_template) > 900:
                        raw_template = raw_template[:900] + " …"
                    pairs.append({"template": raw_template, "expected_json": expected_str})
                    capturing_template = False
                    template_lines = []
                else:
                    template_lines.append(line)
        return pairs

    collected: List[Dict[str, str]] = []
    try:
        for path in candidate_paths:
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            collected.extend(parse_pairs(lines))
    except Exception:
        pass

    # Keep most recent
    if collected:
        collected = collected[-max_examples:]
        collected.reverse()
    return collected


def _build_llm_prompt(query_text: str) -> str:
    """Construct a concise instruction for the LLM to output strict JSON, with few-shot examples from recent errors."""
    
    id_list_only = "\n".join(f"- {rule_id}" for rule_id in RULE_DESCRIPTIONS.keys())

    examples = _extract_recent_mistake_examples()
    examples_text = ""
    if examples:
        blocks: List[str] = []
        for idx, ex in enumerate(examples, start=1):
            blocks.append(
                "Example {}:\nINPUT:\n{}\nOUTPUT:\n{}\n".format(
                    idx, ex["template"], ex["expected_json"]
                )
            )
        examples_text = (
            "Past failures and correct outputs (follow these patterns):\n"
            + "\n".join(blocks)
            + "\n"
        )

    return f"""
You are extracting parameters from a natural-language query about generating name variations.

Return ONLY a JSON object with fields:
{{
  "variation_count": <int not 0, valid range: 1-15>,
  "phonetic_similarity": {{"Light": <0.0-1.0>, "Medium": <0.0-1.0>, "Far": <0.0-1.0>}},
  "orthographic_similarity": {{"Light": <0.0-1.0>, "Medium": <0.0-1.0>, "Far": <0.0-1.0>}},
  "rule_based_rules": [<up to 3 identifiers from the list below>],
  "rule_percentage": <0.0 if not mentioned in query, otherwise 0.1-0.6>
}}

Rules:
- Use ONLY explicitly stated values. If values are given as percents, convert to decimals 0.0-1.0.
- If orthographic similarity is mentioned without per-level numbers, set {{"Light": 1.0}} unless the text clearly says significant/major/heavy (then {{"Far": 1.0}}).
- List at most 3 rule identifiers, only if explicitly requested.
- If rule percentage is not mentioned in query, set rule_percentage to 0.0 (don't guess/infer a value)
- For variation_count, extract explicit numbers like "generate 5 variations" or "create 10 names", "10 excutor vectors"
- For rule_based_rules:
  - Only include rules explicitly requested in query
  - Don't infer/guess rules that aren't clearly stated
  - Maximum 3 rules total
  - Must match exactly from the rule ID list
  - If no rules mentioned, return empty list []

- Identifiers must be EXACTLY one of:
{id_list_only}

{examples_text}
Input query:
{query_text}

Output: JSON only, no extra text.
"""


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Find the last JSON object in the text and parse it."""
    try:
        candidates = re.findall(r"\{[\s\S]*\}", text)
        if not candidates:
            return None
        for candidate in reversed(candidates):
            try:
                return json.loads(candidate)
            except Exception:
                continue
    except Exception:
        return None
    return None


def _normalize_llm_result(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the LLM JSON into the expected dictionary structure."""
    result: Dict[str, Any] = {
        "variation_count": None,
        "phonetic_similarity": {},
        "orthographic_similarity": {},
        "rule_percentage": None,
        "selected_rules": [],
    }

    # Variation count
    vc = obj.get("variation_count")
    if not vc:
        vc = 5
    elif vc > 15:
        vc = 15
    result["variation_count"] = int(vc)

    def normalize_distribution(dist: Any) -> Dict[str, float]:
        if not isinstance(dist, dict):
            return {}
        values = {k.capitalize(): float(v) for k, v in dist.items() if k and isinstance(v, (int, float))}
        # Keep only recognized levels
        values = {k: v for k, v in values.items() if k in {"Light", "Medium", "Far"}}
        # If any value > 1, assume percentages and normalize by total to 0..1
        if any(v > 1.0 for v in values.values()):
            total = sum(values.values())
            if total > 0:
                values = {k: round(v / total, 6) for k, v in values.items()}
        return values

    result["phonetic_similarity"] = normalize_distribution(obj.get("phonetic_similarity", {}))
    result["orthographic_similarity"] = normalize_distribution(obj.get("orthographic_similarity", {}))

    # Rules: expect identifiers only; drop anything not in allowed list
    selected_rules: List[str] = []
    if isinstance(obj.get("rule_based_rules"), list):
        for item in obj["rule_based_rules"]:
            if isinstance(item, str) and item in RULE_DESCRIPTIONS:
                selected_rules.append(item)
    result["selected_rules"] = selected_rules

    rp = obj.get("rule_percentage")
    if isinstance(rp, (int, float)):
        # If >1, assume percentage and convert to decimal
        if rp == 0.0:
            rp = (len(selected_rules) + 0.1) / result["variation_count"]
        
        result["rule_percentage"] = min(0.6, float(rp) / 100.0 if rp > 1 else float(rp))
    if result["phonetic_similarity"]["Light"] + result["phonetic_similarity"]["Medium"] + result["phonetic_similarity"]["Far"] < 1.0:
        if result["phonetic_similarity"]["Light"] == 0.2:
            result["phonetic_similarity"]["Medium"] = 0.6
            result["phonetic_similarity"]["Far"] = 0.2
    return result


def parse_query_sync_safe(query_text: str, max_retries: int = 3) -> Dict[str, Any]:
    """Synchronous safe query parser that avoids gRPC conflicts"""
    client = _get_gemini_client()
    
    if not client:
        # Fallback to default values if no client available
        return {
            "variation_count": 10,
            "phonetic_similarity": {"Light": 1.0, "Medium": 0.0, "Far": 0.0},
            "orthographic_similarity": {"Light": 1.0, "Medium": 0.0, "Far": 0.0},
            "rule_percentage": 0,
            "selected_rules": [],
        }
    
    prompt = _build_llm_prompt(query_text)
    
    for attempt in range(max_retries):
        try:
            # Use synchronous call to avoid async issues in multiprocessing
            response = client.generate_content(prompt)
            
            if response and response.text:
                obj = _extract_json_from_text(response.text)
                if obj:
                    return _normalize_llm_result(obj)
            
            # Add delay between retries
            if attempt < max_retries - 1:
                time.sleep(1)
                
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            continue
    
    # Return default values if all attempts failed
    return {
        "variation_count": 5,
        "phonetic_similarity": {"Light": 1.0, "Medium": 0.0, "Far": 0.0},
        "orthographic_similarity": {"Light": 1.0, "Medium": 0.0, "Far": 0.0},
        "rule_percentage": 0.0,
        "selected_rules": [],
    }

async def parse_query_async_safe(query_text: str, max_retries: int = 3) -> Dict[str, Any]:
    """Async wrapper for safe query parser"""
    # Run the sync function in a thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, parse_query_sync_safe, query_text, max_retries)

# Public API functions
def query_parser_sync(query_text: str, max_retries: int = 3) -> Dict[str, Any]:
    """Synchronous public API"""
    return parse_query_sync_safe(query_text, max_retries)

async def query_parser(query_text: str, max_retries: int = 3) -> Dict[str, Any]:
    """Async public API"""
    return await parse_query_async_safe(query_text, max_retries)

__all__ = [
    "query_parser",
    "query_parser_sync",
]
