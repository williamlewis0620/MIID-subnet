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
from collections import Counter, defaultdict
logger = logging.getLogger(__name__)

# Suppress gRPC warnings
os.environ['GRPC_PYTHON_LOG_LEVEL'] = 'error'
os.environ['ABSL_LOGGING_MIN_LEVEL'] = '2'

# Thread-local storage for Gemini clients
_thread_local = threading.local()

_IGNORED_WORDS = {
    # Articles
    "a", "an", "the",
    # Common prepositions/conjunctions often inserted without changing meaning
    "in", "on", "at", "to", "for", "with", "by", "of", "from", "into",
    "onto", "upon", "over", "under", "between", "among", "about", "around",
    "after", "before", "within", "without", "across", "through", "per",
    "via", "as", "like", "and", "or"
}

def _normalize_and_tokenize(text: str, remove_braces: bool = False) -> List[str]:
    """Lowercase, remove braces if requested, strip punctuation, split into tokens, and drop ignored words."""
    if remove_braces:
        text = text.replace("{", " ").replace("}", " ")
    # Replace any non-alphanumeric (unicode word chars) with space
    text = re.sub(r"[^\w]+", " ", text.lower())
    tokens = [t for t in text.split() if t and t not in _IGNORED_WORDS]
    return tokens

def _select_rules_from_query(query_text: str) -> List[str]:
    """Select rule IDs whose description tokens (order-agnostic, ignoring prepositions/articles) are present in the query.

    - Removes '{' and '}' from the query during comparison.
    - Allows different word orders and insertion of common prepositions/articles.
    - Returns up to 3 IDs ordered by first appearance of any description token in the query.
    """
    query_tokens = _normalize_and_tokenize(query_text, remove_braces=True)
    if not query_tokens:
        return []

    query_counts = Counter(query_tokens)
    token_positions: Dict[str, List[int]] = defaultdict(list)
    for idx, tok in enumerate(query_tokens):
        token_positions[tok].append(idx)

    matches: List[tuple[int, str]] = []  # (first_position, rule_id)
    for rule_id, description in RULE_DESCRIPTIONS.items():
        desc_tokens = _normalize_and_tokenize(str(description), remove_braces=True)
        if not desc_tokens:
            continue
        desc_counts = Counter(desc_tokens)
        # Multiset containment check
        if all(query_counts.get(tok, 0) >= count for tok, count in desc_counts.items()):
            # Determine earliest occurrence index for ordering
            earliest_positions = [token_positions[tok][0] for tok in desc_counts.keys() if tok in token_positions]
            first_pos = min(earliest_positions) if earliest_positions else len(query_tokens)
            matches.append((first_pos, rule_id))

    matches.sort(key=lambda x: x[0])
    # Return up to 3 rule IDs
    return [rule_id for _, rule_id in matches[:3]]

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
                _thread_local.gemini_client = genai.GenerativeModel('gemini-2.5-flash')
            else:
                _thread_local.gemini_client = None
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini client: {e}")
            _thread_local.gemini_client = None
    
    return _thread_local.gemini_client
# -----------------------------
# LLM-BASED PARSING (Gemini)
# -----------------------------

def _load_parse_examples() -> List[Dict[str, str]]:
    """Load parse examples from file"""
    try:
        with open(os.path.join(os.path.dirname(__file__), "correct_parse_examples.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def _build_llm_prompt(query_text: str) -> str:
    """Construct a concise instruction for the LLM to output strict JSON, with few-shot examples from recent errors."""
    
    id_list_only = "\n".join(f"- {rule_id}" for rule_id in RULE_DESCRIPTIONS.keys())
    rules_descriptions_json = json.dumps(RULE_DESCRIPTIONS, indent=4)

    examples = _load_parse_examples()
    examples_text = ""
    if examples:
        blocks: List[str] = []
        for idx, ex in enumerate(examples, start=1):
            blocks.append(
                "Example {}:\nINPUT:\n{}\nOUTPUT:\n{}\n".format(
                    idx, ex["template"], ex["label"]
                )
            )
        examples_text = (
            "Past failures and correct outputs (follow these patterns):\n"
            + "\n".join(blocks)
            + "\n"
        )
        
    return f"""
You are a deterministic parameter extractor for natural-language queries about generating name variations.
Operate in STRICT MODE: never infer, generalize, or guess beyond what is explicitly present in the query text.

Return ONLY a JSON object with fields in this exact order:
{{
  "variation_count": <int not 0, valid range: 1-15>,
  "phonetic_similarity": {{"Light": <0.0-1.0>, "Medium": <0.0-1.0>, "Far": <0.0-1.0>}},
  "orthographic_similarity": {{"Light": <0.0-1.0>, "Medium": <0.0-1.0>, "Far": <0.0-1.0>}},
  "rule_based_rules": [<up to 3 rule IDs from the catalog below>],
  "rule_percentage": <0.0 if not mentioned in query, otherwise 0.1-0.6>
}}

Rule catalog (IDs and authoritative descriptions):
    - RULE_IDS: {id_list_only}
    - RULE_DESCRIPTIONS (JSON object, ID -> exact description string):
    {rules_descriptions_json}

    STRICT MATCH POLICY for "rule_based_rules":
    - Match ONLY when the **full description string** from RULE_DESCRIPTIONS appears in the Input query.
    - Matching is case-insensitive and whitespace-insensitive (collapse runs of spaces/tabs/newlines to a single space on both the query and the description before comparison).
    - NEVER match by ID alone, synonyms, paraphrases, examples, implications, or world knowledge.
    - If uncertain, DO NOT match. If no exact matches exist, return [].
    - If more than 3 descriptions match, keep the first 3 by **order of first appearance** in the query. Remove duplicates.

    Rules for other fields:
    - Use ONLY explicitly stated numeric values. If percents are given, convert to decimals (e.g., 30% -> 0.3).
    - For "orthographic_similarity":
    - If mentioned without per-level numbers, set to {{"Light": 1.0}} only (other levels 0.0).
    - If the query clearly says significant/major/heavy, set to {{"Far": 1.0}} only (other levels 0.0).
    - For "variation_count", extract explicit numbers like "generate 5 variations" or "create 10 names".
    - For "rule_percentage", if not mentioned in the query, set to 0.0 (do not guess).

    Validation BEFORE output (do not print your reasoning):
    - For every ID you plan to output in "rule_based_rules", first locate and internally record the exact matched substring from the query that equals the rule's description (after the normalization above). If you cannot find such a substring for an ID, REMOVE that ID.
    - Ensure numeric ranges are respected; if a provided number is out of range, clamp to the valid range only if the query explicitly demands an out-of-range value (otherwise leave unspecified fields at their required defaults).
    - Output must be valid JSON. Do not include comments, code fences, or extra text.

    {examples_text}

    Input query:
    {query_text}

    Output: JSON only, no extra text.
    """

#     return f"""
# You are extracting parameters from a natural-language query about generating name variations.

# Return ONLY a JSON object with fields:
# {{
#   "variation_count": <int not 0, valid range: 1-15>,
#   "phonetic_similarity": {{"Light": <0.0-1.0>, "Medium": <0.0-1.0>, "Far": <0.0-1.0>}},
#   "orthographic_similarity": {{"Light": <0.0-1.0>, "Medium": <0.0-1.0>, "Far": <0.0-1.0>}},
#   "rule_based_rules": [<up to 3 identifiers from the list below>],
#   "rule_percentage": <0.0 if not mentioned in query, otherwise 0.1-0.6>
# }}

# Rule identifiers and RULE_DESCRIPTIONS (use EXACTLY these IDs):
# {id_list_only}

# Rules:
# - Use ONLY explicitly stated values. If values are given as percents, convert to decimals 0.0-1.0.
# - For `orthographic_similarity`:
#   - If mentioned without per-level numbers, set {{"Light": 1.0}}.
#   - If the query clearly says significant/major/heavy, then set {{"Far": 1.0}}.
# - For `variation_count`, extract explicit numbers like "generate 5 variations" or "create 10 names".
# - For `rule_percentage`, if not mentioned in the query, set to 0.0 (do not guess).
# - For `rule_based_rules`:
#   - RULE_DESCRIPTIONS will be provided as `{{id: description}}`.
#   - Search the query text for the **description** strings.
#   - If a description appears in the query, include the corresponding `id` in the list.
#   - Include at most 3 IDs.
#   - If no description matches are found, return an empty list [].
#   - Do not infer or guess rules beyond exact matches.

# {examples_text}
# Input query:
# {query_text}

# Output: JSON only, no extra text.
# """


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
            "selected_rules": _select_rules_from_query(query_text),
        }
    
    prompt = _build_llm_prompt(query_text)
    
    # print(f"prompt: {prompt}")
    
    for attempt in range(max_retries):
        try:
            # Use synchronous call to avoid async issues in multiprocessing
            response = client.generate_content(prompt)
            
            if response and response.text:
                obj = _extract_json_from_text(response.text)
                if obj:
                    result = _normalize_llm_result(obj)
                    # Override/augment with deterministic selection from query text per relaxed matching
                    result["selected_rules"] = _select_rules_from_query(query_text)
                    return result
            
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
        "selected_rules": _select_rules_from_query(query_text),
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
