#!/usr/bin/env python3
"""
Phonetic-aware expansion with fixed radius, deep Metaphone/NYSIIS heuristics,
tight timeout checks, and exact length-diff indexing:

  vpools[ld][orth_level][phon_class] -> TopK(set)

Phon classes (vs seed):
  P0: s!=, m!=, n!=
  P1: s==, m!=, n!=
  P2: s!=, m==, n!=
  P3: s!=, m!=, n==
  P4: s!=, m==, n==
  P5: s==, m!=, n==
  P6: s==, m==, n!=
  P7: s==, m==, n==

Radius is single-pass: R = min(4, len(name) - 1).
"""

from __future__ import annotations

import time
import psutil
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial

# --- deps ---
try:
    import jellyfish  # for soundex/metaphone/nysiis
except Exception as exc:
    raise ImportError("Install 'jellyfish' (pip install jellyfish)") from exc

try:
    import Levenshtein as _lev  # optional speed-up for orth similarity
except Exception:
    _lev = None

def get_ram_usage() -> float:
    """Get current RAM usage percentage"""
    return psutil.virtual_memory().percent

def is_ram_over_limit(limit: float = 80.0) -> bool:
    """Check if RAM usage is over the limit"""
    return get_ram_usage() > limit

# ---------------- Orthographic similarity (0..1) ----------------
def orth_sim(a: str, b: str) -> float:
    a = a or ""; b = b or ""
    if not a and not b:
        return 1.0
    if _lev:
        d = _lev.distance(a, b)
        m = max(len(a), len(b)) or 1
        return 1.0 - (d / m)
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()


ORTH_BOUNDS: Dict[int, Tuple[float, float]] = {
    0: (0.70, 1.00),
    1: (0.50, 0.69),
    2: (0.20, 0.49),
    3: (0.00, 0.19),
}
def orth_level(x: float) -> int:
    for lvl, (mn, mx) in ORTH_BOUNDS.items():
        if mn <= x <= mx:
            return lvl
    return 3  # safety


# ---------------- Phonetic class P0..P7 ----------------
@dataclass(frozen=True)
class SeedCodes:
    sdx: str
    met: str
    nys: str

def seed_codes(name: str) -> SeedCodes:
    return SeedCodes(
        sdx=jellyfish.soundex(name),
        met=jellyfish.metaphone(name),
        nys=jellyfish.nysiis(name),
    )

def phon_class(s: SeedCodes, v: str) -> int:
    se = jellyfish.soundex(v)   == s.sdx
    me = jellyfish.metaphone(v) == s.met
    ne = jellyfish.nysiis(v)    == s.nys
    # pack bits: s(4) + m(2) + n(1)
    idx = (4 if se else 0) + (2 if me else 0) + (1 if ne else 0)
    table = {0:7, 4:6, 2:5, 1:4, 3:3, 5:2, 6:1, 7:0}
    return table[idx]


# ---------------- Deep phonetic-aware edits ----------------
VOWELS = tuple("aeiou")
YHW = tuple("yhw")

# Soundex consonant groups – stabilizes Soundex
SDX_GROUPS: List[Tuple[str, ...]] = [
    tuple("bfpv"),            # 1
    tuple("cgjkqsxz"),        # 2
    tuple("dt"),              # 3
    tuple("l"),               # 4
    tuple("mn"),              # 5
    tuple("r"),               # 6
]
SDX_MAP: Dict[str, int] = {c: i for i, grp in enumerate(SDX_GROUPS) for c in grp}

def consonant_group(c: str) -> Optional[Tuple[str, ...]]:
    i = SDX_MAP.get(c)
    return SDX_GROUPS[i] if i is not None else None

def optimize_replacements(base_alpha: str = "abcdefghijklmnopqrstuvwxyz") -> Dict[str, List[str]]:
    """
    Replacement candidates tuned for codes:
      - Vowels -> other vowels + y/h/w (Metaphone/NYSIIS treat vowels specially).
      - Consonants -> same Soundex group + y/h/w (preserve Soundex while nudging Metaphone).
      - Add key single letters that often map similarly (c<->k, q->k).
    """
    alpha = list(dict.fromkeys(base_alpha.lower()))
    repl: Dict[str, List[str]] = {}
    for ch in alpha:
        if ch in VOWELS:
            repl[ch] = [v for v in VOWELS if v != ch] + list(YHW)
        else:
            grp = consonant_group(ch) or ()
            cand = [c for c in grp if c != ch] + list(YHW)
            if ch == "c": cand += ["k", "s"]
            if ch == "k": cand += ["c", "q"]
            if ch == "q": cand += ["k", "c"]
            if ch == "g": cand += ["j"]  # g<->j impacts Metaphone before e/i/y
            if ch == "x": cand += ["s"]  # x ~ s/ks
            repl[ch] = list(dict.fromkeys(cand))
    return repl

# Digraph/contextual transforms impacting Metaphone/NYSIIS
def contextual_variants(w: str) -> Iterable[str]:
    n = len(w)
    lower = w
    # start-boundary transforms (Metaphone/NYSIIS)
    if lower.startswith(("kn", "gn", "pn")):
        yield lower[1:]                  # "knight" -> "night"
    if lower.startswith("wr"):
        yield "r" + lower[2:]            # "write" -> "rite"
    if lower.startswith("x"):
        yield "s" + lower[1:]            # "xeno" -> "seno"

    # PH <-> F (Metaphone; NYSIIS PH->FF)
    for i in range(n - 1):
        if lower[i:i+2] == "ph":
            yield lower[:i] + "f" + lower[i+2:]
    for i in range(n):
        if lower[i] == "f":
            yield lower[:i] + "ph" + lower[i+1:]

    # SCH -> SS/SH (NYSIIS)
    for i in range(n - 2):
        if lower[i:i+3] == "sch":
            yield lower[:i] + "ss" + lower[i+3:]
            yield lower[:i] + "sh" + lower[i+3:]

    # CK -> K; C->S before I/E/Y; G->J before I/E/Y (Metaphone)
    for i in range(n - 1):
        if lower[i:i+2] == "ck":
            yield lower[:i] + "k" + lower[i+2:]
    for i in range(n - 1):
        c, nxt = lower[i], lower[i+1]
        if c == "c" and nxt in "iey":
            yield lower[:i] + "s" + lower[i+1:]
        if c == "g" and nxt in "iey":
            yield lower[:i] + "j" + lower[i+1:]

    # X -> KS (Metaphone), S -> X (rarely, but explore)
    for i in range(n):
        if lower[i] == "x":
            yield lower[:i] + "ks" + lower[i+1:]
        if lower[i] == "s":
            yield lower[:i] + "x" + lower[i+1:]

    # Endings per NYSIIS
    if lower.endswith(("ee", "ie")):
        yield lower[:-2] + "y"
    for suf in ("dt", "rt", "rd", "nt", "nd"):
        if lower.endswith(suf):
            yield lower[:-2] + "d"

    # MAC -> MCC (NYSIIS)
    if lower.startswith("mac"):
        yield "mcc" + lower[3:]

# Main phonetic-aware generator
def neighbors_once_phonetic(w: str, repl_map: Dict[str, List[str]]) -> Iterator[str]:
    n = len(w)
    # replacements (highest yield; heavily shapes codes)
    for i in range(n):
        ci = w[i]
        cand = repl_map.get(ci, ())
        if cand:
            wi, wj = w[:i], w[i+1:]
            for ch in cand:
                if ch != ci:
                    yield wi + ch + wj

    # contextual (digraph + boundary) edits
    for v in contextual_variants(w):
        yield v

    # insertions near vowels (affects Metaphone/NYSIIS) and Y/H/W anywhere
    for i in range(n + 1):
        left = w[i-1] if i-1 >= 0 else ""
        right = w[i] if i < n else ""
        local: List[str] = []
        if left in VOWELS or right in VOWELS:
            local.extend(VOWELS)
        local.extend(YHW)
        for ch in local:
            yield w[:i] + ch + w[i:]

    # deletions of Y/H/W (often neutral in codes)
    for i in range(n):
        if w[i] in YHW:
            yield w[:i] + w[i+1:]

    # limited transpositions when a vowel is involved
    for i in range(n - 1):
        a, b = w[i], w[i+1]
        if a != b and (a in VOWELS or b in VOWELS):
            yield w[:i] + b + a + w[i+2:]

    # duplication (stutter) of previous char
    for i in range(1, n):
        yield w[:i] + w[i-1] + w[i:]


# ---------------- Top-K per (ld, o, p) cell ----------------
class TopK:
    __slots__ = ("k", "heap", "seen")
    def __init__(self, k: int) -> None:
        self.k = k
        self.heap: List[Tuple[float, str]] = []  # (-score, value)
        self.seen: Set[str] = set()
    def add(self, v: str, score: float) -> None:
        if v in self.seen: return
        neg = -score
        if len(self.heap) < self.k:
            heappush(self.heap, (neg, v)); self.seen.add(v); return
        if neg > self.heap[0][0]:  # worse than worst
            return
        _, worst = heappop(self.heap); self.seen.remove(worst)
        heappush(self.heap, (neg, v)); self.seen.add(v)
    def to_set(self) -> Set[str]:
        return set(self.seen)


# ---------------- Parallel Processing Functions ----------------
def parallel_neighbors_worker(word: str, repl_map: Dict[str, List[str]], timeout_at: Optional[float]) -> Set[str]:
    """Worker function to generate neighbors for a single word"""
    try:
        neighbors = set()
        for neighbor in neighbors_once_phonetic(word, repl_map):
            if timeout_at and time.time() > timeout_at:
                break
            neighbors.add(neighbor)
        return neighbors
    except Exception as e:
        print(f"Error in parallel_neighbors_worker for word '{word}': {e}")
        return set()

def parallel_bfs_layer_worker(
    words: List[str], 
    repl_map: Dict[str, List[str]], 
    timeout_at: Optional[float],
    max_workers: int = None
) -> Set[str]:
    """Worker function to process a layer of words in parallel"""
    if not words:
        return set()
    
    # Use ThreadPoolExecutor for I/O-bound neighbor generation
    max_workers = max_workers or min(len(words), mp.cpu_count())
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create partial function with fixed arguments
            worker_func = partial(parallel_neighbors_worker, repl_map=repl_map, timeout_at=timeout_at)
            
            # Submit all words for parallel processing
            future_to_word = {executor.submit(worker_func, word): word for word in words}
            
            # Collect results
            all_neighbors = set()
            for future in future_to_word:
                try:
                    neighbors = future.result()
                    all_neighbors.update(neighbors)
                except Exception as e:
                    word = future_to_word[future]
                    print(f"Error processing word '{word}': {e}")
                    continue
            
            return all_neighbors
    except Exception as e:
        print(f"Error in parallel_bfs_layer_worker: {e}")
        return set()

def parallel_bfs_layers(
    seed: str, 
    R: int, 
    repl_map: Dict[str, List[str]],
    timeout_at: Optional[float], 
    check_terminate: Optional[Callable] = None,
    max_workers: int = None
) -> Tuple[List[Set[str]], str]:
    """
    Parallel version of bfs_layers that distributes work across multiple CPUs.
    Returns (layers, reason) where reason is "good", "timed_out", "ram_overhead", or "terminated"
    """
    if R <= 0:
        return [], "good"
    
    # Set default max_workers if not provided
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)  # Limit to 8 workers to avoid overhead
    
    seen: Set[str] = {seed}
    layers: List[Set[str]] = [set()] * (R + 1)  # index 0 unused
    prev = {seed}
    
    for depth in range(1, R + 1):
        if timeout_at and time.time() > timeout_at:
            return layers[:depth], "timed_out"
        if is_ram_over_limit():
            return layers[:depth], "ram_overhead"
        if check_terminate and check_terminate():
            return layers[:depth], "terminated"
        
        # Determine source words for this depth
        source = prev if depth == 1 else layers[depth - 1]
        
        # Process this layer in parallel
        cur = parallel_bfs_layer_worker(
            list(source), 
            repl_map, 
            timeout_at, 
            max_workers
        )
        
        # Filter out already seen words
        cur = {v for v in cur if v not in seen}
        seen.update(cur)
        
        layers[depth] = cur
        if not cur:
            return layers[:depth], "good"
        prev = cur
    
    return layers, "good"


# ---------------- Single-pass BFS up to fixed R ----------------
def bfs_layers(seed: str, R: int, repl_map: Dict[str, List[str]],
               timeout_at: Optional[float], check_terminate: Optional[Callable] = None) -> Tuple[List[Set[str]], str]:
    """
    Return exact-distance layers [1..R], with timeout, RAM, and termination checks.
    Returns (layers, reason) where reason is "good", "timed_out", "ram_overhead", or "terminated"
    """
    if R <= 0:
        return [], "good"
    seen: Set[str] = {seed}
    layers: List[Set[str]] = [set()] * (R + 1)  # index 0 unused
    prev = {seed}
    for depth in range(1, R + 1):
        if timeout_at and time.time() > timeout_at:
            return layers[:depth], "timed_out"
        if is_ram_over_limit():
            return layers[:depth], "ram_overhead"
        if check_terminate and check_terminate():
            return layers[:depth], "terminated"
        cur: Set[str] = set()
        source = prev if depth == 1 else layers[depth - 1]
        for w in source:
            for v in neighbors_once_phonetic(w, repl_map):
                if timeout_at and time.time() > timeout_at:
                    return layers[:depth], "timed_out"
                if is_ram_over_limit():
                    return layers[:depth], "ram_overhead"
                if check_terminate and check_terminate():
                    return layers[:depth], "terminated"
                if v in seen:
                    continue
                seen.add(v)
                cur.add(v)
        layers[depth] = cur
        if not cur:
            return layers[:depth], "good"
        prev = cur
    return layers, "good"


# ---------------- Driver: fixed radius, direct cell writes ----------------
def expand_fixed_radius(
    name: str,
    *,
    alphabet: str = "abcdefghijklmnopqrstuvwxyz",
    bucket_k: int = 15,
    len_cap: Optional[int] = None,     # if set, discard cells with ld > len_cap
    timeout_seconds: Optional[float] = None,
    check_terminate: Optional[Callable] = None,
    use_parallel: bool = True,         # whether to use parallel processing
    max_workers: int = None,           # max workers for parallel processing
) -> Tuple[List[List[List[Set[str]]]], Dict[str, int]]:
    """
    Build vpools[ld][o][p] with TopK sets.
    """
    t0 = time.time()
    timeout_at = t0 + timeout_seconds if timeout_seconds else None

    seed = name.lower()
    # R = max(1, min(4, len(seed) - 1))  # per requirement
    R = len(seed)  # per requirement
    repl_map = optimize_replacements(alphabet)
    sc = seed_codes(seed)

    # Calculate maximum ld needed
    max_ld = len(seed) if len_cap is None else min(len(seed), len_cap)
    
    # Initialize vpools as 3D array: [ld][o][p]
    
    vpools: List[List[List[TopK]]] = [
        [
            [
                TopK(bucket_k) for _ in range(8)
            ] for _ in range(4)
        ] for _ in range(max_ld + 1)
    ]

    def ensure_cell(ld: int, o: int, p: int) -> TopK:
        return vpools[ld][o][p]

    # Use parallel or sequential BFS based on configuration
    if use_parallel:
        print(f"🔄 Using parallel BFS with up to {max_workers or 'auto'} workers")
        layers, reason = parallel_bfs_layers(seed, R, repl_map, timeout_at, check_terminate, max_workers)
    else:
        print("🔄 Using sequential BFS")
        layers, reason = bfs_layers(seed, R, repl_map, timeout_at, check_terminate)
    
    # Calculate actual radius reached (number of layers generated)
    actual_radius = len(layers) - 1  # Subtract 1 because layers[0] is empty
    # Generate additional layer with same-length variations using only different characters
    same_length_variations: Set[str] = set()
    used_chars = set(seed.lower())
    available_chars = [c for c in alphabet if c not in used_chars]
    
    if available_chars:  # Only generate if we have available characters
        target_count = 20
        max_attempts = target_count * 10  # Prevent infinite loops
        attempts = 0
        
        while len(same_length_variations) < target_count and attempts < max_attempts:
            # Generate a variation of the same length using only different characters
            variation = ''.join(random.choice(available_chars) for _ in range(len(seed)))
            same_length_variations.add(variation)
            attempts += 1
        
        # Add this as a new layer if we generated any variations
        if same_length_variations:
            layers.append(same_length_variations)
    
    # --- corrected write path (your requested change) ---
    # print(f"layers: {layers}")
    for cur in layers[1:]:  # ignore index 0
        for v in cur:
            if not v.isalpha():
                continue
            p = phon_class(sc, v)
            ld = abs(len(seed) - len(v))
            if len_cap is not None and ld > len_cap:
                continue
            o = orth_level(orth_sim(seed, v))
            score = 1.0 / (1.0 + ld)  # prefer closer length
            ensure_cell(ld, o, p).add(v, score)

    # materialize sets
    out: List[List[List[Set[str]]]] = []
    for ld in range(len(vpools)):
        ld_level = []
        for o in range(4):
            o_level = []
            for p in range(8):
                o_level.append(vpools[ld][o][p].to_set())
            ld_level.append(o_level)
        out.append(ld_level)
    
    # Determine pool_stats based on the reason
    if reason == "ram_overhead":
        pool_stats = "ram_overhead"
    elif reason == "timed_out":
        pool_stats = "timed_out"
    elif reason == "terminated":
        pool_stats = "terminated"
    else:
        pool_stats = "good"
    
    stats = {
        "max_r": R,  # Maximum radius that was attempted
        "r": actual_radius,  # Actual radius reached when process ended
        "ld_groups": len(out),
        "variants_total": sum(len(out_ld[o][p]) for out_ld in out for o in range(4) for p in range(8)),
        "timeout_hit": bool(timeout_at and time.time() > timeout_at),
        "pool_stats": pool_stats,  # Reason for stopping: "good", "timed_out", or "ram_overhead"
    }
    return out, stats
