#!/usr/bin/env python3
"""
Test script for parallel BFS processing in the name variant generation service.
This script benchmarks the performance difference between sequential and parallel BFS.
"""

import time
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from pool_generator import expand_fixed_radius, bfs_layers, parallel_bfs_layers
from config import BUCKET_K

def benchmark_bfs_methods(name: str, timeout_seconds: float = 60):
    """Benchmark sequential vs parallel BFS methods"""
    
    print(f"🔍 Benchmarking BFS methods for name: '{name}'")
    print(f"   Timeout: {timeout_seconds}s")
    print(f"   Bucket K: {BUCKET_K}")
    print("=" * 60)
    
    # Test sequential BFS
    print("🔄 Testing Sequential BFS...")
    start_time = time.time()
    
    try:
        pools_seq, stats_seq = expand_fixed_radius(
            name=name,
            timeout_seconds=timeout_seconds,
            bucket_k=BUCKET_K,
            use_parallel=False  # Sequential
        )
        seq_time = time.time() - start_time
        
        print(f"✅ Sequential BFS completed in {seq_time:.2f}s")
        print(f"   Variants generated: {stats_seq.get('variants_total', 0)}")
        print(f"   Actual radius: {stats_seq.get('r', 0)}")
        print(f"   Pool stats: {stats_seq.get('pool_stats', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Sequential BFS failed: {e}")
        seq_time = float('inf')
        pools_seq = None
        stats_seq = {}
    
    print()
    
    # Test parallel BFS
    print("🔄 Testing Parallel BFS...")
    start_time = time.time()
    
    try:
        pools_par, stats_par = expand_fixed_radius(
            name=name,
            timeout_seconds=timeout_seconds,
            bucket_k=BUCKET_K,
            use_parallel=True,  # Parallel
            max_workers=8
        )
        par_time = time.time() - start_time
        
        print(f"✅ Parallel BFS completed in {par_time:.2f}s")
        print(f"   Variants generated: {stats_par.get('variants_total', 0)}")
        print(f"   Actual radius: {stats_par.get('r', 0)}")
        print(f"   Pool stats: {stats_par.get('pool_stats', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Parallel BFS failed: {e}")
        par_time = float('inf')
        pools_par = None
        stats_par = {}
    
    print()
    print("=" * 60)
    print("📊 Performance Comparison:")
    
    if seq_time != float('inf') and par_time != float('inf'):
        speedup = seq_time / par_time
        print(f"   Sequential time: {seq_time:.2f}s")
        print(f"   Parallel time:   {par_time:.2f}s")
        print(f"   Speedup:         {speedup:.2f}x")
        
        if speedup > 1.0:
            print(f"   🎉 Parallel BFS is {speedup:.2f}x faster!")
        elif speedup < 1.0:
            print(f"   ⚠️  Sequential BFS is {1/speedup:.2f}x faster (overhead)")
        else:
            print(f"   ⚖️  Both methods have similar performance")
        
        # Compare variant counts
        seq_variants = stats_seq.get('variants_total', 0)
        par_variants = stats_par.get('variants_total', 0)
        
        if seq_variants == par_variants:
            print(f"   ✅ Both methods generated the same number of variants: {seq_variants}")
        else:
            print(f"   ⚠️  Variant count differs: Sequential={seq_variants}, Parallel={par_variants}")
    
    elif seq_time == float('inf') and par_time == float('inf'):
        print("   ❌ Both methods failed")
    elif seq_time == float('inf'):
        print("   ✅ Only parallel BFS succeeded")
    else:
        print("   ✅ Only sequential BFS succeeded")

def test_different_name_lengths():
    """Test BFS performance with different name lengths"""
    
    test_names = [
        "a",           # Very short
        "ab",          # Short
        "abc",         # Short
        "john",        # Medium
        "wilson",      # Medium
        "christopher", # Long
        "alexandria",  # Long
    ]
    
    print("🧪 Testing BFS performance with different name lengths")
    print("=" * 80)
    
    for name in test_names:
        print(f"\n📝 Testing name: '{name}' (length: {len(name)})")
        benchmark_bfs_methods(name, timeout_seconds=30)
        print("-" * 40)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test parallel BFS processing")
    parser.add_argument("--name", "-n", type=str, default="wilson",
                       help="Name to test (default: wilson)")
    parser.add_argument("--timeout", "-t", type=float, default=60,
                       help="Timeout in seconds (default: 60)")
    parser.add_argument("--all-lengths", "-a", action="store_true",
                       help="Test different name lengths")
    
    args = parser.parse_args()
    
    if args.all_lengths:
        test_different_name_lengths()
    else:
        benchmark_bfs_methods(args.name, args.timeout)

if __name__ == "__main__":
    main()
