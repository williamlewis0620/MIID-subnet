#!/usr/bin/env python3
"""
Resource Monitoring Script for Name Variant Generation Service

This script monitors CPU and RAM usage to ensure optimal resource utilization.
"""

import psutil
import time
import json
from datetime import datetime

def get_resource_usage():
    """Get current resource usage"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "cpu_percent": cpu_percent,
        "ram_total_gb": round(memory.total / (1024**3), 2),
        "ram_used_gb": round(memory.used / (1024**3), 2),
        "ram_available_gb": round(memory.available / (1024**3), 2),
        "ram_percent": memory.percent,
        "ram_free_gb": round(memory.free / (1024**3), 2)
    }

def monitor_resources(interval=5, duration=None):
    """Monitor resources continuously"""
    print("🔍 Resource Monitoring Started")
    print("=" * 60)
    print(f"{'Time':<20} {'CPU%':<8} {'RAM%':<8} {'Used GB':<10} {'Available GB':<12}")
    print("=" * 60)
    
    start_time = time.time()
    count = 0
    
    try:
        while True:
            usage = get_resource_usage()
            count += 1
            
            # Print formatted output
            time_str = usage["timestamp"].split("T")[1][:8]
            print(f"{time_str:<20} {usage['cpu_percent']:<8.1f} {usage['ram_percent']:<8.1f} "
                  f"{usage['ram_used_gb']:<10.1f} {usage['ram_available_gb']:<12.1f}")
            
            # Check if duration exceeded
            if duration and (time.time() - start_time) > duration:
                break
                
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
    
    print("=" * 60)
    print(f"📊 Monitoring completed: {count} samples taken")

def get_system_info():
    """Get system information"""
    cpu_count = psutil.cpu_count()
    cpu_count_logical = psutil.cpu_count(logical=True)
    memory = psutil.virtual_memory()
    
    print("🖥️  System Information:")
    print(f"   CPU Cores (Physical): {cpu_count}")
    print(f"   CPU Threads (Logical): {cpu_count_logical}")
    print(f"   Total RAM: {memory.total / (1024**3):.1f} GB")
    print(f"   Available RAM: {memory.available / (1024**3):.1f} GB")
    print()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor system resources")
    parser.add_argument("--interval", "-i", type=int, default=5, 
                       help="Monitoring interval in seconds (default: 5)")
    parser.add_argument("--duration", "-d", type=int, 
                       help="Monitoring duration in seconds (optional)")
    parser.add_argument("--info", action="store_true", 
                       help="Show system information only")
    
    args = parser.parse_args()
    
    if args.info:
        get_system_info()
        return
    
    get_system_info()
    monitor_resources(interval=args.interval, duration=args.duration)

if __name__ == "__main__":
    main()
