#!/usr/bin/env python3
"""
Test script for the Name Variant Generation Service
"""

import array
import requests
import time
import json

BASE_URL = "http://localhost:8001"

def test_get_variants(name):
    """Test getting name variants"""
    print(f"\n=== Testing GET /pool?original_name={name} ===")
    try:
        response = requests.get(f"{BASE_URL}/pool", params={"original_name": name})
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success: Got {data.get('source unknown')} variants")
            print(f"   Consumed count: {data.get('consumed_count', 0)}")
            
            # Count total variants
            total_variants = 0
            pools = data.get('pools', [])
            for ld_level in pools:
                for orth_level in ld_level:
                    for variants in orth_level:
                        total_variants += len(variants)
            
            print(f"   Total variants: {total_variants}")
            return data.get('pools', {})
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

def test_mark_consumed(name, variants):
    """Test marking variants as consumed"""
    print(f"\n=== Testing POST /consumed?original_name={name} ===")
    try:
        response = requests.post(
            f"{BASE_URL}/consumed",
            params={"original_name": name},
            json={"variants": variants}
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success: {data.get('message Unknown')}")
            print(f"   Expires at: {data.get('expires_at Unknown')}")
            print(f"   Total consumed: {data.get('consumed_count', 0)}")
            return True
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def test_status():
    """Test getting service status"""
    print(f"\n=== Testing GET /status ===")
    try:
        response = requests.get(f"{BASE_URL}/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success: Service is {data.get('status unknown')}")
            print(f"   RAM usage: {data.get('ram_usage_percent', 0):.1f}%")
            print(f"   Cached names: {data.get('cached_names', 0)}")
            print(f"   Consumed variants: {data.get('consumed_variants', 0)}")
            print(f"   Queue size: {data.get('queue_size', 0)}")
            print(f"   Currently generating: {data.get('currently_generating', [])}")
            return True
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def test_request_whole_pool(name):
    """Test requesting whole pool generation"""
    print(f"\n=== Testing POST /pool?name={name} ===")
    try:
        response = requests.post(f"{BASE_URL}/pool", params={"name": name})
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success: {data.get('message Unknown')}")
            print(f"   Status: {data.get('status Unknown')}")
            return True
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def test_concurrent_requests(names):
    """Test concurrent requests for the same name"""
    print(f"\n=== Testing concurrent requests for {names} ===")
    try:
        # Make multiple requests simultaneously
        import threading
        import time
        if isinstance(names, str):
            names = [names]
        
        results = []
        errors = []
        
        def make_request(name):
            try:
                import random
                name = "".join(list("hahaha") + random.sample(list("abcde"), 5) + list(name))
                response = requests.get(f"{BASE_URL}/pool", params={"original_name": name}, timeout=10)
                results.append(response.json())
            except Exception as e:
                import traceback
                traceback.print_exc()
                errors.append(str(e))
        
        # Start multiple threads
        threads = []
        # for i in range(len(names)):
        #     thread = threading.Thread(target=make_request, args=(names[i],))
        #     threads.append(thread)
        #     thread.start()
        # for i in range(len(names)):
        for i in range(30):
            thread = threading.Thread(target=make_request, args=(f"{i}",))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        if results:
            print(f"✅ Success: {len(results)} requests completed")
            print(f"   Errors: {len(errors)}")
            return True
        else:
            print(f"❌ All requests failed")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def test_stop_generation():
    """Test stopping pool generation"""
    print(f"\n=== Testing POST /stop-generation ===")
    try:
        response = requests.post(f"{BASE_URL}/stop-generation")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success: {data.get('message Unknown')}")
            print(f"   Status: {data.get('status Unknown')}")
            return True
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def test_pool_stats():
    """Test checking pool_stats in responses"""
    print(f"\n=== Testing pool_stats field ===")
    try:
        # Test with a simple name
        response = requests.get(f"{BASE_URL}/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success: Got status response")
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Instance generating: {data.get('currently_generating_instance', [])}")
            print(f"   Whole generating: {data.get('currently_generating_whole', [])}")
            return True
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def extract_variants(pools, max_variants=5):
    """Extract some variants from pools for testing"""
    variants = []
    for ld_level in pools:
        for orth_level in ld_level:
            for variant_list in orth_level:
                variants.extend(variant_list[:max_variants])
                if len(variants) >= max_variants:
                    return variants[:max_variants]
    return variants

def test_concurrent_instance_whole_generation():
    """Test that instance pool generation works even when whole pool is being generated"""
    print(f"\n=== Testing concurrent instance/whole pool generation ===")
    try:
        test_name = "concurrent_test"
        
        # First, request whole pool generation (this will create a cache file)
        print(f"   Requesting whole pool generation for '{test_name}'...")
        response = requests.post(f"{BASE_URL}/pool", params={"name": test_name})
        if response.status_code != 200:
            print(f"   ❌ Failed to request whole pool: {response.status_code}")
            return False
        
        # Wait a moment for the worker to start processing
        time.sleep(2)
        
        # Check status to see if whole pool generation is running
        status_response = requests.get(f"{BASE_URL}/status")
        if status_response.status_code == 200:
            status_data = status_response.json()
            whole_generating = status_data.get('currently_generating_whole', [])
            if test_name in whole_generating:
                print(f"   ✅ Whole pool generation is running for '{test_name}'")
            else:
                print(f"   ⚠️  Whole pool generation not yet started for '{test_name}'")
        
        # Now try to get instance pool (should work even if whole pool is generating)
        print(f"   Requesting instance pool for '{test_name}'...")
        instance_response = requests.get(f"{BASE_URL}/pool", params={"original_name": test_name})
        
        if instance_response.status_code == 200:
            instance_data = instance_response.json()
            source = instance_data.get('source', 'unknown')
            print(f"   ✅ Instance pool request successful: {source}")
            
            # Check if we got actual pool data
            pools = instance_data.get('pools', [])
            if pools:
                print(f"   ✅ Got pool data with {len(pools)} levels")
                return True
            else:
                print(f"   ⚠️  Got response but no pool data")
                return False
        else:
            print(f"   ❌ Instance pool request failed: {instance_response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False

def test_timeout_parameter():
    """Test the timeout parameter in GET /pool request"""
    print(f"\n=== Testing timeout parameter ===")
    try:
        test_name = "timeout_test"
        
        # Test with custom timeout (shorter than default)
        print(f"   Testing with custom timeout (30 seconds)...")
        response = requests.get(f"{BASE_URL}/pool", params={"original_name": test_name, "timeout": 30.0})
        
        if response.status_code == 200:
            data = response.json()
            source = data.get('source', 'unknown')
            print(f"   ✅ Request successful: {source}")
            
            # Check if we got pool data
            pools = data.get('pools', [])
            if pools:
                print(f"   ✅ Got pool data with {len(pools)} levels")
                return True
            else:
                print(f"   ⚠️  Got response but no pool data")
                return False
        else:
            print(f"   ❌ Request failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False

def test_termination_mechanism():
    """Test the termination mechanism for worker processes"""
    print(f"\n=== Testing termination mechanism ===")
    try:
        test_name = "termination_test"
        
        # Start a whole pool generation
        print(f"   Starting whole pool generation for '{test_name}'...")
        response = requests.post(f"{BASE_URL}/pool", params={"name": test_name})
        if response.status_code != 200:
            print(f"   ❌ Failed to start whole pool generation: {response.status_code}")
            return False
        
        # Wait a moment for generation to start
        time.sleep(2)
        
        # Check status to see if whole pool generation is running
        status_response = requests.get(f"{BASE_URL}/status")
        if status_response.status_code == 200:
            status_data = status_response.json()
            whole_generating = status_data.get('currently_generating_whole', [])
            if test_name in whole_generating:
                print(f"   ✅ Whole pool generation is running for '{test_name}'")
            else:
                print(f"   ⚠️  Whole pool generation not yet started for '{test_name}'")
        
        # Send termination signal to test graceful shutdown
        print(f"   Testing graceful termination...")
        terminate_response = requests.post(f"{BASE_URL}/stop-generation")
        if terminate_response.status_code == 200:
            print(f"   ✅ Termination signal sent successfully")
        else:
            print(f"   ❌ Failed to send termination signal: {terminate_response.status_code}")
        
        # Wait for termination to take effect
        time.sleep(3)
        
        # Check status again
        status_response = requests.get(f"{BASE_URL}/status")
        if status_response.status_code == 200:
            status_data = status_response.json()
            generation_stopped = status_data.get('generation_stopped', False)
            if generation_stopped:
                print(f"   ✅ Generation stopped successfully")
                return True
            else:
                print(f"   ⚠️  Generation not stopped yet")
                return False
        else:
            print(f"   ❌ Failed to get status: {status_response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting Name Variant Generation Service Tests")
    
    # Test names
    test_names = [
        "john",
        "alice",
        "bob",
        "melinda",
        "hill",
        "kevin",
        "brian",
        "angela",
        "tony",
        "rodgers",
        "jessica",
        "coates",
        "arthur",
        "briana",
        "morrow",
        "kenneth",
        "mack",
        "phillip",
        "walker",
        "pamela",
        "michelle",
        "miller",
        "joy",
        "serrano",
        "terence",
        "hudson",
        "ashley",
        "william",
        "apple",
        "orange",
        "jason",
        "renee",
        "coles",
        "ernest",
        "adams",
        "jacob",
        "mallory",
        "kimberly",
        "moore",
        "jeremy",
        "barber",
        "joshua",
        "patterson",
        "benjamin",
        "kelsey",
        "leach",
        "ashley",
        "matthews",
        "joe",
        "dylan",
        "carlos",
        "williams",
        "elizabeth",
        "mary",
        "marshall",
        "leigh",
        "ellis",
        "lee",
        "frances",
        "kristen",
        "heather",
        "butler",
        "desiree",
        "henry",
        "christopher",
        "williams",
        "gerald",
        "young",
        "carol",
        "hooper",
        "thomas",
        "justin",
        "johnson",
        "jay",
        "lauren",
        "angela",
        "ryan"
    ]
    
    # for name in test_names:
    #     print(f"\n{'='*50}")
    #     print(f"Testing with name: {name}")
    #     print(f"{'='*50}")
        
    #     # Test 1: Get variants
    #     pools = test_get_variants(name)
    #     if pools:
    #         # Extract some variants for testing
    #         variants = extract_variants(pools, 3)
    #         if variants:
    #             print(f"   Extracted variants for testing: {variants}")
                
    #             # Test 2: Mark some variants as consumed
    #             test_mark_consumed(name, variants[:2])
                
    #             # Test 3: Get variants again (should have fewer)
    #             time.sleep(1)  # Small delay
    #             test_get_variants(name)
    
    # # Test 5: Request whole pool generation
    # for name in test_names:
    #     test_request_whole_pool(name)
    
    # # Test 6: Concurrent requests
    test_concurrent_requests(test_names)
    
    # # Test 4: Service status
    # test_status()

    # # # Test 7: Stop generation
    # # test_stop_generation()
    
    # # # Test 8: Check pool_stats
    # test_pool_stats()

    # # Test 9: Concurrent instance/whole pool generation
    # test_concurrent_instance_whole_generation()
    
    # # Test 10: Timeout parameter
    # test_timeout_parameter()
    
    # # Test 11: Termination mechanism
    # test_termination_mechanism()
    
    print(f"\n{'='*50}")
    print("🎉 All tests completed!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
