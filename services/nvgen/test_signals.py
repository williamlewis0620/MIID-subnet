#!/usr/bin/env python3
"""
Test script to verify signal handling in the nvgen service.
This script simulates various termination signals to test graceful shutdown.
"""

import os
import signal
import subprocess
import time
import requests
import sys

BASE_URL = "http://localhost:8000"

def test_service_startup():
    """Test if service starts correctly"""
    print("🧪 Testing service startup...")
    
    try:
        response = requests.get(f"{BASE_URL}/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Service is running: {data.get('status', 'unknown')}")
            return True
        else:
            print(f"❌ Service returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Service not responding: {e}")
        return False

def test_signal_handling():
    """Test signal handling by sending various signals"""
    print("\n🧪 Testing signal handling...")
    
    # Find the service process
    try:
        result = subprocess.run(
            ["pgrep", "-f", "python.*main.py"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("❌ Service process not found")
            return False
        
        pids = result.stdout.strip().split('\n')
        if not pids or not pids[0]:
            print("❌ No service process found")
            return False
        
        service_pid = int(pids[0])
        print(f"✅ Found service process: PID {service_pid}")
        
        # Test different signals
        signals_to_test = [
            (signal.SIGTERM, "SIGTERM"),
            (signal.SIGINT, "SIGINT"),
            (signal.SIGQUIT, "SIGQUIT"),
            (signal.SIGHUP, "SIGHUP"),
        ]
        
        for sig, sig_name in signals_to_test:
            print(f"\n📡 Testing {sig_name}...")
            
            # Send signal
            os.kill(service_pid, sig)
            
            # Wait a moment for signal processing
            time.sleep(2)
            
            # Check if service is still responding
            try:
                response = requests.get(f"{BASE_URL}/status", timeout=3)
                if response.status_code == 200:
                    data = response.json()
                    status = data.get('status', 'unknown')
                    print(f"   Service status: {status}")
                    
                    if status == "shutting_down":
                        print(f"   ✅ {sig_name} handled correctly - service is shutting down")
                    else:
                        print(f"   ⚠️  {sig_name} sent but service still running")
                else:
                    print(f"   ❌ Service not responding after {sig_name}")
            except requests.exceptions.RequestException:
                print(f"   ✅ {sig_name} handled correctly - service stopped")
                break
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing signals: {e}")
        return False

def test_ctrl_c_simulation():
    """Simulate Ctrl+C by sending SIGINT"""
    print("\n🧪 Testing Ctrl+C simulation...")
    
    try:
        # Start service in background
        print("   Starting service in background...")
        process = subprocess.Popen(
            [sys.executable, "main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for service to start
        time.sleep(3)
        
        # Check if service is running
        try:
            response = requests.get(f"{BASE_URL}/status", timeout=5)
            if response.status_code != 200:
                print("   ❌ Service failed to start")
                process.terminate()
                return False
        except requests.exceptions.RequestException:
            print("   ❌ Service not responding")
            process.terminate()
            return False
        
        print("   ✅ Service started successfully")
        
        # Send SIGINT (Ctrl+C)
        print("   📡 Sending SIGINT (Ctrl+C)...")
        process.send_signal(signal.SIGINT)
        
        # Wait for graceful shutdown
        time.sleep(5)
        
        # Check if process terminated gracefully
        if process.poll() is not None:
            print("   ✅ Service terminated gracefully after SIGINT")
            return True
        else:
            print("   ⚠️  Service still running after SIGINT, forcing termination...")
            process.terminate()
            time.sleep(2)
            if process.poll() is None:
                process.kill()
            return False
            
    except Exception as e:
        print(f"   ❌ Error in Ctrl+C test: {e}")
        return False

def test_worker_process_termination():
    """Test that worker processes are properly terminated on service shutdown"""
    print("\n🧪 Testing worker process termination...")
    
    try:
        # Start a whole pool generation to create worker processes
        print("   Starting whole pool generation to create worker processes...")
        response = requests.post(f"{BASE_URL}/pool", params={"name": "worker_test"})
        if response.status_code != 200:
            print(f"   ❌ Failed to start whole pool generation: {response.status_code}")
            return False
        
        # Wait a moment for worker processes to start
        time.sleep(3)
        
        # Check status to see if worker processes are running
        status_response = requests.get(f"{BASE_URL}/status")
        if status_response.status_code == 200:
            status_data = status_response.json()
            worker_processes = status_data.get('active_worker_processes', [])
            if worker_processes:
                print(f"   ✅ Worker processes running: {worker_processes}")
            else:
                print(f"   ⚠️  No worker processes detected")
        
        # Find the service process
        result = subprocess.run(
            ["pgrep", "-f", "python.*main.py"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("   ❌ Service process not found")
            return False
        
        service_pid = int(result.stdout.strip().split('\n')[0])
        print(f"   ✅ Found service process: PID {service_pid}")
        
        # Send SIGTERM to the service
        print("   📡 Sending SIGTERM to service...")
        os.kill(service_pid, signal.SIGTERM)
        
        # Wait for service to terminate
        time.sleep(5)
        
        # Check if service process is still running
        try:
            os.kill(service_pid, 0)  # Check if process exists
            print(f"   ⚠️  Service process {service_pid} still running")
            return False
        except OSError:
            print(f"   ✅ Service process {service_pid} terminated successfully")
        
        # Check if any worker processes are still running
        for worker_pid in worker_processes:
            try:
                os.kill(worker_pid, 0)  # Check if process exists
                print(f"   ❌ Worker process {worker_pid} still running")
                return False
            except OSError:
                print(f"   ✅ Worker process {worker_pid} terminated successfully")
        
        print("   ✅ All worker processes terminated successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Error in worker process termination test: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Signal Handling Test Suite")
    print("=" * 50)
    
    # Test 1: Service startup
    if not test_service_startup():
        print("\n❌ Service startup test failed")
        return False
    
    # Test 2: Signal handling
    if not test_signal_handling():
        print("\n❌ Signal handling test failed")
        return False
    
    # Test 3: Ctrl+C simulation
    if not test_ctrl_c_simulation():
        print("\n❌ Ctrl+C simulation test failed")
        return False
    
    # Test 4: Worker process termination
    if not test_worker_process_termination():
        print("\n❌ Worker process termination test failed")
        return False
    
    print("\n✅ All signal handling tests passed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
