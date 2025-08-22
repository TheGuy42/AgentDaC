#!/usr/bin/env python3
"""
Example usage of the Code Execution Server and Client

This file demonstrates how to use both the server and client components.
"""

import time
import subprocess
import sys
from experiments.big_code_bench.server.code_client import CodeClient


def example_basic_usage():
    """Basic usage example"""
    print("=== Basic Usage Example ===")
    
    # Create a client (assumes server is running on localhost:8000)
    client = CodeClient()
    
    # Check if server is healthy
    if not client.health_check():
        print("❌ Server is not running. Please start the server first.")
        print("Run: python code_server.py")
        return
    
    print("✅ Server is healthy")
    
    # Execute simple code
    result = client.execute_code("print('Hello from remote execution!')")
    print("Result:", result)
    
    # Execute code with variables
    code = """
x = 10
y = 20
print(f"x + y = {x + y}")
for i in range(3):
    print(f"Loop iteration {i}")
"""
    result = client.execute_code(code)
    print("Multi-line result:", result)


def example_error_handling():
    """Example showing error handling"""
    print("\n=== Error Handling Example ===")
    
    client = CodeClient()
    
    # Test 1: NameError (execution error) - now returns ExecutionResult
    print("1. Testing NameError (returns ExecutionResult):")
    result = client.execute_code("print(undefined_variable)")
    print(f"  Success: {result.success}")
    print(f"  Error Type: {result.error_type}")
    print(f"  Error Message: {result.error_message}")
    print(f"  STDERR: {result.stderr[:100]}...")
    
    # Test 2: Timeout error - now returns ExecutionResult
    print("\n2. Testing Timeout (returns ExecutionResult):")
    code = """
import time
print("Starting...")
time.sleep(2)
print("This won't be reached")
"""
    result = client.execute_code(code, execution_timeout=1)
    print(f"  Success: {result.success}")
    print(f"  Timeout occurred: {result.timeout_occurred}")
    print(f"  Error Type: {result.error_type}")
    print(f"  Error Message: {result.error_message}")
    print(f"  Partial STDOUT: {result.stdout}")
    
    # Test 3: Syntax error - returns ExecutionResult with stderr
    print("\n3. Testing Syntax Error (returns ExecutionResult):")
    result = client.execute_code("print('unclosed string")
    print(f"  Success: {result.success}")
    print(f"  Return code: {result.returncode}")
    print(f"  STDERR: {result.stderr[:100]}...")
    
    # Test 4: Import error - returns ExecutionResult with stderr
    print("\n4. Testing Import Error (returns ExecutionResult):")
    result = client.execute_code("import nonexistent_module")
    print(f"  Success: {result.success}")
    print(f"  Return code: {result.returncode}")
    print(f"  STDERR: {result.stderr[:100]}...")
    
    # Test 5: Connection error - still raises exception
    print("\n5. Testing Connection Error (still raises exception):")
    try:
        disconnected_client = CodeClient(port=9999)  # Non-existent server
        result = disconnected_client.execute_code("print('hello')")
        print("  Unexpected success:", result)
    except ConnectionError as e:
        print(f"  Caught ConnectionError: {e}")
    except Exception as e:
        print(f"  Caught other error: {type(e).__name__}: {e}")


def example_with_timeout():
    """Example with custom timeout"""
    print("\n=== Timeout Example ===")
    
    client = CodeClient()
    
    # Execute code with a short timeout
    code = """
import time
print("Starting long operation...")
time.sleep(2)  # This might timeout if execution_timeout is set to 1
print("Finished!")
"""
    
    try:
        result = client.execute_code(code, execution_timeout=1)
        print("Result with 1s timeout:", result)
    except Exception as e:
        print(f"Timeout error: {e}")
    
    # Try again with longer timeout
    try:
        result = client.execute_code(code, execution_timeout=5)
        print("Result with 5s timeout:", result)
    except Exception as e:
        print(f"Error: {e}")


def example_server_info():
    """Example showing how to get server information"""
    print("\n=== Server Info Example ===")
    
    client = CodeClient()
    
    try:
        info = client.get_server_info()
        print("Server info:", info)
    except Exception as e:
        print(f"Could not get server info: {e}")


def start_server_demo():
    """Start server for demonstration (in a separate process)"""
    print("=== Starting Server Demo ===")
    print("Starting server on localhost:8001 for demo...")
    
    # Start server in background
    server_process = subprocess.Popen([
        sys.executable, "code_server.py", 
        "--host", "localhost", 
        "--port", "8001"
    ])
    
    # Wait a moment for server to start
    time.sleep(2)
    
    try:
        # Create client for the demo server
        client = CodeClient(host="localhost", port=8001)
        
        # Test the demo server
        if client.health_check():
            print("✅ Demo server is running")
            
            # Execute some code
            result = client.execute_code("print('Demo server working!')")
            print("Demo result:", result)
        else:
            print("❌ Demo server failed to start")
    
    finally:
        # Clean up - terminate the server
        server_process.terminate()
        server_process.wait()
        print("Demo server stopped")


if __name__ == "__main__":
    print("Code Execution Server/Client Example")
    print("=====================================")
    
    # Check if user wants to run the server demo
    if len(sys.argv) > 1 and sys.argv[1] == "--demo-server":
        start_server_demo()
    else:
        print("Make sure to start the server first:")
        print("  python code_server.py")
        print("\nOr run with --demo-server to start a temporary server:")
        print("  python example_usage.py --demo-server")
        print("\n" + "="*50)
        
        # Run the examples
        example_basic_usage()
        example_error_handling()
        example_with_timeout()
        example_server_info()
        
        print("\n=== Interactive Mode Demo ===")
        print("To try interactive mode, run:")
        print("  python code_client.py interactive")
