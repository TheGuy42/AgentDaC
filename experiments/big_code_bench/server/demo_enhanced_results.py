#!/usr/bin/env python3
"""
Simple demo showing the enhanced ExecutionResult behavior

This demonstrates how the client now returns ExecutionResult objects
instead of raising exceptions for most server-side errors.
"""

import sys
from pathlib import Path

# Add the current directory to the Python path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from code_client import CodeClient


def demo_new_behavior():
    """Demonstrate the new ExecutionResult-based error handling"""
    print("Enhanced ExecutionResult Demo")
    print("=" * 40)
    
    client = CodeClient()
    
    # Check server health
    if not client.health_check():
        print("❌ Server is not running. Please start the server first:")
        print("   python code_server.py")
        return
    
    print("✅ Server is running\n")
    
    # 1. Successful execution
    print("1. Successful execution:")
    result = client.execute_code("print('Hello, World!')")
    print(f"   Success: {result.success}")
    print(f"   Execution time: {result.execution_time:.3f}s")
    print(f"   Output: {result.stdout.strip()}")
    print()
    
    # 2. Runtime error (NameError) - now returns ExecutionResult
    print("2. Runtime error (NameError):")
    result = client.execute_code("print(undefined_variable)")
    print(f"   Success: {result.success}")
    print(f"   Execution time: {result.execution_time:.3f}s")
    print(f"   Return code: {result.returncode}")
    print(f"   Error type: {result.error_type}")
    print(f"   STDERR: {result.stderr.strip()[:80]}...")
    print()
    
    # 3. Syntax error - returns ExecutionResult
    print("3. Syntax error:")
    result = client.execute_code("print('unclosed string")
    print(f"   Success: {result.success}")
    print(f"   Execution time: {result.execution_time:.3f}s")
    print(f"   Return code: {result.returncode}")
    print(f"   STDERR: {result.stderr.strip()[:80]}...")
    print()
    
    # 4. Timeout error - returns ExecutionResult with timeout info
    print("4. Timeout error:")
    result = client.execute_code("""
import time
print("Starting...")
time.sleep(3)
print("Should not reach here")
""", execution_timeout=1)
    print(f"   Success: {result.success}")
    print(f"   Execution time: {result.execution_time:.3f}s")
    print(f"   Timeout occurred: {result.timeout_occurred}")
    print(f"   Error type: {result.error_type}")
    print(f"   Error message: {result.error_message}")
    print(f"   Partial STDOUT: {repr(result.stdout)}")
    print()
    
    # 5. Import error - returns ExecutionResult
    print("5. Import error:")
    result = client.execute_code("import nonexistent_module_xyz")
    print(f"   Success: {result.success}")
    print(f"   Execution time: {result.execution_time:.3f}s")
    print(f"   Return code: {result.returncode}")
    print(f"   STDERR: {result.stderr.strip()[:80]}...")
    print()
    
    # 6. Connection error - still raises exception
    print("6. Connection error (still raises exception):")
    try:
        disconnected_client = CodeClient(port=9999)
        result = disconnected_client.execute_code("print('hello')")
        print(f"   Unexpected success: {result}")
    except ConnectionError as e:
        print(f"   ✅ Correctly raised ConnectionError: {e}")
    except Exception as e:
        print(f"   ❌ Unexpected error: {type(e).__name__}: {e}")
    
    print("\n" + "=" * 40)
    print("Key Benefits:")
    print("• Server-side errors return ExecutionResult objects")
    print("• No need for try/except for code execution errors")
    print("• Consistent API for handling different error types")
    print("• Client-side errors (network) still raise exceptions")
    print("• Timeout information preserved in results")
    print("• Execution time tracking for performance analysis")
    
    # 7. Performance test
    print("\n=== Performance Examples ===")
    performance_tests = [
        ("Fast operation", "x = 1 + 1"),
        ("Medium operation", "sum(range(10000))"),
        ("Slower operation", "import time; time.sleep(0.1); print('done')"),
    ]
    
    for name, code in performance_tests:
        result = client.execute_code(code)
        print(f"{name}: {result.execution_time:.3f}s")


if __name__ == "__main__":
    demo_new_behavior()
