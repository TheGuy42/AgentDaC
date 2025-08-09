# Code Execution Server and Client

This directory contains a FastAPI-based code execution server and a Python client for remotely executing code.

## Features

### Server (`code_server.py`)
- FastAPI-based REST API for executing Python code
- Configurable host, port, and worker processes
- Built-in health check and info endpoints
- Request timeout handling
- Subprocess isolation for code execution

### Client (`code_client.py`)
- Simple Python client for interacting with the server
- Command-line interface with multiple modes
- Interactive mode for real-time code execution
- File-based code execution
- Health checking and server info retrieval

## Quick Start

### 1. Install Dependencies

```bash
# From the project root
pip install fastapi uvicorn requests
# or if using uv:
uv add fastapi uvicorn requests
```

### 2. Start the Server

```bash
# Basic usage (localhost:8000)
python code_server.py

# With custom configuration
python code_server.py --host 0.0.0.0 --port 8080 --workers 4

# Development mode with auto-reload
python code_server.py --reload
```

### 3. Use the Client

#### Command Line Usage

```bash
# Execute simple code
python code_client.py execute "print('Hello World')"

# Execute code from file
python code_client.py execute --file script.py

# Check server health
python code_client.py health

# Get server information
python code_client.py info

# Interactive mode
python code_client.py interactive
```

#### Programmatic Usage

```python
from code_client import CodeClient

# Create client
client = CodeClient(host="localhost", port=8000)

# Execute code
result = client.execute_code("print('Hello from remote!')")
print(result.stdout)  # Output: Hello from remote!

# Check if execution was successful
if result.success:
    print("Code executed successfully")
else:
    print(f"Error: {result.stderr}")
```

## API Endpoints

### Server Endpoints

- `POST /execute` - Execute Python code
- `GET /health` - Health check
- `GET /` - Server information
- `GET /docs` - Auto-generated API documentation

### Execute Endpoint

**Request:**
```json
{
    "code": "print('Hello World')",
    "timeout": 10
}
```

**Successful Response:**
```json
{
    "stdout": "Hello World\n",
    "stderr": "",
    "returncode": 0
}
```

**Error Responses:**

The server returns detailed error information with appropriate HTTP status codes:

**Timeout Error (408):**
```json
{
    "detail": {
        "error_type": "timeout",
        "message": "Code execution timed out after 5 seconds",
        "timeout": 5,
        "partial_stdout": "Starting...\n",
        "partial_stderr": ""
    }
}
```

**Execution Error (422):**
```json
{
    "detail": {
        "error_type": "execution_error", 
        "message": "Code execution failed",
        "returncode": 1,
        "stdout": "",
        "stderr": "NameError: name 'undefined_variable' is not defined\n"
    }
}
```

**Permission Error (403):**
```json
{
    "detail": {
        "error_type": "permission_error",
        "message": "Permission denied during code execution",
        "details": "Access denied to system resource"
    }
}
```

**System Error (500):**
```json
{
    "detail": {
        "error_type": "runtime_error",
        "message": "Python interpreter not found or system error",
        "details": "Command not found"
    }
}
```

## Configuration Options

### Server Configuration

```bash
python code_server.py --help
```

Options:
- `--host`: Host to bind the server to (default: localhost)
- `--port`: Port to bind the server to (default: 8000)
- `--workers`: Number of worker processes (default: 1)
- `--reload`: Enable auto-reload for development

### Client Configuration

```bash
python code_client.py --help
```

Options:
- `--host`: Server host (default: localhost)
- `--port`: Server port (default: 8000)
- `--timeout`: Request timeout in seconds (default: 30)

## Examples

### Example 1: Basic Math Calculation

```bash
python code_client.py execute "result = 2 + 2; print(f'2 + 2 = {result}')"
```

### Example 2: Multi-line Code

```bash
python code_client.py execute "
for i in range(5):
    print(f'Number: {i}')
"
```

### Example 3: Interactive Session

```bash
python code_client.py interactive
>>> x = 10
>>> y = 20
>>> print(x + y)
30
>>> exit
```

### Example 4: Error Handling

The client now returns `ExecutionResult` objects instead of raising exceptions for most server-side errors, making error handling more consistent:

```python
from code_client import CodeClient

client = CodeClient()

# Handle timeout - returns ExecutionResult
result = client.execute_code("import time; time.sleep(10)", execution_timeout=2)
if result.timeout_occurred:
    print(f"Code timed out: {result.error_message}")
    print(f"Partial output: {result.stdout}")

# Handle execution errors - returns ExecutionResult
result = client.execute_code("print(undefined_variable)")
if not result.success:
    print(f"Execution failed: {result.stderr}")
    print(f"Error type: {result.error_type}")

# Connection errors still raise exceptions
try:
    client = CodeClient(port=9999)  # Non-existent server
    result = client.execute_code("print('hello')")
except ConnectionError as e:
    print(f"Connection failed: {e}")
```

### Example 5: Enhanced ExecutionResult

The `ExecutionResult` object now contains additional error information and execution timing:

```python
result = client.execute_code("x = 1/0")  # Division by zero
print(f"Success: {result.success}")           # False
print(f"Return code: {result.returncode}")    # 1
print(f"Execution time: {result.execution_time:.3f}s")  # e.g., 0.125s
print(f"Error type: {result.error_type}")     # None (normal execution error)
print(f"Timeout occurred: {result.timeout_occurred}")  # False
print(f"Error output: {result.stderr}")       # ZeroDivisionError details

# For timeouts:
result = client.execute_code("import time; time.sleep(5)", execution_timeout=1)
print(f"Timeout occurred: {result.timeout_occurred}")  # True
print(f"Error type: {result.error_type}")     # "timeout"
print(f"Error message: {result.error_message}")  # Timeout details
print(f"Execution time: {result.execution_time:.3f}s")  # Time until timeout

# Performance monitoring:
result = client.execute_code("sum(range(100000))")
print(f"Computation took: {result.execution_time:.3f}s")
```

```python
# The client now provides detailed error information
result = client.execute_code("x = 1/0")  # This will succeed but return error info
print(f"Success: {result.success}")
print(f"Return code: {result.returncode}")
print(f"Error output: {result.stderr}")
```

## Security Considerations

⚠️ **Warning**: This server executes arbitrary Python code. Only use in trusted environments and consider the following:

1. **Network Security**: Bind to localhost only unless needed
2. **Code Restrictions**: Consider implementing code filtering/sandboxing
3. **Resource Limits**: Set appropriate timeouts and resource constraints
4. **Authentication**: Add authentication for production use

## Development

### Running the Example

```bash
# Run example with explanations
python example_usage.py

# Run example with temporary server
python example_usage.py --demo-server
```

### Testing

1. Start the server: `python code_server.py`
2. Run the client tests: `python example_usage.py`
3. Try interactive mode: `python code_client.py interactive`

## File Structure

```
Server/
├── code_server.py      # FastAPI server implementation
├── code_client.py      # Python client implementation
├── example_usage.py    # Usage examples and demos
└── README.md          # This file
```
