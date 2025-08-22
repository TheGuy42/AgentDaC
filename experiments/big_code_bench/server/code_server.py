from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import sys
import os
from pathlib import Path
import argparse
import uvicorn
from typing import Optional

app = FastAPI(title="Code Execution Server", description="A server for executing Python code remotely")
current_dir = Path(__file__).parent.resolve()
trash_dir = os.path.join(current_dir, "trash")
os.makedirs(trash_dir, exist_ok=True)

class CodeRequest(BaseModel):
    code: str
    timeout: int = 10


class ServerConfig:
    def __init__(self, host: str = "localhost", port: int = 8002, workers: int = 1):
        self.host = host
        self.port = port
        self.workers = workers


@app.post("/execute")
async def execute_code(request: CodeRequest):
    try:
        # Run the code in a subprocess
        result = subprocess.run(
            [sys.executable, "-c", request.code],
            capture_output=True,
            text=True,
            timeout=request.timeout,
            cwd=trash_dir,
        )

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired as e:
        raise HTTPException(
            status_code=408,  # Request Timeout
            detail={
                "error_type": "timeout",
                "message": f"Code execution timed out after {request.timeout} seconds",
                "timeout": request.timeout,
                "partial_stdout": e.stdout.decode() if e.stdout else "",
                "partial_stderr": e.stderr.decode() if e.stderr else "",
            }
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=422,  # Unprocessable Entity
            detail={
                "error_type": "execution_error",
                "message": "Code execution failed",
                "returncode": e.returncode,
                "stdout": e.stdout.decode() if e.stdout else "",
                "stderr": e.stderr.decode() if e.stderr else "",
            }
        )
    except PermissionError as e:
        raise HTTPException(
            status_code=403,  # Forbidden
            detail={
                "error_type": "permission_error",
                "message": "Permission denied during code execution",
                "details": str(e)
            }
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,  # Internal Server Error
            detail={
                "error_type": "runtime_error",
                "message": "Python interpreter not found or system error",
                "details": str(e)
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,  # Internal Server Error
            detail={
                "error_type": "unknown_error",
                "message": "An unexpected error occurred during code execution",
                "details": str(e)
            }
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Code execution server is running"}


@app.get("/")
async def root():
    """Root endpoint with basic information"""
    return {
        "message": "Code Execution Server",
        "endpoints": {
            "execute": "/execute - POST - Execute Python code",
            "health": "/health - GET - Health check",
            "docs": "/docs - GET - API documentation"
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Run the code execution server")
    parser.add_argument("--host", default="localhost", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8002, help="Port to bind the server to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    config = ServerConfig(host=args.host, port=args.port, workers=args.workers)
    
    print(f"Starting Code Execution Server on {config.host}:{config.port}")
    
    uvicorn.run(
        "code_server:app",
        host=config.host,
        port=config.port,
        workers=config.workers,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
