import requests
import json
import argparse
import sys
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """Result of code execution"""
    stdout: str
    stderr: str
    returncode: int
    success: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    timeout_occurred: bool = False
    execution_time: Optional[float] = None  # Execution time in seconds
    
    def __str__(self) -> str:
        result = f"Return Code: {self.returncode}\n"
        if self.execution_time is not None:
            result += f"Execution Time: {self.execution_time:.3f}s\n"
        if self.error_type:
            result += f"Error Type: {self.error_type}\n"
        if self.error_message:
            result += f"Error Message: {self.error_message}\n"
        if self.timeout_occurred:
            result += "⚠️  Execution timed out\n"
        if self.stdout:
            result += f"STDOUT:\n{self.stdout}\n"
        if self.stderr:
            result += f"STDERR:\n{self.stderr}\n"
        return result


class CodeClient:
    """Client for interacting with the code execution server"""
    
    def __init__(self, host: str = "localhost", port: int = 8002, timeout_buffer: int = 2):
        self.base_url = f"http://{host}:{port}"
        self.timeout_buffer = timeout_buffer
        
    def execute_code(self, code: str, execution_timeout: int = 10) -> ExecutionResult:
        """
        Execute Python code on the remote server
        
        Args:
            code: Python code to execute
            execution_timeout: Timeout for code execution in seconds
            
        Returns:
            ExecutionResult containing the output and status
        """
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/execute",
                json={"code": code, "timeout": execution_timeout},
                # timeout=self.timeout
                timeout=execution_timeout + self.timeout_buffer  # Add a small buffer to avoid timeout issues
            )
            response.raise_for_status()
            
            execution_time = time.time() - start_time
            
            data = response.json()
            return ExecutionResult(
                stdout=data["stdout"],
                stderr=data["stderr"],
                returncode=data["returncode"],
                success=True,
                execution_time=execution_time
            )
            
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Could not connect to server at {self.base_url}")
        except requests.exceptions.Timeout:
            # Client-side timeout - still raise as this is a network issue
            raise TimeoutError(f"Request timed out after {execution_timeout + self.timeout_buffer} seconds")
        except requests.exceptions.HTTPError as e:
            execution_time = time.time() - start_time
            
            # Parse the detailed error response from the server
            try:
                error_detail = e.response.json().get("detail", {})
                if isinstance(error_detail, dict):
                    error_type = error_detail.get("error_type", "unknown")
                    message = error_detail.get("message", str(e))
                    
                    if error_type == "timeout":
                        # Server-side timeout - return ExecutionResult
                        timeout_val = error_detail.get("timeout", "unknown")
                        partial_stdout = error_detail.get("partial_stdout", "")
                        partial_stderr = error_detail.get("partial_stderr", "")
                        return ExecutionResult(
                            stdout=partial_stdout,
                            stderr=partial_stderr,
                            returncode=-1,
                            success=False,
                            error_type="timeout",
                            error_message=f"Code execution timed out after {timeout_val} seconds",
                            timeout_occurred=True,
                            execution_time=execution_time
                        )
                    elif error_type == "execution_error":
                        # Execution error - return ExecutionResult
                        stdout = error_detail.get("stdout", "")
                        stderr = error_detail.get("stderr", "")
                        returncode = error_detail.get("returncode", -1)
                        return ExecutionResult(
                            stdout=stdout,
                            stderr=stderr,
                            returncode=returncode,
                            success=False,
                            error_type="execution_error",
                            error_message=message,
                            execution_time=execution_time
                        )
                    elif error_type == "permission_error":
                        # Permission error - return ExecutionResult
                        return ExecutionResult(
                            stdout="",
                            stderr=f"Permission denied: {message}",
                            returncode=-1,
                            success=False,
                            error_type="permission_error",
                            error_message=message,
                            execution_time=execution_time
                        )
                    elif error_type == "runtime_error":
                        # Runtime error - return ExecutionResult
                        details = error_detail.get("details", "")
                        return ExecutionResult(
                            stdout="",
                            stderr=f"Runtime error: {message}\nDetails: {details}",
                            returncode=-1,
                            success=False,
                            error_type="runtime_error",
                            error_message=message,
                            execution_time=execution_time
                        )
                    else:
                        # Unknown server error - return ExecutionResult
                        return ExecutionResult(
                            stdout="",
                            stderr=f"Server error ({error_type}): {message}",
                            returncode=-1,
                            success=False,
                            error_type=error_type,
                            error_message=message,
                            execution_time=execution_time
                        )
                else:
                    # Fallback for simple string error details
                    return ExecutionResult(
                        stdout="",
                        stderr=f"Server returned error: {e.response.status_code} - {error_detail}",
                        returncode=-1,
                        success=False,
                        error_type="server_error",
                        error_message=str(error_detail),
                        execution_time=execution_time
                    )
            except (ValueError, KeyError):
                # Fallback if response is not JSON or doesn't have expected structure
                return ExecutionResult(
                    stdout="",
                    stderr=f"Server returned error: {e.response.status_code} - {e.response.text}",
                    returncode=-1,
                    success=False,
                    error_type="server_error",
                    error_message=f"HTTP {e.response.status_code}",
                    execution_time=execution_time
                )
        except Exception as e:
            # Unexpected client-side error - still raise as this is likely a programming error
            raise RuntimeError(f"Unexpected error: {str(e)}")
    
    def health_check(self) -> bool:
        """
        Check if the server is healthy
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_server_info(self) -> Dict[str, Any]:
        """
        Get server information
        
        Returns:
            Dictionary with server information
        """
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Could not get server info: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Code execution client")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8002, help="Server port")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Execute command
    execute_parser = subparsers.add_parser("execute", help="Execute Python code")
    execute_parser.add_argument("code", help="Python code to execute (or use --file)")
    execute_parser.add_argument("--file", help="Execute code from file instead")
    execute_parser.add_argument("--exec-timeout", type=int, default=10, help="Code execution timeout")
    
    # Health check command
    subparsers.add_parser("health", help="Check server health")
    
    # Info command
    subparsers.add_parser("info", help="Get server information")
    
    # Interactive mode
    subparsers.add_parser("interactive", help="Start interactive mode")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    client = CodeClient(host=args.host, port=args.port, timeout_buffer=args.timeout)
    
    try:
        if args.command == "execute":
            if args.file:
                try:
                    with open(args.file, 'r') as f:
                        code = f.read()
                except FileNotFoundError:
                    print(f"Error: File '{args.file}' not found")
                    sys.exit(1)
            else:
                code = args.code
            
            print(f"Executing code on {args.host}:{args.port}...")
            result = client.execute_code(code, args.exec_timeout)
            print(result)
            
            if not result.success:
                sys.exit(1)
                
        elif args.command == "health":
            if client.health_check():
                print(f"✅ Server at {args.host}:{args.port} is healthy")
            else:
                print(f"❌ Server at {args.host}:{args.port} is not responding")
                sys.exit(1)
                
        elif args.command == "info":
            info = client.get_server_info()
            print("Server Information:")
            print(json.dumps(info, indent=2))
            
        elif args.command == "interactive":
            interactive_mode(client)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def interactive_mode(client: CodeClient):
    """Interactive mode for executing multiple code snippets"""
    print("🐍 Interactive Code Execution Mode")
    print("Enter Python code (type 'exit' to quit, 'help' for commands)")
    print("Use '###' on a new line to execute multi-line code")
    print("-" * 50)
    
    while True:
        try:
            # Check if server is still healthy
            if not client.health_check():
                print("❌ Lost connection to server")
                break
                
            code_lines = []
            print(">>> ", end="")
            
            while True:
                line = input()
                
                if line.strip() == "exit":
                    print("Goodbye! 👋")
                    return
                elif line.strip() == "help":
                    print("Commands:")
                    print("  exit    - Exit interactive mode")
                    print("  help    - Show this help")
                    print("  ###     - Execute multi-line code")
                    print("  Single line code is executed immediately")
                    break
                elif line.strip() == "###":
                    if code_lines:
                        code = "\n".join(code_lines)
                        execute_and_display(client, code)
                        code_lines = []
                    break
                else:
                    code_lines.append(line)
                    if len(code_lines) == 1 and line.strip():
                        # Single line - execute immediately
                        execute_and_display(client, line)
                        break
                    else:
                        # Multi-line mode
                        print("... ", end="")
                        
        except KeyboardInterrupt:
            print("\nUse 'exit' to quit")
        except EOFError:
            print("\nGoodbye! 👋")
            break


def execute_and_display(client: CodeClient, code: str):
    """Execute code and display results in interactive mode"""
    try:
        result = client.execute_code(code.strip())
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"Error: {result.stderr}")
        if result.returncode != 0:
            print(f"Exit code: {result.returncode}")
    except Exception as e:
        print(f"Execution error: {e}")


if __name__ == "__main__":
    main()