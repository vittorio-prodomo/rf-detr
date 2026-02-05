"""
GPU Test Runner for RF-DETR
Run tests on Modal's GPU infrastructure with pytest interface similar to local execution.

Usage:
    modal run .modal/test_runner.py --test-path tests/ --pytest-args "-v"
    modal run .modal/test_runner.py --test-path tests/datasets/ --pytest-args "-v -k synthetic"
    
Environment Variables:
    MODAL_GPU: GPU type to use (default: L4, options: L4, T4, A10G, A100, etc.)
"""

import sys
import os
from pathlib import Path
import modal

# Create Modal app with profile name from environment or default
app = modal.App("rf-detr-gpu-tests")

# Get GPU type from environment or default to L4
GPU_TYPE = os.environ.get("MODAL_GPU", "L4")

# Start with PyTorch public image for faster builds
# Using PyTorch 2.8 with CUDA 12.6
# Note: copy=True is required when running build commands after add_local_dir
image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/pytorch:25.01-py3",  # PyTorch 2.8 with CUDA 12.6
        add_python="3.10",
    )
    .apt_install("git")
    .pip_install("uv")
    # Copy project files, excluding large/unnecessary files using .gitignore patterns
    # Using copy=True so we can install dependencies during build
    .add_local_dir(
        ".",
        remote_path="/root/project",
        copy=True,
        # todo: improve this ignoring to be dynamic
        ignore=[
            ".git",
            "__pycache__",
            "*.pyc",
            ".pytest_cache",
            ".venv",
            "venv",
            "*.egg-info",
            ".DS_Store",
            "*.log",
            ".modal",
            "debugging",
            "docs",
            "test_synthetic_output",
            "uv.lock",
        ]
    )
    .workdir("/root/project")
    # Install dependencies during image build for faster execution
    # Note: tests dependencies are in [dependency-groups] not [project.optional-dependencies]
    .run_commands(
        "uv pip install -e . --group tests --system"
    )
)


@app.function(
    image=image,
    gpu=GPU_TYPE,  # GPU type from environment variable
    timeout=3600,  # Hard 1 hour timeout safety limit
)
def run_tests(test_path: str = "tests/", pytest_args: str = "-v"):
    """Run pytest on Modal GPU infrastructure."""
    import subprocess
    import os
    
    # Change to project directory
    os.chdir("/root/project")
    
    # Verify GPU availability
    print("\n" + "="*80)
    print("GPU ENVIRONMENT CHECK")
    print("="*80)
    try:
        import torch
        print(f"🎮 GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🎮 GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"🎮 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    except Exception as e:
        print(f"⚠️  GPU check failed: {e}")
    print("="*80 + "\n")
    
    # Build pytest command
    pytest_cmd = ["pytest", test_path]
    
    # Add user-provided pytest arguments
    if pytest_args:
        # Split args properly, handling quoted strings
        import shlex
        pytest_cmd.extend(shlex.split(pytest_args))
    
    # Force colored output for better readability
    pytest_cmd.append("--color=yes")
    
    print("="*80)
    print(f"RUNNING TESTS")
    print("="*80)
    print(f"Command: {' '.join(pytest_cmd)}")
    print(f"Working directory: {os.getcwd()}")
    print("="*80 + "\n")
    
    # Create output directory for pytest logs
    output_dir = Path("/root/project/test-outputs")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "pytest-output.log"
    
    # Run pytest and capture output to both console and file
    try:
        with open(output_file, "w") as log_file:
            # Run pytest with output going to both stdout and file
            result = subprocess.run(
                pytest_cmd,
                cwd="/root/project",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            
            # Write to file
            log_file.write(result.stdout)
            # Also print to console for real-time viewing
            print(result.stdout, end="")
        
        print("\n" + "="*80)
        print("TEST EXECUTION COMPLETE")
        print("="*80)
        print(f"Exit code: {result.returncode}")
        print(f"📄 Test output saved to: {output_file}")
        print("="*80 + "\n")
        
        # Read the log file content to return
        with open(output_file, "r") as f:
            pytest_output = f.read()
        
        return {
            "returncode": result.returncode,
            "success": result.returncode == 0,
            "pytest_output": pytest_output,
            "output_file": str(output_file),
        }
    
    except Exception as e:
        print("\n" + "="*80)
        print("ERROR DURING TEST EXECUTION")
        print("="*80)
        print(f"Error: {e}")
        print("="*80 + "\n")
        
        return {
            "returncode": 1,
            "success": False,
            "error": str(e),
            "pytest_output": "",
        }


@app.local_entrypoint()
def main(
    test_path: str = "tests/",
    pytest_args: str = "-v",
):
    """Local entrypoint to run tests on Modal GPU."""
    print("\n" + "="*80)
    print("RF-DETR GPU TEST RUNNER")
    print("="*80)
    print(f"📁 Test Path: {test_path}")
    print(f"⚙️  Pytest Args: {pytest_args}")
    print(f"🎮 GPU: {GPU_TYPE}")
    print(f"⏱️  Timeout: 1 hour")
    print("="*80 + "\n")
    
    # Run tests remotely with streaming output
    result = run_tests.remote(test_path=test_path, pytest_args=pytest_args)
    
    # Save output to local file
    local_output_dir = Path("test-outputs")
    local_output_dir.mkdir(exist_ok=True)
    local_output_file = local_output_dir / "pytest-output.log"
    
    if result.get("pytest_output"):
        with open(local_output_file, "w") as f:
            f.write(result["pytest_output"])
        print(f"📄 Test output saved to: {local_output_file}")
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Return Code: {result['returncode']}")
    print(f"Success: {result['success']}")
    
    if not result["success"]:
        if "error" in result:
            print(f"Error: {result['error']}")
        print("="*80 + "\n")
        sys.exit(result["returncode"])
    
    print("="*80 + "\n")
    print("✅ All tests passed!")
