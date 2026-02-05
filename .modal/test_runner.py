"""
GPU Test Runner for RF-DETR
Run tests on Modal's GPU infrastructure with pytest interface similar to local execution.

Usage:
    modal run .modal/test_runner.py --test-path tests/ --pytest-args "-v"
    modal run .modal/test_runner.py --test-path tests/datasets/ --pytest-args "-v -k synthetic"
"""

import sys
from pathlib import Path
import modal

# Create Modal app
app = modal.App("rf-detr-gpu-tests")

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
    gpu="L4",  # NVIDIA L4 GPU
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
    
    # Run pytest and stream output
    try:
        result = subprocess.run(
            pytest_cmd,
            cwd="/root/project",
        )
        
        print("\n" + "="*80)
        print("TEST EXECUTION COMPLETE")
        print("="*80)
        print(f"Exit code: {result.returncode}")
        print("="*80 + "\n")
        
        return {
            "returncode": result.returncode,
            "success": result.returncode == 0,
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
    print(f"🎮 GPU: L4")
    print(f"⏱️  Timeout: 1 hour")
    print("="*80 + "\n")
    
    # Run tests remotely with streaming output
    result = run_tests.remote(test_path=test_path, pytest_args=pytest_args)
    
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
