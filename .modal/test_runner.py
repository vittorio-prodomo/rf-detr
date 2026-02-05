"""
GPU Test Runner for RF-DETR

Run tests on Modal's GPU infrastructure with pytest interface similar to local execution.

Usage:
    # Run all tests
    modal run .modal/test_runner.py

    # Run specific test file
    modal run .modal/test_runner.py --test-path tests/datasets/test_synthetic.py

    # Run with pytest flags (e.g., verbose, specific tests)
    modal run .modal/test_runner.py --pytest-args "-v -k test_name"

    # Run tests in a specific directory
    modal run .modal/test_runner.py --test-path tests/datasets/

Environment Variables Required:
    - MODAL_TOKEN_ID: Your Modal token ID
    - MODAL_TOKEN_SECRET: Your Modal token secret
"""

import sys
import modal

# Create Modal app
app = modal.App("rf-detr-gpu-tests")

# Start with PyTorch public image for faster builds
# Using PyTorch 2.8 with CUDA 12.6
image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/pytorch:25.01-py3",  # PyTorch 2.8 with CUDA 12.6
        add_python="3.10",
    )
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        # Install the package with tests dependency group
        "cd /root && uv pip install --system --group tests ."
    )
)


@app.function(
    image=image,
    gpu="L4",  # NVIDIA L4 GPU
    timeout=3600,  # Hard 1 hour timeout safety limit - session will be terminated after this
)
def run_tests(test_path: str = "tests/", pytest_args: str = "-v"):
    """
    Run pytest on Modal GPU infrastructure.
    
    Args:
        test_path: Path to test file or directory (default: "tests/")
        pytest_args: Additional pytest arguments (default: "-v")
    
    Returns:
        dict: Test results with stdout, stderr, and return code
    """
    import subprocess
    
    # Verify GPU availability
    try:
        import torch
        print(f"🎮 GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🎮 GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"🎮 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    except Exception as e:
        print(f"⚠️  GPU check failed: {e}")
    
    # Build pytest command
    pytest_cmd = ["pytest", test_path]
    
    # Add user-provided pytest arguments
    if pytest_args:
        pytest_cmd.extend(pytest_args.split())
    
    # Force colored output for better readability
    pytest_cmd.append("--color=yes")
    
    print(f"\n{'='*80}")
    print(f"🚀 Running command: {' '.join(pytest_cmd)}")
    print(f"{'='*80}\n")
    
    # Run pytest
    try:
        result = subprocess.run(
            pytest_cmd,
            capture_output=False,  # Stream output in real-time
            text=True
        )
        
        print(f"\n{'='*80}")
        print(f"✅ Tests completed with exit code: {result.returncode}")
        print(f"{'='*80}\n")
        
        return {
            "returncode": result.returncode,
            "success": result.returncode == 0,
        }
    
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ Test execution failed: {e}")
        print(f"{'='*80}\n")
        
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
    """
    Local entrypoint to run tests on Modal GPU.
    
    Args:
        test_path: Path to test file or directory (default: "tests/")
        pytest_args: Additional pytest arguments (default: "-v")
    """
    print(f"\n{'='*80}")
    print(f"🚀 RF-DETR GPU Test Runner")
    print(f"{'='*80}")
    print(f"📁 Test Path: {test_path}")
    print(f"⚙️  Pytest Args: {pytest_args}")
    print(f"🎮 GPU: L4")
    print(f"{'='*80}\n")
    
    # Run tests remotely with streaming output
    result = run_tests.remote(test_path=test_path, pytest_args=pytest_args)
    
    print(f"\n{'='*80}")
    print(f"📊 Final Results")
    print(f"{'='*80}")
    print(f"Return Code: {result['returncode']}")
    print(f"Success: {result['success']}")
    
    if not result["success"]:
        if "error" in result:
            print(f"Error: {result['error']}")
        print(f"{'='*80}\n")
        sys.exit(result["returncode"])
    
    print(f"{'='*80}\n")
    print("✅ All tests passed!")
