# GPU Test Runner for RF-DETR

Run your pytest tests on Modal's GPU infrastructure (L4) with the same interface as local execution.

## Features

- 🎮 **L4 GPU** with 24GB VRAM
- 🚀 **PyTorch pre-installed** using NVIDIA official image for faster builds
- 📦 **Automatic dependency installation** from `pyproject.toml`
- 🎨 **Colored output** with real-time streaming
- ⚙️ **Respects pytest configuration** from your project
- 🔄 **Same interface as local pytest**

## Setup

### 1. Install Modal CLI

```bash
pip install modal
```

### 2. Set up Modal credentials

```bash
export MODAL_TOKEN_ID=your_token_id
export MODAL_TOKEN_SECRET=your_token_secret
```

### 3. Configure Modal profile (optional)

The wrapper script automatically activates the Modal profile specified by the `MODAL_PROFILE` environment variable (defaults to `rf-detr`).

```bash
export MODAL_PROFILE=rf-detr  # or your custom profile name
```

The script will run `modal profile activate rf-detr` before executing tests.

## Usage

### Using the convenience wrapper (recommended)

```bash
# Run all tests
.modal/pytest.sh

# Run specific test directory
.modal/pytest.sh -v tests/datasets/

# Run with pytest markers
.modal/pytest.sh "-m modal" tests/

# Run specific test file
.modal/pytest.sh -v tests/datasets/test_synthetic.py
```

### Using Modal CLI directly

```bash
# Run all tests
modal run .modal/test_runner.py

# Run specific tests
modal run .modal/test_runner.py \
    --test-path tests/datasets/ \
    --pytest-args "-v"

# Run with custom pytest flags
modal run .modal/test_runner.py \
    --test-path tests/ \
    --pytest-args "-v -s -k test_name"
```

## How It Works

1. **Image Building**: Starts with NVIDIA PyTorch image (`pytorch:25.01-py3`)
2. **Dependency Installation**: Uses `uv` to install your package with tests dependency group from `pyproject.toml`
3. **Test Execution**: Runs pytest with your specified arguments, streaming output in real-time
4. **GPU Info**: Reports GPU availability and specs before running tests

### Pytest Configuration
The runner respects your project's pytest configuration in `pyproject.toml`. It doesn't override any settings unless explicitly passed via `--pytest-args`.

## Examples

### Run all tests verbosely
```bash
.modal/pytest.sh -v tests/
```

### Run specific test pattern
```bash
.modal/pytest.sh "-v -k synthetic" tests/
```

### Run with markers
```bash
.modal/pytest.sh "-m modal" tests/
```

### Stop on first failure
```bash
.modal/pytest.sh "-x -v" tests/
```

### Show print statements
```bash
.modal/pytest.sh "-v -s" tests/
```

## Troubleshooting

### Image build fails
The first build may take a few minutes as it downloads the PyTorch base image and installs dependencies. Subsequent builds use cached layers and are much faster.

### Tests fail locally but pass on GPU
Some tests may require GPU resources that aren't available locally. Check the test markers in your `conftest.py`.

## CI Integration

To use in GitHub Actions, add the workflow:

```yaml
name: GPU Tests
on: [push, pull_request]

jobs:
  gpu-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - name: Install Modal
        run: pip install modal
      - name: Run GPU tests
        env:
          MODAL_TOKEN_ID: ${{ secrets.MODAL_TOKEN_ID }}
          MODAL_TOKEN_SECRET: ${{ secrets.MODAL_TOKEN_SECRET }}
        run: .modal/pytest.sh -v tests/
```
