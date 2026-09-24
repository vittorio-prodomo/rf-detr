# RF-DETR Baseline Download Failures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all three known RF-DETR baseline failures pass while preserving centralized cache resolution and optional no-pretraining initialization.

**Architecture:** Keep the production cache resolver and downloader contract unchanged, and correct the two stale tests to assert the configured resolved path. Guard the high-level RF-DETR initialization hook so an explicitly absent pretrained checkpoint skips downloading, while the low-level downloader remains strict about receiving strings.

**Tech Stack:** Python 3.12, Pydantic, PyTorch, Pytest, unittest.mock, uv.

---

### Task 1: Align download tests with the centralized cache contract

**Files:**
- Modify: `tests/assets/test_downloads.py`

- [ ] **Step 1: Reproduce the two stale assertions**

Run:

```bash
uv run pytest -q \
  tests/assets/test_downloads.py::TestDownloadPretrainWeights::test_download_from_local_model_weights \
  tests/assets/test_downloads.py::TestDownloadIntegration::test_download_flow_for_real_model
```

Expected: both tests fail because `_download_file(filename=...)` receives the resolved cache path instead of the bare `rf-detr-base.pth` name.

- [ ] **Step 2: Make each test supply and assert a temporary cache directory**

Add `tmp_path` to both test signatures and patch the cache-directory lookup around the download call:

```python
def test_download_from_local_model_weights(self, mock_file_operations, tmp_path):
    with patch("rfdetr.cache.get_cache_dir", return_value=tmp_path):
        result = download_pretrain_weights("rf-detr-base.pth")

    expected_path = str(tmp_path / "rf-detr-base.pth")
    call_kwargs = mock_file_operations["download"].call_args[1]
    assert call_kwargs["filename"] == expected_path
    assert result == expected_path
```

Apply the same temporary-cache arrangement in `test_download_flow_for_real_model`, using its existing `mock_download` object:

```python
def test_download_flow_for_real_model(
    self, mock_download, mock_validate, mock_exists, tmp_path
):
    mock_exists.return_value = False
    mock_validate.return_value = True

    with patch("rfdetr.cache.get_cache_dir", return_value=tmp_path):
        result = download_pretrain_weights("rf-detr-base.pth")

    expected_path = str(tmp_path / "rf-detr-base.pth")
    call_kwargs = mock_download.call_args[1]
    assert call_kwargs["filename"] == expected_path
    assert result == expected_path
```

Retain the existing URL and MD5 assertions in both tests.

- [ ] **Step 3: Run the complete asset-download suite**

Run:

```bash
uv run pytest -q tests/assets/test_downloads.py tests/assets/test_model_weights.py
```

Expected: all asset tests pass without network access.

- [ ] **Step 4: Commit the cache-contract test correction**

```bash
git add tests/assets/test_downloads.py
git commit -m "test: align downloads with weight cache"
```

### Task 2: Skip checkpoint download when pretraining is disabled

**Files:**
- Modify: `src/rfdetr/detr.py`
- Modify: `tests/models/test_predict.py`

- [ ] **Step 1: Write the focused failing tests**

Add `Mock` and `patch` imports from `unittest.mock`, then exercise the base method without constructing a heavyweight model:

```python
def test_pretrain_download_is_skipped_when_weights_are_none():
    model = object.__new__(RFDETR)
    model.model_config = SimpleNamespace(pretrain_weights=None)

    with patch("rfdetr.detr.download_pretrain_weights") as download:
        model.maybe_download_pretrain_weights()

    download.assert_not_called()


def test_pretrain_download_receives_configured_weight_path():
    model = object.__new__(RFDETR)
    model.model_config = SimpleNamespace(pretrain_weights="/tmp/model.pth")

    with patch("rfdetr.detr.download_pretrain_weights") as download:
        model.maybe_download_pretrain_weights()

    download.assert_called_once_with("/tmp/model.pth")
```

- [ ] **Step 2: Run the new tests and verify the `None` case fails**

Run:

```bash
uv run pytest -q \
  tests/models/test_predict.py::test_pretrain_download_is_skipped_when_weights_are_none \
  tests/models/test_predict.py::test_pretrain_download_receives_configured_weight_path
```

Expected: the `None` test fails because the downloader is called once with `None`; the configured-path test passes.

- [ ] **Step 3: Add the minimal high-level guard**

Change `RFDETR.maybe_download_pretrain_weights()` to:

```python
def maybe_download_pretrain_weights(self):
    """Download configured pretrained weights, if any."""
    if self.model_config.pretrain_weights is not None:
        download_pretrain_weights(self.model_config.pretrain_weights)
```

Do not change `download_pretrain_weights()` or `resolve_weight_path()` to accept `None`.

- [ ] **Step 4: Run model and asset regression tests**

Run:

```bash
uv run pytest -q \
  tests/models/test_predict.py \
  tests/models/test_postprocess.py \
  tests/assets/test_downloads.py \
  tests/assets/test_model_weights.py
```

Expected: all focused tests pass.

- [ ] **Step 5: Commit the optional-pretraining fix**

```bash
git add src/rfdetr/detr.py tests/models/test_predict.py
git commit -m "fix: allow initialization without pretrained weights"
```

### Task 3: Verify the former failures and complete suite selections

**Files:**
- No product changes expected.

- [ ] **Step 1: Run the synthetic convergence benchmark on physical GPU 1**

```bash
CUDA_VISIBLE_DEVICES=1 uv run pytest -q \
  tests/benchmarks/test_synthetic_convergence.py::test_synthetic_training_improves_performance
```

Expected: the model initializes without downloading a checkpoint, completes training, and passes its convergence assertions.

- [ ] **Step 2: Run all non-GPU tests**

```bash
CUDA_VISIBLE_DEVICES=1 uv run pytest -q -m "not gpu"
```

Expected: the two former asset failures are gone and all non-GPU tests pass, apart from any explicitly skipped network test.

- [ ] **Step 3: Run all GPU-marked selections on physical GPU 1**

```bash
CUDA_VISIBLE_DEVICES=1 uv run pytest -q -m gpu
```

Expected: all GPU-marked tests pass. If the aggregate command exceeds five minutes, preserve its output and finish any remaining selections through bounded explicit invocations, recording an aggregate unique-test result.

- [ ] **Step 4: Verify branch integrity**

```bash
git diff --check "$(git merge-base main HEAD)"..HEAD
git status --short --branch
```

Expected: no whitespace errors and a clean `feat/filtered-mask-logits` worktree.

- [ ] **Step 5: Stop at the integration boundary**

Report the commits and exact test evidence. Do not merge, push, restart Iris, publish frontend assets, or deploy without explicit authorization.
