# RF-DETR Baseline Download Failures Design

**Date:** 2026-09-24
**Status:** Approved

## Goal

Repair the three RF-DETR baseline test failures without weakening the centralized weight-cache contract or changing model behavior when pretrained weights are configured.

## Root causes

1. Two download tests still expect `_download_file()` to receive the original bare filename. Production now intentionally resolves bare weight names into the configured Roboflow cache directory before downloading.
2. `ModelConfig.pretrain_weights` explicitly permits `None`, but `RFDETR.__init__()` unconditionally asks the downloader to process that value. The path resolver then receives `None` and raises `TypeError` before synthetic training starts.

## Design

### Cache-path tests

Keep the production downloader unchanged. Update the two stale assertions so they verify that `_download_file()` receives `rf-detr-base.pth` inside the configured cache directory. Tests will supply a temporary cache directory rather than depend on a developer's home directory.

### Optional pretrained weights

Treat `pretrain_weights=None` as an explicit request to initialize without downloaded pretrained weights. `RFDETR.maybe_download_pretrain_weights()` will call `download_pretrain_weights()` only when the configured value is not `None`.

The low-level downloader remains strict and continues accepting only a string path/name. This avoids silently converting unrelated caller bugs into no-ops.

## Testing

- Preserve the existing failing path assertions as the RED evidence, then update them to the configured-cache contract.
- Add a focused failing unit test proving `maybe_download_pretrain_weights()` does not invoke the downloader for `None`.
- Verify the asset download suite, the focused RF-DETR model tests, and the synthetic training test on GPU 1.
- Run the complete RF-DETR test selections with GPU work constrained to physical GPU 1.

## Non-goals

- Reverting centralized cache resolution.
- Changing checkpoint contents, model initialization, training hyperparameters, or synthetic benchmark expectations.
- Making `download_pretrain_weights(None)` a supported public API.
