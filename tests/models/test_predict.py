# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import socket
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import supervision as sv
import torch

from rfdetr.detr import RFDETR

_HTTP_IMAGE_URL = "http://images.cocodataset.org/val2017/000000397133.jpg"
_HTTP_HOST = "images.cocodataset.org"
_HTTP_PORT = 80


def _is_online(host: str, port: int, timeout_s: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


class _DummyModel:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.resolution = 32
        self.model = torch.nn.Identity()

    def postprocess(
        self, predictions: Any, target_sizes: torch.Tensor
    ) -> list[dict[str, torch.Tensor]]:
        batch = target_sizes.shape[0]
        results = []
        for _ in range(batch):
            results.append(
                {
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([1]),
                    "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                }
            )
        return results


class _DummyRFDETR(RFDETR):
    def maybe_download_pretrain_weights(self) -> None:
        return None

    def get_model_config(self, **kwargs) -> SimpleNamespace:
        return SimpleNamespace()

    def get_model(self, config: SimpleNamespace) -> _DummyModel:
        return _DummyModel()


class _MaskLogitsDummyModel(_DummyModel):
    def postprocess(
        self, predictions: Any, target_sizes: torch.Tensor
    ) -> list[dict[str, torch.Tensor]]:
        batch = target_sizes.shape[0]
        results = []
        for _ in range(batch):
            results.append(
                {
                    "scores": torch.tensor([0.9, 0.1]),
                    "labels": torch.tensor([1, 2]),
                    "boxes": torch.tensor(
                        [[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]]
                    ),
                    "mask_logits": torch.tensor(
                        [
                            [[1.0, -1.0], [0.5, -0.5]],
                            [[2.0, -2.0], [1.5, -1.5]],
                        ]
                    ),
                }
            )
        return results


class _MaskLogitsDummyRFDETR(_DummyRFDETR):
    def get_model(self, config: SimpleNamespace) -> _MaskLogitsDummyModel:
        return _MaskLogitsDummyModel()


def test_predict_accepts_image_url() -> None:
    if not _is_online(_HTTP_HOST, _HTTP_PORT):
        pytest.skip("Offline environment, skipping HTTP predict URL test.")
    model = _DummyRFDETR()
    detections = model.predict(_HTTP_IMAGE_URL)
    assert isinstance(detections, sv.Detections)
    assert detections.xyxy.shape == (1, 4)


def test_predict_carries_float_mask_logits_in_detection_data() -> None:
    model = _MaskLogitsDummyRFDETR()

    detections = model.predict(torch.zeros((3, 2, 2)), threshold=0.5)

    assert detections.mask is None
    np.testing.assert_array_equal(
        detections.data["mask_logits"],
        np.array([[[1.0, -1.0], [0.5, -0.5]]], dtype=np.float32),
    )
    assert detections.data["mask_logits"].dtype == np.float32
    assert len(detections) == 1
