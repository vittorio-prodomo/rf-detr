# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""PostProcess inference-time filters + independent mask resolution.

These consolidate a monkey-patch that previously lived in three verbatim
copies outside this repo. Two properties matter most and are pinned here:

1. **Stock behaviour is untouched** when nothing is configured -- byte-identical
   to the pre-consolidation decode, so every existing caller is unaffected.
2. **Masks are decoded only for survivors** -- the entire reason the filters
   live inside PostProcess rather than in the caller.
"""
import pytest
import torch

from rfdetr.models.lwdetr import PostProcess


NUM_SELECT = 6
QUERIES, CLASSES = 8, 3
MASK_H = MASK_W = 8


@pytest.fixture
def outputs():
    """Deterministic raw model outputs with a segmentation head."""
    g = torch.Generator().manual_seed(0)
    return {
        "pred_logits": torch.randn(2, QUERIES, CLASSES, generator=g),
        "pred_boxes": torch.rand(2, QUERIES, 4, generator=g) * 0.5 + 0.25,
        "pred_masks": torch.randn(2, QUERIES, MASK_H, MASK_W, generator=g),
    }


@pytest.fixture
def target_sizes():
    return torch.tensor([[32, 32], [32, 32]])


def _stock_forward(outputs, target_sizes, num_select):
    """The pre-consolidation decode, inlined verbatim as a reference oracle."""
    import torch.nn.functional as F

    from rfdetr.util import box_ops

    out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
    out_masks = outputs.get("pred_masks", None)
    prob = out_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(
        prob.view(out_logits.shape[0], -1), num_select, dim=1
    )
    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]

    results = []
    if out_masks is not None:
        for i in range(out_masks.shape[0]):
            res_i = {"scores": scores[i], "labels": labels[i], "boxes": boxes[i]}
            k_idx = topk_boxes[i]
            masks_i = torch.gather(
                out_masks[i], 0,
                k_idx.unsqueeze(-1).unsqueeze(-1).repeat(
                    1, out_masks.shape[-2], out_masks.shape[-1]
                ),
            )
            h, w = target_sizes[i].tolist()
            masks_i = F.interpolate(
                masks_i.unsqueeze(1), size=(int(h), int(w)),
                mode="bilinear", align_corners=False,
            )
            res_i["masks"] = masks_i > 0.0
            results.append(res_i)
    else:
        results = [
            {"scores": s, "labels": l, "boxes": b}
            for s, l, b in zip(scores, labels, boxes)
        ]
    return results


# --------------------------------------------------------- stock equivalence


def test_unconfigured_output_is_identical_to_the_stock_decode(outputs, target_sizes):
    """The load-bearing guarantee: adding these knobs must change nothing for
    a caller that does not use them."""
    got = PostProcess(num_select=NUM_SELECT)(outputs, target_sizes)
    want = _stock_forward(outputs, target_sizes, NUM_SELECT)

    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert g.keys() == w.keys()
        for key in w:
            assert torch.equal(g[key], w[key]), key


def test_unconfigured_output_is_identical_without_a_mask_head(outputs, target_sizes):
    det_only = {k: v for k, v in outputs.items() if k != "pred_masks"}
    got = PostProcess(num_select=NUM_SELECT)(det_only, target_sizes)
    want = _stock_forward(det_only, target_sizes, NUM_SELECT)

    for g, w in zip(got, want):
        assert "masks" not in g
        for key in w:
            assert torch.equal(g[key], w[key]), key


# ------------------------------------------------------------------ filters


def test_score_threshold_drops_low_scoring_queries(outputs, target_sizes):
    pp = PostProcess(num_select=NUM_SELECT)
    baseline = pp(outputs, target_sizes)[0]
    cutoff = baseline["scores"].median().item()

    pp.score_threshold = cutoff
    kept = pp(outputs, target_sizes)[0]

    assert kept["scores"].numel() < baseline["scores"].numel()
    assert (kept["scores"] > cutoff).all()
    # masks shrink with the survivors -- that is the point
    assert kept["masks"].shape[0] == kept["scores"].numel()


def test_target_class_ids_keeps_only_wanted_classes(outputs, target_sizes):
    pp = PostProcess(num_select=NUM_SELECT)
    pp.target_class_ids = {1}
    out = pp(outputs, target_sizes)[0]

    assert out["labels"].numel() > 0, "fixture must exercise the filter"
    assert (out["labels"] == 1).all()


def test_per_class_threshold_overrides_the_base_floor(outputs, target_sizes):
    """A rare class needs a lower floor than a common one -- the reason this is
    a per-class dict rather than one scalar."""
    pp = PostProcess(num_select=NUM_SELECT)
    everything = pp(outputs, target_sizes)[0]
    high = everything["scores"].max().item() + 1.0

    pp.score_threshold = high              # nothing can clear this...
    pp.per_class_threshold = {2: 0.0}      # ...except class 2
    out = pp(outputs, target_sizes)[0]

    assert out["labels"].numel() > 0
    assert (out["labels"] == 2).all()


def test_per_class_threshold_alone_defaults_the_base_to_zero(outputs, target_sizes):
    """Matches the pre-consolidation default so ported call sites that set only
    the per-class dict behave as they did."""
    pp = PostProcess(num_select=NUM_SELECT)
    pp.per_class_threshold = {0: 1.1}      # unreachable for sigmoid scores
    out = pp(outputs, target_sizes)[0]

    assert (out["labels"] != 0).all()
    assert out["labels"].numel() > 0, "other classes keep the 0.0 base floor"


def test_nms_iou_suppresses_overlapping_same_class_boxes(target_sizes):
    """Two near-identical class-0 boxes -> one survivor."""
    logits = torch.full((1, 2, CLASSES), -10.0)
    logits[0, 0, 0] = 5.0
    logits[0, 1, 0] = 4.0
    outs = {
        "pred_logits": logits,
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.4, 0.4], [0.51, 0.51, 0.4, 0.4]]]),
    }
    sizes = torch.tensor([[32, 32]])

    pp = PostProcess(num_select=2, score_threshold=0.5)
    assert pp(outs, sizes)[0]["scores"].numel() == 2

    pp.nms_iou = 0.5
    kept = pp(outs, sizes)[0]
    assert kept["scores"].numel() == 1
    assert kept["scores"].item() == pytest.approx(torch.sigmoid(torch.tensor(5.0)).item())


def test_filters_compose(outputs, target_sizes):
    pp = PostProcess(
        num_select=NUM_SELECT, score_threshold=0.0, target_class_ids={0, 1}, nms_iou=0.9
    )
    out = pp(outputs, target_sizes)[0]
    assert set(out["labels"].tolist()) <= {0, 1}


def test_filtering_everything_out_yields_well_formed_empties(outputs, target_sizes):
    """Empty must still carry the right rank and mask resolution, or downstream
    metric code sees ragged shapes."""
    pp = PostProcess(num_select=NUM_SELECT, score_threshold=1.1)
    out = pp(outputs, target_sizes)[0]

    assert out["scores"].numel() == 0
    assert out["boxes"].shape == (0, 4)
    assert out["masks"].shape == (0, 1, 32, 32)


# ------------------------------------------------------------- mask_size


def test_mask_size_native_skips_the_interpolation(outputs, target_sizes):
    """`native` must return the head's own resolution, NOT an upsample-then-
    downsample round trip, which resamples every boundary twice."""
    pp = PostProcess(num_select=NUM_SELECT, mask_size="native")
    out = pp(outputs, target_sizes)[0]
    assert out["masks"].shape[-2:] == (MASK_H, MASK_W)


def test_native_masks_equal_thresholding_the_raw_logits(outputs, target_sizes):
    """Proves `native` is a genuine no-op on the mask tensor rather than an
    identity-shaped interpolate."""
    pp = PostProcess(num_select=NUM_SELECT, mask_size="native")
    out = pp(outputs, target_sizes)[0]

    prob = outputs["pred_logits"].sigmoid()
    _, topk = torch.topk(prob.view(2, -1), NUM_SELECT, dim=1)
    query_idx = (topk // CLASSES)[0]
    expect = (outputs["pred_masks"][0][query_idx] > 0.0).unsqueeze(1)
    assert torch.equal(out["masks"], expect)


def test_native_differs_from_the_round_trip(outputs, target_sizes):
    """Sanity that the distinction is real: upsampling to 32 and coming back to
    8 is not the same mask as thresholding at 8."""
    native = PostProcess(num_select=NUM_SELECT, mask_size="native")(
        outputs, target_sizes
    )[0]["masks"]
    upsampled = PostProcess(num_select=NUM_SELECT)(outputs, target_sizes)[0]["masks"]
    round_tripped = torch.nn.functional.interpolate(
        upsampled.float(), size=(MASK_H, MASK_W), mode="nearest"
    ).bool()
    assert not torch.equal(native, round_tripped)


def test_mask_size_accepts_an_explicit_hw(outputs, target_sizes):
    pp = PostProcess(num_select=NUM_SELECT, mask_size=(16, 20))
    assert pp(outputs, target_sizes)[0]["masks"].shape[-2:] == (16, 20)


def test_mask_size_is_independent_of_the_box_frame(outputs, target_sizes):
    """The decoupling itself: boxes stay in the image frame while masks are
    scored somewhere else."""
    stock = PostProcess(num_select=NUM_SELECT)(outputs, target_sizes)[0]
    decoupled = PostProcess(num_select=NUM_SELECT, mask_size="native")(
        outputs, target_sizes
    )[0]
    assert torch.equal(stock["boxes"], decoupled["boxes"])
    assert stock["masks"].shape[-2:] != decoupled["masks"].shape[-2:]


def test_unknown_mask_size_string_fails_fast(outputs, target_sizes):
    pp = PostProcess(num_select=NUM_SELECT, mask_size="full")
    with pytest.raises(ValueError, match="mask_size"):
        pp(outputs, target_sizes)


def test_empty_survivors_honour_an_explicit_mask_size(outputs, target_sizes):
    pp = PostProcess(num_select=NUM_SELECT, score_threshold=1.1, mask_size=(16, 16))
    assert pp(outputs, target_sizes)[0]["masks"].shape == (0, 1, 16, 16)


# ------------------------------------------------------- migration guardrail


@pytest.mark.parametrize(
    "legacy,replacement",
    [
        ("_score_threshold", "score_threshold"),
        ("_per_class_threshold", "per_class_threshold"),
        ("_target_class_ids", "target_class_ids"),
        ("_nms_iou", "nms_iou"),
    ],
)
def test_legacy_monkeypatch_attributes_raise(legacy, replacement):
    """A missed migration must fail loudly. Silently accepting the old name
    would restore the unfiltered all-300-query mask decode with no symptom
    other than being ~13x slower."""
    pp = PostProcess(num_select=NUM_SELECT)
    with pytest.raises(AttributeError, match=replacement):
        setattr(pp, legacy, 0.5)


def test_bridge_patched_flag_disarms_stale_patch_installers():
    """Every copy of the old installer guards on this flag, so a straggler
    becomes a no-op instead of replacing forward and reverting the class."""
    assert PostProcess._bridge_patched is True


def test_capability_flag_is_detectable_on_the_class():
    """Consumers that may load an older vendored rf-detr (the handoff bundles
    pin by commit) need a check that works BEFORE constructing anything -- the
    instance attributes cannot serve, since setting them on an old PostProcess
    succeeds and silently does nothing."""
    assert getattr(PostProcess, "SUPPORTS_INFERENCE_FILTERS", False) is True
