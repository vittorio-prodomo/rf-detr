# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import os
from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv
import torch
from PIL import Image
from torchvision.datasets import VisionDataset

from rfdetr.datasets.coco import (
    make_coco_transforms,
    make_coco_transforms_square_div_64,
)

REQUIRED_YOLO_YAML_FILE = "data.yaml"
REQUIRED_SPLIT_DIRS = ["train", "valid"]
REQUIRED_DATA_SUBDIRS = ["images", "labels"]


def _detect_yolo_layout(dataset_dir: str):
    """Detect whether the dataset uses YOLO-A or YOLO-B directory layout.

    YOLO-A (Ultralytics standard): ``images/{split}/`` + ``labels/{split}/``
    YOLO-B (Roboflow style):       ``{split}/images/`` + ``{split}/labels/``

    Also accepts ``val`` as an alias for ``valid``.

    Returns:
        ``"yolo-a"``, ``"yolo-b"``, or ``None`` if neither layout is detected.
    """
    # YOLO-B: {split}/images/ + {split}/labels/
    for val_name in ("valid", "val"):
        splits = ["train", val_name]
        if all(
            os.path.exists(os.path.join(dataset_dir, s, d))
            for s in splits
            for d in REQUIRED_DATA_SUBDIRS
        ):
            return "yolo-b"

    # YOLO-A: images/{split}/ + labels/{split}/
    for val_name in ("val", "valid"):
        splits = ["train", val_name]
        if all(
            os.path.exists(os.path.join(dataset_dir, d, s))
            for d in REQUIRED_DATA_SUBDIRS
            for s in splits
        ):
            return "yolo-a"

    return None


def _resolve_val_split_name(dataset_dir: str, layout: str):
    """Return the actual validation split name on disk (``val`` or ``valid``)."""
    if layout == "yolo-b":
        if os.path.isdir(os.path.join(dataset_dir, "valid")):
            return "valid"
        return "val"
    else:  # yolo-a
        if os.path.isdir(os.path.join(dataset_dir, "images", "val")):
            return "val"
        return "valid"


def is_valid_yolo_dataset(dataset_dir: str) -> bool:
    """
    Checks if the specified dataset directory is in yolo format.

    Supports both YOLO-A (``images/{split}/``) and YOLO-B (``{split}/images/``)
    layouts, and accepts ``val`` as an alias for ``valid``.

    Returns a boolean indicating whether the dataset is in correct yolo format.
    """
    contains_required_data_yaml = os.path.exists(os.path.join(dataset_dir, REQUIRED_YOLO_YAML_FILE))
    layout = _detect_yolo_layout(dataset_dir)
    return contains_required_data_yaml and layout is not None


class ConvertYolo:
    """
    Converts supervision Detections to the target dict format expected by RF-DETR.

    Args:
        include_masks: whether to include segmentation masks

    Examples:
        >>> import numpy as np
        >>> import supervision as sv
        >>> from PIL import Image
        >>> # Create a sample image and target
        >>> image = Image.new("RGB", (100, 100))
        >>> detections = sv.Detections(
        ...     xyxy=np.array([[10, 20, 30, 40]]),
        ...     class_id=np.array([0])
        ... )
        >>> target = {"image_id": 0, "detections": detections}
        >>> # Create converter
        >>> converter = ConvertYolo(include_masks=False)
        >>> # Call converter
        >>> img, result = converter(image, target)
        >>> sorted(result.keys())
        ['area', 'boxes', 'image_id', 'iscrowd', 'labels', 'orig_size', 'size']
        >>> result["boxes"].shape
        torch.Size([1, 4])
        >>> result["labels"].tolist()
        [0]
        >>> result["image_id"].tolist()
        [0]
    """

    def __init__(self, include_masks: bool = False):
        self.include_masks = include_masks

    def __call__(self, image: Image.Image, target: dict) -> tuple:
        """
        Convert image and YOLO detections to RF-DETR format.

        Args:
            image: PIL Image
            target: dict with 'image_id' and 'detections' (sv.Detections)

        Returns:
            tuple of (image, target_dict)
        """
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        detections = target["detections"]

        if len(detections) > 0:
            boxes = torch.from_numpy(detections.xyxy).to(torch.float32)
            classes = torch.from_numpy(detections.class_id).to(torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            classes = torch.zeros((0,), dtype=torch.int64)

        # clamp and filter
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target_out = {}
        target_out["boxes"] = boxes
        target_out["labels"] = classes
        target_out["image_id"] = image_id

        # compute area after clamp
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target_out["area"] = area

        iscrowd = torch.zeros((classes.shape[0],), dtype=torch.int64)
        target_out["iscrowd"] = iscrowd

        if self.include_masks:
            if detections.mask is not None and np.size(detections.mask) > 0:
                masks = torch.from_numpy(detections.mask[keep.cpu().numpy()]).to(torch.uint8)
                target_out["masks"] = masks
            else:
                target_out["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)

            target_out["masks"] = target_out["masks"].bool()

        target_out["orig_size"] = torch.as_tensor([int(h), int(w)])
        target_out["size"] = torch.as_tensor([int(h), int(w)])

        return image, target_out


class _MockSvDataset:
    """Mock supervision dataset for testing CocoLikeAPI."""

    classes = ["cat", "dog"]

    def __len__(self):
        return 2

    def __getitem__(self, i):
        import numpy as np
        import supervision as sv

        det = sv.Detections(xyxy=np.array([[10 * i, 20, 30, 40]]), class_id=np.array([i]))
        return f"img_{i}.jpg", np.zeros((100, 100, 3), dtype=np.uint8), det


class CocoLikeAPI:
    """
    A minimal COCO-compatible API wrapper for YOLO datasets.

    This provides the necessary interface for CocoEvaluator to work with
    YOLO format datasets.

    Examples:
        >>> mock = _MockSvDataset()
        >>> coco = YoloCoco(mock.classes, mock)
        >>> # dataset structure
        >>> len(coco.dataset["images"]), len(coco.dataset["categories"]), len(coco.dataset["annotations"])
        (2, 2, 2)
        >>> # getAnnIds
        >>> coco.getAnnIds()
        [0, 1]
        >>> coco.getAnnIds(imgIds=[0])
        [0]
        >>> coco.getAnnIds(catIds=[1])
        [1]
        >>> # getCatIds
        >>> sorted(coco.getCatIds())
        [0, 1]
        >>> coco.getCatIds(catNms=["cat"])
        [0]
        >>> # getImgIds
        >>> sorted(coco.getImgIds())
        [0, 1]
        >>> coco.getImgIds(catIds=[0])
        [0]
        >>> # loadAnns
        >>> ann = coco.loadAnns([0])[0]
        >>> ann["category_id"], ann["image_id"]
        (0, 0)
        >>> # loadCats
        >>> coco.loadCats([0])[0]["name"]
        'cat'
        >>> len(coco.loadCats())
        2
        >>> # loadImgs
        >>> coco.loadImgs([1])[0]["file_name"]
        'img_1.jpg'
    """

    def __init__(self, classes: list, dataset: sv.DetectionDataset):
        self.classes = classes
        self.sv_dataset = dataset

        # Build the dataset dict that COCO API expects
        self.dataset = self._build_coco_dataset()
        self.imgs = {img["id"]: img for img in self.dataset["images"]}
        self.anns = {ann["id"]: ann for ann in self.dataset["annotations"]}
        self.cats = {cat["id"]: cat for cat in self.dataset["categories"]}

        # Build imgToAnns index
        self.imgToAnns = {}
        for ann in self.dataset["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.imgToAnns:
                self.imgToAnns[img_id] = []
            self.imgToAnns[img_id].append(ann)

        # Ensure all images have an entry
        for img_id in self.imgs:
            if img_id not in self.imgToAnns:
                self.imgToAnns[img_id] = []

        # Build catToImgs index
        self.catToImgs = {}
        for cat_id in self.cats:
            self.catToImgs[cat_id] = []
        for ann in self.dataset["annotations"]:
            cat_id = ann["category_id"]
            img_id = ann["image_id"]
            if img_id not in self.catToImgs[cat_id]:
                self.catToImgs[cat_id].append(img_id)

    def _build_coco_dataset(self) -> dict:
        """Build a COCO-format dataset dict from YOLO data."""
        images = []
        annotations = []
        categories = []

        # Build categories (0-indexed class IDs in YOLO)
        for idx, class_name in enumerate(self.classes):
            categories.append({"id": idx, "name": class_name, "supercategory": "none"})

        ann_id = 0
        for img_id in range(len(self.sv_dataset)):
            image_path, cv2_image, detections = self.sv_dataset[img_id]
            h, w = cv2_image.shape[:2]

            images.append({"id": img_id, "file_name": str(image_path), "height": h, "width": w})

            if len(detections) == 0:
                continue
            for i in range(len(detections)):
                bbox_x, bbox_y, bbox_w, bbox_h = sv.xyxy_to_xywh(detections.xyxy[i : i + 1])[0]

                ann = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(detections.class_id[i]),
                    "bbox": [float(bbox_x), float(bbox_y), float(bbox_w), float(bbox_h)],
                    "area": float(bbox_w * bbox_h),
                    "iscrowd": 0,
                }

                # Add segmentation if available
                if detections.mask is not None:
                    # For now, use empty polygon - evaluation will still work for bbox
                    ann["segmentation"] = []

                annotations.append(ann)
                ann_id += 1

        return {
            "info": {"description": "RF-DETR YOLO dataset"},
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

    def getAnnIds(self, imgIds=None, catIds=None, areaRng=None, iscrowd=None):
        """Get annotation IDs that satisfy given filter conditions.

        Args:
            imgIds: Filter by image IDs (list or single ID)
            catIds: Filter by category IDs (list or single ID)
            areaRng: Filter by area range [min, max]
            iscrowd: Filter by iscrowd flag (0 or 1)

        Returns:
            List of annotation IDs matching the filter conditions
        """
        imgIds = imgIds or []
        catIds = catIds or []
        areaRng = areaRng or []

        imgIds = imgIds if isinstance(imgIds, list) else [imgIds]
        catIds = catIds if isinstance(catIds, list) else [catIds]

        if len(imgIds) == 0:
            anns = self.dataset["annotations"]
        else:
            anns = []
            for img_id in imgIds:
                anns.extend(self.imgToAnns.get(img_id, []))

        if len(catIds) > 0:
            anns = [ann for ann in anns if ann["category_id"] in catIds]

        if len(areaRng) == 2:
            anns = [ann for ann in anns if ann["area"] >= areaRng[0] and ann["area"] <= areaRng[1]]

        if iscrowd is not None:
            anns = [ann for ann in anns if ann["iscrowd"] == iscrowd]

        return [ann["id"] for ann in anns]

    def getCatIds(self, catNms=None, supNms=None, catIds=None):
        """Get category IDs that satisfy given filter conditions.

        Args:
            catNms: Filter by category names (list)
            supNms: Filter by supercategory names (list, not used)
            catIds: Filter by category IDs (list)

        Returns:
            List of category IDs matching the filter conditions
        """
        catNms = catNms or []
        # supNms = supNms or []
        catIds = catIds or []

        cats = self.dataset["categories"]

        if len(catNms) > 0:
            cats = [cat for cat in cats if cat["name"] in catNms]
        if len(catIds) > 0:
            cats = [cat for cat in cats if cat["id"] in catIds]

        return [cat["id"] for cat in cats]

    def getImgIds(self, imgIds=None, catIds=None):
        """Get image IDs that satisfy given filter conditions.

        Args:
            imgIds: Filter to these image IDs (list)
            catIds: Filter by images containing these category IDs (list)

        Returns:
            List of image IDs matching the filter conditions
        """
        imgIds = imgIds or []
        catIds = catIds or []
        imgIds = set(imgIds) if imgIds else set(self.imgs.keys())

        if len(catIds) > 0:
            # Find all images that contain at least one of the specified categories
            matching_img_ids = set()
            for cat_id in catIds:
                matching_img_ids.update(self.catToImgs.get(cat_id, []))

            # Intersect with existing imgIds filter
            imgIds &= matching_img_ids

        return list(imgIds)

    def loadAnns(self, ids=None):
        """Load annotations with the specified IDs.

        Args:
            ids: Annotation IDs to load (list or single ID)

        Returns:
            List of annotation dicts with keys: id, image_id, category_id, bbox, area, iscrowd
        """
        if ids is None:
            return []
        ids = ids if isinstance(ids, list) else [ids]
        return [self.anns[ann_id] for ann_id in ids if ann_id in self.anns]

    def loadCats(self, ids=None):
        """Load categories with the specified IDs.

        Args:
            ids: Category IDs to load (list or single ID). If None, returns all categories.

        Returns:
            List of category dicts with keys: id, name, supercategory
        """
        if ids is None:
            return list(self.cats.values())
        ids = ids if isinstance(ids, list) else [ids]
        return [self.cats[cat_id] for cat_id in ids if cat_id in self.cats]

    def loadImgs(self, ids=None):
        """Load images with the specified IDs.

        Args:
            ids: Image IDs to load (list or single ID)

        Returns:
            List of image dicts with keys: id, file_name, height, width
        """
        if ids is None:
            return []
        ids = ids if isinstance(ids, list) else [ids]
        return [self.imgs[img_id] for img_id in ids if img_id in self.imgs]


class YoloDetection(VisionDataset):
    """
    YOLO format dataset using supervision.DetectionDataset.from_yolo().

    This class provides a VisionDataset interface compatible with RF-DETR training,
    matching the API of CocoDetection.

    Args:
        img_folder: Path to the directory containing images
        lb_folder: Path to the directory containing YOLO annotation .txt files
        data_file: Path to data.yaml file containing class names and dataset info
        transforms: Optional transforms to apply to images and targets
        include_masks: Whether to load segmentation masks (for YOLO segmentation format)
        data_fraction: Fraction of samples to use (0.0–1.0). Applied before loading
            to avoid opening every image file. Default: 1.0 (use all).
        seed: Random seed for reproducible subsampling. Default: 0.
    """

    _IMAGE_EXTENSIONS = {
        ".bmp", ".dng", ".jpg", ".jpeg", ".mpo",
        ".png", ".tif", ".tiff", ".webp",
    }

    def __init__(
        self,
        img_folder: str,
        lb_folder: str,
        data_file: str,
        transforms=None,
        include_masks: bool = False,
        data_fraction: float = 1.0,
        seed: int = 0,
    ):
        super(YoloDetection, self).__init__(img_folder)
        self._transforms = transforms
        self.include_masks = include_masks
        self.prepare = ConvertYolo(include_masks=include_masks)

        if data_fraction < 1.0:
            self.sv_dataset = self._load_fraction(
                img_folder, lb_folder, data_file,
                include_masks, data_fraction, seed,
            )
        else:
            # Load dataset using supervision's from_yolo method
            self.sv_dataset = sv.DetectionDataset.from_yolo(
                images_directory_path=img_folder,
                annotations_directory_path=lb_folder,
                data_yaml_path=data_file,
                force_masks=include_masks,
            )

        self.classes = self.sv_dataset.classes
        self.ids = list(range(len(self.sv_dataset)))

        # Create COCO-compatible API for evaluation
        self.coco = CocoLikeAPI(self.classes, self.sv_dataset)

    def _load_fraction(self, img_folder, lb_folder, data_file,
                       include_masks, fraction, seed):
        """Load only a fraction of the dataset to avoid slow full scans."""
        import random as _random
        import yaml

        # Read class names from data.yaml
        with open(data_file, "r") as f:
            data = yaml.safe_load(f)
        if isinstance(data["names"], dict):
            classes = [data["names"][i] for i in sorted(data["names"].keys())]
        else:
            classes = data["names"]

        # List image files
        img_dir = Path(img_folder)
        all_images = sorted([
            p for p in img_dir.iterdir()
            if p.suffix.lower() in self._IMAGE_EXTENSIONS
        ])

        # Subsample
        n_subset = max(1, int(len(all_images) * fraction))
        rng = _random.Random(seed)
        subset_images = sorted(rng.sample(all_images, n_subset))

        # Parse only the subset annotations (mirrors supervision's logic)
        from supervision.dataset.formats.yolo import (
            yolo_annotations_to_detections,
            _with_mask,
        )
        from supervision.utils.file import read_txt_file

        image_paths = []
        annotations = {}
        for img_path in subset_images:
            img_str = str(img_path)
            label_path = os.path.join(lb_folder, f"{img_path.stem}.txt")
            if not os.path.exists(label_path):
                image_paths.append(img_str)
                annotations[img_str] = sv.Detections.empty()
                continue

            image = Image.open(img_path)
            w, h = image.size
            lines = read_txt_file(file_path=label_path, skip_empty=True)
            with_masks = _with_mask(lines=lines)
            with_masks = include_masks if include_masks else with_masks
            det = yolo_annotations_to_detections(
                lines=lines, resolution_wh=(w, h),
                with_masks=with_masks, is_obb=False,
            )
            image_paths.append(img_str)
            annotations[img_str] = det

        return sv.DetectionDataset(
            classes=classes, images=image_paths, annotations=annotations
        )

    def __len__(self) -> int:
        return len(self.sv_dataset)

    def __getitem__(self, idx: int):
        image_id = self.ids[idx]
        image_path, cv2_image, detections = self.sv_dataset[idx]

        # Convert BGR (OpenCV) to RGB (PIL)
        rgb_image = cv2_image[:, :, ::-1]
        img = Image.fromarray(rgb_image)

        target = {"image_id": image_id, "detections": detections}
        img, target = self.prepare(img, target)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target


def build_roboflow_from_yolo(image_set: str, args: Any, resolution: int) -> YoloDetection:
    """Build a Roboflow YOLO-format dataset.

    Supports both YOLO-A (``images/{split}/``) and YOLO-B (``{split}/images/``)
    directory layouts, and accepts ``val`` as an alias for ``valid``.
    """
    root = Path(args.dataset_dir)
    assert root.exists(), f"provided Roboflow path {root} does not exist"

    layout = _detect_yolo_layout(str(root))
    val_name = _resolve_val_split_name(str(root), layout) if layout else "valid"

    if layout == "yolo-a":
        # YOLO-A: images/{split}/ + labels/{split}/
        PATHS = {
            "train": (root / "images" / "train", root / "labels" / "train"),
            "val": (root / "images" / val_name, root / "labels" / val_name),
            "test": (root / "images" / "test", root / "labels" / "test"),
        }
    else:
        # YOLO-B (default): {split}/images/ + {split}/labels/
        PATHS = {
            "train": (root / "train" / "images", root / "train" / "labels"),
            "val": (root / val_name / "images", root / val_name / "labels"),
            "test": (root / "test" / "images", root / "test" / "labels"),
        }

    data_file = root / "data.yaml"
    img_folder, lb_folder = PATHS[image_set.split("_")[0]]
    square_resize_div_64 = getattr(args, "square_resize_div_64", False)
    include_masks = getattr(args, "segmentation_head", False)
    multi_scale = getattr(args, "multi_scale", False)
    expanded_scales = getattr(args, "expanded_scales", None)
    do_random_resize_via_padding = getattr(args, "do_random_resize_via_padding", False)
    patch_size = getattr(args, "patch_size", None)
    num_windows = getattr(args, "num_windows", None)

    # Only subsample the training split
    data_fraction = getattr(args, "data_fraction", 1.0)
    fraction_for_split = data_fraction if image_set == "train" else 1.0
    seed = getattr(args, "seed", 0)

    if square_resize_div_64:
        dataset = YoloDetection(
            img_folder=str(img_folder),
            lb_folder=str(lb_folder),
            data_file=str(data_file),
            transforms=make_coco_transforms_square_div_64(
                image_set,
                resolution,
                multi_scale=multi_scale,
                expanded_scales=expanded_scales,
                skip_random_resize=not do_random_resize_via_padding,
                patch_size=patch_size,
                num_windows=num_windows,
            ),
            include_masks=include_masks,
            data_fraction=fraction_for_split,
            seed=seed,
        )
    else:
        dataset = YoloDetection(
            img_folder=str(img_folder),
            lb_folder=str(lb_folder),
            data_file=str(data_file),
            transforms=make_coco_transforms(
                image_set,
                resolution,
                multi_scale=multi_scale,
                expanded_scales=expanded_scales,
                skip_random_resize=not do_random_resize_via_padding,
                patch_size=patch_size,
                num_windows=num_windows,
            ),
            include_masks=include_masks,
            data_fraction=fraction_for_split,
            seed=seed,
        )
    return dataset
