import os
from ultralytics import YOLO
import hashlib
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import json
import uuid
import secrets
import time
import tempfile
import shutil
import torch
from collections import defaultdict
from torchvision.ops import nms
from sahi.postprocess.combine import nms
import gc
from config import BASE_DIR
import openslide
from openslide import OpenSlideUnsupportedFormatError
import math
from sqlalchemy import select, inspect
from models import (
    TestEnum,
    User,
    Patient,
    Sample,
    Report,
    CompleteBloodCount,
    AnnotationBox,
    Annotation,
)
from fastapi import Depends, HTTPException, status
from sahi.postprocess.combine import nms
import gc
from config import BASE_DIR
import math

from PIL import Image, ImageEnhance, ImageFilter
from typing import Union


def generate_uuid7() -> uuid.UUID:
    if hasattr(uuid, "uuid7"):
        return uuid.uuid7()

    unix_ts_ms = int(time.time() * 1000) & ((1 << 48) - 1)
    rand_a = secrets.randbits(12)
    rand_b = secrets.randbits(62)

    uuid_int = 0
    uuid_int |= unix_ts_ms << 80
    uuid_int |= 0x7 << 76
    uuid_int |= rand_a << 64
    uuid_int |= 0b10 << 62
    uuid_int |= rand_b

    return uuid.UUID(int=uuid_int)


def compute_sha256(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_checksum(file_path: str, expected_checksum: str) -> bool:
    if not expected_checksum:
        return True
    return compute_sha256(file_path) == expected_checksum


def process_results(results, save_crops=True, crop_dir="pipeline_output/crops"):
    image_detections = []

    for result in results:
        image_name = os.path.basename(result.path)
        image = cv2.imread(result.path)
        img_h, img_w = image.shape[:2]

        boxes = result.boxes.xywh.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        class_names = result.names

        detections = []

        for idx, (xywh, conf, cls_id) in enumerate(zip(boxes, confs, class_ids)):
            x_center, y_center, w, h = xywh
            label = class_names[cls_id]

            # ------------------------------------------------------------------
            # 1. DEFINE SQUARE CROP SIZE (MIN 256)
            # ------------------------------------------------------------------
            side = int(np.ceil(max(w, h)))
            final_side = max(side, 256)

            # ------------------------------------------------------------------
            # 2. DEFINE SQUARE SOURCE WINDOW (NO CLIPPING)
            # ------------------------------------------------------------------
            x1 = int(np.floor(x_center - side / 2))
            y1 = int(np.floor(y_center - side / 2))
            x2 = x1 + side
            y2 = y1 + side

            crop_path = None
            if save_crops:
                # ------------------------------------------------------------------
                # 3. CREATE FINAL BLACK CANVAS
                # ------------------------------------------------------------------
                crop_img = np.zeros((final_side, final_side, 3), dtype=np.uint8)

                # Offset to center square inside final canvas (when side < 256)
                offset = (final_side - side) // 2

                # ------------------------------------------------------------------
                # 4. COMPUTE VALID IMAGE REGION
                # ------------------------------------------------------------------
                src_x1 = max(0, x1)
                src_y1 = max(0, y1)
                src_x2 = min(img_w, x2)
                src_y2 = min(img_h, y2)

                # ------------------------------------------------------------------
                # 5. DESTINATION COORDS IN CANVAS
                # ------------------------------------------------------------------
                dst_x1 = offset + (src_x1 - x1)
                dst_y1 = offset + (src_y1 - y1)
                dst_x2 = dst_x1 + (src_x2 - src_x1)
                dst_y2 = dst_y1 + (src_y2 - src_y1)

                # ------------------------------------------------------------------
                # 6. COPY PIXELS
                # ------------------------------------------------------------------
                if src_x1 < src_x2 and src_y1 < src_y2:
                    crop_img[dst_y1:dst_y2, dst_x1:dst_x2] = image[
                        src_y1:src_y2, src_x1:src_x2
                    ]

                # ------------------------------------------------------------------
                # 7. SAVE
                # ------------------------------------------------------------------
                class_folder = os.path.join(crop_dir, label)
                os.makedirs(class_folder, exist_ok=True)

                crop_filename = f"{os.path.splitext(image_name)[0]}_{idx}.png"
                crop_path = os.path.join(class_folder, crop_filename)
                cv2.imwrite(crop_path, crop_img)

            # ------------------------------------------------------------------
            # 8. METADATA
            # ------------------------------------------------------------------
            detections.append(
                {
                    "object_id": str(generate_uuid7()),
                    "bbox": [float(x_center), float(y_center), float(w), float(h)],
                    "main_class": {
                        "label": str(label),
                        "confidence": round(float(conf), 4),
                    },
                    "sub_class": {"label": None, "confidence": None},
                    "segmentation": {"mask": None, "size": None},
                    "crop_path": crop_path,
                }
            )

        image_detections.append(
            {
                "image_name": image_name,
                "original_size": [img_w, img_h],
                "detections": detections,
            }
        )

    return image_detections


def run_pipeline(
    model_path,
    image_dir,
    batch_size=768,
    save_detections=False,
    output_json="pipeline_output/detections.json",
    save_crops=True,
):
    model = YOLO(model_path)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    results = model(
        image_dir, imgsz=608, batch=batch_size, save=save_detections, verbose=False
    )
    detections = process_results(results, save_crops)
    torch.cuda.empty_cache()
    del model
    gc.collect()
    # with open(output_json, "w") as f:
    #   json.dump(detections, f, indent=2)

    return detections


def scale_annotations(all_annotations):
    scaled_annotations = []
    for annotation in all_annotations:
        imagename = annotation["image_name"]
        row_str, col_str = imagename.replace(".jpeg", "").split("_")
        row_no = int(row_str)
        col_no = int(col_str)

        for box in annotation["detections"]:
            x1, y1, w, h = box["bbox"]
            x = row_no * 512 + x1
            y = col_no * 512 + y1

            x = x - w / 2
            y = y - h / 2

            if col_no == 0 and row_no != 0:
                x = x - 40
            elif col_no != 0 and row_no == 0:
                y = y - 40
            elif col_no != 0 and row_no != 0:
                x = x - 40
                y = y - 40

            data = {
                "id": box["object_id"],
                "boxes": [x, y, w, h],
                "main_class": box["main_class"],
                "sub_class": box["sub_class"],
                "segmentation": box["segmentation"],
            }
            scaled_annotations.append(data)

    return scaled_annotations


# sahi NMS
def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def is_near_boundary(box, tile_size=512, margin=80, extra_margin=20):
    margin = margin + extra_margin  # Make margin "thicker"
    x, y, w, h = box
    x2, y2 = x + w, y + h
    return (
        x % tile_size <= margin
        or x2 % tile_size >= (tile_size - margin)
        or y % tile_size <= margin
        or y2 % tile_size >= (tile_size - margin)
    )


def apply_nms_on_boundaries(
    all_annotations, iou_threshold=0.1, tile_size=512, margin=40, class_agnostic=True
):
    boundary_idxs = []
    boxes = []
    scores = []
    category_ids = []
    inner_idxs = []

    # Only keep indices, not the whole dicts
    for idx, annotation in enumerate(all_annotations):
        if is_near_boundary(annotation["boxes"], tile_size, margin, extra_margin=60):
            boundary_idxs.append(idx)
            boxes.append(xywh_to_xyxy(annotation["boxes"]))
            scores.append(annotation["main_class"]["confidence"])
            # Use 0 for class-agnostic, or actual category_id for class-aware
            category_id = (
                0 if class_agnostic else annotation["main_class"].get("category_id", 0)
            )
            category_ids.append(category_id)
        else:
            inner_idxs.append(idx)

    if not boundary_idxs:
        return all_annotations

    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    category_tensor = torch.tensor(category_ids, dtype=torch.float32).unsqueeze(1)

    input_tensor = torch.cat(
        [boxes_tensor, scores_tensor.unsqueeze(1), category_tensor], dim=1
    )
    keep = nms(input_tensor, match_metric="IOU", match_threshold=iou_threshold)

    kept_boundary_idxs = [boundary_idxs[i] for i in keep]
    # Only build the output list at the end
    return [all_annotations[i] for i in inner_idxs + kept_boundary_idxs]


def apply_global_nms(annotations, iou_threshold=0.2, class_agnostic=True):
    """
    Apply global NMS to remove all duplicate detections.

    Args:
        annotations (list): List of detection annotation dictionaries
        iou_threshold (float): IoU threshold for NMS (lower = more aggressive removal)
        class_agnostic (bool): Whether to ignore class labels during NMS

    Returns:
        list: List of annotations after NMS
    """
    if not annotations:
        return annotations

    boxes = []
    scores = []
    category_ids = []

    # Extract boxes, scores and categories from all annotations
    for ann in annotations:
        boxes.append(xywh_to_xyxy(ann["boxes"]))
        scores.append(ann["main_class"]["confidence"])
        # If class_agnostic is True, use 0 for all boxes to ignore class
        # Otherwise use the actual category_id
        category_id = 0 if class_agnostic else ann["main_class"].get("category_id", 0)
        category_ids.append(category_id)

    # Convert to tensors
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    category_tensor = torch.tensor(category_ids, dtype=torch.float32).unsqueeze(1)

    # Format for NMS function: [x1, y1, x2, y2, score, category_id]
    input_tensor = torch.cat(
        [boxes_tensor, scores_tensor.unsqueeze(1), category_tensor], dim=1
    )

    # Apply NMS
    keep = nms(input_tensor, match_metric="IOU", match_threshold=iou_threshold)

    # Return only the kept annotations
    return [annotations[i] for i in keep]


def crop_region_from_slide(
    slide_path,
    x,
    y,
    width,
    height,
    save_path=None,
    level=0,
    padding=40,
    min_size=300,
    scale=None,
):
    """
    Returns or saves a square crop.

    Rules:
    - Black padding appears ONLY where the requested crop lies outside the slide.
    - No resizing is ever performed.
    - min_size is enforced by expanding the crop region.
    """

    # --------------------------------------------------------------
    # CLASS-BASED PADDING OVERRIDE
    # --------------------------------------------------------------
    if save_path:
        if scale == '40x':
            if "WBC" in save_path:
                padding = 80
                min_size = 350
            elif "RBC" in save_path:
                padding = 40
                min_size = 200
            elif "Platelet" in save_path:
                padding = 20
                min_size = 100
            elif "Artefact" in save_path:
                padding = 80
                min_size = 300
            else:
                padding = 40
                min_size = 100
        elif scale == '20x':
            if "WBC" in save_path:
                padding = 40
                min_size = 175
            elif "RBC" in save_path:
                padding = 20
                min_size = 100
            elif "Platelet" in save_path:
                padding = 10
                min_size = 100
            elif "Artefact" in save_path:
                padding = 40
                min_size = 175
            else:
                padding = 20
                min_size = 100
        else:
            if "WBC" in save_path:
                padding = 40
                min_size = 175
            elif "RBC" in save_path:
                padding = 20
                min_size = 100
            elif "Platelet" in save_path:
                padding = 10
                min_size = 100
            elif "Artefact" in save_path:
                padding = 40
                min_size = 175
            else:
                padding = 20
                min_size = 100

    # --------------------------------------------------------------
    # 1. DEFINE SQUARE CROP IN SLIDE COORDS (NO RESIZING)
    # --------------------------------------------------------------
    padded_w = width + 2 * padding
    padded_h = height + 2 * padding

    side = int(math.ceil(max(padded_w, padded_h)))

    # Enforce min_size by EXPANDING CROP AREA
    side = max(side, min_size)

    x1 = int(math.floor(x + width / 2 - side / 2))
    y1 = int(math.floor(y + height / 2 - side / 2))
    x2 = x1 + side
    y2 = y1 + side

    # --------------------------------------------------------------
    # 2. READ AVAILABLE REGION FROM SLIDE / IMAGE
    # --------------------------------------------------------------
    try:
        slide = openslide.OpenSlide(slide_path)
        slide_width, slide_height = slide.dimensions

        src_x1 = max(0, x1)
        src_y1 = max(0, y1)
        src_x2 = min(slide_width, x2)
        src_y2 = min(slide_height, y2)

        if src_x1 >= src_x2 or src_y1 >= src_y2:
            return None

        region = slide.read_region(
            (src_x1, src_y1),
            level,
            (src_x2 - src_x1, src_y2 - src_y1),
        ).convert("RGB")

    except OpenSlideUnsupportedFormatError:
        image = Image.open(slide_path).convert("RGB")
        img_width, img_height = image.size

        src_x1 = max(0, x1)
        src_y1 = max(0, y1)
        src_x2 = min(img_width, x2)
        src_y2 = min(img_height, y2)

        if src_x1 >= src_x2 or src_y1 >= src_y2:
            return None

        region = image.crop((src_x1, src_y1, src_x2, src_y2))

    crosses_boundary = src_x1 > x1 or src_y1 > y1 or src_x2 < x2 or src_y2 < y2

    # --------------------------------------------------------------
    # 3. HANDLE BOUNDARIES (BLACK ONLY WHERE NEEDED)
    # --------------------------------------------------------------
    if not crosses_boundary:
        # Fully inside slide → NO black padding
        crop_img = region
    else:
        # Crosses slide boundary → black ONLY where pixels are missing
        crop_img = Image.new("RGB", (side, side), (0, 0, 0))

        dst_x = src_x1 - x1
        dst_y = src_y1 - y1

        crop_img.paste(region, (dst_x, dst_y))

    # --------------------------------------------------------------
    # 4. SAVE OR RETURN
    # --------------------------------------------------------------
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        final_path = os.path.join(save_path, f"{x}_{y}.png")
        crop_img.save(final_path)
        return final_path

    return crop_img


async def crop_and_save(
    db,
    patient_id: str,
    test_id: str,
    filename: str,
    annotationId: str,
    all_annotations: list,
    isRBC: bool = False,
    isWBC: bool = False,
    isPlatelet: bool = False,
):
    if not isRBC and not isWBC and not isPlatelet:
        return False
    else:
        try:
            target_dir = os.path.join(BASE_DIR, "samples", patient_id, test_id)
            slide_path = None
            for f in os.listdir(target_dir):
                if os.path.splitext(f)[0] == os.path.splitext(filename)[0]:
                    slide_path = os.path.join(target_dir, f)
                    break

            if slide_path is None:
                print(f"No matching slide file found for {filename} in {target_dir}")
                return False

            # Get sample to retrieve scale
            sample_query = select(Sample).where(
                Sample.test_id == test_id,
                Sample.sample_location.like(f"%/{filename}.%"),
            )
            sample_result = await db.execute(sample_query)
            sample = sample_result.scalar_one_or_none()
            if not sample:
                print(f"Sample not found for {filename} in test {test_id}")
                return False

            # Get existing annotation boxes from database
            existing_boxes_query = select(AnnotationBox).where(
                AnnotationBox.annotation_id == annotationId
            )
            existing_boxes_result = await db.execute(existing_boxes_query)
            existing_boxes = existing_boxes_result.scalars().all()
            if isWBC:
                crop_path_wbc = os.path.join(
                    BASE_DIR,
                    "crops",
                    patient_id,
                    test_id,
                    filename,
                    annotationId,
                    "WBC",
                )
                os.makedirs(crop_path_wbc, exist_ok=True)

                for box in existing_boxes:
                    if box.class_name == "WBC":
                        x, y, w, h = box.boxes
                        path = crop_region_from_slide(
                            slide_path, x, y, w, h, crop_path_wbc, padding=80, scale=sample.scale
                        )

                        # Update crop_path in database
                        box.crop_path = path
                        db.add(box)  # Explicitly add to session

            if isRBC:
                crop_path_rbc = os.path.join(
                    BASE_DIR,
                    "crops",
                    patient_id,
                    test_id,
                    filename,
                    annotationId,
                    "RBC",
                )
                os.makedirs(crop_path_rbc, exist_ok=True)

                for box in existing_boxes:
                    if box.class_name == "RBC":
                        x, y, w, h = box.boxes
                        path = crop_region_from_slide(
                            slide_path, x, y, w, h, crop_path_rbc, padding=40, scale=sample.scale
                        )

                        # Update crop_path in database
                        box.crop_path = path
                        db.add(box)  # Explicitly add to session

            if isPlatelet:
                crop_path_platelet = os.path.join(
                    BASE_DIR,
                    "crops",
                    patient_id,
                    test_id,
                    filename,
                    annotationId,
                    "Platelet",
                )
                os.makedirs(crop_path_platelet, exist_ok=True)

                for box in existing_boxes:
                    if box.class_name == "Platelets":
                        x, y, w, h = box.boxes
                        path = crop_region_from_slide(
                            slide_path, x, y, w, h, crop_path_platelet, padding=20, scale=sample.scale
                        )

                        # Update crop_path in database
                        box.crop_path = path
                        db.add(box)  # Explicitly add to session

            # Commit changes to database
            await db.commit()
        except Exception as e:
            print(f"Error in crop_and_save: {e}")
            await db.rollback()
            return False
    return True


async def update_annotations_db(
    db, patient_id, test_id, filename, annotationId, all_annotations
):
    from sqlalchemy.orm import selectinload

    # Use JOIN to fetch Sample with Annotations in a single query
    sample_query = (
        select(Sample)
        .where(
            Sample.test_id == test_id,
            Sample.sample_location.like(f"%/{filename}.%"),
        )
        .options(selectinload(Sample.annotations))
    )
    sample_result = await db.execute(sample_query)
    sample = sample_result.scalar_one_or_none()
    if not sample:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sample not found or does not belong to this test.",
        )

    # Find the annotation from preloaded annotations
    existing_annotation = None
    for anno in sample.annotations:
        if str(anno.id) == annotationId:
            existing_annotation = anno
            break

    if not existing_annotation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Annotation instance not found for this image.",
        )
    # Update existing annotation
    existing_annotation.tool = "point"
    existing_annotation.isClassDetected = True

    await db.flush()
    # Get existing annotation boxes
    existing_boxes_query = select(AnnotationBox).where(
        AnnotationBox.annotation_id == annotationId
    )
    existing_boxes_result = await db.execute(existing_boxes_query)
    existing_boxes = existing_boxes_result.scalars().all()

    # Delete all existing annotation boxes for this annotation
    for box in existing_boxes:
        await db.delete(box)
    await db.flush()
    # Create a lookup dictionary for existing boxes by ID
    existing_box_dict = {box.id: box for box in existing_boxes}
    # Process incoming annotations - update existing or add new
    for anno in all_annotations:
        anno_id = uuid.UUID(anno["id"].lstrip("#"))
        if anno_id in existing_box_dict:
            # Update existing box
            box = existing_box_dict[(anno_id)]
            box.boxes = anno["boxes"]
            box.class_name = anno["main_class"]["label"]
            box.sub_class = anno["sub_class"]["label"]
            box.class_confidence = anno["main_class"]["confidence"]
            box.sub_class_confidence = anno["sub_class"]["confidence"]
            box.crop_path = anno.get("crop_path", None)
            # db.merge(box)  # Use merge to update the existing box
        else:
            # Add new box
            new_box = AnnotationBox(
                id=anno_id,
                annotation_id=annotationId,
                boxes=anno["boxes"],
                class_name=anno["main_class"]["label"],
                sub_class=anno["sub_class"]["label"],
                class_confidence=anno["main_class"]["confidence"],
                sub_class_confidence=anno["sub_class"]["confidence"],
                crop_path=anno.get("crop_path", None),
                isModelClass=True,
            )
            db.add(new_box)
    await db.commit()
    return True


def apply_filters(
    image_path: str,
    brightness_factor: float = 1.0,
    saturation_factor: float = 1.0,
    contrast_factor: float = 1.0,
    method: str = "",
    strength: Union[int, float] = 2,
) -> np.ndarray:
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    result = image.astype(np.float32) / 255.0

    if contrast_factor != 1.0:
        result = (result - 0.5) * contrast_factor + 0.5
        result = np.clip(result, 0, 1)

    if brightness_factor != 1.0:
        result = result * brightness_factor
        result = np.clip(result, 0, 1)

    if saturation_factor != 1.0:
        hsv = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(
            np.float32
        )
        hsv[..., 1] *= saturation_factor
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        result = (
            cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
            / 255.0
        )

    result = (result * 255).astype(np.uint8)

    # 🧹 Apply filter if specified
    if method == "gaussian":
        ksize = int(strength) * 2 + 1
        result = cv2.GaussianBlur(result, (ksize, ksize), sigmaX=0)
    elif method == "box":
        ksize = int(strength) * 2 + 1
        result = cv2.blur(result, (ksize, ksize))
    elif method == "median":
        ksize = int(strength)
        if ksize % 2 == 0:
            ksize += 1
        result = cv2.medianBlur(result, ksize)

    return result
