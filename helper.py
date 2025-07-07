import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import json
import uuid
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
    Test,
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

def process_results(results, save_crops=True, crop_dir="pipeline_output/crops"):
    image_detections = []
    for result in results:
        image_name = os.path.split(result.path)[-1]
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

            # Convert xywh to xyxy
            x1 = int(max(0, x_center - w / 2))
            y1 = int(max(0, y_center - h / 2))
            x2 = int(min(img_w, x_center + w / 2))
            y2 = int(min(img_h, y_center + h / 2))

            crop_path = None
            if save_crops:
                crop_img = image[y1:y2, x1:x2]
                class_folder = os.path.join(crop_dir, label)
                os.makedirs(class_folder, exist_ok=True)
                crop_filename = f"{os.path.splitext(image_name)[0]}_{idx}.jpg"
                crop_path = os.path.join(class_folder, crop_filename)
                cv2.imwrite(crop_path, crop_img)

            detections.append(
                {
                    "object_id": str(uuid.uuid4()),
                    "bbox": [float(x_center), float(y_center), float(w), float(h)],
                    "main_class": {
                        "label": str(label),
                        "confidence": round(float(conf), 4),
                    },
                    "sub_class": {"label": None, "confidence": None},
                    "segmentation": {"mask": None, "size": None},
                    "crop_path": str(crop_path) if crop_path else None,
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
    results = model(image_dir, imgsz=608, batch=batch_size, save=save_detections, verbose=False)
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


# Convert to point annotations form bounding boxes
def create_point_annotations_json(boxes_and_annotations, source_url, path):

    annotations = []

    for anno in boxes_and_annotations:
        x, y, w, h = anno["boxes"]
        x = x + w / 2
        y = y + h

        annotation = {
            "@context": "http://www.w3.org/ns/anno.jsonld",
            "type": "Annotation",
            "id": f"#{anno['id']}",
            "body": [],
            "target": {
                "source": source_url,
                "selector": {
                    "type": "FragmentSelector",
                    "conformsTo": "http://www.w3.org/TR/media-frags/",
                    "value": f"xywh=pixel:{x},{y},0,0",
                },
                "renderedVia": {"name": "point"},
            },
        }

        # Add class and subclass bodies
        if anno["main_class"]["label"] != None:
            annotation["body"].append(
                {
                    "type": "TextualBody",
                    "purpose": "tagging",
                    "value": anno["main_class"]["label"],
                    "source": "class",
                }
            )
        if anno["sub_class"]["label"] != None:
            annotation["body"].append(
                {
                    "type": "TextualBody",
                    "purpose": "tagging",
                    "value": anno["sub_class"]["label"],
                    "source": "subclass",
                }
            )

        annotations.append(annotation)
    with open(path, "w") as f:
        json.dump(annotations, f, indent=2)

    # print(f"Annotations saved to {filename}")
    return True


def create_box_annotations_json(boxes_and_annotations, source_url, path):

    annotations = []

    for anno in boxes_and_annotations:
        x_min, y_min, width, height = anno["boxes"]

        annotation = {
            "@context": "http://www.w3.org/ns/anno.jsonld",
            "type": "Annotation",
            "id": f"#{anno['id']}",
            "body": [],
            "target": {
                "source": source_url,
                "selector": {
                    "type": "FragmentSelector",
                    "conformsTo": "http://www.w3.org/TR/media-frags/",
                    "value": f"xywh=pixel:{x_min},{y_min},{width},{height}",
                },
            },
        }

        # Add class and subclass bodies
        if anno["main_class"]["label"] != None:
            annotation["body"].append(
                {
                    "type": "TextualBody",
                    "purpose": "tagging",
                    "value": anno["main_class"]["label"],
                    "source": "class",
                }
            )
        if anno["sub_class"]["label"] != None:
            annotation["body"].append(
                {
                    "type": "TextualBody",
                    "purpose": "tagging",
                    "value": anno["sub_class"]["label"],
                    "source": "subclass",
                }
            )

        annotations.append(annotation)
    with open(path, "w") as f:
        json.dump(annotations, f, indent=2)

    # print(f"Annotations saved to {filename}")
    return True


def crop_region_from_slide(
    slide_path, x, y, width, height, save_path, level=0, padding=40
):
    try:
        slide = openslide.OpenSlide(slide_path)
        slide_width, slide_height = slide.dimensions
        x_padded = int(math.floor(max(x - padding, 0)))
        y_padded = int(math.floor(max(y - padding, 0)))
        x_max = int(math.ceil(min(x + width + padding, slide_width)))
        y_max = int(math.ceil(min(y + height + padding, slide_height)))
        w_padded = x_max - x_padded
        h_padded = y_max - y_padded

        region = slide.read_region(
            (x_padded, y_padded), level, (w_padded, h_padded)
        ).convert("RGB")
    except OpenSlideUnsupportedFormatError:
        # Fallback to PIL if the file isn't an OpenSlide-compatible slide
        image = Image.open(slide_path).convert("RGB")
        img_width, img_height = image.size
        x_padded = int(math.floor(max(x - padding, 0)))
        y_padded = int(math.floor(max(y - padding, 0)))
        x_max = int(math.ceil(min(x + width + padding, img_width)))
        y_max = int(math.ceil(min(y + height + padding, img_height)))
        region = image.crop((x_padded, y_padded, x_max, y_max))

    if save_path:
        final_path = os.path.join(save_path, f"{x_padded}_{y_padded}.png")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        region.save(final_path)
        return final_path
    else:
        return region


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
                            slide_path, x, y, w, h, crop_path_wbc, padding=80
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
                            slide_path, x, y, w, h, crop_path_rbc, padding=40
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
                            slide_path, x, y, w, h, crop_path_platelet, padding=20
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
    test_query = select(Test).where(Test.id == test_id)
    test_result = await db.execute(test_query)
    test = test_result.scalar_one_or_none()
    if not test:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Test not found or does not belong to this patient.",
        )
    # Check if sample exists and belongs to the test
    sample_query = select(Sample).where(
        Sample.test_id == test_id,
        Sample.sample_location.like(f"%/{filename}.%"),
    )
    sample_result = await db.execute(sample_query)
    sample = sample_result.scalar_one_or_none()
    if not sample:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Test not found or does not belong to this patient.",
        )
    annotation_query = select(Annotation).where(
        Annotation.id == annotationId,
        Annotation.sample_id == sample.id,
    )
    annotation_result = await db.execute(annotation_query)
    existing_annotation = annotation_result.scalar_one_or_none()
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
            )
            db.add(new_box)
    await db.commit()
    return True


# def apply_filters(
#     image_path: str,
#     brightness_factor: float = 1.0,
#     saturation_factor: float = 1.0,
#     contrast_factor: float = 1.0,
#     method: str = '',
#     strength: Union[int, float] = 2
# ) -> Image.Image:
#     """
#     Apply brightness, contrast, saturation, and a low-pass filter to an image.

#     Parameters:
#         image_path (str): Path to the input image.
#         brightness_factor (float): Brightness multiplier (1.0 = original).
#         saturation_factor (float): Saturation multiplier (1.0 = original).
#         contrast_factor (float): Contrast multiplier (1.0 = original).
#         method (str): Low-pass filter type ('gaussian', 'box', 'median').
#         strength (int or float): Strength of the filter (radius or kernel size).

#     Returns:
#         PIL.Image.Image: The processed image.
#     """
#     image = Image.open(image_path)
#     filtered = image.copy()

#     # Apply brightness
#     if not brightness_factor == 1: 
#         bright_enhancer = ImageEnhance.Brightness(image)
#         filtered = bright_enhancer.enhance(brightness_factor)

#     # Apply contrast
#     if not contrast_factor == 1: 
#         contrast_enhancer = ImageEnhance.Contrast(filtered)
#         filtered = contrast_enhancer.enhance(contrast_factor)

#     # Apply saturation
#     if not saturation_factor == 1: 
#         color_enhancer = ImageEnhance.Color(filtered)
#         filtered = color_enhancer.enhance(saturation_factor)

#     # Apply selected low-pass filter
#     if method == 'gaussian':
#         filtered = filtered.filter(ImageFilter.GaussianBlur(radius=strength))
#     elif method == 'box':
#         filtered = filtered.filter(ImageFilter.BoxBlur(radius=strength))
#     elif method == 'median':
#         size = strength if isinstance(strength, int) and strength % 2 == 1 else int(strength) + 1
#         filtered = filtered.filter(ImageFilter.MedianFilter(size=size))
#     else:
#         pass

#     return filtered

import cv2
import numpy as np

def apply_filters(
    image_path: str,
    brightness_factor: float = 1.0,
    saturation_factor: float = 1.0,
    contrast_factor: float = 1.0,
    method: str = '',
    strength: Union[int, float] = 2
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
        hsv = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] *= saturation_factor
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    result = (result * 255).astype(np.uint8)

    # 🧹 Apply filter if specified
    if method == 'gaussian':
        ksize = int(strength) * 2 + 1
        result = cv2.GaussianBlur(result, (ksize, ksize), sigmaX=0)
    elif method == 'box':
        ksize = int(strength) * 2 + 1
        result = cv2.blur(result, (ksize, ksize))
    elif method == 'median':
        ksize = int(strength)
        if ksize % 2 == 0:
            ksize += 1
        result = cv2.medianBlur(result, ksize)

    return result

