import shutil
from contextlib import asynccontextmanager
import cv2
from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    Path,
    Request,
    Response,
    Depends,
)
from fastapi import status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from config import BASE_DIR
import os
from schemas import DetectBoxesRequest
from helper import (
    scale_annotations,
    run_pipeline,
    apply_nms_on_boundaries,
    apply_global_nms,
    crop_and_save,
    update_annotations_db,
    apply_filters,
)
import torch
import gc
from typing import Annotated
from sqlalchemy.ext.asyncio import AsyncSession
from database import initialize_database, get_db
import logging
from fastapi import Request
from datetime import datetime


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI application"""
    # Startup
    try:
        await initialize_database()
        print("Database initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize database: {e}")
        raise e

    yield

    # Shutdown
    try:
        from database import get_engine

        engine = get_engine()
        await engine.dispose()
        print("Database connections closed successfully!")
    except Exception as e:
        print(f"Error during database cleanup: {e}")


# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Database dependency
db_dependency = Annotated[AsyncSession, Depends(get_db)]


# Constants and Configuration
TEMP_FOLDER = os.path.join(BASE_DIR, "temp")


os.makedirs(TEMP_FOLDER, exist_ok=True)

# DETECTION_MODEL_PATH = "./models/model.pt"
DETECTION_MODEL_PATH = os.path.join(BASE_DIR, "models", "detection_models", "best_new_2.pt")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Root endpoint
@app.get("/")
async def root():
    return {"message": "Detection Model API is running"}


@app.post("/detect_points")
async def detect_boxes(
    background_tasks: BackgroundTasks,
    data: DetectBoxesRequest,
    db: db_dependency,
):

    try:
        patient_id = data.patient_id
        test_id = data.test_id
        filename = data.filename
        annotationId = data.annotationId
        all_images = data.all_images
        source_url = data.source_url
        model_path = data.model_path
        filter_data = data.filter_data

        if not os.path.exists(model_path) or not model_path or model_path == "":        
            model_path = DETECTION_MODEL_PATH

        temp_dir = os.path.join(
            TEMP_FOLDER, f"{patient_id}_{test_id}_{filename}_{annotationId}"
        )
        os.makedirs(temp_dir, exist_ok=True)
        count = 0
        for image_path in all_images:
            if os.path.exists(image_path):
                count += 1
                # filtered_image = apply_filters(
                #     image_path,
                #     brightness_factor=filter_data.get('brightness_factor'),
                #     saturation_factor=filter_data.get('saturation_factor'), 
                #     contrast_factor=filter_data.get('contrast_factor'), 
                #     method=filter_data.get('method'),
                #     strength= filter_data.get('strength'),
                # )
                # image_name = os.path.basename(image_path)
                # image_path = os.path.join(temp_dir, image_name)
                # output_bgr = cv2.cvtColor(filtered_image, cv2.COLOR_RGB2BGR)
                # cv2.imwrite(image_path, output_bgr)
                shutil.copy(image_path, temp_dir)
            else:
                print(f"File not found: {image_path}")
        # Now the temp_dir can be used for detection
        # print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(1, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        unscaled_detections = run_pipeline(
            model_path, temp_dir, batch_size=400, save_crops=False
        )
        torch.cuda.empty_cache()
        print(2, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        scaled_anno = scale_annotations(unscaled_detections)
        del unscaled_detections
        gc.collect()
        print(3, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        nms_applied = apply_nms_on_boundaries(scaled_anno)
        del scaled_anno
        gc.collect()
        print(4, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        final_deduced = apply_global_nms(nms_applied, iou_threshold=0.2)
        del nms_applied
        await update_annotations_db(
            db, patient_id, test_id, filename, annotationId, final_deduced
        )
        # background_tasks.add_task(crop_and_save, patient_id, test_id, filename, annotationId, final_deduced, True, True, True)
        print(5, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        await crop_and_save(
            db,
            patient_id,
            test_id,
            filename,
            annotationId,
            final_deduced,
            isRBC=False,
            isWBC=True,
            isPlatelet=True,
        )
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print(6, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        return JSONResponse(
            content={"message": f"Boxes detected and saved Successfully."},
            status_code=status.HTTP_200_OK,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error in detect_boxes: {e}",
        )
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
    # uvicorn.run("main:app", host="0.0.0.0", port=8201, reload=True)
