# PebSI-detection

## Files Created
- `config.py`: Contains configuration files for the application.
- `database.py`: Database connection and models.
- `helpers.py`: Helper functions for various tasks.
- `main.py`: Main entry point for the application.
- `models.py`: Contains the model definitions and training logic.
- `schemas.py`: Pydantic schemas for request and response validation.
- `requirements.txt`: List of Python dependencies.
- `.env.example`: Example environment configuration file.


## Prerequisites

- **Python**: 3.11
- **PostgreSQL**: 12 or higher
- **Conda**: For managing Python environments
  - OpenSlide (for medical image formats)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Pebsi-yolov12
```


### 2. Create conda environment

**Ubuntu/Debian:**

```bash
conda create --name pebsi-detection python=3.11
conda activate pebsi-detection
```

### 3. Install Python Dependencies

```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install -r requirements.txt
pip install -e .
```

### 4. Install Pytorch

Note: Ensure you have the correct CUDA version installed. The following command is for CUDA 12.6. Adjust the version as necessary based on your system configuration.

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 5. Environment Configuration

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Edit the `.env` file to set your environment variables:

```
URL_DATABASE=postgresql+asyncpg://<user>:<password>@<localhost>:<port>/<db_name>
BASE_DIR=/path/to/base/dir
```

For Example:

```
URL_DATABASE=postgresql+asyncpg://postgres:admin@localhost:5432/pebsi_db
BASE_DIR=/home/storage/pebsi
```

### 6. Run the Application

```bash
python main.py
```

The API will be available at `http://localhost:8001`
API Documentation: `http://localhost:8001/docs`


## Configuration

### Environment Variables

- `URL_DATABASE`: PostgreSQL connection string
- `BASE_DIR`: Base directory for file storage

---

## Function Documentation

### config.py

Configuration module containing application settings and environment variables.

**Variables:**
- `SECRET_KEY`: JWT secret key for token authentication
- `ALGORITHM`: JWT algorithm used for token signing (HS256)
- `ACCESS_TOKEN_EXPIRE_MINUTES`: Token expiration time in minutes (1440 = 24 hours)
- `URL_DATABASE`: PostgreSQL database connection URL from environment variables
- `BASE_DIR`: Base directory path for file storage from environment variables

### database.py

Database connection and session management module for async PostgreSQL operations.

**Functions:**

#### `get_async_db_url(url: str) -> str`
Converts a PostgreSQL URL from psycopg2 format to asyncpg format.

**Parameters:**
- `url`: The database URL string to convert

**Returns:**
- The converted URL with the asyncpg driver prefix if applicable, otherwise the original URL

#### `async initialize_database() -> AsyncEngine`
Initializes the database by creating the async engine. This should be called at application startup.

**Returns:**
- The initialized async SQLAlchemy engine

#### `get_engine() -> AsyncEngine`
Returns the async engine. If not initialized, raises a RuntimeError.

**Returns:**
- The async SQLAlchemy engine

**Raises:**
- `RuntimeError`: If database is not initialized

#### `get_session_factory() -> sessionmaker`
Returns the async session factory. Creates it if not already initialized.

**Returns:**
- The async session factory for creating database sessions

#### `async get_db() -> AsyncSession`
Dependency function to get an async database session. Used as FastAPI dependency.

**Yields:**
- `AsyncSession`: An async SQLAlchemy session for database operations

### helper.py

Helper functions for image processing, model inference, and database operations.

**Functions:**

#### `process_results(results, save_crops: bool = True, crop_dir: str = "pipeline_output/crops") -> list`
Processes YOLO model results and extracts detection information.

**Parameters:**
- `results`: YOLO model inference results
- `save_crops`: Whether to save cropped regions of detections
- `crop_dir`: Directory to save cropped images

**Returns:**
- List of image detection dictionaries

#### `run_pipeline(model_path: str, image_dir: str, batch_size: int = 768, save_detections: bool = False, output_json: str = "pipeline_output/detections.json", save_crops: bool = True) -> list`
Runs the complete detection pipeline on a directory of images.

**Parameters:**
- `model_path`: Path to the YOLO model file
- `image_dir`: Directory containing images to process
- `batch_size`: Batch size for model inference
- `save_detections`: Whether to save detection visualizations
- `output_json`: Path to save detection results JSON
- `save_crops`: Whether to save cropped detections

**Returns:**
- List of all detections from processed images

#### `scale_annotations(all_annotations: list) -> list`
Scales annotation coordinates based on image processing parameters.

**Parameters:**
- `all_annotations`: List of annotation dictionaries

**Returns:**
- List of scaled annotation dictionaries

#### `xywh_to_xyxy(box: list) -> list`
Converts bounding box from XYWH format to XYXY format.

**Parameters:**
- `box`: Bounding box in [x, y, width, height] format

**Returns:**
- Bounding box in [x1, y1, x2, y2] format

#### `is_near_boundary(box: list, tile_size: int = 512, margin: int = 80, extra_margin: int = 20) -> bool`
Checks if a bounding box is near tile boundaries.

**Parameters:**
- `box`: Bounding box in XYWH format
- `tile_size`: Size of image tiles
- `margin`: Boundary margin distance
- `extra_margin`: Additional margin buffer

**Returns:**
- True if box is near boundary, False otherwise

#### `apply_nms_on_boundaries(all_annotations: list, iou_threshold: float = 0.1, tile_size: int = 512, margin: int = 40, class_agnostic: bool = True) -> list`
Applies Non-Maximum Suppression specifically on boundary detections to remove duplicates.

**Parameters:**
- `all_annotations`: List of all annotation dictionaries
- `iou_threshold`: IoU threshold for NMS
- `tile_size`: Size of image tiles
- `margin`: Boundary margin distance
- `class_agnostic`: Whether to ignore class labels during NMS

**Returns:**
- List of annotations after boundary NMS

#### `apply_global_nms(annotations: list, iou_threshold: float = 0.2, class_agnostic: bool = True) -> list`
Applies global Non-Maximum Suppression to remove all duplicate detections.

**Parameters:**
- `annotations`: List of detection annotation dictionaries
- `iou_threshold`: IoU threshold for NMS (lower = more aggressive removal)
- `class_agnostic`: Whether to ignore class labels during NMS

**Returns:**
- List of annotations after global NMS

#### `create_point_annotations_json(boxes_and_annotations: list, source_url: str, path: str) -> bool`
Creates point annotations JSON file from bounding box detections.

**Parameters:**
- `boxes_and_annotations`: List of detection data
- `source_url`: Source URL of the image
- `path`: Output file path for JSON

**Returns:**
- True if successful

#### `create_box_annotations_json(boxes_and_annotations: list, source_url: str, path: str) -> bool`
Creates bounding box annotations JSON file from detections.

**Parameters:**
- `boxes_and_annotations`: List of detection data
- `source_url`: Source URL of the image
- `path`: Output file path for JSON

**Returns:**
- True if successful

#### `crop_region_from_slide(slide_path: str, x: int, y: int, width: int, height: int, save_path: str, level: int = 0, padding: int = 40) -> Image`
Crops a region from a slide image (supports various medical image formats).

**Parameters:**
- `slide_path`: Path to the slide image file
- `x, y`: Top-left coordinates of crop region
- `width, height`: Dimensions of crop region
- `save_path`: Path to save cropped image
- `level`: Pyramid level to read from
- `padding`: Additional padding around crop region

**Returns:**
- Cropped PIL Image object

#### `async crop_and_save(db: AsyncSession, patient_id: str, test_id: str, filename: str, annotationId: str, all_annotations: list, isRBC: bool = False, isWBC: bool = False, isPlatelet: bool = False) -> bool`
Crops and saves detected cells from annotations to database.

**Parameters:**
- `db`: Database session
- `patient_id`: Patient identifier
- `test_id`: Test identifier
- `filename`: Image filename
- `annotationId`: Annotation identifier
- `all_annotations`: List of detection annotations
- `isRBC`: Whether to process Red Blood Cells
- `isWBC`: Whether to process White Blood Cells
- `isPlatelet`: Whether to process Platelets

**Returns:**
- True if successful

#### `async update_annotations_db(db: AsyncSession, patient_id: str, test_id: str, filename: str, annotationId: str, all_annotations: list)`
Updates annotation data in the database with detection results.

**Parameters:**
- `db`: Database session
- `patient_id`: Patient identifier
- `test_id`: Test identifier
- `filename`: Image filename
- `annotationId`: Annotation identifier
- `all_annotations`: List of detection annotations

#### `apply_filters(image_path: str, brightness_factor: float = 1.0, saturation_factor: float = 1.0, contrast_factor: float = 1.0, method: str = '', strength: Union[int, float] = 2) -> np.ndarray`
Applies image enhancement filters including brightness, contrast, saturation, and low-pass filters.

**Parameters:**
- `image_path`: Path to input image
- `brightness_factor`: Brightness multiplier (1.0 = original)
- `saturation_factor`: Saturation multiplier (1.0 = original)
- `contrast_factor`: Contrast multiplier (1.0 = original)
- `method`: Low-pass filter type ('gaussian', 'box', 'median')
- `strength`: Strength/radius of the filter

**Returns:**
- Filtered image as numpy array

### main.py

Main FastAPI application with endpoints for blood cell detection.

**Functions:**

#### `async lifespan(app: FastAPI)`
Lifespan event handler for FastAPI application startup and shutdown.

**Parameters:**
- `app`: FastAPI application instance

#### `async root() -> dict`
Root endpoint returning API status message.

**Returns:**
- Dictionary with status message

#### `async detect_boxes(background_tasks: BackgroundTasks, data: DetectBoxesRequest, db: AsyncSession) -> JSONResponse`
Main detection endpoint that processes images and detects blood cells.

**Parameters:**
- `background_tasks`: FastAPI background tasks handler
- `data`: Request data containing patient info, images, and model parameters
- `db`: Database session dependency

**Returns:**
- JSON response with detection results

**Raises:**
- `HTTPException`: On processing errors

---

