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

### models.py

SQLAlchemy database models for the medical imaging application.

**Enums:**

#### `RoleEnum`
User role enumeration with values: Admin, Editor, Viewer, SuperAdmin

#### `StatusEnum` 
User status enumeration with values: active, inactive

#### `GenderEnum`
Gender enumeration with values: male, female, other, ratherNotSay

#### `TestEnum`
Medical test type enumeration with values: cbc (Complete Blood Count)

**Models:**

#### `User`
Database model for system users.

**Attributes:**
- `id`: UUID primary key
- `name`: User full name
- `phone_no`: Phone number
- `employee_id`: Unique employee identifier
- `password`: Hashed password
- `role`: User role from RoleEnum
- `status`: User status from StatusEnum

**Relationships:**
- `reports`: Reports generated by this user
- `assignments_created`: Test assignments created by this user
- `assignments_received`: Test assignments received by this user

#### `Patient`
Database model for patient information.

**Attributes:**
- `id`: UUID primary key
- `name`: Patient full name
- `uhid`: Unique hospital identifier
- `dob`: Date of birth
- `phone_no`: Phone number
- `aadhar`: Aadhar card number
- `gender`: Gender from GenderEnum
- `created_at`: Record creation timestamp

**Relationships:**
- `tests`: Medical tests for this patient
- `samples`: Samples collected from this patient
- `reports`: Medical reports for this patient

#### `Test`
Database model for medical tests.

**Attributes:**
- `id`: UUID primary key
- `patient_id`: Foreign key to Patient
- `custom_sample_id`: Custom identifier for the test
- `condition`: Medical condition being tested
- `test_name`: Type of test from TestEnum
- `test_datetime`: Test execution timestamp

**Relationships:**
- `patient`: Associated patient
- `samples`: Samples collected for this test
- `report`: Generated report for this test
- `blood_counts`: Blood count results
- `assignments`: Test assignments

#### `Sample`
Database model for biological samples.

**Attributes:**
- `id`: UUID primary key
- `patient_id`: Foreign key to Patient
- `test_id`: Foreign key to Test
- `isProcessed`: Processing status flag
- `sample_location`: File system location of sample
- `sample_datetime`: Sample collection timestamp

**Relationships:**
- `patient`: Associated patient
- `test`: Associated test
- `cbc`: Complete blood count results
- `annotations`: Image annotations for this sample

#### `Report`
Database model for medical reports.

**Attributes:**
- `id`: UUID primary key
- `patient_id`: Foreign key to Patient
- `test_id`: Foreign key to Test
- `diagnosis`: Medical diagnosis
- `report_location`: File system location of report
- `report_datetime`: Report generation timestamp
- `generated_by_id`: Foreign key to User who generated report

**Relationships:**
- `patient`: Associated patient
- `test`: Associated test
- `generated_by`: User who generated the report

#### `CompleteBloodCount`
Database model for Complete Blood Count test results.

**Attributes:**
- `id`: UUID primary key
- `sample_id`: Foreign key to Sample
- `test_id`: Foreign key to Test
- Multiple blood parameter fields (haemoglobin, pcv, rbc_count, etc.)
- `created_at`: Record creation timestamp

**Relationships:**
- `sample`: Associated sample
- `test`: Associated test

#### `Annotation`
Database model for image annotations.

**Attributes:**
- `id`: String primary key
- `sample_id`: Foreign key to Sample
- `source`: Source of the annotation
- `type`: Annotation type
- `tool`: Tool used for annotation
- `height`, `width`: Image dimensions
- Processing status flags (isProcessed, isClassDetected, etc.)
- `created_at`: Creation timestamp

**Relationships:**
- `sample`: Associated sample
- `annotation_boxes`: Bounding box annotations

#### `AnnotationBox`
Database model for individual detection bounding boxes.

**Attributes:**
- `id`: UUID primary key
- `annotation_id`: Foreign key to Annotation
- `boxes`: JSON field containing bounding box coordinates
- `class_name`: Detected class name
- `class_confidence`: Detection confidence score
- `sub_class`: Detected sub-class
- `sub_class_confidence`: Sub-class confidence score
- `crop_path`: Path to cropped image
- `segmentation_polygon`: JSON field for segmentation data
- `created_at`: Creation timestamp

**Relationships:**
- `annotation`: Parent annotation
- `wbc_sub_class_confidences`: White blood cell sub-classification data

#### `WBCSubClassConfidences`
Database model for White Blood Cell sub-classification confidence scores.

**Attributes:**
- `id`: UUID primary key
- `annotation_box_id`: Foreign key to AnnotationBox
- `neutrophil_confidence`: Confidence score for neutrophil classification

**Relationships:**
- `annotation_box`: Associated annotation box

#### `CellClasses`
Database model for cell class definitions (implementation incomplete in provided code).

#### `CellSubClasses`
Database model for cell sub-class definitions (implementation incomplete in provided code).

#### `TestAssignment`
Database model for test assignments (implementation incomplete in provided code).

### schemas.py

Pydantic schemas for request and response validation.

**Classes:**

#### `DetectBoxesRequest`
Pydantic model for blood cell detection API requests.

**Attributes:**
- `patient_id`: String identifier for the patient
- `test_id`: String identifier for the test
- `filename`: Name of the image file to process
- `annotationId`: String identifier for the annotation
- `all_images`: List of image file paths to process
- `source_url`: Source URL of the images
- `model_path`: Path to the YOLO detection model
- `filter_data`: Dictionary containing image filter parameters

**Configuration:**
- `from_attributes = True`: Allows creation from ORM objects

---

