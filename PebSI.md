# PebSI Detection Service (`PebSI-yolo12`)

This document is aligned with the current implementation in:

- `main.py`
- `helper.py`
- `middleware/security_middleware.py`
- `audit.py`
- `config.py`
- `models.py`

## What this service does

`PebSI-yolo12` is the detection microservice used by PebSI backend to:

- run detection inference on tiled images,
- post-process detections with boundary and global NMS,
- write annotation boxes into the database,
- optionally generate RBC / WBC / platelet crops,
- enforce internal-service security controls (mTLS + user-agent checks),
- verify model integrity via SHA-256 checksums stored in `ml_models`.

## Runtime and startup

Application entrypoint: `main.py`.

At startup (`lifespan`):

1. Initializes DB engine (`initialize_database()`),
2. Creates tables (`create_tables()`),
3. Starts FastAPI app with middleware chain.

At shutdown:

- disposes async SQLAlchemy engine safely.

The service runs with `uvicorn` and is deployed behind mandatory HTTPS with mTLS, plus a mandatory internal service-to-service JWT.

## API endpoints

### `GET /`

Health endpoint:

```json
{
  "message": "Detection Model API is running"
}
```

### `POST /detect_points`

Main detection endpoint.

Request model (`DetectBoxesRequest` in `schemas.py`):

- `patient_id: str`
- `test_id: str`
- `filename: str`
- `annotationId: str`
- `all_images: list[str]`
- `source_url: str`
- `model_id: str | null`
- `filter_data: dict`
- `generate_rbc_crops: bool = true`
- `generate_wbc_crops: bool = true`
- `generate_platelets_crops: bool = true`

Response on success:

```json
{
  "message": "Boxes detected and saved Successfully."
}
```

## Detection flow (`/detect_points`)

1. Captures audit user context (`x-user-id` header or request body value if present).
2. Resolves model by:
   - explicit `model_id` in category `detection_models`, or
   - default active model (`is_default = true`) in `detection_models`.
3. Verifies model file exists and checksum matches `ml_models.checksum_sha256`.
4. Copies incoming image paths to a request-scoped temp folder under `{BASE_DIR}/temp`.
5. Runs inference (`run_pipeline`) using YOLO model path.
6. Post-processes detections:
   - `scale_annotations(...)`
   - `apply_nms_on_boundaries(...)`
   - `apply_global_nms(...)`
7. Persists annotations (`update_annotations_db`).
8. Generates crops (`crop_and_save`) according to RBC/WBC/platelet flags.
9. Cleans temp folder and GPU cache.

## Security controls

Implemented in `middleware/security_middleware.py` and mounted in `main.py`.

- Exempt paths: `/docs`, `/redoc`, `/openapi.json`.
- Non-HTTPS requests are rejected (`403 mTLS required`).
- A valid internal JWT (`Authorization: Bearer <token>`) is required on all non-exempt paths (`401` on missing/invalid token).
- Requests with blocked user-agent keywords are rejected (`403 Client not allowed`).

Blocked user-agent list is configurable by `SECURITY_BLOCKED_USER_AGENTS`.

## TLS / mTLS behavior

`main.py` configures uvicorn SSL options from:

- `SSL_CERT_FILE`
- `SSL_KEY_FILE`
- `SSL_CA_FILE`

If cert values are missing, the service raises a startup error.

## Logging

The service creates a dedicated logger `pebsi_yolo12` and writes:

- file log: `{BASE_DIR}/logs/detection_api_calls.log`
- console log: stdout

Per-request middleware logs request path, status code, and duration.

## Audit context

`audit.py` defines a request-local context variable:

- `set_current_user(user_id)`
- `get_current_user()`

`main.py` sets this context in `/detect_points` so DB audit logic can attribute CRUD/READ activity to the invoking user when available.

## Data model touchpoints

Primary model used for model selection and integrity checks:

### `MLModel` (`models.py`)

Relevant columns:

- `id`
- `model_name`
- `model_path`
- `model_category`
- `model_type`
- `model_size_mb`
- `version`
- `checksum_sha256` (unique)
- `is_active`
- `is_default`
- `created_at`

Expected detection category in this service:

- `model_category = "detection_models"`

## Environment variables (`config.py`)

### Required/expected

- `URL_DATABASE`
- `BASE_DIR`

### Service runtime

- `SERVICE_HOST` (default: `0.0.0.0`)
- `SERVICE_PORT` (default: `8001`)
- `SERVICE_REFRESH` (`false` by default)

### Security / mTLS

- `SSL_CA_FILE`
- `SSL_CERT_FILE`
- `SSL_KEY_FILE`
- `SECURITY_BLOCKED_USER_AGENTS` (CSV)

### Security / internal JWT

- `INTERNAL_JWT_SECRET` (required)
- `INTERNAL_JWT_ISSUER` (default `pebsi-backend`)
- `INTERNAL_JWT_AUDIENCE` (default `pebsi-internal`)
- `INTERNAL_JWT_ALGORITHM` (default `HS256`)

### Optional shared secrets

Present in config and loaded from env when used by integrated flows:

- `SECRET_KEY`
- `ALGORITHM`
- `REFRESH_SECRET_KEY`

## Local run

```bash
cd /home/pebsi/harsh/PebSI/PebSI-yolo12
conda activate pebsi-detection
pip install -r requirements.txt
python main.py
```

Default API URL:

- `https://localhost:8001`

## Notes

- `filter_data` is accepted by the request schema; image filtering is currently not applied in active pipeline code (filter block is commented).
- Request temp data is cleaned in `finally` to avoid residual disk growth.
