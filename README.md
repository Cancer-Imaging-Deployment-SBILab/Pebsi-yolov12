# PebSI YOLO12 Detection Service

FastAPI microservice for blood-cell detection in the PebSI platform.

## Features

- Detection inference from tiled image inputs
- Boundary NMS + global NMS deduplication
- Annotation box updates in PostgreSQL
- Optional crop generation (`RBC`, `WBC`, `Platelet`)
- mTLS-ready internal service transport
- Model resolution from `ml_models` with SHA-256 integrity verification
- Request logging and audit-context propagation

## Main endpoint

- `POST /detect_points`

Request schema (`DetectBoxesRequest`):

- `patient_id`
- `test_id`
- `filename`
- `annotationId`
- `all_images`
- `source_url`
- `model_id` (optional)
- `filter_data`
- `generate_rbc_crops`
- `generate_wbc_crops`
- `generate_platelets_crops`

Health endpoint:

- `GET /` → `{"message": "Detection Model API is running"}`

## Setup

```bash
conda create -n pebsi-detection python=3.11
conda activate pebsi-detection
pip install -r requirements.txt
```

## Environment

Copy and update:

```bash
cp .env.example .env
```

Minimum required values:

- `URL_DATABASE`
- `BASE_DIR`

Common runtime values:

- `SERVICE_HOST` (default `0.0.0.0`)
- `SERVICE_PORT` (default `8001`)
- `SERVICE_REFRESH` (default `false`)

mTLS values:

- `MTLS_ENABLED`
- `SSL_REQUIRE_CLIENT_CERT`
- `SSL_CA_FILE`
- `SSL_CERT_FILE`
- `SSL_KEY_FILE`

Security hardening:

- `SECURITY_BLOCKED_USER_AGENTS` (CSV)

## Run

```bash
python main.py
```

With default settings, service binds to port `8001`.

## Related docs

- `PebSI.md` — detailed implementation-aligned technical documentation
- `documentation/routes.md` — API and processing flow details
