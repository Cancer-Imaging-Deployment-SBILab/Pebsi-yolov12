# Detection API Routes

## Base URL

Default local URL: `http://localhost:8001`

When `MTLS_ENABLED=true`, clients are expected to use HTTPS and present valid certs per deployment policy.

## 1) Health

- **Method**: `GET`
- **Path**: `/`
- **Response**:

```json
{
  "message": "Detection Model API is running"
}
```

## 2) Detect points

- **Method**: `POST`
- **Path**: `/detect_points`
- **Body schema**: `DetectBoxesRequest`

Fields:

- `patient_id: string`
- `test_id: string`
- `filename: string`
- `annotationId: string`
- `all_images: string[]`
- `source_url: string`
- `model_id?: string`
- `filter_data: object`
- `generate_rbc_crops: boolean` (default `true`)
- `generate_wbc_crops: boolean` (default `true`)
- `generate_platelets_crops: boolean` (default `true`)

### Processing steps

1. Resolve detection model (`model_id` or active default in `ml_models` category `detection_models`).
2. Validate model state (`is_active`) and file checksum (`checksum_sha256`) when present.
3. Copy all input image paths into request temp folder under `{BASE_DIR}/temp`.
4. Run inference pipeline (`run_pipeline`).
5. Apply scaling and NMS:
   - `scale_annotations`
   - `apply_nms_on_boundaries`
   - `apply_global_nms`
6. Persist annotation boxes (`update_annotations_db`).
7. Optionally create crops (`crop_and_save`) based on RBC/WBC/platelet flags.
8. Clean temp folder and GPU cache.

### Success response

```json
{
  "message": "Boxes detected and saved Successfully."
}
```

### Common error responses

- `404`: model not found on disk or in DB, sample/annotation not found downstream
- `400`: inactive model or general detect pipeline failure
- `409`: checksum mismatch

## Middleware behavior

### Security middleware

- Exempts `/docs`, `/redoc`, `/openapi.json`
- Enforces HTTPS transport when `MTLS_ENABLED=true`
- Rejects blocked user-agents from `SECURITY_BLOCKED_USER_AGENTS`

### Logging middleware

Each request writes:

- request line (`method`, `path`, client IP)
- response status and latency
- error logs with elapsed time when exceptions occur
