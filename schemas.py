import json
from typing import List, Optional, Union
from fastapi import Form, HTTPException
from pydantic import UUID4, BaseModel
from datetime import datetime, date
from uuid import UUID
from enum import Enum


# Schema for DetectBoxes request
class DetectBoxesRequest(BaseModel):
    patient_id: str
    test_id: str
    filename: str
    annotationId: str
    all_images: List[str]
    source_url: str
    model_path: str
    filter_data: dict

    class Config:
        from_attributes = True
