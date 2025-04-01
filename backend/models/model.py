from typing import Optional
from pydantic import BaseModel

class OCRExtractResponse(BaseModel):
    extraction_id: str
    extracted_text: str
    processing_time: float

class OCRApproveRequest(BaseModel):
    extraction_id: str
    approved: bool

class OCRApproveResponse(BaseModel):
    final_text: str
