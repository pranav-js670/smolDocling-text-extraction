from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from backend.models.model import OCRExtractResponse, OCRApproveRequest, OCRApproveResponse
from backend.services.ocr_service import extract_text, store_extraction, get_extraction

router = APIRouter(prefix="/ocr", tags=["OCR"])

@router.post("/extract", response_model=OCRExtractResponse)
async def extract_endpoint(
    file: UploadFile = File(...),
    prompt_text: str = Form("Convert this page to docling.")
):
    file_ext = file.filename.split(".")[-1].lower()
    try:
        file_bytes = await file.read()
        doctags, md_content, proc_time = extract_text(file_bytes, file_ext, prompt_text)
        extraction_id = store_extraction(md_content)
        return OCRExtractResponse(
            extraction_id=extraction_id,
            extracted_text=md_content,
            processing_time=proc_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/approve", response_model=OCRApproveResponse)
async def approve_endpoint(request: OCRApproveRequest):
    approved_text = get_extraction(request.extraction_id)
    if not approved_text:
        raise HTTPException(status_code=404, detail="Extraction ID not found")
    return OCRApproveResponse(final_text=approved_text)