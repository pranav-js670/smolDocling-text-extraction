import os
import time
import torch
import tempfile
import uuid
import fitz 
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from huggingface_hub import login
from backend.config import settings

try:
    from docling_core.types.doc import DoclingDocument
    from docling_core.types.doc.document import DocTagsDocument
    docling_available = True
except ImportError:
    docling_available = False

# Global in-memory store for extracted text (for demo purposes)
extracted_text_store = {}

# Load model and processor once (could also use FastAPI startup event)
device = "cuda" if torch.cuda.is_available() else "cpu"

if settings.HF_TOKEN:
    login(token=settings.HF_TOKEN)

processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")
model = AutoModelForVision2Seq.from_pretrained(
    "ds4sd/SmolDocling-256M-preview",
    torch_dtype=torch.float32
).to(device)

def process_single_image(image, prompt_text="Convert this page to docling."):
    start_time = time.time()
    # Create input messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text}
            ]
        },
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = inputs.to(device)
    # Generate outputs
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    prompt_length = inputs.input_ids.shape[1]
    trimmed_generated_ids = generated_ids[:, prompt_length:]
    doctags = processor.batch_decode(trimmed_generated_ids, skip_special_tokens=False)[0].lstrip()
    doctags = doctags.replace("<end_of_utterance>", "").strip()

    if docling_available:
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
        doc = DoclingDocument(name="Document")
        doc.load_from_doctags(doctags_doc)
        md_content = doc.export_to_markdown()
    else:
        md_content = doctags  # fallback if docling-core is not available

    processing_time = time.time() - start_time
    return doctags, md_content, processing_time

def process_pdf(pdf_file, prompt_text="Convert this PDF to docling."):
    # Save uploaded PDF to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(pdf_file)
    temp_file.close()
    pdf_path = temp_file.name
    doc = fitz.open(pdf_path)

    all_doctags = []
    all_md_content = []
    total_processing_time = 0

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doctags, md_content, processing_time = process_single_image(image, prompt_text)
        all_doctags.append(doctags)
        all_md_content.append(md_content)
        total_processing_time += processing_time

    combined_doctags = "\n\n".join(all_doctags)
    combined_md_content = "\n\n".join(all_md_content)
    return combined_doctags, combined_md_content, total_processing_time

def extract_text(file_bytes: bytes, file_type: str, prompt_text: str):
    """
    Processes the file (image or pdf) and returns extracted text and processing time.
    """
    if file_type in ["jpg", "jpeg", "png"]:
        from io import BytesIO
        image = Image.open(BytesIO(file_bytes)).convert("RGB")
        doctags, md_content, proc_time = process_single_image(image, prompt_text)
    elif file_type == "pdf":
        doctags, md_content, proc_time = process_pdf(file_bytes, prompt_text)
    else:
        raise ValueError("Unsupported file type")
    return doctags, md_content, proc_time

def store_extraction(extracted_text: str) -> str:
    """Stores the extracted text with a unique ID and returns the ID."""
    extraction_id = str(uuid.uuid4())
    extracted_text_store[extraction_id] = extracted_text
    return extraction_id

def get_extraction(extraction_id: str) -> str:
    return extracted_text_store.get(extraction_id, "")
