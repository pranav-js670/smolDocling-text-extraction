import streamlit as st
import requests
import json

API_URL = "http://localhost:8000/ocr"

st.set_page_config(page_title="SmolDocling OCR Frontend", layout="wide")
st.title("SmolDocling OCR - User Review")

upload_option = st.radio("Choose file type to upload:", ["Image", "PDF"])
if upload_option == "Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
elif upload_option == "PDF":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

prompt_text = st.text_input("Enter prompt text:", "Convert this page to docling.")

if uploaded_file is not None:
    if upload_option == "Image":
        from PIL import Image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)

    if st.button("Extract Text"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        data = {"prompt_text": prompt_text}
        with st.spinner("Extracting text..."):
            response = requests.post(f"{API_URL}/extract", files=files, data=data)
        if response.status_code == 200:
            result = response.json()
            extraction_id = result["extraction_id"]
            extracted_text = result["extracted_text"]
            proc_time = result["processing_time"]

            st.success(f"Extraction completed in {proc_time:.2f} seconds!")
            st.subheader("Extracted Text (Review)")
            st.text_area("Review the extracted text:", extracted_text, height=300)
            st.session_state["extraction_id"] = extraction_id
        else:
            st.error(f"Error: {response.json()['detail']}")

if "extraction_id" in st.session_state:
    if st.button("Approve Extraction"):
        extraction_id = st.session_state["extraction_id"]
        payload = {"extraction_id": extraction_id, "approved": True}
        with st.spinner("Approving..."):
            approve_resp = requests.post(f"{API_URL}/approve", json=payload)
        if approve_resp.status_code == 200:
            final_text = approve_resp.json()["final_text"]
            st.success("Extraction approved!")
            st.subheader("Final Extracted Text")
            st.markdown(final_text)
        else:
            st.error("Approval failed.")
