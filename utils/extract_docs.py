import os
import subprocess
from docx import Document
from pathlib import Path

# ---------------------------
# Configurable Absolute Path to Resources Folder
# ---------------------------
RESOURCES_PATH = "agenticrag/Resources"

# ---------------------------
# Convert .doc to .docx using LibreOffice (if needed)
# ---------------------------
def convert_doc_to_docx(doc_path):
    """
    Converts a .doc file to .docx using LibreOffice in headless mode.
    """
    try:
        # Expected output file path
        output_path = os.path.splitext(doc_path)[0] + ".docx"

        # Run LibreOffice conversion
        result = subprocess.run(["soffice", "--headless", "--convert-to", "docx", doc_path, "--outdir", RESOURCES_PATH],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        # Check if the converted .docx file exists
        converted_file_path = os.path.join(RESOURCES_PATH, os.path.basename(output_path))
        if os.path.exists(converted_file_path):
            print(f"✔ Successfully converted: {doc_path} → {converted_file_path}")
            return converted_file_path
        else:
            print(f"❌ Conversion failed for: {doc_path}\nError: {result.stderr.decode()}")
            return None
    except Exception as e:
        print(f"❌ Error converting {doc_path} to .docx: {e}")
        return None

# ---------------------------
# Extract text from all DOCX files in Resources folder
# ---------------------------
def extract_text_from_docs():
    if not os.path.exists(RESOURCES_PATH):
        raise FileNotFoundError(f"❌ Folder 'Resources' not found at {RESOURCES_PATH}")

    doc_texts = []
    for filename in os.listdir(RESOURCES_PATH):
        file_path = os.path.join(RESOURCES_PATH, filename)

        # Convert .doc to .docx if necessary
        if filename.endswith(".doc"):
            new_docx_path = convert_doc_to_docx(file_path)
            if new_docx_path:
                file_path = new_docx_path  # Update path to newly converted .docx file
            else:
                print(f"Skipping {filename}, conversion failed.")
                continue

        # Extract text from .docx
        if file_path.endswith(".docx"):
            try:
                doc = Document(file_path)
                text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
                doc_texts.append({"filename": filename, "text": text})
            except Exception as e:
                print(f"❌ Error reading {filename}: {e}")

    return doc_texts

# ---------------------------
# Run and Debug Extraction
# ---------------------------
if __name__ == "__main__":
    extracted_docs = extract_text_from_docs()
    for doc in extracted_docs:
        print(f"\n📄 Extracted from {doc['filename']}:\n{doc['text'][:500]}...\n")
