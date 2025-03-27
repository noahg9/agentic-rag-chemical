from document_loader import load_sop_documents
import os

directory= r"C:\Users\mnmhy\PycharmProjects\agenticrag\Resources"

if os.path.isdir(directory):
    print("[DEBUG] Directory exists, loading SOP documents...")
    sop_docs = load_sop_documents(directory)
    print(f"[SUCCESS] Loaded {len(sop_docs)} SOP documents")
else:
    print(f"[ERROR] Invalid directory: {directory}")