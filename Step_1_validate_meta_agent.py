from meta_agent import run_meta_agent
import os
from dotenv import load_dotenv
from langsmith import Client, traceable
import re

# Load environment variables from .env
load_dotenv()

# LangSmith Configuration
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "workstations")
client = Client()

print("🔹 Choose LLM Model:")
print("1: Azure")
print("2: Ollama")
print("3: Qwen2.5")
model_choice = input("Enter 1, 2, or 3: ").strip()

if model_choice == "1":
    selected_model = "azure"
elif model_choice == "2":
    selected_model = "ollama"
elif model_choice == "3":
    selected_model = "qwen2.5"
else:
    print("❌ Invalid choice! Defaulting to 'ollama'.")
    selected_model = "ollama"

print(f"\n✅ Using {selected_model} as the LLM model.")

# Define test queries to validate routing
test_queries = [
    "How do I handle hypophosphorous acid?",
    "What safety measures are required for unloading a tanker?",
    "What are the steps for HPLC Sulfated Analysis?",
    "How do I properly clean the reactor after use?",
    "What is the required PPE for handling sulfuric acid?"
]

# # --- Functions to Save Results to DOCX and PDF ---
# from docx import Document as DocxDocument
# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen import canvas
#
# def save_results_to_docx(results: dict, filename: str = "meta_agent_results.docx"):
#     try:
#         doc = DocxDocument()
#         doc.add_heading("Meta-Agent Routing Validation Results", 0)
#         for query, details in results.items():
#             doc.add_heading(f"Query: {query}", level=1)
#             doc.add_paragraph(f"Combined Answer: {details.get('combined_answer', '')}")
#             doc.add_paragraph(f"Confidence: {details.get('confidence', 0)}%")
#             doc.add_paragraph(f"Evaluation Feedback: {details.get('evaluation_feedback', '')}")
#             doc.add_paragraph("Work Result:")
#             doc.add_paragraph(str(details.get("work_result", {})))
#             doc.add_paragraph("Retrieved Documents:")
#             for doc_name in details.get("source_documents", []):
#                 doc.add_paragraph(f"- {doc_name}")
#             doc.add_page_break()
#         doc.save(filename)
#         print(f"Results saved to DOCX: {filename}")
#         return True
#     except Exception as e:
#         print("Failed to save DOCX:", e)
#         return False
#
# def save_results_to_pdf(results: dict, filename: str = "meta_agent_results.pdf"):
#     try:
#         c = canvas.Canvas(filename, pagesize=letter)
#         width, height = letter
#         y = height - 50
#         c.setFont("Helvetica-Bold", 16)
#         c.drawString(50, y, "Meta-Agent Routing Validation Results")
#         y -= 30
#         c.setFont("Helvetica", 12)
#         for query, details in results.items():
#             if y < 100:
#                 c.showPage()
#                 y = height - 50
#             c.drawString(50, y, f"Query: {query}")
#             y -= 20
#             c.drawString(70, y, f"Combined Answer: {details.get('combined_answer', '')}")
#             y -= 20
#             c.drawString(70, y, f"Confidence: {details.get('confidence', 0)}%")
#             y -= 20
#             c.drawString(70, y, f"Evaluation Feedback: {details.get('evaluation_feedback', '')}")
#             y -= 20
#             c.drawString(70, y, "Work Result:")
#             y -= 20
#             c.drawString(90, y, str(details.get("work_result", {})))
#             y -= 20
#             c.drawString(70, y, "Retrieved Documents:")
#             y -= 20
#             for doc_name in details.get("source_documents", []):
#                 c.drawString(90, y, f"- {doc_name}")
#                 y -= 20
#                 if y < 100:
#                     c.showPage()
#                     y = height - 50
#             y -= 20
#             c.line(50, y, width - 50, y)
#             y -= 30
#         c.save()
#         print(f"Results saved to PDF: {filename}")
#         return True
#     except Exception as e:
#         print("Failed to save PDF:", e)
#         return False
#
# def format_results_as_string(results: dict) -> str:
#     lines = []
#     for query, details in results.items():
#         lines.append(f"Query: {query}")
#         lines.append(f"Combined Answer: {details['combined_answer']}")
#         lines.append(f"Confidence: {details['confidence']}%")
#         lines.append(f"Evaluation Feedback: {details['evaluation_feedback']}")
#         lines.append(f"Work Result: {details['work_result']}")
#         lines.append("Retrieved Documents:")
#         for doc_name in details["source_documents"]:
#             lines.append(f"  - {doc_name}")
#         lines.append("-" * 60)
#     return "\n".join(lines)
#
# # --- End Saving Functions ---
#
# # Running validation for Meta-Agent routing
# routing_results = {}
#
# for query in test_queries:
#     result = run_meta_agent(query, model_choice=selected_model)  # Run with selected model
#     routing_results[query] = {
#         "combined_answer": result.get("combined_answer", ""),
#         "work_result": result.get("work_result", {}),
#         "source_documents": [
#             doc.metadata.get("Document Name", "Unknown") for doc in result.get("source_documents", [])
#         ],
#         "confidence": result.get("confidence", 0),
#         "evaluation_feedback": result.get("evaluation_feedback", "No feedback")
#     }
#
# # Pretty-print results to the terminal (plain text)
# print("\n✅ Meta-Agent Routing Validation Results:\n")
# for query, details in routing_results.items():
#     print(f"🔍 Query: {query}")
#     print(f"   ➤ Combined Answer: {details['combined_answer']}")
#     print(f"   ➤ Confidence: {details['confidence']}%")
#     print(f"   ➤ Evaluation Feedback: {details['evaluation_feedback']}")
#     print(f"   ➤ Work Result: {details['work_result']}")
#     print("   ➤ Retrieved Documents:")
#     for doc_name in details["source_documents"]:
#         print(f"     - {doc_name}")
#     print("-" * 60)
#
# # Attempt to save results to DOCX and PDF.
# saved_docx = save_results_to_docx(routing_results, filename="meta_agent_results.docx")
# saved_pdf = save_results_to_pdf(routing_results, filename="meta_agent_results.pdf")
#
# if not (saved_docx or saved_pdf):
#     print("\n❌ Could not save to DOCX or PDF. Printing results as plain text:\n")
#     print(format_results_as_string(routing_results))


# Running validation for SOP Meta-Agent routing
routing_results_sop = {}

# List of test queries related to SOP validation
test_queries_sop = [
    "What are the steps for HPLC Sulfated Analysis?",  # SOP
    "How do I properly clean the reactor after use?",  # SOP
    "What safety measures are required for unloading a tanker?",  # SOP
]

print("\n🔍 **Meta-Agent SOP Routing Validation**")

for query in test_queries_sop:
    result = run_meta_agent(query, query_type="sop", model_choice=selected_model)  # Ensure query_type is "sop"

    routing_results_sop[query] = {
        "combined_answer": result.get("combined_answer", ""),
        "sop_result": result.get("sop_result", {}),  # Changed from work_result to sop_result
        "source_documents": [
            doc.metadata.get("Document Name", "Unknown") for doc in result.get("source_documents", [])
        ],
        "confidence": result.get("confidence", 0),
        "evaluation_feedback": result.get("evaluation_feedback", "No feedback")
    }

# Pretty-print results to the terminal (plain text)
print("\n✅ Meta-Agent SOP Routing Validation Results:\n")
for query, details in routing_results_sop.items():
    print(f"🔍 Query: {query}")
    print(f"   ➤ Combined Answer: {details['combined_answer']}")
    print(f"   ➤ Confidence: {details['confidence']}%")
    print(f"   ➤ Evaluation Feedback: {details['evaluation_feedback']}")
    print(f"   ➤ SOP Result: {details['sop_result']}")  # Changed from Work Result
    print("   ➤ Retrieved Documents:")
    for doc_name in details["source_documents"]:
        print(f"     - {doc_name}")
    print("-" * 60)
