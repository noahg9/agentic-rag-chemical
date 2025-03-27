import json
import pandas as pd
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.schema import Document
from llm import get_llm
from vector_db import setup_qdrant, setup_chroma
from langchain.schema import Document


def filter_complex_metadata(metadata: dict) -> dict:
    """
    Filters metadata so that only simple types (str, int, float, bool) remain.
    Any None values are replaced with an empty string.
    """
    cleaned_metadata = {}

    for key, value in metadata.items():
        if value is None:
            cleaned_metadata[key] = ""
        elif isinstance(value, (str, int, float, bool)):  # Allowed types
            cleaned_metadata[key] = value
        elif isinstance(value, (list, dict, pd.DataFrame)):  # Convert complex types
            cleaned_metadata[key] = json.dumps(value, default=str)  # Convert to JSON-safe string
        else:
            cleaned_metadata[key] = str(value)  # Convert unknown types to string

    return cleaned_metadata

def escape_curly_braces(text: str) -> str:
    """
    Escapes curly braces so they are not interpreted as prompt input variables.
    """
    return text.replace("{", "{{").replace("}", "}}")


def setup_work_retriever(documents):
    """
    Set up a retriever for work instructions documents.

    Args:
        documents: List of documents to index

    Returns:
        A RetrievalQA chain that returns source documents
    """
    # Use default fallback chain: Azure -> Ollama -> Qwen2.5
    model_chain = ["ollama", "qwen2.5"]
    llm_instance = None

    # Try models in sequence until one works
    for model in model_chain:
        try:
            llm_instance = get_llm(model)
            break
        except Exception as e:
            print(f"Failed to load {model}, trying next model. Error: {str(e)}")
            continue

    if not llm_instance:
        raise Exception("All models failed to load")


    # Setup vector store with proper configuration
    vector_store = setup_qdrant(documents)

    # Create a properly configured retriever
    # Only pass standard parameters to avoid the error
    retriever = vector_store.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance for diversity
        search_kwargs={
            "k": 4,  # Number of documents to retrieve
            "fetch_k": 20,  # Number of documents to consider for diversity
            "lambda_mult": 0.7  # Balance between relevance and diversity (0.0-1.0)
        }
    )

    return RetrievalQA.from_chain_type(
        llm=llm_instance,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )



def setup_sop_retriever(documents):
    """
    Sets up retrieval for SOPs using Chroma with model fallback chain.
    """
    # Use default fallback chain
    model_chain = ["ollama", "qwen2.5"]
    llm_instance = None

    for model in model_chain:
        try:
            llm_instance = get_llm(model)
            break
        except Exception as e:
            print(f"Failed to load {model}, trying next model. Error: {str(e)}")
            continue

    if not llm_instance:
        raise Exception("All models failed to load")

    filtered_documents = [
        Document(
            page_content=doc.page_content,
            metadata=filter_complex_metadata(doc.metadata)
        ) for doc in documents if hasattr(doc, "page_content") and hasattr(doc, "metadata")
    ]
    vector_store = setup_chroma(filtered_documents)
    retriever = vector_store.as_retriever()
    return RetrievalQA.from_chain_type(
        llm=llm_instance,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )



def select_qa_chain(query: str, documents) -> (RetrievalQA, str):
    """
    Routes queries to appropriate vector store based on keywords.
    Uses internal model fallback chain.
    """
    sop_keywords = [
        "sop", "standard operating procedure", "sap code", "raw material", "quality", "review", "document",
        "procedure", "synthesis", "reaction", "setpoint", "concentration", "temperature control", "pH level",
        "neutralization", "chemical composition", "hazard classification", "toxicity", "regulatory",
        "batch process", "formulation", "purity", "stoichiometry", "yield", "storage", "msds", "wastewater",
        "alkylbenzen sulfonic acid", "an-84", "as-42", "deto pk-45", "texapon", "tomperlan"
    ]
    work_keywords = [
        "safety", "handling", "process", "manufacturing", "container", "acid", "instructions", "ppe", "addition",
        "unloading", "sampling", "injection", "hplc", "tank", "cleaning", "loading", "valve", "pump",
        "quality check", "inspection", "equipment", "checklist", "shutdown", "emergency", "flow rate",
        "pressure", "filtration", "hypophosphorous", "manual load", "tanker", "sulfated analysis"
    ]

    work_score = sum(1 for kw in work_keywords if kw in query.lower())
    sop_score = sum(1 for kw in sop_keywords if kw in query.lower())

    if sop_score > work_score:
        print("Selected SOP vector store based on query keywords.")
        qa_chain = setup_sop_retriever(documents)
        return qa_chain, "SOP"
    else:
        print("Selected Work Instructions vector store based on query keywords.")
        qa_chain = setup_work_retriever(documents)
        return qa_chain, "Work Instructions"