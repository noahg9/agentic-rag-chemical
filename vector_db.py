import os.path

from langchain_chroma import Chroma
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from langchain.schema import Document
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from config import WORK_COLLECTION, SOP_COLLECTION

from qdrant_client import models
from langchain.vectorstores.qdrant import Qdrant as QdrantVectorStore

from qdrant_client import models
from langchain.vectorstores.qdrant import Qdrant as QdrantVectorStore



def setup_qdrant(documents: list[Document]):
    # Let Qdrant handle collection creation automatically
    return QdrantVectorStore.from_documents(
        documents=documents,
        embedding=OllamaEmbeddings(model="paraphrase-multilingual"),
        url="http://localhost:6333",
        collection_name=WORK_COLLECTION,
        content_payload_key="page_content",
        metadata_payload_key="metadata"
    )

def setup_chroma(documents):
    """
    Sets up Chroma vector store for SOPs using the "mxbai-embed-large" embedding model.
    """
    chroma_persist_dir = "chroma_db_sop_new"

    # Ensure hte directory exists
    if not os.path.exists(chroma_persist_dir):
        os.makedirs(chroma_persist_dir)

    embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

    chroma_vector_store = Chroma(
        collection_name=SOP_COLLECTION,
        embedding_function=embedding_model,
        persist_directory=chroma_persist_dir
    )
    if documents:
        print(f"[DEBUG] Adding {len(documents)} SOP documents to ChromaDB.")
        chroma_vector_store.add_documents(documents)

    return chroma_vector_store


