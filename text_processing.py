import pickle
import hashlib
from pathlib import Path
from langchain_ollama import OllamaEmbeddings

from config import RESOURCES_PATH
from extract_metadata import filter_metadata
from config import RESOURCES_PATH
from document_loader import load_work_instructions

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)  # Ensure cache directory exists


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure you download stopwords and punkt once
nltk.download('stopwords')
nltk.download('punkt')

def remove_stop_words(text: str) -> str:
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)


import re
from langchain.schema import Document


def dynamic_text_splitter(documents, default_chunk_size=800):
    """
    Splits documents into semantically meaningful chunks while preserving structure.
    Uses section headers, paragraphs, and sentence boundaries to maintain context.
    """
    chunks = []
    for doc in documents:
        text = doc.page_content
        metadata = doc.metadata  # Preserve metadata for retrieval

        # Split at section headers (e.g., "Step X", "Procedure", "Safety Measures")
        sections = re.split(r"\n\s*[A-Z ]{5,}\s*\n", text)

        current_chunk = ""
        for section in sections:
            sentences = re.split(r'(?<=[.!?])\s+', section)  # Sentence boundary split

            for sentence in sentences:
                if len(current_chunk) + len(sentence) > default_chunk_size:
                    if current_chunk:
                        chunks.append(Document(page_content=current_chunk.strip(), metadata=metadata))
                        current_chunk = sentence
                    else:
                        # Force split long sentences
                        for i in range(0, len(sentence), default_chunk_size):
                            chunk_text = sentence[i:i + default_chunk_size]
                            chunks.append(Document(page_content=chunk_text.strip(), metadata=filter_metadata(metadata)))
                        current_chunk = ""
                else:
                    current_chunk += " " + sentence if current_chunk else sentence

            if current_chunk:
                chunks.append(Document(page_content=current_chunk.strip(), metadata=metadata))

    return chunks


def get_or_cache_embeddings(chunks, cache_prefix: str, model):
    """
    Stores or retrieves embeddings from cache to optimize retrieval.
    """
    cache_key = hashlib.md5("".join(sorted([str(doc.metadata) for doc in chunks])).encode()).hexdigest()
    embedding_cache_path = CACHE_DIR / f"{cache_prefix}_{cache_key}_embeddings.pkl"
    if embedding_cache_path.exists():
        with embedding_cache_path.open("rb") as f:
            embeddings = pickle.load(f)
        print(f"Loaded {cache_prefix} embeddings from cache.")
    else:
        print(f"Generating {cache_prefix} embeddings...")
        embeddings = model.embed_documents([chunk.page_content for chunk in chunks])
        with embedding_cache_path.open("wb") as f:
            pickle.dump(embeddings, f)
        print(f"{cache_prefix} embeddings cached.")
    return embeddings

# Ensure loading work instructions separately
work_instruction_docs = load_work_instructions(RESOURCES_PATH)
work_chunks = dynamic_text_splitter(work_instruction_docs)
embedding_model_instance = OllamaEmbeddings(model="paraphrase-multilingual")
get_or_cache_embeddings(work_chunks, "workinst", embedding_model_instance)
