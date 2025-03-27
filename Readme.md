# Agentic RAG System

## Overview

The Agentic RAG (Retrieval-Augmented Generation) System is designed to process user queries by routing them through a meta-agent, performing vector-based semantic search, and generating concise answers with confidence ratings. The system is divided into several stages, each responsible for a specific part of the query processing pipeline.

## Architecture

### 1. User Input → Meta-Agent Routing

The user's query is routed by the meta-agent based on keyword analysis to determine whether it pertains to Standard Operating Procedures (SOP) or Work Instructions (WI). The routing is now based on vector/semantic search with similarity.

###  2. Document Agent Processing

The Document Agent ingests, splits, and embeds all documents. It then retrieves and reranks relevant chunks using semantic search.

The Document Agent processes the documents by:
- Ingesting and splitting the documents into chunks.
- Embedding the chunks using a semantic embedding model.
- Retrieving and reranking relevant chunks based on the user's query.

### 3. Query Engine Multi-Stage QA

The retrieved chunks are processed through:
- Decomposition (if needed).
- Synthesis.
- Evaluation/refinement.
- Summarization to generate a concise final answer with a confidence rating.

### 4. Meta-Agent Aggregation

The meta-agent aggregates the outputs from both branches (SOP and WI) and selects the best final answer along with source documents, evaluation feedback, and refinement if needed.

### 5. Final Output Display

The main interface displays the final concise answer, source document details, and the confidence rating.

## Validation by Jupyter notebook
We validate the following steps for each work instruction and SOP workflows.

1. **Select one work instruction document (the easiest).**
2. **Validate that step 1 (Meta-agent routing) of the agent's pipeline works fine.** Check that the question is correctly routed.
   - If it does not work fine, modify the LLM/modify the code.
3. **Once validated for one work instruction, validate step 2 (Split and embed the documents, creation of vector database) of the agent's pipeline.**
4. **Continue validating each step until step 5.**
5. **When all steps work, generalize for more documents of the work instruction group.**
6. **When all works for work instructions, do the same for SOP documents.**

## File Descriptions

### `AgenticRag/document_loader.py`
- Loads PDF and DOCX work instructions with metadata and structured data.

### `AgenticRag/document_agent.py`
- Handles the loading and preparation of work documents.
- Splits documents into chunks and embeds them.
- Provides tools for local retrieval.

### `AgenticRag/extract_metadata.py`
- Extracts metadata from PDF and DOCX files.

### `AgenticRag/text_processing.py`
- Processes text by removing stop words and splitting documents into meaningful chunks.
- Caches embeddings for efficient retrieval.

### `AgenticRag/meta_agent.py`
- Manages the meta-agent workflow for routing queries and aggregating results.

### `AgenticRag/query_engine.py`
- Implements the multi-stage QA process, including query decomposition, document retrieval, synthesis, and summarization.

### `AgenticRag/llm.py`
- Provides functions to initialize and use different language models (Azure, Ollama, Qwen2.5).

### `AgenticRag/reranker.py`
- Reranks document snippets based on relevance using an LLM.

### `AgenticRag/response_generator.py`
- Replaces industry terms with their definitions and processes math expressions.

### `AgenticRag/retriever.py`
- Sets up retrievers for work instructions and SOPs using vector stores.

### `AgenticRag/utils/extract_docs.py`
- Extracts text from DOCX files and converts DOC files to DOCX.

### `AgenticRag/config.py`
- Contains configuration settings and paths.

### `AgenticRag/document_extraction.py`
- Extracts full text and table data from PDF and DOCX files.

### `AgenticRag/utils/utils.py`
- Provides utility functions for caching responses.

### `AgenticRag/memory.py`
- Manages memory for storing and retrieving cached responses.

### `AgenticRag/main.py`
- The main interface for the Agentic RAG System, built with Streamlit.

### `AgenticRag/vector_db.py`
- Sets up vector stores for work instructions and SOPs using Qdrant and Chroma.

### `.env`
- Contains environment variables for configuration.

## Docker Setup

### Docker Compose

Create a `docker-compose.yml` file with the following content to set up both Chroma and Qdrant vector databases:

```yaml
version: '3.8'

services:
  chromadb:
    image: chromadb/chroma
    container_name: chromadb
    ports:
      - "8000:8000"
    volumes:
      - ./chroma_data:/chroma_data

  qdrant:
    image: qdrant/qdrant
    container_name: qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_storage:/qdrant/storage:z
```

Run the following command to start the services:

```sh
docker-compose up -d
```

### Docker Run Commands

Alternatively, you can use the following Docker run commands to start the services individually:

#### Chroma

```sh
docker run -d -p 8000:8000 -v $(pwd)/chroma_data:/chroma_data --name chromadb chromadb/chroma
```

#### Qdrant

```sh
docker run -d -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" --name qdrant qdrant/qdrant
```
### How to Run main.py

```sh
streamlit run main.py
```

## Conclusion

This README provides a comprehensive overview of the Agentic RAG System, detailing the architecture, validation steps, file descriptions, and current status. The system is designed to efficiently process user queries using vector-based semantic search and generate concise answers with confidence ratings.
