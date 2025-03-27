import pickle
import hashlib
from importlib.metadata import metadata
from pathlib import Path
from typing import List, Any, Dict
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.tools import BaseTool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_ollama import OllamaEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from pydantic import PrivateAttr
from tenacity import retry, wait_exponential, stop_after_attempt
from sympy.abc import lamda
from tenacity import wait_exponential, retry, stop_after_attempt

from reranker import rerank_documents
from response_generator import replace_definitions
from document_loader import load_work_instructions
from text_processing import dynamic_text_splitter, get_or_cache_embeddings
from retriever import setup_work_retriever, setup_sop_retriever
from query_engine import QueryEngine
from llm import get_llm
from config import CACHE_DIR, RESOURCES_PATH
from utils.utils import cache_response, retrieve_cached_response
from vector_db import setup_chroma
from langchain.schema import Document
import os
from typing import List, Tuple
from langchain.schema import Document
from document_loader import load_work_instructions, load_sop_documents  # Assuming you have loaders


CACHE_DIR = Path(CACHE_DIR)
CACHE_DIR.mkdir(exist_ok=True)  # Ensure cache directory exists

def load_and_prepare_work_documents() -> List[Document]:
    return load_work_instructions(RESOURCES_PATH)


def load_and_prepare_sop_documents(path: str) -> List[Document]:
    if os.path.isdir(path):
        return load_sop_documents(path)  # Load all SOP docs from directory
    elif os.path.isfile(path) and path.endswith(".docx"):
        return load_sop_documents(path)  # Load a single DOCX file


def load_and_prepare_documents(directory_path) -> Tuple[List[Document], List[Document]]:
    """
    Load only Work Instruction documents.
    Load and prepare documents separately:
    - Work Instructions (PDFs)
    - SOP Documents (DOCX)

    Returns:
        Tuple[List[Document], List[Document]] -> (work_docs, sop_docs)
    """
    return load_work_instructions(RESOURCES_PATH)

    work_docs = []
    sop_docs = []

    if not os.path.isdir(directory_path):
        raise ValueError(f"[ERROR] The given path is not a directory: {directory_path}")

    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)

        if file.endswith(".pdf"):
            work_docs.extend(load_work_instructions(file_path))
        elif file.endswith(".docx"):
            sop_docs.extend(load_sop_documents(file_path))

    return work_docs, sop_docs

def split_documents(documents: List[Document], chunk_size: int = 800) -> List[Document]:
    return dynamic_text_splitter(documents, default_chunk_size=chunk_size)

class LocalRetrievalTool(BaseTool):
    name: str = "local_retrieval"
    description: str = (
        "Use this tool to answer queries using only your ingested local documents. "
        "It calls the advanced multi-stage query function and returns the final answer."
    )
    _retriever_func: callable = PrivateAttr()
    _doc_agent: "DocumentAgentWork" = PrivateAttr()

    def __init__(self, doc_agent: "DocumentAgentWork"):
        super().__init__()
        self._doc_agent = doc_agent

    def _run(self, query: str) -> str:
        return self._retriever_func(query)
        return self._doc_agent.answer_query(query)

    def _arun(self, query: str) -> str:
        raise NotImplementedError("Async mode not implemented.")



class DocumentAgentWork:
    def __init__(self, debug_mode: bool = False, confidence_threshold: int = 80):
        self.debug_mode = debug_mode
        self.confidence_threshold = confidence_threshold
        # Initialize with primary model (Azure) and fallbacks
        self.model_chain = ["ollama", "qwen2.5"]
        self.current_model_index = 0
        self.llm_instance = self._get_llm_with_fallback()
        work_docs = load_and_prepare_work_documents()
        self.work_chunks = split_documents(work_docs, chunk_size=800)
        self.work_embeddings = get_or_cache_embeddings(
            self.work_chunks,
            "workinst",
            OllamaEmbeddings(model="paraphrase-multilingual")
        )
        self.memory = MemorySaver()
        self.tools = [LocalRetrievalTool(self)]
        self.agent_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an autonomous document agent. Use the 'local_retrieval' tool to answer queries using only your ingested local documents. Do not incorporate any external information."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        self.agent = create_tool_calling_agent(self.llm_instance, self.tools, self.agent_prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
        self.query_engine = QueryEngine(debug_mode=True, confidence_threshold=self.confidence_threshold)

    def _get_llm_with_fallback(self):
        while self.current_model_index < len(self.model_chain):
            try:
                model = self.model_chain[self.current_model_index]
                return get_llm(model)
            except Exception as e:
                print(f"Failed to load {model}, trying next model. Error: {str(e)}")
                self.current_model_index += 1
        raise Exception("All models failed to load")

    def _try_next_model(self):
        self.current_model_index = (self.current_model_index + 1) % len(self.model_chain)
        self.llm_instance = self._get_llm_with_fallback()

    @retry(wait=wait_exponential(multiplier=1, min=2, max=60), stop=stop_after_attempt(5))
    def answer_query(self, query: str) -> str:
        cached = retrieve_cached_response(query)
        if cached:
            print("Returning cached response.")
            return cached["result"]

        for attempt in range(len(self.model_chain)):
            try:
                retriever = setup_work_retriever(self.work_chunks)
                result = self.query_engine.query_documents_advanced(query, retriever)

                if result.source_documents:
                    # Debug print to check metadata before reranking
                    if self.debug_mode:
                        print("\n=== Metadata Before Reranking ===")
                        for i, doc in enumerate(result.source_documents[:3]):
                            print(
                                f"Document {i + 1} Metadata: {doc if isinstance(doc, dict) else 'Document object'}")

                    # Convert source_documents to the format expected by rerank_documents
                    doc_dicts = []
                    for doc in result.source_documents:
                        if isinstance(doc, dict):
                            # Already a dictionary
                            doc_dict = {
                                "page_content": doc.get("page_content", ""),
                                "metadata": doc
                            }
                        else:
                            # Document object
                            doc_dict = {
                                "page_content": doc.page_content if hasattr(doc, "page_content") else "",
                                "metadata": doc.metadata if hasattr(doc, "metadata") and doc.metadata else {}
                            }
                        doc_dicts.append(doc_dict)

                    # Rerank documents
                    reranked_docs = rerank_documents(query, doc_dicts, self.llm_instance)

                    # Debug print after reranking
                    if self.debug_mode:
                        print("\n=== After Reranking ===")
                        for i, doc in enumerate(reranked_docs[:3]):
                            print(f"Reranked Document {i + 1} Metadata: {doc.get('metadata', {})}")

                    # Convert back to Document objects if needed
                    from langchain.schema import Document
                    result.source_documents = [
                        Document(page_content=doc["page_content"], metadata=doc["metadata"])
                        for doc in reranked_docs
                    ]

                cache_response(query, {
                    "result": result.answer,
                    "confidence": result.confidence,
                    "source_documents": result.source_documents,
                    "refined": result.refined
                })
                return result.answer
            except Exception as e:
                print(f"Error with model {self.model_chain[self.current_model_index]}: {str(e)}")
                self._try_next_model()

        return "I couldn't process your query due to technical issues. Please try again later."

    def agent_answer_query(self, query: str, thread_id: str = "default_thread") -> str:
        config = {"configurable": {"thread_id": thread_id}}
        result = self.agent_executor.invoke({"input": query}, config=config)
        return result["output"]


class DocumentAgentSop:
    def __init__(self, debug_mode: bool = False, confidence_threshold: int = 80):
        self.debug_mode = debug_mode
        self.confidence_threshold= confidence_threshold
        self.model_chain = ["ollama", "qwen2.5"]
        self.current_model_index = 0
        self.llm_instance = self._get_llm_with_fallback()
        self.sop_docs = load_and_prepare_sop_documents(RESOURCES_PATH)
        self.sop_chunks = split_documents(self.sop_docs, chunk_size=800)
        self.sop_vector_store = setup_chroma(self.sop_chunks)
        self.sop_embeddings = get_or_cache_embeddings(self.sop_chunks, "sop", OllamaEmbeddings(model="mxbai-embed-large"))
        self.memory = MemorySaver()
        self.tools = [LocalRetrievalTool(lambda query: self.answer_query(query))]
        self.agent_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an autonomous document agent. Use the 'local_retrieval' tool to answer queries using only your ingested local documents. Do not incorporate any external information."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        self.agent = create_tool_calling_agent(self.llm_instance, self.tools, self.agent_prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
        self.query_engine = QueryEngine(debug_mode=True, confidence_threshold=80)

    def _get_llm_with_fallback(self):
        while self.current_model_index < len(self.model_chain):
            try:
                model = self.model_chain[self.current_model_index]
                return get_llm(model)
            except Exception as e:
                print(f"Failed to load {model}, trying next model. Error: {str(e)}")
                self.current_model_index += 1
        raise Exception("All models failed to load")

    def _try_next_model(self):
        self.current_model_index = (self.current_model_index + 1) % len(self.model_chain)
        self.llm_instance = self._get_llm_with_fallback()


    def answer_query(self, query: str) -> str:
        # Retrieve the most relevant SOP documents and generate an answer
        cached = retrieve_cached_response(query)
        if cached:
            print(f"[DEBUG] Retrieving cached response")
            return cached["result"]

        for attempt in range(len(self.model_chain)):
            try:
                retriever = setup_sop_retriever(self.sop_chunks)
                result = self.query_engine.query_documents_advanced(query, retriever)

                if result.source_documents:
                    # Debug print to check metadat abefore reranking
                    if self.debug_mode:
                        print("\n=== Metadata Before Reranking ===")
                        for i, doc in enumerate(result.source_documents[:3]):
                            print(f"Document {i+1} Metadata: {doc if isinstance(doc, dict) else 'Document object'}")

                    doc_dicts = []
                    for doc in result.source_documents:
                        if isinstance(doc, dict):
                            # Already a dictionary
                            doc_dict = {
                                "page_content": doc.get("page_content", ""),
                                "metadata": doc
                            }
                        else:
                            doc_dict = {
                                "page_content": doc.page_content if hasattr(doc, "page_content") else "",
                                "metadata": doc.metadata if hasattr(doc, "metadata") and doc.metadata else {}
                            }
                        doc_dicts.append(doc_dict)

                    # Rerank documents
                    reranked_docs = rerank_documents(query, doc_dicts, self.llm_instance)

                    # Debug print after reranking
                    if self.debug_mode:
                        print("\n=== After Reranking ===")
                        for i, doc in enumerate(reranked_docs[:3]):
                            print(f"Reranked Document {i + 1} Metadata: {doc.get('metadata', {})}")

                    # Convert back to Document objects if needed
                    result.source_documents = [
                        Document(page_content=doc["page_content"], metadata=doc["metadata"])
                        for doc in reranked_docs
                    ]

                cache_response(query, {
                    "result": result.answer,
                    "confidence": result.confidence,
                    "source_documents": result.source_documents,
                    "refined": result.refined
                    })
                return result.answer

            except Exception as e:
                print(f"Error with model {self.model_chain[self.current_model_index]}: {str(e)}")
                self._try_next_model()

        return "I couldn't process your query due to technical issues. Please try again later."

        # if result.source_documents:
        #     top_docs = sorted(result.source_documents, key=lambda doc: doc.metadata.get("score", 0), reverse=True)[:10]
        #     result.source_documents = top_docs
        #
        # # Store result for caching
        # cache_response(query, {
        #     "result": result.answer,
        #     "confidence": result.confidence,
        #     "source_documents": result.source_documents,
        #     "refined": result.refined
        # })
        # return result.answer

    def agent_answer_query(self, query: str, thread_id: str = "default_thread") -> str:
        # uses the AI agent to process the query with SOP documents
        config = {"configurable": {"thread_id": thread_id}}
        result = self.agent_executor.invoke({"input": query}, config=config)
        return result["output"]


def select_question() -> str:
    default_questions = [
        "Which personal protective equipment do I need to use during Hypophosphorous Acid Addition?",
        "Describe in detail the steps that I need to do for the Sulfated Analysis by HPLC.",
        "Provide me the raw materials to be used for the synthesis of Alkylbenzen Sulfonic Acid and their SAP Code.",
        "When synthesizing Alkylbenzen sulfonic acid, which should be the setpoint for the sulfur trioxide when doing the sulfonation?",
        "Which range of humidity values are acceptable for the Alkylbenzen Sulfonic Acid?",
        "Describe the hazard classification of the AN-84 product.",
        "Describe the amidation reaction for the production of AN-84.",
        "Describe, in detail, the operational method for the production of Texapon S80.",
        "Which of the following products require water as raw material: AS-42, DETON PK-45, Texapon S80, TOMPERLAN OCD, Alkylbenzen Sulfonic Acid and AN-84?",
        "Can I produce AS-42 at the R003 reactor?",
        "Is there wastewater generation for AS-42 production?",
        "Which are the raw materials used for the production of AS-42? How much of each raw material should be used for the production of 1 Ton of AS-42?",
        "How much of each raw material should be used for the production of 3 Tons of AS-42?"
    ]
    print("\nPlease select a question:")
    print("0: Enter your own custom question")
    for i, question in enumerate(default_questions, start=1):
        print(f"{i}: {question}")
    try:
        choice = int(input("Enter the number of your choice (0 for custom): "))
    except ValueError:
        print("Invalid input. Please enter a number.")
        return select_question()
    return default_questions[choice - 1] if choice != 0 else input("Enter your custom question: ")