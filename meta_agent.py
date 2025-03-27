from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
from langsmith import traceable, Client
from reportlab.lib.pagesizes import elevenSeventeen

from document_agent import DocumentAgentSop
from memory import load_memory, update_memory
from retriever import setup_work_retriever, setup_sop_retriever
from query_engine import QueryEngine
from document_agent import DocumentAgentWork

client = Client()

class QAState(TypedDict):
    query: str
    query_type: str
    work_result: dict
    sop_result: dict
    combined_answer: str
    source_documents: List
    evaluation_feedback: str
    confidence: int
    memory: dict


def retrieve_work_instructions(state: QAState) -> QAState:
    try:
        # Initialize Document Agent with default fallback chain
        agent = DocumentAgentWork(debug_mode=True)
        retriever = setup_work_retriever(agent.work_chunks)
        query_engine = QueryEngine(debug_mode=True, confidence_threshold=80)

        # Check memory first
        memory = state["memory"]
        if state["query"] in memory:
            print("Returning cached response from memory.")
            cached_result = memory[state["query"]]
            state["work_result"] = cached_result
            state["combined_answer"] = cached_result["answer"]
            state["confidence"] = cached_result["confidence"]
            state["source_documents"] = cached_result["source_documents"]
            return state

        # Process query with fallback handling
        result = query_engine.query_documents_advanced(state["query"], retriever)

        # Update memory and state
        result_data = {
            "answer": result.answer,
            "confidence": result.confidence,
            "source_documents": result.source_documents,
            "refined": result.refined
        }
        update_memory(memory, state["query"], result_data)

        state["work_result"] = result_data
        state["combined_answer"] = result.answer
        state["confidence"] = result.confidence
        state["source_documents"] = result.source_documents

        return state
    except Exception as e:
        print(f"Error in work instructions retrieval: {str(e)}")
        state["work_result"] = {
            "result": "Unable to process query at this time.",
            "confidence": 0,
            "source_documents": [],
            "refined": False
        }
        return state



def retrieve_sop_documents(state: QAState) -> dict:
    """
    Retrieves SOP documents from the document retriever
    """
    try:
        agent = DocumentAgentSop(True)
        retriever = setup_sop_retriever(agent.sop_chunks)
        query_engine = QueryEngine(debug_mode=True, confidence_threshold=80)

        # Check memory first
        memory = state["memory"]
        if state["query"] in memory:
            print("Returning cached SOP response from memory.")
            cached_result = memory[state["query"]]
            state["sop_result"] = cached_result
            return state

        # Retrieve SOP documents
        retrieval_result = query_engine.query_documents_advanced(state["query"], retriever)

        # Store in Memory Cache
        result_data = {
            "answer": retrieval_result.answer,
            "confidence": retrieval_result.confidence,
            "source_documents": retrieval_result.source_documents,
            "refined": retrieval_result.refined
        }
        update_memory(memory, state["query"], result_data)

        state["sop_result"] = result_data

        return state
    except Exception as e:
        print(f"Error in SOP document retrieval: {str(e)}")
        state["sop_result"] = {
            "result": "Unable to retrieve SOP documents.",
            "confidence": 0,
            "source_documents": [],
            "refined": False
        }
        return state





def detect_query_type(query: str) -> str:
    """
    Detects whether a query is related to work instructions, SOPs, or both.

    :param query: The user input
    :return: str : either "work", "sop", or "both"
    """
    work_keywords = {
        "personal protective equipment", "steps", "step", "PPE", "manual", "HPLC",
        "solids", "Hypophosphorous", "safety", "handling", "process", "manufacturing", "container", "acid",
        "instructions", "ppe", "addition",
        "unloading", "sampling", "injection", "hplc", "tank", "cleaning", "loading", "valve", "pump",
        "quality check", "inspection", "equipment", "checklist", "shutdown", "emergency", "flow rate",
        "pressure", "filtration", "hypophosphorous", "manual load", "tanker", "sulfated analysis", "manual load",
        "portable hopper", "valve operation", "hplc analysis",
        "sample injection", "vacuum setup", "ppe requirements", "connect hose",
        "disposal procedure", "cleanup steps", "work instruction", "pdf steps",
        "immediate action", "equipment setup", "short procedure", "field operation"
    }

    sop_keywords = {
        "detail", "step", "steps", "SAP", "SAP code", "process",
        "raw material", "material", "reactor", "production", "setpoint", "concentration", "temperature control",
        "pH level",
        "neutralization", "chemical composition", "hazard classification", "toxicity", "regulatory",
        "batch process", "formulation", "purity", "stoichiometry", "yield", "storage", "msds", "wastewater",
        "alkylbenzen sulfonic acid", "an-84", "as-42", "deto pk-45", "texapon", "tomperlan", "manufacturing procedure",
        "hazard classification", "process parameters",
        "regulatory compliance", "material storage", "sap code", "raw material list",
        "reaction conditions", "environmental impact", "batch record", "quality control",
        "version control", "effective date", "document revision", "risk assessment",
        "change control", "validation protocol"
    }

    query_lower = query.lower()

    is_work = any(keyword in query_lower for keyword in work_keywords)
    is_sop = any(keyword in query_lower for keyword in sop_keywords)

    if is_work and is_sop:
        query_type = "both"
    elif is_work:
        query_type = "work"
    elif is_sop:
        query_type = "sop"
    else:
        query_type = "both"  # Default fallback



    # Print statement for debugging in local testing (remove in production)
    print(f"[DEBUG] Query Type Detected: {query_type.upper()} for query: \"{query}\"")

    return query_type


def combine_results(state: QAState) -> dict:
    """Merge results from work and SOP retrieval based on confidence levels."""

    work_confidence = state["work_result"].get("confidence", 0)
    sop_confidence = state["sop_result"].get("confidence", 0)

    if state["query_type"] == "both":
        if work_confidence > sop_confidence:
            combined_answer = f"Primary Source (Work Instructions - {work_confidence}% confidence):\n{state['work_result'].get('answer', 'No answer found')}\n\n"
            combined_answer += f"Additional SOP Context ({sop_confidence}% confidence):\n{state['sop_result'].get('answer', 'No answer found')}"
        else:
            combined_answer = f"Primary Source (SOP - {sop_confidence}% confidence):\n{state['sop_result'].get('answer', 'No answer found')}\n\n"
            combined_answer += f"Additional Work Instructions ({work_confidence}% confidence):\n{state['work_result'].get('answer', 'No answer found')}"

        source_docs = state["work_result"].get("source_documents", []) + state["sop_result"].get("source_documents", [])

    elif state["query_type"] == "work":
        combined_answer = state["work_result"].get("answer", "")
        source_docs = state["work_result"].get("source_documents", [])

    else:  # SOP query
        combined_answer = state["sop_result"].get("answer", "")
        source_docs = state["sop_result"].get("source_documents", [])

    state["combined_answer"] = combined_answer
    state["source_documents"] = source_docs
    return state


workflow = StateGraph(QAState)

workflow.add_node("retrieve_work_instructions", retrieve_work_instructions)
workflow.add_node("retrieve_sop_documents", retrieve_sop_documents)
workflow.add_node("combine_results", combine_results)

workflow.add_conditional_edges(
    START,
    lambda state: (
    ["retrieve_work_instructions", "retrieve_sop_documents"] if state["query_type"] == "both" else
    ["retrieve_work_instructions"] if state["query_type"] == "work" else
    ["retrieve_sop_documents"]
))

# Remove these lines:
# workflow.add_edge("retrieve_work_instructions", END)
# workflow.add_edge("retrieve_sop_documents", END)

# Add these instead:
workflow.add_edge("retrieve_work_instructions", "combine_results")
workflow.add_edge("retrieve_sop_documents", "combine_results")

meta_agent_chain = workflow.compile()

@traceable(project_name="workstations", client=client)
def run_meta_agent(query: str) -> QAState:
    memory = load_memory()

    query_type = detect_query_type(query)

    state = {"query": query,
             "query_type": query_type,
             "work_result": {},
             "sop_result": {},
             "combined_answer": "",
             "source_documents": [],
             "evaluation_feedback": "",
             "confidence": 0,
             "memory": memory
    }
    result_state = meta_agent_chain.invoke(state)
    return result_state
