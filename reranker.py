from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


def rerank_documents(query: str, documents: list, llm_instance, top_n=5):
    """
    Rerank documents based on relevance to query.

    Args:
        query: The user query
        documents: List of document dicts with page_content and metadata
        llm_instance: Language model for scoring
        top_n: Number of top documents to consider

    Returns:
        List of reranked document dicts with scores in metadata
    """
    if not documents:
        return []

    # Limit to top_n documents to avoid token limits
    doc_subset = documents[:top_n] if len(documents) > top_n else documents

    prompt = PromptTemplate(
        template="""
You are a document reranker. Given the user query and the following document snippets,
assign a relevance score between 0 (not relevant) and 10 (highly relevant).
Return your answer as a comma-separated list of scores.

User Query: {query}
Document Snippets:
{snippets}
Scores (comma-separated):
""",
        input_variables=["query", "snippets"]
    )

    # Create a list of documents with their indices
    snippet_texts = []
    for i, doc in enumerate(doc_subset):
        snippet = doc["page_content"][:300]
        snippet_texts.append(f"Document {i + 1}:\n{snippet}")

    snippets = "\n---\n".join(snippet_texts)
    chain = LLMChain(llm=llm_instance, prompt=prompt)

    try:
        scores_str = chain.run(query=query, snippets=snippets).strip()
        scores = [float(s.strip()) for s in scores_str.split(",")]

        # If we don't get enough scores, pad with defaults
        if len(scores) < len(doc_subset):
            scores.extend([5.0] * (len(doc_subset) - len(scores)))
        # If we get too many scores, truncate
        elif len(scores) > len(doc_subset):
            scores = scores[:len(doc_subset)]
    except Exception as e:
        print(f"Error during reranking: {str(e)}, defaulting to 5.0 scores")
        scores = [5.0] * len(doc_subset)

    # Create reranked documents with scores in metadata
    reranked_docs = []
    for doc, score in zip(doc_subset, scores):
        # Create a new dict to avoid modifying the original
        new_doc = {
            "page_content": doc["page_content"],
            "metadata": doc["metadata"].copy() if isinstance(doc["metadata"], dict) else {}
        }

        # Add score to metadata
        new_doc["metadata"]["score"] = score
        reranked_docs.append(new_doc)

    # Sort by score
    reranked_docs.sort(key=lambda x: x["metadata"].get("score", 0), reverse=True)

    return reranked_docs