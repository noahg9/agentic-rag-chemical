import re
import concurrent.futures
import logging
import hashlib
import nltk
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from nltk.tokenize import sent_tokenize
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langsmith import traceable, Client
from response_generator import process_math_expressions
from azure.core.exceptions import HttpResponseError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from llm import get_llm
from langchain_core.output_parsers.string import StrOutputParser
from concurrent.futures import TimeoutError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    answer: str
    confidence: int
    source_documents: List[Dict[str, Any]]
    image_text: str = ""  # Optional field for image text
    refined: bool = False
    subqueries_used: List[str] = None


def extract_confidence(text: str) -> int:
    match = re.search(r"Confidence:\s*(\d+)%", text)
    return int(match.group(1)) if match else 80


def clean_final_summary(summary: str) -> str:
    lines = summary.splitlines()
    cleaned_lines = [re.sub(r"Sub-?query:.*", "", line).strip() for line in lines]
    return "\n".join([line for line in cleaned_lines if line]).strip()


class QueryEngine:
    def __init__(self, debug_mode: bool = False, confidence_threshold: int = 80):
        self.debug_mode = debug_mode
        self.client = Client()
        self.confidence_threshold = confidence_threshold
        nltk.download('punkt', quiet=True)
        self.MAX_SUBQUERIES = 3  # Increased from 2 to allow more diverse queries
        self.MAX_SENTENCES_PER_SUMMARY = 3
        self.MODEL_TIMEOUT = 30  # seconds
        # Model chain with priorities for different tasks
        self.model_priorities = {
            "decomposition": ["ollama", "qwen2.5"],  # Fast -> Slower but reliable
            "evaluation": ["qwen2.5", "ollama"],  # Accurate -> Less accurate
            "summarization": ["ollama", "qwen2.5"],  # Quick -> Thorough
            "synthesis": ["qwen2.5", "ollama"],  # Complex -> Simpler
            "refinement": ["qwen2.5", "ollama"]  # Quality -> Speed
        }
        # Default fallback chain
        self.default_model_chain = ["qwen2.5", "ollama"]
        self.previous_qa = {}  # Cache of previous Q&A pairs
        self.query_vectors = {}  # Store query embeddings for semantic comparison
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.similarity_threshold = 0.75  # Threshold for document similarity
        self.answer_similarity_threshold = 0.7  # Threshold for answer similarity

    def _log_debug(self, message: str, data: Any = None) -> None:
        if self.debug_mode:
            logging.debug(f"{message}: {data}")

    def _simple_summarize(self, text: str) -> str:
        sentences = sent_tokenize(text)
        cleaned_sentences = []
        for sentence in sentences:
            if not re.search(r'\b(Document Code|Version:|Date:)\b', sentence):
                cleaned_sentences.append(sentence.strip())
        return " ".join(cleaned_sentences[:self.MAX_SENTENCES_PER_SUMMARY])

    def _get_stage_llm(self, stage: str, attempt: int = 0) -> Any:
        # Get the appropriate model chain for the stage
        models = self.model_priorities.get(stage, self.default_model_chain)
        try:
            model = models[attempt % len(models)]
            return get_llm(model)
        except Exception as e:
            logger.warning(f"Failed to get LLM for {stage} with model {model}: {str(e)}")
            if attempt < len(models) - 1:
                return self._get_stage_llm(stage, attempt + 1)
            raise Exception(f"All models failed for stage {stage}")

    def _detect_stage_from_prompt(self, prompt_template: str) -> str:
        stage_keywords = {
            "Decompose": "decomposition",
            "Evaluate": "evaluation",
            "Summarize": "summarization",
            "Synthesize": "synthesis",
            "Refine": "refinement"
        }

        for keyword, stage in stage_keywords.items():
            if keyword.lower() in prompt_template.lower():
                return stage
        return "synthesis"

    @retry(
        wait=wait_exponential(multiplier=1, min=30, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((HttpResponseError, TimeoutError))
    )
    def _safe_invoke(self, chain: LLMChain, attempt: int = 0, **kwargs) -> str:
        try:
            # Handle different return types from LLMChain invoke
            result = chain.invoke(kwargs)

            # Check the type of result and extract text appropriately
            if hasattr(result, "text"):
                return result.text
            elif isinstance(result, dict) and "text" in result:
                return result["text"]
            elif isinstance(result, str):
                return result
            else:
                # Convert to string as fallback
                return str(result)
        except (HttpResponseError, TimeoutError) as e:
            logger.error(f"Error with LLM invoke: {str(e)}")
            if attempt < 2:
                stage = self._detect_stage_from_prompt(chain.prompt.template)
                fallback_llm = self._get_stage_llm(stage, attempt + 1)
                fallback_chain = LLMChain(llm=fallback_llm, prompt=chain.prompt)
                return self._safe_invoke(fallback_chain, attempt + 1, **kwargs)
            raise

    def _calculate_document_relevance_score(self, query: str, doc_content: str) -> float:
        """
        Calculate document relevance using both semantic and keyword-based scoring.
        Returns a score between 0 and 10.
        """
        # Prepare text
        query_clean = query.lower()
        doc_clean = doc_content.lower()

        # Simple keyword scoring
        query_words = set(re.findall(r'\b\w+\b', query_clean))
        doc_words = set(re.findall(r'\b\w+\b', doc_clean))

        # Count keyword matches (simple overlap)
        keyword_match_score = len(query_words.intersection(doc_words)) / max(1, len(query_words)) * 5

        # LLM-based relevance assessment
        try:
            relevance_llm = self._get_stage_llm("evaluation")
            relevance_prompt = ChatPromptTemplate.from_template("""
            Evaluate if this document is DIRECTLY relevant to answering the query.
            Query: {query}
            Document excerpt: {doc_excerpt}

            On a scale of 1-10, how relevant is this document to the specific query?
            Only return a number between 1-10.
            """)
            relevance_chain = relevance_prompt | relevance_llm | StrOutputParser()
            llm_score = float(relevance_chain.invoke({
                "query": query,
                "doc_excerpt": doc_content[:500]  # Use first 500 chars for efficiency
            }))
        except Exception as e:
            logger.warning(f"Error calculating LLM relevance score: {str(e)}")
            llm_score = 5.0  # Default mid-range score

        # Combine scores (weighted average)
        final_score = (keyword_match_score * 0.4) + (llm_score * 0.6)
        return final_score

    def _filter_relevant_docs(self, query: str, docs: List[Any]) -> List[Any]:
        """Filter documents based on relevance to the specific query."""
        relevant_docs = []
        scores = []

        for doc in docs:
            relevance_score = self._calculate_document_relevance_score(query, doc.page_content)
            if relevance_score >= 6.0:  # Only include highly relevant docs
                relevant_docs.append(doc)
                scores.append(relevance_score)

        # Sort by relevance score in descending order
        if relevant_docs:
            sorted_docs = [x for _, x in sorted(zip(scores, relevant_docs), key=lambda pair: pair[0], reverse=True)]
            return sorted_docs
        return docs[:3]  # Return at least some docs if none meet threshold

    def _mmr_reranking(self, query: str, docs: List[Any], diversity_weight: float = 0.3) -> List[Any]:
        """
        Implement Maximum Marginal Relevance to balance relevance with diversity.

        Args:
            query: The user query
            docs: List of document objects
            diversity_weight: Higher values prioritize diversity (0-1)

        Returns:
            Reranked list of documents
        """
        if not docs:
            return []

        # Extract document content
        doc_contents = [doc.page_content for doc in docs]

        try:
            # Create document vectors
            if len(doc_contents) == 1:
                return docs  # Skip MMR for single document

            # Fit vectorizer on all documents + query
            all_texts = doc_contents + [query]
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)

            # Separate query vector from document vectors
            query_vec = tfidf_matrix[-1]
            doc_vecs = tfidf_matrix[:-1]

            # Calculate query-document similarities
            query_doc_sims = cosine_similarity(query_vec, doc_vecs)[0]

            # Initialize selected documents and remaining candidates
            selected_indices = []
            remaining_indices = list(range(len(docs)))

            # Always select the most relevant document first
            best_idx = np.argmax(query_doc_sims)
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

            # Select remaining documents using MMR
            while remaining_indices and len(selected_indices) < min(5, len(docs)):
                # Calculate document similarities between candidates and selected docs
                doc_doc_sims = cosine_similarity(doc_vecs[remaining_indices], doc_vecs[selected_indices])

                # Calculate MMR scores
                mmr_scores = []
                for i, idx in enumerate(remaining_indices):
                    relevance = query_doc_sims[idx]
                    diversity = 1 - np.max(doc_doc_sims[i])
                    mmr = (1 - diversity_weight) * relevance + diversity_weight * diversity
                    mmr_scores.append(mmr)

                # Select document with highest MMR score
                best_mmr_idx = np.argmax(mmr_scores)
                selected_indices.append(remaining_indices[best_mmr_idx])
                remaining_indices.remove(remaining_indices[best_mmr_idx])

            # Return reranked documents
            return [docs[i] for i in selected_indices]

        except Exception as e:
            logger.warning(f"Error during MMR reranking: {str(e)}")
            return docs  # Return original docs if MMR fails

    def _retrieve_subquery_docs(self, subq: str, qa_chain_instance) -> Dict[str, Any]:
        """Retrieve documents for a subquery with enhanced relevance filtering and MMR."""
        # Get raw documents
        docs = qa_chain_instance.retriever.get_relevant_documents(subq)

        # Apply relevance filtering
        docs = self._filter_relevant_docs(subq, docs)

        # Apply MMR reranking for diversity
        docs = self._mmr_reranking(subq, docs, diversity_weight=0.3)

        for doc in docs:
            logger.info(f"Retrieved document content: {doc.page_content[:500]}")  # Log first 500 chars of content
            logger.info(f"Retrieved document metadata: {doc.metadata}")

        answer_text = "\n".join([self._simple_summarize(doc.page_content) for doc in docs]) if docs else ""
        return {
            "subquery": subq,
            "answer": answer_text,
            "sources": docs
        }

    def _generate_diverse_subqueries(self, query: str) -> List[str]:
        """Generate diverse subqueries that explore different aspects of the original query."""
        decomp_llm = self._get_stage_llm("decomposition")
        decomp_prompt = ChatPromptTemplate.from_template("""
        Decompose the following query into {max_queries} DISTINCT sub-queries that explore DIFFERENT ASPECTS of the original question.
        Each sub-query should cover a UNIQUE angle, perspective, or component of the original query.

        Original Query: {query}

        IMPORTANT INSTRUCTIONS:
        1. Make each sub-query SUBSTANTIALLY DIFFERENT from the others
        2. Ensure each sub-query can stand alone as a complete question
        3. Maximize the DIVERSITY of information that will be retrieved
        4. Avoid simply rephrasing the original query or each other
        5. Focus on different entities, time periods, aspects, or components

        Format your answer as a comma-separated list, like:
        Sub-query about aspect A, Sub-query about aspect B, Sub-query about aspect C
        """)

        decomp_chain = decomp_prompt | decomp_llm | StrOutputParser()
        subqueries_text = decomp_chain.invoke({"query": query, "max_queries": self.MAX_SUBQUERIES})

        # Clean up the output
        subqueries = [s.strip() for s in subqueries_text.split(',') if s.strip()]

        # Add a sub-query that's very close to the original query to ensure direct relevance
        if len(subqueries) < self.MAX_SUBQUERIES:
            subqueries.append(query)

        self._log_debug("Diverse subqueries", subqueries)
        return subqueries[:self.MAX_SUBQUERIES]

    def _is_answer_too_similar(self, query: str, answer: str, previous_qa: Dict[str, str]) -> bool:
        """
        Check if current answer is too similar to previous answers for different questions.
        Uses both vector similarity and LLM judgment.
        """
        # Quick return if no previous Q&A
        if not previous_qa:
            return False

        # Vector similarity check
        try:
            # Vectorize the new answer
            all_answers = list(previous_qa.values()) + [answer]
            answer_vectors = self.vectorizer.fit_transform(all_answers)
            new_answer_vec = answer_vectors[-1]
            prev_answer_vecs = answer_vectors[:-1]

            # Calculate similarities
            similarities = cosine_similarity(new_answer_vec, prev_answer_vecs)[0]
            max_similarity = np.max(similarities) if len(similarities) > 0 else 0

            # If high vector similarity, do deeper LLM check
            if max_similarity > self.answer_similarity_threshold:
                # Find the most similar previous answer
                most_similar_idx = np.argmax(similarities)
                prev_queries = list(previous_qa.keys())
                most_similar_query = prev_queries[most_similar_idx]
                most_similar_answer = previous_qa[most_similar_query]

                # Use LLM to verify similarity
                comparison_llm = self._get_stage_llm("evaluation")
                comparison_prompt = ChatPromptTemplate.from_template("""
                Compare these two question-answer pairs:

                Question 1: {q1}
                Answer 1: {a1}

                Question 2: {q2}
                Answer 2: {a2}

                The questions are different, but might the answers be providing essentially the same information?
                Consider:
                1. Are they covering the same main points?
                2. Do they present the same key information or conclusions?
                3. Would a user learn anything new from the second answer if they've already seen the first?

                Answer "YES" if they are too similar (>70% overlapping content).
                Answer "NO" if they provide distinct information relevant to their respective questions.
                """)

                comparison_chain = comparison_prompt | comparison_llm | StrOutputParser()
                result = comparison_chain.invoke({
                    "q1": most_similar_query, "a1": most_similar_answer,
                    "q2": query, "a2": answer
                })

                return "YES" in result.upper()

            return False

        except Exception as e:
            logger.warning(f"Error checking answer similarity: {str(e)}")
            # Fallback to simple LLM check for one random previous answer
            if previous_qa:
                prev_query, prev_answer = next(iter(previous_qa.items()))
                comparison_llm = self._get_stage_llm("evaluation")
                comparison_prompt = ChatPromptTemplate.from_template("""
                Are these two answers providing essentially the same information despite being for different questions?
                Answer 1: {a1}
                Answer 2: {a2}
                Only answer "YES" or "NO".
                """)
                comparison_chain = comparison_prompt | comparison_llm | StrOutputParser()
                result = comparison_chain.invoke({"a1": prev_answer, "a2": answer})
                return "YES" in result.upper()
            return False

    def _deduplicate_sources(self, sources: List[Any]) -> List[Any]:
        seen_hashes = set()
        unique_sources = []

        for source in sources:
            content_hash = hashlib.md5(source.page_content.encode()).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_sources.append(source)
                logger.info(f"Deduplicated document metadata: {source.metadata}")

        return unique_sources

    def _refine_answer(self, answer: str) -> Tuple[str, int]:
        refinement_llm = self._get_stage_llm("refinement")
        refinement_prompt = PromptTemplate(
            template="Refine this answer to improve clarity and accuracy while keeping it concise.\nOriginal Answer: {answer}\nRefined Answer (with confidence):",
            input_variables=["answer"]
        )
        refinement_chain = LLMChain(llm=refinement_llm, prompt=refinement_prompt)
        refined = self._safe_invoke(refinement_chain, answer=answer)
        new_confidence = extract_confidence(refined)
        return refined, new_confidence

    def _enhance_answer_diversity(self, query: str, answer: str, previous_qa: Dict[str, str]) -> str:
        """
        Enhance the diversity of the answer if it's too similar to previous answers.
        """
        # Check if there are any previous answers to compare with
        if not previous_qa:
            return answer

        # Find if any previous answers are semantically similar
        similar_answers = []
        for prev_query, prev_answer in previous_qa.items():
            if prev_query != query:  # Skip comparing to the same query
                comparison_llm = self._get_stage_llm("evaluation")
                comparison_prompt = ChatPromptTemplate.from_template("""
                Do these two answers contain substantially similar information?
                Answer 1: {a1}
                Answer 2: {a2}
                Only respond with "YES" or "NO".
                """)
                comparison_chain = comparison_prompt | comparison_llm | StrOutputParser()
                result = comparison_chain.invoke({"a1": prev_answer, "a2": answer})

                if "YES" in result.upper():
                    similar_answers.append((prev_query, prev_answer))

        # If there are similar previous answers, request a more diverse answer
        if similar_answers:
            diversification_llm = self._get_stage_llm("refinement")

            # Format the similar answers for context
            similar_qa_text = "\n\n".join([
                f"Similar Q: {prev_q}\nSimilar A: {prev_a}"
                for prev_q, prev_a in similar_answers[:2]  # Limit to 2 examples
            ])

            diversification_prompt = ChatPromptTemplate.from_template("""
            I need you to create a more UNIQUE and DISTINCTIVE answer to this question.

            Current Question: {query}
            Current Answer: {answer}

            This answer is too similar to previous answers:
            {similar_qa}

            Please create a NEW answer that:
            1. Is specifically tailored to the current question
            2. Focuses on unique aspects not covered in the similar answers
            3. Uses different phrasing, structure, and examples
            4. Offers distinct insights while remaining accurate
            5. Addresses the query from a different angle

            End with a confidence score like "Confidence: X%" where X is between 1-100.

            New Distinctive Answer:
            """)

            diversification_chain = diversification_prompt | diversification_llm | StrOutputParser()
            diversified_answer = diversification_chain.invoke({
                "query": query,
                "answer": answer,
                "similar_qa": similar_qa_text
            })

            return diversified_answer

        return answer

    def _query_focused_synthesis(self, query: str, context: str, image_text: str = "") -> str:
        """Generate a response that's tightly focused on the specific query."""
        # Create combined context
        combined_context = context + "\n" + image_text if image_text else context
        synthesis_llm = self._get_stage_llm("synthesis")
        synthesis_prompt = ChatPromptTemplate.from_template("""
        Create a focused answer that PRECISELY addresses the original question and ONLY the original question.

        Original Question: {query}

        Context Information:
        {combined_context}

        STRICT INSTRUCTIONS:
        1. Focus EXCLUSIVELY on the specific question asked - do not add tangential information
        2. Use ONLY facts found in the provided context
        3. Structure your response to directly answer exactly what was asked
        4. Include clear bullet points for clarity
        5. Do not repeat the same information in different ways
        6. Avoid generic statements that could apply to any similar query
        7. Ensure each sentence contains information that directly helps answer the specific question

        End with a confidence score like "Confidence: X%" where X is between 1-100.

        Final Answer:
        """)

        synthesis_chain = synthesis_prompt | synthesis_llm | StrOutputParser()
        return synthesis_chain.invoke({"query": query, "combined_context": combined_context})

    @traceable(project_name="workstations")
    def query_documents_advanced(self, query: str, qa_chain_instance) -> QueryResult:
        # Stage 1: Generate diverse subqueries
        subqueries = self._generate_diverse_subqueries(query)
        self._log_debug("Diverse subqueries", subqueries)

        # Stage 2: Parallel Retrieval with enhanced filtering
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.MAX_SUBQUERIES) as executor:
            subanswers = list(executor.map(
                lambda sq: self._retrieve_subquery_docs(sq, qa_chain_instance),
                subqueries[:self.MAX_SUBQUERIES]
            ))

            # Stage 3: Query-focused Synthesis
            context = "\n".join(sa["answer"] for sa in subanswers)
            image_text = "\n".join(sa.get("image_text", "") for sa in subanswers)  # Collect image text if available
            final_answer = self._query_focused_synthesis(query, context, image_text)
            confidence = extract_confidence(final_answer)

        # Stage 4: Collect and deduplicate sources with metadata
        all_sources = []
        for sa in subanswers:
            all_sources.extend(sa["sources"])

        # Deduplicate sources
        unique_sources = self._deduplicate_sources(all_sources)
        sources_with_metadata = [
            source for source in unique_sources
            if hasattr(source, "metadata") and source.metadata
        ]

        # Stage 5: Check for similarity to previous answers
        if self._is_answer_too_similar(query, final_answer, self.previous_qa):
            # Enhance diversity if too similar
            final_answer = self._enhance_answer_diversity(query, final_answer, self.previous_qa)
            confidence = extract_confidence(final_answer)

        # Stage 6: Optional Refinement for low confidence
        refined_flag = False
        if confidence < self.confidence_threshold:
            refined_answer, confidence = self._refine_answer(final_answer)
            final_answer = refined_answer
            refined_flag = True

        # Stage 7: Final Summary
        summary_prompt = ChatPromptTemplate.from_template(
            "Using the answer below, produce a final response in plain text using clear bullet points. "
            "Do not include any JSON or dictionary formatting, only plain text. At the end, output a separate line "
            "that exactly reads 'Confidence: X%', where X is the confidence score as a whole number.\n\n"
            "Answer: {refined_answer}\n\nFinal Response:"
        )
        summary_chain = summary_prompt | self._get_stage_llm("summarization") | StrOutputParser()
        final_summary_raw = summary_chain.invoke({"refined_answer": final_answer})
        self._log_debug("Final summary raw", final_summary_raw)
        final_summary = clean_final_summary(process_math_expressions(final_summary_raw))

        # Prepare source documents for return
        source_documents = []
        logger.info(f"No. of sources with metadata: {len(sources_with_metadata)}")
        for doc in sources_with_metadata:
            if hasattr(doc, "metadata"):
                logger.info(f"Final document metadata: {doc.metadata}")
                source_documents.append(doc.metadata if hasattr(doc, "metadata") else {})
            else:
                logger.warning(f"Document without metadata: {doc}")

        # Store this Q&A pair for future comparison
        self.previous_qa[query] = final_summary

        return QueryResult(
            answer=final_summary,
            confidence=confidence,
            source_documents=source_documents,
            refined=refined_flag,
            subqueries_used=subqueries  # Return subqueries used for debugging
        )


def search_tables_for_answer(documents, keyword):
    """
    Searches for a specific keyword in structured table data across all extracted tables.

    Returns:
        A list of tuples (Document Name, Matching Table Data).
    """
    results = []

    for doc in documents:
        if "tables" in doc.metadata:
            for table in doc.metadata["tables"]:
                df = pd.DataFrame(table)  # Ensure it's a DataFrame
                if keyword in df.values:
                    matches = df[df.apply(lambda row: row.astype(str).str.contains(keyword, case=False, na=False).any(), axis=1)]
                    if not matches.empty:
                        results.append((doc.metadata.get("source", "Unknown Document"), matches))

    return results
