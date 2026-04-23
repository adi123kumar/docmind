from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    question: str
    document_name: str
    retrieved_chunks: list
    relevance_score: float
    web_search_results: list
    final_answer: str
    query_rewritten: bool
    use_web_search: bool