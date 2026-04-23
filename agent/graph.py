import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import (
    router_node,
    retriever_node,
    grader_node,
    generator_node,
    web_search_node
)
from config import RELEVANCE_THRESHOLD

def should_retry_or_search(state: AgentState) -> str:
    use_web_search = state["use_web_search"]
    query_rewritten = state["query_rewritten"]
    relevance_score = state["relevance_score"]
    
    if use_web_search:
        return "web_search"
    elif query_rewritten and relevance_score < RELEVANCE_THRESHOLD:
        return "web_search"
    elif not use_web_search and query_rewritten:
        return "retrieve"
    elif relevance_score >= RELEVANCE_THRESHOLD:
        return "generate"
    else:
        return "retrieve"
def build_graph():
    graph = StateGraph(AgentState)
    
    graph.add_node("router", router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("grader", grader_node)
    graph.add_node("generator", generator_node)
    graph.add_node("web_search", web_search_node)
    
    graph.set_entry_point("router")
    
    graph.add_edge("router", "retriever")
    graph.add_edge("retriever", "grader")
    graph.add_edge("web_search", "generator")
    graph.add_edge("generator", END)
    
    graph.add_conditional_edges(
        "grader",
        should_retry_or_search,
        {
            "retrieve": "retriever",
            "web_search": "web_search",
            "generate": "generator"
        }
    )
    
    compiled_graph = graph.compile()
    return compiled_graph

def run_agent(question: str, doc_name: str) -> dict:
    graph = build_graph()
    
    initial_state = {
        "question": question,
        "document_name": doc_name,
        "retrieved_chunks": [],
        "relevance_score": 0.0,
        "web_search_results": [],
        "final_answer": "",
        "query_rewritten": False,
        "use_web_search": False
    }
    
    result = graph.invoke(initial_state)
    
    return {
        "answer": result["final_answer"],
        "retrieved_chunks": result["retrieved_chunks"],
        "relevance_score": result["relevance_score"],
        "used_web_search": len(result["web_search_results"]) > 0,
        "web_results": result["web_search_results"]
    }
