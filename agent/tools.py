import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from vectorstore.chroma_store import search_chunks
from config import GROQ_API_KEY, TAVILY_API_KEY, LLM_MODEL

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=LLM_MODEL,
    temperature=0.1
)

tavily_tool = TavilySearch(
    api_key=TAVILY_API_KEY,
    max_results=3
)
def retrieve_from_db(question: str, doc_name: str) -> list:
    results = search_chunks(question, doc_name)
    return results

def search_web(query: str) -> list:
    response = tavily_tool.invoke({"query": query})
    if isinstance(response, list):
        return response
    return response.get("results", [])

def get_llm():
    return llm