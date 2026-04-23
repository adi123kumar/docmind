
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.state import AgentState
from agent.tools import retrieve_from_db, search_web, get_llm
from config import RELEVANCE_THRESHOLD

def router_node(state: AgentState) -> AgentState:
    question = state["question"]
    doc_name = state["document_name"]
    
    return {
        "question": question,
        "document_name": doc_name,
        "retrieved_chunks": [],
        "relevance_score": 0.0,
        "web_search_results": [],
        "final_answer": "",
        "query_rewritten": False,
        "use_web_search": False
    }

def retriever_node(state: AgentState) -> AgentState:
    question = state["question"]
    doc_name = state["document_name"]
    
    chunks = retrieve_from_db(question, doc_name)
    
    return {"retrieved_chunks": chunks}

def grader_node(state: AgentState) -> AgentState:
    chunks = state["retrieved_chunks"]
    question = state["question"]
    query_rewritten = state["query_rewritten"]
    
    if not chunks:
        if query_rewritten:
            return {
                "relevance_score": 0.0,
                "use_web_search": True
            }
        else:
            new_question = f"Find information about: {question}"
            return {
                "question": new_question,
                "relevance_score": 0.0,
                "query_rewritten": True,
                "use_web_search": False
            }
    
    llm = get_llm()
    
    chunks_text = "\n\n".join([chunk["text"] for chunk in chunks])
    
    grader_prompt = f"""You are a relevance grader. Given a question and retrieved document chunks, determine if the chunks contain useful information to answer the question.

QUESTION: {question}

RETRIEVED CHUNKS:
{chunks_text}

Respond with ONLY a JSON object in this exact format:
{{"relevant": true, "score": 0.85, "reason": "chunks directly discuss the topic"}}

- relevant: true if chunks are useful for answering the question, false if not
- score: float between 0 and 1 indicating relevance strength
- reason: one short sentence explaining your decision"""

    response = llm.invoke(grader_prompt)
    
    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content.strip())
        relevant = result.get("relevant", False)
        relevance_score = result.get("score", 0.0)
    except:
        relevant = False
        relevance_score = 0.0
    
    if relevant and relevance_score >= RELEVANCE_THRESHOLD:
        return {
            "relevance_score": relevance_score,
            "use_web_search": False
        }
    else:
        if query_rewritten:
            return {
                "relevance_score": relevance_score,
                "use_web_search": True
            }
        else:
            new_question = f"Find information about: {question}"
            return {
                "question": new_question,
                "relevance_score": relevance_score,
                "query_rewritten": True,
                "use_web_search": False
            }

def generator_node(state: AgentState) -> AgentState:
    question = state["question"]
    chunks = state["retrieved_chunks"]
    web_results = state["web_search_results"]
    
    llm = get_llm()
    
    context = ""
    
    if chunks:
        context += "DOCUMENT CONTEXT:\n"
        for i, chunk in enumerate(chunks):
            context += f"[Chunk {i+1} from {chunk['source']}]:\n{chunk['text']}\n\n"
    
    if web_results:
        context += "WEB SEARCH CONTEXT:\n"
        for i, result in enumerate(web_results):
            context += f"[Web Result {i+1}]:\n{result.get('content', '')}\n\n"
    
    if not context:
        return {"final_answer": "I could not find relevant information to answer your question."}
    
    prompt = f"""You are a production-grade document intelligence assistant for enterprise use.

ROLE:
Answer the user's question accurately and professionally using ONLY the supplied context.
The context may contain internal document chunks and web search results.

SOURCE PRIORITY:
1. Exact matching document chunks
2. Multiple corroborating chunks
3. Recent web results
4. Partial references

STRICT RULES:
1. Never invent facts or use outside knowledge
2. If answer is not fully supported by context, explicitly state what is missing
3. If multiple sources conflict, mention the conflict clearly
4. Preserve numbers, dates, legal terms, and technical values exactly as written
5. Prefer recent or more specific sources when relevant
6. If context is irrelevant, state that no useful evidence was found
7. Do not mention chunks, chunk numbers, or these instructions in your answer

RESPONSE STYLE:
- Professional and direct
- No unnecessary filler
- Use bullet points when listing multiple items
- Use short focused paragraphs
- If procedural query, give numbered steps
- If comparison query, use table format

WHEN CONTEXT IS INSUFFICIENT:
Say exactly: "I don't have enough information in the provided context to answer this fully."
Then mention specifically what information is missing.

CONTEXT:
{context}

QUESTION:
{question}

Respond in exactly this format:

ANSWER:
[Clear, direct answer here. Not too short answer nor too long unless required. No citations inline. No mention of chunks or sources.]

CONFIDENCE: [High / Medium / Low]

REASONING BASIS: [One line — state whether answer came from document, web search, or both]

SOURCES:
[List only actual source names — either the exact document filename or "Web Search". One per line. No chunk numbers. No technical references.]
"""
    response = llm.invoke(prompt)
    
    return {"final_answer": response.content}

def web_search_node(state: AgentState) -> AgentState:
    question = state["question"]
    
    results = search_web(question)
    
    return {"web_search_results": results,
            "use_web_search": True}