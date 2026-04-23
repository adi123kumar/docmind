# 🧠 DocMind — Agentic Document Intelligence System

A production-grade RAG (Retrieval-Augmented Generation) system powered by an AI agent that reasons, self-corrects, and falls back to web search when needed.

## 🚀 Live Demo
[Click here to try DocMind](https://docmind-aditya.streamlit.app/) 

## 🎯 What DocMind Does
Upload any PDF or Word document and ask questions about it in plain English. DocMind doesn't just search — it uses an AI agent to reason about how to find the best answer, self-correct if results are poor, and fall back to web search when the document doesn't contain the answer.

## 🏗️ Architecture
```
User Question
↓
Router Node
↓
Retriever Node ← ChromaDB Vector Search
↓
Grader Node (LLM-as-Judge)
↙           ↘
Good          Not relevant
↓               ↓
Generator    Rewrite Query → Retry
↓               ↓
Answer       Still not good → Web Search (Tavily)
↓
Generator
↓
Final Answer
```

## 🤖 AI Concepts Demonstrated
- **RAG** — Retrieval Augmented Generation with ChromaDB vector store
- **NLP** — Semantic chunking, text embeddings, cosine similarity search
- **LLM** — Llama 3.3 70B via Groq API for generation and grading
- **AI Agents** — LangGraph StateGraph with conditional routing and self-correction loop
- **LLM-as-Judge** — LLM grades its own retrieval quality instead of raw cosine distance

## 🛠️ Tech Stack
| Component | Technology |
|-----------|------------|
| LLM | Llama 3.3 70B (Meta, Open Source) via Groq |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers, local) |
| Vector Store | ChromaDB (local, persistent) |
| Agent Framework | LangGraph |
| Web Search Fallback | Tavily API |
| Document Parsing | PyMuPDF, python-docx |
| UI | Streamlit |

## 🔄 Agent Flow
1. **Router** — Initializes state for new question
2. **Retriever** — Semantic search across uploaded documents in ChromaDB
3. **Grader** — LLM judges if retrieved chunks are actually relevant
4. **Self-correction** — If irrelevant, rewrites query and retries once
5. **Web Search Fallback** — If retry fails, searches web via Tavily
6. **Generator** — Synthesizes cited answer from document and web context

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- Groq API key (free at console.groq.com)
- Tavily API key (free at tavily.com)

### Installation

```bash
git clone https://github.com/yourusername/docmind.git
cd docmind
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### Run
```bash
streamlit run app.py

```

## 📁 Project Structure
```
docmind/
├── app.py                  # Streamlit UI
├── config.py               # Centralized settings
├── ingestion/
│   ├── loader.py           # PDF/DOCX text extraction
│   ├── chunker.py          # Recursive text splitting
│   └── embedder.py         # Sentence transformer embeddings
├── vectorstore/
│   └── chroma_store.py     # ChromaDB operations
├── agent/
│   ├── state.py            # LangGraph AgentState TypedDict
│   ├── tools.py            # LLM, retriever, web search tools
│   ├── nodes.py            # Agent node functions
│   └── graph.py            # LangGraph StateGraph definition
└── requirements.txt
```

## 🎓 Key Design Decisions
- **LLM-as-Judge grading** over cosine distance — more semantically accurate relevance scoring
- **Full document chunking** over page-by-page — preserves context across page boundaries
- **Local embeddings** over OpenAI — zero cost, privacy preserving, works offline
- **ChromaDB** over FAISS — built-in metadata filtering, automatic persistence, simpler API
- **Self-RAG loop** — agent rewrites query once before falling back to web search

## 👨‍💻 Author
Aditya Kumar — Final Year Student, AIML Intern