import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL="llama-3.3-70b-versatile"

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# EMBEDDING_DIMENSION = 384
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIMENSION = 1024

CHUNK_SIZE=500
CHUNK_OVERLAP=150


TOP_K_RESULTS = 5
RELEVANCE_THRESHOLD = 0.50

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "docmind_collection"
