from sentence_transformers import SentenceTransformer
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBEDDING_MODEL

model = SentenceTransformer(EMBEDDING_MODEL)

def embed_chunks(chunks: list) -> list:
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings.tolist()

def embed_query(query: str) -> list:
    embedding = model.encode([query])
    return embedding[0].tolist()

# def embed_chunks(chunks: list) -> list:
#     prefixed = ["Represent this passage for retrieval: " + chunk 
#                 for chunk in chunks]
#     embeddings = model.encode(prefixed, show_progress_bar=True)
#     return embeddings.tolist()

# def embed_query(query: str) -> list:
#     prefixed = "Represent this sentence for searching relevant passages: " + query
#     embedding = model.encode([prefixed])
#     return embedding[0].tolist()