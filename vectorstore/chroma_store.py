import chromadb
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHROMA_DB_PATH, COLLECTION_NAME, TOP_K_RESULTS
from ingestion.embedder import embed_chunks, embed_query

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

def get_collection():
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

# def store_chunks(chunks: list, doc_name: str):
#     collection = get_collection()
#     embeddings = embed_chunks(chunks)
    
#     ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]
    
#     collection.add(
#         embeddings=embeddings,
#         documents=chunks,
#         ids=ids,
#         metadatas=[{"source": doc_name, "chunk_index": i} 
#                    for i in range(len(chunks))]
#     )
    
#     return len(chunks)

def store_chunks(chunks: list, doc_name: str):
    if not chunks:
        raise ValueError("No chunks to store. Document may be empty or unreadable.")
    
    collection = get_collection()
    embeddings = embed_chunks(chunks)
    
    if not embeddings:
        raise ValueError("Embedding failed. No embeddings generated.")
    
    ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]
    
    collection.add(
        embeddings=embeddings,
        documents=chunks,
        ids=ids,
        metadatas=[{"source": doc_name, "chunk_index": i} 
                   for i in range(len(chunks))]
    )
    
    return len(chunks)

def search_chunks(query: str, doc_name: str = None) -> list:
    collection = get_collection()
    query_embedding = embed_query(query)
    
    where_filter = {"source": doc_name} if doc_name else None
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K_RESULTS,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )
    
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "distance": results["distances"][0][i]
        })
    
    return chunks


def delete_collection():
    client.delete_collection(COLLECTION_NAME)