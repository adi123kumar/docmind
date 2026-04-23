from langchain_text_splitters import RecursiveCharacterTextSplitter
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_text(text: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = splitter.split_text(text)
    return chunks

def chunk_documents(documents: list) -> list:
    all_chunks = []
    
    for doc in documents:
        chunks = chunk_text(doc.page_content)
        all_chunks.extend(chunks)
    
    return all_chunks