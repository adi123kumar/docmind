from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from pathlib import Path

def load_document(file_path: str) -> list:
    path = Path(file_path)
    extension = path.suffix.lower()
    
    if extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif extension == ".docx":
        loader = Docx2txtLoader(file_path)
    elif extension == ".txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")
    
    documents = loader.load()
    return documents

def get_document_text(file_path: str) -> str:
    documents = load_document(file_path)
    full_text = ""
    
    for doc in documents:
        full_text += doc.page_content + "\n"
    
    return full_text.strip()