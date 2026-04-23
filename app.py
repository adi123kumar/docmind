import streamlit as st
import os
import tempfile
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingestion.loader import get_document_text
from ingestion.chunker import chunk_text
from vectorstore.chroma_store import store_chunks, delete_collection
from agent.graph import run_agent


st.set_page_config(
    page_title="DocMind",
    page_icon="🧠",
    layout="wide"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False

if "doc_names" not in st.session_state:
    st.session_state.doc_names = []

with st.sidebar:
    st.title("📄 DocMind")
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "docx", "txt"],
        help="Upload a PDF, Word, or text document"
    )
    
    if uploaded_file is not None:
        if st.button("Process Document", type="primary"):
            with st.spinner("Processing document..."):
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=f".{uploaded_file.name.split('.')[-1]}"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                text = get_document_text(tmp_path)
                chunks = chunk_text(text)
                store_chunks(chunks, uploaded_file.name)
                
                os.unlink(tmp_path)
                
                st.session_state.doc_loaded = True
                if uploaded_file.name not in st.session_state.doc_names:
                    st.session_state.doc_names.append(uploaded_file.name)
                
            st.success(f"✅ Processed {len(chunks)} chunks from {uploaded_file.name}")
    
    if st.session_state.doc_loaded:
        st.markdown("---")
        st.markdown("**Loaded documents:**")
        for name in st.session_state.doc_names:
            st.markdown(f"- {name}")
        
        if st.button("Clear All Documents", type="secondary"):
            delete_collection()
            st.session_state.doc_loaded = False
            st.session_state.doc_names = []
            st.session_state.messages = []
            st.rerun()

st.title("🧠 DocMind — Agentic Document Intelligence")
st.markdown("Upload a document in the sidebar, then ask questions about it.")

if not st.session_state.doc_loaded:
    st.info("👈 Please upload and process a document to get started.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Agent is thinking..."):
                result = run_agent(prompt, None)
            
            st.markdown(result["answer"])
            
            with st.expander("🔍 Reasoning Details"):
                st.markdown(f"**Relevance Score:** {result['relevance_score']:.2f}")
                st.markdown(f"**Used Web Search:** {result['used_web_search']}")
                st.markdown(f"**Chunks Retrieved:** {len(result['retrieved_chunks'])}")
                
                if result["retrieved_chunks"]:
                    st.markdown("**Source Chunks:**")
                    for i, chunk in enumerate(result["retrieved_chunks"]):
                        st.markdown(f"*Chunk {i+1} from {chunk['source']}:*")
                        st.markdown(f"> {chunk['text'][:200]}...")
                
                if result["web_results"]:
                    st.markdown("**Web Sources:**")
                    for i, web in enumerate(result["web_results"]):
                        st.markdown(f"- [{web.get('title', 'Web Result')}]({web.get('url', '')})")
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"]
        })