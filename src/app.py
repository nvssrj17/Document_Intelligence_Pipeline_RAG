from pathlib import Path
import time
import streamlit as st
from rag_pipeline import RAGPipeline
import json
from logger import logger

st.set_page_config(page_title="Document Intelligence RAG", layout="wide")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
uploaded_files = st.file_uploader("Upload your documents", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        pdf_path = UPLOAD_DIR / uploaded_file.name
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")

st.title("📄 Document Intelligence RAG Pipeline")

if "history" not in st.session_state:
    st.session_state.history = []
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

query = st.text_input("Enter your question:")
if st.button("Submit"):
    if query:
        start = time.time()
        rag = RAGPipeline()
        result = rag.answer_query(query)
        latency = time.time() - start
        st.session_state.latest_result = result
        st.session_state.history.append({"question": query, "answer": result["answer"], "latency": latency, "sources": result["sources"]})
        logger.info(f"Question: {query}")
        logger.info(f"Latency: {latency:.2f} seconds")

for item in reversed(st.session_state.history):
    st.markdown("---")
    st.markdown(f"**Q:**\n {item['question']}")
    st.markdown(f"**A:**\n {item['answer']}")
    st.markdown(f"**Response Time:**\n {item['latency']:.2f} seconds")
    st.markdown(f"**Sources:**\n {', '.join(map(str, item['sources']))}")
    st.markdown("---")

if st.session_state.latest_result:
    json_output = json.dumps(st.session_state.latest_result, indent=2)
    st.download_button(label="Download JSON Result", data=json_output, file_name="result.json", mime="application/json")

    st.markdown("### Source Pages")
    for page in st.session_state.latest_result["sources"]:
        st.write(f"Page {page}")

with st.sidebar:
    st.header("System")
    st.write("Embedding Model")
    st.write("BAAI/bge-small-en-v1.5")
    st.write("LLM")
    st.write("Gemma 3 4B")