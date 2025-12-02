# app.py
import os
import time
import threading
from pathlib import Path
from typing import List
from dotenv import load_dotenv

import streamlit as st

# PDF / Docx
import fitz  # PyMuPDF
import docx

# Optional OCR
try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# LangChain (split packages for langchain 1.1.x)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------------------
# Utility: file -> text extraction
# ---------------------------
def extract_text_from_pdf(path: Path, ocr_if_needed: bool = True) -> str:
    try:
        doc = fitz.open(str(path))
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception:
        text = ""

    # If no text and OCR requested & available => try OCR
    if (not text.strip()) and ocr_if_needed and OCR_AVAILABLE:
        try:
            # convert first N pages to images (adjust dpi if needed)
            images = convert_from_path(str(path), dpi=200)
            ocr_text = []
            for img in images:
                ocr_text.append(pytesseract.image_to_string(img))
            text = "\n".join(ocr_text)
        except Exception as e:
            st.warning(f"OCR failed for {path.name}: {e}")
            text = ""
    return text

def extract_text_from_docx(path: Path) -> str:
    try:
        doc = docx.Document(str(path))
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""

def extract_text_from_txt(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return ""

# ---------------------------
# Document loader (with metadata)
# ---------------------------
def load_documents_from_folder(folder: Path, allowed_ext=(".pdf", ".docx", ".txt"), ocr_if_needed=True) -> List[Document]:
    docs = []
    files = sorted([p for p in folder.glob("*") if p.suffix.lower() in allowed_ext])
    for p in files:
        text = ""
        if p.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(p, ocr_if_needed)
        elif p.suffix.lower() == ".docx":
            text = extract_text_from_docx(p)
        elif p.suffix.lower() == ".txt":
            text = extract_text_from_txt(p)

        if not text or len(text.strip()) < 20:
            # Skip empty / scanned (unless OCR rescued)
            st.info(f"Skipping (no text): {p.name}")
            continue

        meta = {"source": str(p), "name": p.name}
        docs.append(Document(page_content=text, metadata=meta))
    return docs

# ---------------------------
# Chunking, embedding, vectorstore builder
# ---------------------------
def build_vectorstore(documents: List[Document], chunk_size=1200, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()  # uses OPENAI_API_KEY env
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, chunks

# ---------------------------
# Folder watcher (watchdog-based background thread)
# ---------------------------
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class FolderChangeHandler(FileSystemEventHandler):
    def __init__(self, folder_path: Path, on_change_callback):
        super().__init__()
        self.folder_path = folder_path
        self.on_change_callback = on_change_callback
        self._debounce_time = 1.0
        self._last_trigger = 0.0

    def on_any_event(self, event):
        # Debounce rapid events
        now = time.time()
        if now - self._last_trigger < self._debounce_time:
            return
        self._last_trigger = now
        # Call the callback in a thread-safe way
        threading.Thread(target=self.on_change_callback, daemon=True).start()

def start_folder_watcher(folder: Path, on_change_callback):
    event_handler = FolderChangeHandler(folder, on_change_callback)
    observer = Observer()
    observer.schedule(event_handler, str(folder), recursive=False)
    observer.daemon = True
    observer.start()
    return observer

# ---------------------------
# Streamlit UI + state
# ---------------------------
st.set_page_config(page_title="Phase-2 Document Chatbot", layout="wide")
st.title("Phase-2 — Advanced RAG Chatbot (Local Folder)")

# Sidebar settings
st.sidebar.header("Settings")
folder_input = st.sidebar.text_input("Folder path (local):", value=r"E:\My Projects\Chatbot For excisting model\Server")
use_ocr = st.sidebar.checkbox("Enable OCR for scanned PDFs (requires Tesseract & Poppler)", value=False)
auto_watch = st.sidebar.checkbox("Enable Auto-watch folder (rebuild on changes)", value=False)
chunk_size = st.sidebar.number_input("Chunk size (words)", value=1200, min_value=200, max_value=5000, step=100)
chunk_overlap = st.sidebar.number_input("Chunk overlap (words)", value=200, min_value=0, max_value=1000, step=50)
top_k = st.sidebar.slider("Retrieval top K", value=4, min_value=1, max_value=10)

# Session state
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "chunks" not in st.session_state:
    st.session_state["chunks"] = []
if "last_build" not in st.session_state:
    st.session_state["last_build"] = None
if "watcher" not in st.session_state:
    st.session_state["watcher"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

folder = Path(folder_input)

# Build/Rebuild function
def rebuild_action():
    if not folder.exists():
        st.session_state["last_build"] = f"Folder not found: {folder}"
        return
    st.session_state["last_build"] = "Building..."
    docs = load_documents_from_folder(folder, ocr_if_needed=use_ocr)
    if not docs:
        st.session_state["last_build"] = "No readable documents found to build."
        return
    vs, chunks = build_vectorstore(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    st.session_state["vectorstore"] = vs
    st.session_state["chunks"] = chunks
    st.session_state["last_build"] = f"Built vectorstore with {len(chunks)} chunks at {time.strftime('%Y-%m-%d %H:%M:%S')}"

# Start / Stop watcher handlers
def start_watcher():
    if not folder.exists():
        st.warning("Folder does not exist — cannot start watcher.")
        return
    if st.session_state["watcher"] is not None:
        return
    def on_change():
        # small sleep to allow file writes to finish
        time.sleep(1.0)
        rebuild_action()
    observer = start_folder_watcher(folder, on_change)
    st.session_state["watcher"] = observer
    st.session_state["last_build"] = (st.session_state.get("last_build") or "") + "\nWatcher started."

def stop_watcher():
    obs = st.session_state.get("watcher")
    if obs is not None:
        obs.stop()
        obs.join(timeout=0.5)
    st.session_state["watcher"] = None
    st.session_state["last_build"] = (st.session_state.get("last_build") or "") + "\nWatcher stopped."

# Controls
col1, col2, col3 = st.columns([1,1,1])
with col1:
    if st.button("Build / Rebuild Now"):
        rebuild_action()
with col2:
    if auto_watch and st.session_state["watcher"] is None:
        start_watcher()
    elif (not auto_watch) and st.session_state["watcher"] is not None:
        stop_watcher()
with col3:
    if st.button("Clear Chat History"):
        st.session_state["chat_history"] = []

st.markdown("### Build log")
st.text_area("Log", value=st.session_state.get("last_build") or "No builds yet.", height=120)

# Query area
st.markdown("---")
st.header("Ask the Documents")
query = st.text_input("Question:", key="query_input")
if st.button("Ask"):
    if st.session_state["vectorstore"] is None:
        st.error("No vectorstore built — click 'Build / Rebuild Now' first.")
    else:
        vs = st.session_state["vectorstore"]
        results = vs.similarity_search(query, k=top_k)
        # Prepare context
        context = "\n\n".join([r.page_content for r in results])
        # Build a prompt that uses chat history
        history_snippet = ""
        if st.session_state["chat_history"]:
            # include last 4 messages (user+assistant pairs) to keep token usage reasonable
            hist = st.session_state["chat_history"][-8:]
            history_snippet = "\n".join(hist)
        prompt = f"""You are a helpful company assistant. Use ONLY the information from DOCUMENT CONTEXT to answer. If not found, say "I could not find this information in the documents."

Chat history:
{history_snippet}

USER QUESTION:
{query}

DOCUMENT CONTEXT:
{context}
"""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        response = llm.invoke(prompt)
        answer = response.content

        # Save to history
        st.session_state["chat_history"].append(f"User: {query}")
        st.session_state["chat_history"].append(f"Assistant: {answer}")

        # Display answer and sources
        st.markdown("**Answer:**")
        st.write(answer)
        st.markdown("**Sources:**")
        for r in results:
            src = r.metadata.get("source") or r.metadata.get("name")
            st.write("-", src)

# Chat history display
st.markdown("### Chat history (latest first)")
for msg in reversed(st.session_state["chat_history"][-50:]):
    st.write(msg)
