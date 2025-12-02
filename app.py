# app_improved.py — PHASE 3 IMPROVED: Production-Ready Multi-User RAG Chatbot
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from pathlib import Path
import hashlib
import sqlite3
import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import fitz
import pytesseract
from pdf2image import convert_from_path
import threading
from contextlib import contextmanager
import logging

# LangChain + Vector DB
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# === FIX INVISIBLE SOURCES - INJECT CUSTOM CSS ===
st.markdown("""
<style>
    .source-box {
        padding: 12px 16px;
        margin: 8px 0;
        border-left: 4px solid #4CAF50;
        border-radius: 0 8px 8px 0;
        background-color: var(--text-color) !important;   /* this is the trick */
        background-color: rgba(0, 0, 0, 0.05) !important; /* fallback */
        color: var(--text-color) !important;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .source-box strong {
        color: #4CAF50;
    }

    /* Force correct colors in both themes */
    [data-testid="stExpander"] .source-box {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
    }

    /* Very safe fallback for light theme */
    @media (prefers-color-scheme: light) {
        .source-box {
            background-color: #f8f9fa !important;
            color: #212529 !important;
            border-left-color: #28a745 !important;
        }
        .source-box strong { color: #28a745 !important; }
    }

    /* Very safe fallback for dark theme */
    @media (prefers-color-scheme: dark) {
        .source-box {
            background-color: #2d2d2d !important;
            color: #e0e0e0 !important;
            border-left-color: #4ade80 !important;
        }
        .source-box strong { color: #4ade80 !important; }
    }
</style>
""", unsafe_allow_html=True)

# === LOGGING SETUP ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === CONFIG ===
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"E:\Downloads\Release-25.11.0-0\poppler-25.11.0\Library\bin"
DOCS_FOLDER = Path(r"E:\My Projects\Chatbot For excisting model\Server")
DB_PATH = Path("phase3_chatbot.db")
INDEX_PATH = Path("faiss_index")
INDEX_PATH.mkdir(exist_ok=True)

# Rate limiting config
RATE_LIMIT_QUERIES = 20  # queries per time window
RATE_LIMIT_WINDOW = 60  # seconds

# Embedding & LLM with error handling
try:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
except Exception as e:
    logger.error(f"Failed to initialize OpenAI clients: {e}")
    st.error("⚠️ Failed to initialize AI models. Please check your API keys in secrets.")
    st.stop()

# Thread lock for vector store operations
vs_lock = threading.Lock()

# === DATABASE SETUP ===
def init_db():
    """Initialize SQLite database with proper schema"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Users table with UUID
    c.execute("""CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                created_at TEXT
    )""")
    
    # Chats table with user isolation
    c.execute("""CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                timestamp TEXT,
                role TEXT,
                content TEXT,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
    )""")
    
    # Documents table with content hash
    c.execute("""CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE,
                file_hash TEXT,
                content_hash TEXT,
                processed_at TEXT,
                chunk_count INTEGER,
                status TEXT DEFAULT 'active'
    )""")
    
    # Document chunks for tracking
    c.execute("""CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER,
                chunk_index INTEGER,
                content_hash TEXT,
                FOREIGN KEY (doc_id) REFERENCES documents(id)
    )""")
    
    # Rate limiting table
    c.execute("""CREATE TABLE IF NOT EXISTS rate_limits (
                user_id TEXT,
                timestamp TEXT,
                PRIMARY KEY (user_id, timestamp)
    )""")
    
    # Create indexes for performance
    c.execute("CREATE INDEX IF NOT EXISTS idx_chats_user ON chats(user_id, timestamp)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_rate_limits ON rate_limits(user_id, timestamp)")
    
    conn.commit()
    conn.close()

init_db()

# === RATE LIMITING ===
def check_rate_limit(user_id: str) -> Tuple[bool, int]:
    """Check if user has exceeded rate limit. Returns (allowed, remaining_queries)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    cutoff_time = (datetime.now() - timedelta(seconds=RATE_LIMIT_WINDOW)).isoformat()
    
    # Clean old entries
    c.execute("DELETE FROM rate_limits WHERE timestamp < ?", (cutoff_time,))
    
    # Count recent queries
    c.execute("SELECT COUNT(*) FROM rate_limits WHERE user_id = ? AND timestamp >= ?", 
              (user_id, cutoff_time))
    count = c.fetchone()[0]
    
    if count >= RATE_LIMIT_QUERIES:
        conn.close()
        return False, 0
    
    # Log this query
    c.execute("INSERT INTO rate_limits (user_id, timestamp) VALUES (?, ?)",
              (user_id, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    
    return True, RATE_LIMIT_QUERIES - count - 1

# === USER SESSION (Improved with UUID) ===
def get_or_create_user() -> Tuple[str, str]:
    """Get or create user with UUID-based authentication"""
    if "user_id" not in st.session_state:
        # Check for existing session
        if "temp_name" in st.session_state:
            name = st.session_state.temp_name
        else:
            name = st.text_input("👤 Enter your name to start:", key="name_input")
            if not name:
                st.info("Please enter your name to begin chatting with documents.")
                st.stop()
            st.session_state.temp_name = name
        
        # Generate UUID for new users
        user_id = str(uuid.uuid4())
        
        # Store in database
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT OR IGNORE INTO users (user_id, name, created_at) VALUES (?, ?, ?)",
                  (user_id, name, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        
        st.session_state.user_id = user_id
        st.session_state.user_name = name
        st.rerun()
    
    return st.session_state.user_id, st.session_state.user_name

user_id, user_name = get_or_create_user()

# === VECTOR STORE MANAGEMENT ===
@contextmanager
def vector_store_session():
    """Context manager for thread-safe vector store operations"""
    vs_lock.acquire()
    try:
        yield
    finally:
        vs_lock.release()

def load_or_create_vectorstore() -> FAISS:
    """Load existing FAISS index or create new one"""
    index_file = INDEX_PATH / "index.faiss"
    if index_file.exists():
        try:
            return FAISS.load_local(
                str(INDEX_PATH), 
                embeddings, 
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            # Backup corrupted index
            import shutil
            backup_path = INDEX_PATH / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.move(str(INDEX_PATH), str(backup_path))
            logger.info(f"Corrupted index backed up to {backup_path}")
    
    # Create new index
    dummy_doc = Document(page_content="Initialization document", metadata={"source": "system"})
    return FAISS.from_documents([dummy_doc], embeddings)

# Initialize vector store in session state for thread safety
if "vectorstore" not in st.session_state:
    with vector_store_session():
        st.session_state.vectorstore = load_or_create_vectorstore()

# === OCR + TEXT EXTRACTION (Improved with error handling) ===
def extract_text(pdf_path: Path, timeout: int = 300) -> Optional[str]:
    """
    Extract text from PDF with OCR fallback and timeout protection
    
    Args:
        pdf_path: Path to PDF file
        timeout: Maximum processing time in seconds
    
    Returns:
        Extracted text or None if failed
    """
    try:
        # Try native text extraction first
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        
        # If sufficient text found, return it
        if len(text.strip()) > 100:
            logger.info(f"Extracted {len(text)} chars from {pdf_path.name} (native)")
            return text
    except Exception as e:
        logger.warning(f"Native extraction failed for {pdf_path.name}: {e}")
    
    # Fallback to OCR
    try:
        logger.info(f"Starting OCR for {pdf_path.name}")
        images = convert_from_path(
            str(pdf_path), 
            dpi=300,  # Reduced from 350 for performance
            poppler_path=POPPLER_PATH,
            timeout=timeout
        )
        
        ocr_text = []
        for i, img in enumerate(images):
            try:
                page_text = pytesseract.image_to_string(img, lang='eng', timeout=30)
                ocr_text.append(page_text)
                logger.info(f"OCR page {i+1}/{len(images)} complete")
            except Exception as e:
                logger.error(f"OCR failed on page {i+1}: {e}")
                continue
        
        full_text = "\n\n".join(ocr_text)
        logger.info(f"OCR complete: {len(full_text)} chars from {pdf_path.name}")
        return full_text if full_text.strip() else None
        
    except Exception as e:
        logger.error(f"OCR failed for {pdf_path.name}: {e}")
        return None

def get_content_hash(text: str) -> str:
    """Generate hash of document content (not file)"""
    return hashlib.sha256(text.encode()).hexdigest()

def get_file_hash(path: Path) -> str:
    """Generate hash of file bytes"""
    return hashlib.md5(path.read_bytes()).hexdigest()

# === DOCUMENT PROCESSING ===
def rebuild_vectorstore():
    """Rebuild/update vector store with new or modified documents"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    with st.status("🔄 Updating knowledge base...") as status:
        docs_to_add = []
        docs_to_remove = []
        
        # Check for removed documents
        c.execute("SELECT id, filename FROM documents WHERE status='active'")
        existing_docs = {row[1]: row[0] for row in c.fetchall()}
        
        current_files = {pdf.name for pdf in DOCS_FOLDER.glob("*.pdf")}
        removed_files = set(existing_docs.keys()) - current_files
        
        if removed_files:
            status.write(f"📤 Removing {len(removed_files)} deleted documents...")
            for filename in removed_files:
                c.execute("UPDATE documents SET status='deleted' WHERE filename=?", (filename,))
                logger.info(f"Marked {filename} as deleted")
        
        # Process new/updated documents
        processed = 0
        failed = 0
        
        for pdf in DOCS_FOLDER.glob("*.pdf"):
            file_hash = get_file_hash(pdf)
            
            c.execute("SELECT file_hash, content_hash FROM documents WHERE filename=?", (pdf.name,))
            row = c.fetchone()
            
            # Skip if unchanged
            if row and row[0] == file_hash:
                continue
            
            status.write(f"📄 Processing: {pdf.name}")
            
            try:
                # Extract text
                text = extract_text(pdf)
                if not text:
                    logger.error(f"No text extracted from {pdf.name}")
                    failed += 1
                    continue
                
                content_hash = get_content_hash(text)
                
                # Skip if content unchanged (file metadata changed but not content)
                if row and row[1] == content_hash:
                    c.execute("UPDATE documents SET file_hash=? WHERE filename=?", 
                             (file_hash, pdf.name))
                    continue
                
                # Split into chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,  # Increased from 600 for better context
                    chunk_overlap=150,  # Increased overlap
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                chunks = splitter.split_text(text)
                
                # Create documents with rich metadata
                for i, chunk in enumerate(chunks):
                    docs_to_add.append(Document(
                        page_content=chunk,
                        metadata={
                            "source": pdf.name,
                            "chunk": i + 1,
                            "total_chunks": len(chunks),
                            "updated": datetime.now().isoformat(),
                            "content_hash": get_content_hash(chunk)
                        }
                    ))
                
                # Update database
                c.execute("""INSERT OR REPLACE INTO documents 
                            (filename, file_hash, content_hash, processed_at, chunk_count, status)
                            VALUES (?, ?, ?, ?, ?, 'active')""",
                         (pdf.name, file_hash, content_hash, datetime.now().isoformat(), len(chunks)))
                
                processed += 1
                logger.info(f"Processed {pdf.name}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Failed to process {pdf.name}: {e}")
                failed += 1
                continue
        
        # Update vector store
        if docs_to_add:
            status.write(f"🔧 Adding {len(docs_to_add)} chunks to vector store...")
            
            with vector_store_session():
                # CRITICAL FIX: Use add_documents instead of from_documents
                st.session_state.vectorstore.add_documents(docs_to_add)
                st.session_state.vectorstore.save_local(str(INDEX_PATH))
            
            status.update(label=f"✅ Knowledge base updated! ({processed} docs processed)", 
                         state="complete")
            st.success(f"✅ Processed {processed} documents, {len(docs_to_add)} chunks added")
            if failed > 0:
                st.warning(f"⚠️ {failed} documents failed to process")
        else:
            status.update(label="✅ All documents up to date", state="complete")
            st.info("ℹ️ No new documents to process")
    
    conn.commit()
    conn.close()

# === CHAT HISTORY MANAGEMENT ===
def load_chat_history(limit: int = 50) -> List[Dict[str, str]]:
    """Load recent chat history for current user"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""SELECT role, content FROM chats 
                WHERE user_id=? 
                ORDER BY timestamp DESC 
                LIMIT ?""", (user_id, limit))
    messages = [{"role": r[0], "content": r[1]} for r in reversed(c.fetchall())]
    conn.close()
    return messages

def save_message(role: str, content: str):
    """Save message to database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""INSERT INTO chats (user_id, timestamp, role, content) 
                VALUES (?, ?, ?, ?)""",
              (user_id, datetime.now().isoformat(), role, content))
    conn.commit()
    conn.close()

def get_conversation_context(messages: List[Dict], max_turns: int = 5) -> str:
    """Build conversation context from recent messages"""
    if not messages:
        return ""
    
    recent = messages[-max_turns*2:] if len(messages) > max_turns*2 else messages
    context_lines = []
    
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        context_lines.append(f"{role}: {msg['content']}")
    
    return "\n".join(context_lines)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# === MAIN UI ===
st.set_page_config(
    page_title="Enterprise RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage { padding: 1rem; }
    .source-box { 
        background: #f0f2f6; 
        padding: 0.5rem; 
        border-radius: 0.5rem; 
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("📚 Enterprise Document Assistant")
st.caption(f"Multi-user • Persistent • Auto-updating • OCR-enabled | User: **{user_name}**")

# === SIDEBAR ===
with st.sidebar:
    st.header("⚙️ Controls")
    
    if st.button("🔄 Rebuild Knowledge Base", use_container_width=True):
        rebuild_vectorstore()
        st.rerun()
    
    if st.button("🗑️ Clear My Chat History", use_container_width=True):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM chats WHERE user_id=?", (user_id,))
        conn.commit()
        conn.close()
        st.session_state.messages = []
        st.success("Chat history cleared!")
        st.rerun()
    
    st.markdown("---")
    st.header("📊 Statistics")
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM documents WHERE status='active'")
    doc_count = c.fetchone()[0]
    
    c.execute("SELECT SUM(chunk_count) FROM documents WHERE status='active'")
    chunk_count = c.fetchone()[0] or 0
    
    c.execute("SELECT COUNT(*) FROM chats WHERE user_id=?", (user_id,))
    msg_count = c.fetchone()[0]
    
    conn.close()
    
    st.metric("📄 Active Documents", doc_count)
    st.metric("🧩 Total Chunks", chunk_count)
    st.metric("💬 Your Messages", msg_count)
    
    # Rate limit info
    allowed, remaining = check_rate_limit(user_id + "_check")  # Don't count this check
    st.metric("⏱️ Queries Remaining", f"{remaining}/{RATE_LIMIT_QUERIES}")

# Auto-rebuild on first load
if not (INDEX_PATH / "index.faiss").exists():
    with st.spinner("Building initial knowledge base..."):
        rebuild_vectorstore()

# === CHAT DISPLAY ===
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# === CHAT INPUT ===
if prompt := st.chat_input("💬 Ask about your documents...", key="chat_input"):
    # Check rate limit
    allowed, remaining = check_rate_limit(user_id)
    if not allowed:
        st.error(f"⚠️ Rate limit exceeded. Please wait {RATE_LIMIT_WINDOW} seconds before sending more messages.")
        st.stop()
    
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documents..."):
            try:
                # Perform semantic search
                with vector_store_session():
                    results = st.session_state.vectorstore.similarity_search_with_score(
                        prompt, 
                        k=8
                    )
                
                # Filter by relevance score (lower is better for FAISS)
                relevant_results = [(doc, score) for doc, score in results if score < 1.0]
                
                if not relevant_results:
                    answer = "I couldn't find relevant information in the current documents to answer your question."
                    st.markdown(answer)
                else:
                    # Build context
                    context_parts = []
                    for doc, score in relevant_results:
                        source = doc.metadata.get('source', 'Unknown')
                        chunk_num = doc.metadata.get('chunk', '?')
                        context_parts.append(
                            f"[Source: {source}, Chunk {chunk_num}, Relevance: {1-score:.2f}]\n{doc.page_content}"
                        )
                    
                    context = "\n\n---\n\n".join(context_parts)
                    conversation_history = get_conversation_context(st.session_state.messages[:-1])
                    
                    # Build prompt with conversation context
                    system_prompt = f"""You are an expert document assistant. Answer questions using ONLY the provided context.

CONVERSATION HISTORY:
{conversation_history if conversation_history else "No previous conversation"}

DOCUMENT CONTEXT:
{context}

INSTRUCTIONS:
- Provide accurate, professional answers based solely on the context
- If information is not in the context, say: "I don't have this information in the current documents"
- Cite sources when possible (e.g., "According to [document name]...")
- Keep answers concise but complete
- Maintain conversation continuity when relevant

QUESTION: {prompt}"""
                    
                    # Get LLM response
                    response = llm.invoke(system_prompt)
                    answer = response.content
                    
                    st.markdown(answer)
                    
                    # Show sources
                    with st.expander("View Sources", expanded=False):
                        for i, (doc, score) in enumerate(relevant_results, 1):
                            source = doc.metadata.get('source', 'Unknown document')
                            chunk = doc.metadata.get('chunk', '?')
                            preview = doc.page_content.replace("\n", " ").strip()
                            
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>#{i} — {source}</strong> (Chunk {chunk})<br>
                                <small>Relevance: <b>{1-score:.3f}</b></small><br>
                                {preview[:450]}{"..." if len(preview) > 450 else ""}
                            </div>
                            """, unsafe_allow_html=True)
                
                # Save assistant message
                st.session_state.messages.append({"role": "assistant", "content": answer})
                save_message("assistant", answer)
                
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                error_msg = "I encountered an error while processing your question. Please try again."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                save_message("assistant", error_msg)

# Footer
st.markdown("---")
st.caption(f"🔒 Session ID: `{user_id[:8]}...` | Queries remaining: {remaining}/{RATE_LIMIT_QUERIES}")