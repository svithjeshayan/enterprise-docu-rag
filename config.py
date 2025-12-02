# ============================================
# config.py — Centralized Configuration
# ============================================
from pathlib import Path
import os
from typing import Optional

class Config:
    """Centralized configuration for the RAG chatbot"""
    
    # === PATHS ===
    BASE_DIR = Path(__file__).parent
    DOCS_FOLDER = Path(os.getenv("DOCS_FOLDER", r"E:\My Projects\Chatbot For excisting model\Server"))
    DB_PATH = Path(os.getenv("DB_PATH", "phase3_chatbot.db"))
    INDEX_PATH = Path(os.getenv("INDEX_PATH", "faiss_index"))
    LOG_PATH = Path(os.getenv("LOG_PATH", "chatbot.log"))
    
    # === OCR CONFIGURATION ===
    TESSERACT_CMD = os.getenv(
        "TESSERACT_CMD",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
    POPPLER_PATH = os.getenv(
        "POPPLER_PATH",
        r"E:\Downloads\Release-25.11.0-0\poppler-25.11.0\Library\bin"
    )
    OCR_DPI = int(os.getenv("OCR_DPI", "300"))
    OCR_TIMEOUT = int(os.getenv("OCR_TIMEOUT", "300"))
    OCR_PAGE_TIMEOUT = int(os.getenv("OCR_PAGE_TIMEOUT", "30"))
    
    # === AI MODEL CONFIGURATION ===
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
    
    # === CHUNKING CONFIGURATION ===
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
    
    # === RETRIEVAL CONFIGURATION ===
    RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "8"))
    RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "1.0"))
    
    # === RATE LIMITING ===
    RATE_LIMIT_QUERIES = int(os.getenv("RATE_LIMIT_QUERIES", "20"))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
    
    # === CONVERSATION CONFIGURATION ===
    MAX_CONVERSATION_TURNS = int(os.getenv("MAX_CONVERSATION_TURNS", "5"))
    MAX_CHAT_HISTORY_LOAD = int(os.getenv("MAX_CHAT_HISTORY_LOAD", "50"))
    
    # === SECURITY ===
    REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "False").lower() == "true"
    SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", "60"))
    
    @classmethod
    def validate(cls) -> bool:
        """Validate critical configuration"""
        issues = []
        
        if not cls.DOCS_FOLDER.exists():
            issues.append(f"Documents folder not found: {cls.DOCS_FOLDER}")
        
        if not Path(cls.TESSERACT_CMD).exists():
            issues.append(f"Tesseract not found: {cls.TESSERACT_CMD}")
        
        if not Path(cls.POPPLER_PATH).exists():
            issues.append(f"Poppler not found: {cls.POPPLER_PATH}")
        
        if not os.getenv("OPENAI_API_KEY"):
            issues.append("OPENAI_API_KEY not set in environment")
        
        if issues:
            for issue in issues:
                print(f"❌ {issue}")
            return False
        
        print("✅ Configuration validated successfully")
        return True
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        cls.INDEX_PATH.mkdir(exist_ok=True)
        print(f"✅ Index directory: {cls.INDEX_PATH}")