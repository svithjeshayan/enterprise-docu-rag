# Enterprise RAG Document Assistant  
**Secure Company-Wide Policy & Document Chatbot**  

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)  
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B.svg)](https://streamlit.io)  
[![OpenAI](https://img.shields.io/badge/Powered%20by-OpenAI-412991.svg)](https://openai.com)

A production-ready, multi-user Retrieval-Augmented Generation (RAG) chatbot that lets employees instantly ask questions about HR policies, leave rules, employee handbooks, EPA letters, compliance docs, and any internal document — and get 100% accurate, source-cited answers directly from the real files.

No hallucinations. No data leaks. Full audit trail.

## Features

- Accurate answers using only your actual documents (PDF, scanned PDF, Word)
- Built-in OCR for scanned/image-based PDFs (Tesseract + Poppler)
- Source citations with relevance scores and expandable previews
- Multi-user authentication (john, sarah, guest — ready for SSO)
- Per-user document access control (public / hr / finance / legal folders)
- Conversation history & context-aware follow-ups
- Rate limiting (20 queries/minute per user)
- Dark/light theme support with beautiful UI
- Full logging and usage statistics
- Automatic re-indexing when documents change

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: OpenAI gpt-4o-mini (temperature=0)
- **Embeddings**: text-embedding-3-large
- **Vector DB**: FAISS (local, blazing fast)
- **Database**: SQLite (ready to migrate to PostgreSQL)
- **Auth**: streamlit-authenticator + bcrypt
- **OCR**: Tesseract + Poppler + PyMuPDF

## Quick Start (Local)

```bash
#  git clone https://github.com/your-org/internal-rag-chatbot.git
  cd internal-rag-chatbot

  # Install dependencies
  pip install -r requirements.txt

  # Set your OpenAI key
  export OPENAI_API_KEY=sk-...

  # Place your documents in the folder (default: Z:\Chat bot testing
  # Or override: export DOCS_FOLDER="/path/to/your/docs"

  # Run
<<<<<<< HEAD
  streamlit run app.py
=======
  streamlit run app.py
>>>>>>> 64ee88716ab1ed871287bafc1a706829bd9839a0
