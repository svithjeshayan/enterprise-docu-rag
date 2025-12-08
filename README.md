# Enterprise RAG Document Assistant  
**Secure Company-Wide Policy & Document Chatbot**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b.svg)](https://streamlit.io)
[![OpenAI](https://img.shields.io/badge/Powered%20by-OpenAI-412991.svg)](https://openai.com)
[![License: Internal Use](https://img.shields.io/badge/License-Internal%20Use-yellow.svg)](#license)

A **production-ready**, multi-user Retrieval-Augmented Generation (RAG) chatbot that lets employees instantly ask questions about HR policies, leave rules, employee handbooks, compliance documents, EPA letters — and get **100% accurate, source-cited answers** directly from your real files.

**No hallucinations. No data leaks. Full audit trail.**

![Chat Interface Preview](assets/demo.gif)
*(Real answers with expandable source citations and relevance scores)*

## Key Features

- Accurate answers grounded **only** in your actual documents (PDF, scanned PDF, Word)
- Built-in OCR for scanned/image-based PDFs (Tesseract + Poppler)
- Source citations with relevance scores & expandable previews
- Multi-user authentication (ready for SSO/Azure AD integration)
- Per-user document access control (public / hr / finance / legal folders)
- Conversation history & context-aware follow-up questions
- Rate limiting (20 queries/min per user)
- Dark/light theme with polished UI
- Full logging + usage statistics
- Automatic re-indexing when documents change or are added

## Tech Stack

| Component           | Technology                                      |
|---------------------|----------------------------------------------------------|
| Frontend            | Streamlit                                               |
| LLM                 | OpenAI `gpt-4o-mini` (temperature=0)                    |
| Embeddings          | `text-embedding-3-large`                                |
| Vector DB           | FAISS (local, blazing fast)                             |
| Database            | SQLite → ready for PostgreSQL migration                 |
| Authentication      | `streamlit-authenticator` + bcrypt                     |
| Document Processing | PyMuPDF, python-docx, Tesseract, Poppler                |
| Chunking            | LangChain RecursiveCharacterTextSplitter               |

## Quick Start (Local)

```bash
# 1. Clone the repository
git clone https://github.com/your-org/internal-rag-chatbot.git
cd internal-rag-chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up your OpenAI API key
cp .env.example .env
# → Edit .env and paste your key:
# OPENAI_API_KEY=sk-...

# 4. (Optional) Override documents folder
# export DOCS_FOLDER="/path/to/your/company/documents"
# Default: Z:\Chat bot testing (Windows) or current folder

# 5. Launch the app
streamlit run app.py
