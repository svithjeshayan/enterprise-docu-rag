import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

import fitz  # PyMuPDF
import docx

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Load OpenAI API key from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ------------------------
# STREAMLIT UI
# ------------------------
st.title("📄 GPT-4o-mini RAG Chatbot (Phase-1)")
st.write("Load documents → build vectorstore → ask questions")

folder_input = st.text_input(
    "Enter folder path containing documents:",
    value=r"E:\My Projects\Chatbot For excisting model\Server"
)

# ------------------------
# HELPER FUNCTIONS
# ------------------------
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except:
        return ""
    return text

def extract_text_from_docx(path):
    try:
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    except:
        return ""

def load_documents(folder_path):
    folder = Path(folder_path)
    docs = []
    for file in folder.glob("*"):
        text = ""
        try:
            if file.suffix.lower() == ".pdf":
                text = extract_text_from_pdf(file)
            elif file.suffix.lower() == ".docx":
                text = extract_text_from_docx(file)
            elif file.suffix.lower() == ".txt":
                text = file.read_text(errors="ignore")
            else:
                continue

            if len(text.strip()) < 20:
                st.warning(f"⚠ {file.name} is empty or scanned, skipping.")
                continue

            docs.append(Document(page_content=text, metadata={"source": str(file)}))
        except Exception as e:
            st.error(f"Error reading {file.name}: {e}")
    return docs

# ------------------------
# LOAD DOCUMENTS & BUILD VECTORSTORE
# ------------------------
if st.button("Load & Build Vector Store"):
    folder = Path(folder_input)
    if not folder.exists():
        st.error("❌ Folder does not exist!")
        st.stop()

    files = list(folder.glob("*"))
    if len(files) == 0:
        st.error("❌ No files found in folder!")
        st.stop()

    with st.spinner("📚 Loading documents..."):
        all_docs = load_documents(folder_input)

    if len(all_docs) == 0:
        st.error("❌ No readable documents found.")
        st.stop()

    st.success(f"✔ Loaded {len(all_docs)} documents.")
    st.write("🔍 Preview first document:")
    st.code(all_docs[0].page_content[:500])

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)
    st.write(f"🔹 Total chunks created: {len(chunks)}")

    # Create embeddings
    with st.spinner("🔍 Creating embeddings and FAISS vectorstore..."):
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)

    st.session_state["vectorstore"] = vectorstore
    st.success("✅ Vectorstore ready! You can now ask questions.")

# ------------------------
# QUERY & ANSWER
# ------------------------
st.subheader("Ask a question from your documents:")
query = st.text_input("Your question:")

if st.button("Ask"):
    if "vectorstore" not in st.session_state:
        st.error("❌ Load documents first!")
        st.stop()

    vectorstore = st.session_state["vectorstore"]

    # Retrieve top 4 relevant chunks
    results = vectorstore.similarity_search(query, k=4)

    if len(results) == 0:
        st.warning("⚠ No relevant information found in documents.")
        st.stop()

    context_text = "\n\n".join([doc.page_content for doc in results])

    # Initialize GPT-4o-mini LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    # Construct prompt for RAG
    prompt = f"""
You are a helpful company assistant. Answer the question using ONLY the information from the documents provided.

QUESTION:
{query}

DOCUMENT CONTEXT:
{context_text}

If the answer is not in the documents, respond: "I could not find this information in the documents."
"""

    response = llm.invoke(prompt)

    # Show answer
    st.write("### 📌 Answer:")
    st.write(response.content)

    # Show sources
    st.write("---")
    st.write("### 📄 Sources:")
    for r in results:
        st.write("-", r.metadata["source"])
