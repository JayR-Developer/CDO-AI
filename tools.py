import os
import pandas as pd
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# === Setup ===
DATA_FOLDER = "./data"
CHROMA_DIR = "./chroma_db/chroma_files_db"

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=2000)

# === Initialize Vector Store ===
file_vector_store = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

def index_files_from_folder():
    """Scan the local data folder and index supported files into ChromaDB."""
    print(f"📂 Scanning folder: {DATA_FOLDER}")

    new_docs = []

    for filename in os.listdir(DATA_FOLDER):
        file_path = os.path.join(DATA_FOLDER, filename)
        if not os.path.isfile(file_path):
            continue

        # === Excel ===
        if filename.lower().endswith((".xlsx", ".xls")):
            try:
                df = pd.read_excel(file_path)
                if df.empty:
                    print(f"⚠️ Skipped empty Excel file: {filename}")
                    continue

                content = df.to_string(index=False)
                doc = Document(page_content=content, metadata={"source": filename})
                splits = text_splitter.split_documents([doc])
                new_docs.extend(splits)
                print(f"📘 Indexed Excel: {filename}")
            except Exception as e:
                print(f"⚠️ Error reading Excel {filename}: {e}")

        # === PDF ===
        elif filename.lower().endswith(".pdf"):
            try:
                loader = PyPDFLoader(file_path)
                pdf_docs = loader.load()
                splits = text_splitter.split_documents(pdf_docs)
                new_docs.extend(splits)
                print(f"📕 Indexed PDF: {filename}")
            except Exception as e:
                print(f"⚠️ Error reading PDF {filename}: {e}")

        # === TXT ===
        elif filename.lower().endswith(".txt"):
            try:
                loader = TextLoader(file_path, encoding="utf-8")
                txt_docs = loader.load()
                splits = text_splitter.split_documents(txt_docs)
                new_docs.extend(splits)
                print(f"📄 Indexed TXT: {filename}")
            except Exception as e:
                print(f"⚠️ Error reading TXT {filename}: {e}")

        else:
            print(f"⏩ Skipped unsupported file: {filename}")

    # === Add to Vector Store ===
    if new_docs:
        file_vector_store.add_documents(new_docs)
        file_vector_store.persist()
        print(f"✅ Indexed {len(new_docs)} document chunks successfully!")
    else:
        print("⚠️ No new documents found or indexed.")

# === Run Indexing ===
if __name__ == "__main__":
    index_files_from_folder()
