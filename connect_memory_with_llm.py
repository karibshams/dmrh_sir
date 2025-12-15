from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, JSONLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from pathlib import Path

DATA_PATH = "data/"

def load_documents(data_path):
    """Load PDF, JSON, and CSV files from directory"""
    documents = []
    
    # Load PDF files
    if any(Path(data_path).glob("*.pdf")):
        pdf_loader = DirectoryLoader(
            data_path,
            glob="*.pdf",
            loader_cls=PyPDFLoader,
        )
        documents.extend(pdf_loader.load())
    
    # Load JSON files
    if any(Path(data_path).glob("*.json")):
        json_loader = DirectoryLoader(
            data_path,
            glob="*.json",
            loader_cls=JSONLoader,
            loader_kwargs={'jq_schema': '.', 'text_content_key': 'content'}
        )
        try:
            documents.extend(json_loader.load())
        except:
            pass
    
    # Load CSV files
    if any(Path(data_path).glob("*.csv")):
        csv_loader = DirectoryLoader(
            data_path,
            glob="*.csv",
            loader_cls=CSVLoader,
            loader_kwargs={'source_column': 'source'}
        )
        try:
            documents.extend(csv_loader.load())
        except:
            pass
    
    return documents

def create_chunks(extracted_data):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return text_splitter.split_documents(extracted_data)

def get_embedding_model():
    """Load embedding model"""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_vectorstore():
    """Create and save FAISS vector store"""
    print("Loading documents...")
    documents = load_documents(DATA_PATH)
    print(f"Total documents loaded: {len(documents)}")
    
    print("Creating chunks...")
    text_chunks = create_chunks(documents)
    print(f"Total chunks created: {len(text_chunks)}")
    
    print("Loading embedding model...")
    embedding_model = get_embedding_model()
    
    print("Creating FAISS vector store...")
    db = FAISS.from_documents(text_chunks, embedding_model)
    
    DB_FAISS_PATH = "vectorstore/db_faiss"
    os.makedirs(DB_FAISS_PATH, exist_ok=True)
    db.save_local(DB_FAISS_PATH)
    print(f"Vector store saved at {DB_FAISS_PATH}")

if __name__ == "__main__":
    create_vectorstore()