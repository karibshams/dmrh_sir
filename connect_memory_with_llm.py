from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, JSONLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from pathlib import Path

class DocumentProcessor:
    def __init__(self, data_path="data/"):
        self.data_path = data_path
        self.documents = []
        self.text_chunks = []
        self.embedding_model = None
        self.db = None
        self.db_path = "vectorstore/db_faiss"
    
    def load_documents(self):
        """Load PDF, JSON, and CSV files"""
        print("📄 Loading documents...")
        
        if any(Path(self.data_path).glob("*.pdf")):
            pdf_loader = DirectoryLoader(
                self.data_path,
                glob="*.pdf",
                loader_cls=PyPDFLoader,
            )
            self.documents.extend(pdf_loader.load())
            print(f"✓ PDFs loaded: {len([d for d in self.documents if 'pdf' in str(d.metadata.get('source', ''))])}")
        
        if any(Path(self.data_path).glob("*.json")):
            json_loader = DirectoryLoader(
                self.data_path,
                glob="*.json",
                loader_cls=JSONLoader,
                loader_kwargs={'jq_schema': '.', 'text_content_key': 'content'}
            )
            try:
                self.documents.extend(json_loader.load())
                print(f"✓ JSON files loaded")
            except Exception as e:
                print(f"⚠ JSON load error: {e}")
        
        if any(Path(self.data_path).glob("*.csv")):
            csv_loader = DirectoryLoader(
                self.data_path,
                glob="*.csv",
                loader_cls=CSVLoader,
                loader_kwargs={'source_column': 'source'}
            )
            try:
                self.documents.extend(csv_loader.load())
                print(f"✓ CSV files loaded")
            except Exception as e:
                print(f"⚠ CSV load error: {e}")
        
        print(f"📊 Total documents: {len(self.documents)}")
        return self.documents
    
    def create_chunks(self):
        """Split documents into chunks"""
        print("\n✂️  Creating chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        self.text_chunks = text_splitter.split_documents(self.documents)
        print(f"📦 Total chunks: {len(self.text_chunks)}")
        return self.text_chunks
    
    def load_embedding_model(self):
        """Load embedding model"""
        print("\n🤖 Loading embedding model...")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("✓ Embedding model loaded")
        return self.embedding_model
    
    def create_vectorstore(self):
        """Create and save FAISS vector store"""
        print("\n🔍 Creating FAISS vector store...")
        self.db = FAISS.from_documents(self.text_chunks, self.embedding_model)
        os.makedirs(self.db_path, exist_ok=True)
        self.db.save_local(self.db_path)
        print(f"✓ Vector store saved at {self.db_path}")
        return self.db
    
    def process(self):
        """Execute full processing pipeline"""
        try:
            self.load_documents()
            self.create_chunks()
            self.load_embedding_model()
            self.create_vectorstore()
            print("\n✅ Processing complete!")
        except Exception as e:
            print(f"❌ Error: {e}")
            raise

if __name__ == "__main__":
    processor = DocumentProcessor()
    processor.process()