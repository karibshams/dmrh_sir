from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import os

class AcademicAssistant:
    """AI-powered Academic Assistant for East West University"""
    
    def __init__(self, groq_api_key, db_path="vectorstore/db_faiss", model="llama-3.3-70b-versatile"):
        self.groq_api_key = groq_api_key
        self.db_path = db_path
        self.model = model
        self.llm = None
        self.db = None
        self.qa_chain = None
        self.embedding_model = None
        
        self.system_prompt = """You are an AI-powered Academic Assistant for East West University.
Your task is to answer student questions strictly using the provided academic context retrieved from official university documents.
Use clear, concise, and student-friendly language.
If the answer is not found in the given context, clearly say that the information is not available in the official documents.
Do not hallucinate or assume any academic rules.
Always prioritize accuracy, clarity, and relevance."""
        
        self.custom_prompt_template = """{system_prompt}

Context:
{{context}}

Question:
{{question}}

Answer the question using only the above context.
If applicable, mention relevant rules, credits, prerequisites, or academic policies clearly."""
    
    def load_embedding_model(self):
        """Load HuggingFace embedding model"""
        try:
            print("🤖 Loading embedding model...")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            print("✓ Embedding model loaded")
            return self.embedding_model
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            raise
    
    def load_vectorstore(self):
        """Load FAISS vector store"""
        try:
            print("📂 Loading vector database...")
            
            if not os.path.exists(self.db_path):
                raise FileNotFoundError(f"Vector store not found at {self.db_path}")
            
            self.db = FAISS.load_local(
                self.db_path,
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
            print("✓ Vector database loaded")
            return self.db
        except Exception as e:
            print(f"❌ Error loading vector database: {e}")
            raise
    
    def load_llm(self):
        """Load Groq LLM with optimal parameters"""
        try:
            print("🚀 Loading Groq LLM...")
            self.llm = ChatGroq(
                api_key=self.groq_api_key,
                model=self.model,
                temperature=0.3,
                max_tokens=1024
            )
            print(f"✓ LLM loaded (Model: {self.model})")
            return self.llm
        except Exception as e:
            print(f"❌ Error loading LLM: {e}")
            raise
    
    def get_prompt_template(self):
        """Create custom prompt template"""
        return PromptTemplate(
            template=self.custom_prompt_template.format(system_prompt=self.system_prompt),
            input_variables=["context", "question"]
        )
    
    def create_qa_chain(self):
        """Create RetrievalQA chain"""
        try:
            print("⛓️  Creating QA chain...")
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.db.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': self.get_prompt_template()}
            )
            print("✓ QA chain created")
            return self.qa_chain
        except Exception as e:
            print(f"❌ Error creating QA chain: {e}")
            raise
    
    def initialize(self):
        """Initialize all components"""
        try:
            print("\n" + "=" * 60)
            print("🔧 Initializing Academic Assistant")
            print("=" * 60 + "\n")
            
            self.load_embedding_model()
            self.load_vectorstore()
            self.load_llm()
            self.create_qa_chain()
            
            print("\n" + "=" * 60)
            print("✅ Assistant initialized successfully!")
            print("=" * 60 + "\n")
            
        except Exception as e:
            print(f"\n❌ Initialization failed: {e}")
            raise
    
    def query(self, question):
        """Process a user query and return answer with sources"""
        try:
            if not self.qa_chain:
                raise ValueError("QA chain not initialized. Call initialize() first.")
            
            response = self.qa_chain.invoke({'query': question})
            return {
                'answer': response["result"],
                'sources': response["source_documents"]
            }
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            raise
    
    def format_output(self, result):
        """Format query result for display"""
        output = "\n" + "=" * 60
        output += "\n📝 ANSWER:\n"
        output += result['answer']
        output += "\n\n" + "-" * 60
        output += "\n📄 SOURCE DOCUMENTS:\n"
        
        for i, doc in enumerate(result['sources'], 1):
            output += f"\n{i}. {doc.metadata.get('source', 'Unknown')}\n"
            output += f"   {doc.page_content[:150]}...\n"
        
        output += "=" * 60 + "\n"
        return output
    
    def interactive_session(self):
        """Run interactive Q&A session"""
        self.initialize()
        print("Type 'quit' or 'exit' to end session\n")
        
        while True:
            try:
                question = input("❓ Question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thank you for using EWU Academic Assistant!")
                    break
                
                if not question:
                    print("⚠️  Please enter a valid question\n")
                    continue
                
                result = self.query(question)
                output = self.format_output(result)
                print(output)
                
            except KeyboardInterrupt:
                print("\n\n👋 Session ended!")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    api_key = "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"
    assistant = AcademicAssistant(groq_api_key=api_key)
    assistant.interactive_session()