import streamlit as st
import os
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

class EWUAcademicUI:
    """Streamlit UI for EWU Academic Assistant"""
    
    def __init__(self):
        self.groq_api_key = "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"
        self.db_path = "vectorstore/db_faiss"
        self.model = "mixtral-8x7b-32768"
        
        # EWU Theme Colors
        self.ewu_primary = "#003d7a"
        self.ewu_secondary = "#0066cc"
        self.ewu_accent = "#ff6600"
        
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
    
    def configure_page(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="EWU Academic Assistant",
            page_icon="🎓",
            layout="wide"
        )
        self.apply_theme()
    
    def apply_theme(self):
        """Apply EWU theme with custom CSS"""
        st.markdown(f"""
        <style>
        :root {{
            --primary-color: {self.ewu_primary};
            --secondary-color: {self.ewu_secondary};
            --accent-color: {self.ewu_accent};
        }}
        .main {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }}
        .stChatMessage {{
            background: white;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .header-title {{
            color: {self.ewu_primary};
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .header-subtitle {{
            color: {self.ewu_secondary};
            text-align: center;
            font-size: 1.2em;
            margin-bottom: 20px;
        }}
        .info-box {{
            background-color: {self.ewu_primary}20;
            border-left: 4px solid {self.ewu_primary};
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .error-box {{
            background-color: #ff6b6b20;
            border-left: 4px solid #ff6b6b;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .success-box {{
            background-color: #51cf6620;
            border-left: 4px solid #51cf66;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        </style>
        """, unsafe_allow_html=True)
    
    @st.cache_resource
    def load_vectorstore(_self):
        """Load FAISS vector store with caching"""
        try:
            if not os.path.exists(_self.db_path):
                return None
            
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            return FAISS.load_local(
                _self.db_path,
                embedding_model,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.error(f"Error loading database: {str(e)}")
            return None
    
    def get_prompt_template(self):
        """Create custom prompt template"""
        return PromptTemplate(
            template=self.custom_prompt_template.format(system_prompt=self.system_prompt),
            input_variables=["context", "question"]
        )
    
    def get_llm(self):
        """Initialize Groq LLM"""
        return ChatGroq(
            api_key=self.groq_api_key,
            model=self.model,
            temperature=0.3,
            max_tokens=1024
        )
    
    def create_qa_chain(self, db):
        """Create RetrievalQA chain"""
        llm = self.get_llm()
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': self.get_prompt_template()}
        )
    
    def render_header(self):
        """Render page header"""
        st.markdown('<div class="header-title">🎓 EWU Academic Assistant</div>', unsafe_allow_html=True)
        st.markdown('<div class="header-subtitle">Retrieval-Augmented Q&A System for East West University</div>', unsafe_allow_html=True)
    
    def check_prerequisites(self):
        """Check system prerequisites"""
        if not os.path.exists(self.db_path):
            st.markdown("""
            <div class="error-box">
            <b>⚠️ Vector Database Not Found</b><br>
            Please run <code>connect_memory_with_llm.py</code> to create the vector store from your documents.
            </div>
            """, unsafe_allow_html=True)
            return False
        
        if not self.groq_api_key or self.groq_api_key == "your_api_key_here":
            st.markdown("""
            <div class="error-box">
            <b>⚠️ API Key Missing</b><br>
            Please configure your Groq API key in the code.
            </div>
            """, unsafe_allow_html=True)
            return False
        
        return True
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
    
    def display_chat_history(self):
        """Display chat message history"""
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    def process_user_input(self, prompt, db):
        """Process user input and generate response"""
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            try:
                with st.spinner("🔍 Searching documents..."):
                    qa_chain = self.create_qa_chain(db)
                    response = qa_chain.invoke({'query': prompt})
                    result = response["result"]
                    
                    st.markdown(result)
                    st.session_state.messages.append({"role": "assistant", "content": result})
                    
                    with st.expander("📄 View Source Documents"):
                        for i, doc in enumerate(response["source_documents"], 1):
                            st.write(f"**Document {i}:** {doc.metadata.get('source', 'Unknown')}")
                            st.write(f"_{doc.page_content[:250]}..._")
                            st.write("---")
                            
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    def run(self):
        """Run the Streamlit application"""
        self.configure_page()
        self.render_header()
        
        if not self.check_prerequisites():
            st.stop()
        
        self.initialize_session_state()
        self.display_chat_history()
        
        db = self.load_vectorstore()
        
        if db is None:
            st.stop()
        
        if prompt := st.chat_input("Ask your question about EWU programs, courses, or policies:"):
            self.process_user_input(prompt, db)

if __name__ == "__main__":
    app = EWUAcademicUI()
    app.run()