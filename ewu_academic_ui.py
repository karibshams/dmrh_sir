import streamlit as st
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
#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
class EWUAcademicUI:
    """Streamlit UI for EWU Academic Assistant"""
    
    def __init__(self):
        self.groq_api_key = "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"
        self.db_path = "vectorstore/db_faiss"
        self.model = "llama-3.3-70b-versatile"
        
        # EWU Theme Colors
        self.ewu_primary = "#003d7a"
        self.ewu_secondary = "#0066cc"
        self.ewu_accent = "#ff6600"
        
        self.system_prompt = """You are a friendly and helpful Academic Assistant for East West University.
Your primary role is to help students with questions about:
- Courses and their details (code, credits, prerequisites, learning outcomes)
- Programs and degree requirements
- Academic policies and rules
- Faculty information

Guidelines:
1. First, check the provided context from official EWU documents for relevant information
2. If specific information is in the context, provide it clearly with course codes and requirements
3. If the context doesn't have complete information, supplement with your general knowledge about academic systems
4. Be conversational, friendly, and encouraging to students
5. If you're unsure, be honest and suggest they consult with academic advisors

Always be helpful and provide useful information, whether from documents or general knowledge."""
        
        self.custom_prompt_template = """{system_prompt}

Available Context from EWU Documents:
{{context}}

Student Question:
{{question}}

Provide a friendly, helpful answer. Use the context when relevant, but feel free to provide general academic knowledge if needed."""
    
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
            temperature=0.5,
            max_tokens=250
        )
    
    def create_qa_chain(self, db):
        """Create RetrievalQA chain"""
        llm = self.get_llm()
        prompt = self.get_prompt_template()
        retriever = db.as_retriever(search_kwargs={'k': 3})
        
        # Create RAG chain using modern langchain approach
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
        )
        return rag_chain
    
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
                    response = qa_chain.invoke(prompt)
                    
                    # Extract text content from response
                    if hasattr(response, 'content'):
                        result = response.content
                    else:
                        result = str(response)
                    
                    st.markdown(result)
                    st.session_state.messages.append({"role": "assistant", "content": result})
                            
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