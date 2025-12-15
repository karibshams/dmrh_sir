import streamlit as st
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

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
            layout="centered",
            initial_sidebar_state="expanded"
        )
        self.apply_theme()
    
    def apply_theme(self):
        """Apply EWU theme with ChatGPT-style responsive layout"""
        st.markdown(f"""
        <style>
        /* ===== Base Layout ===== */
        html, body {{
            height: 100%;
            overflow-x: hidden;
        }}
        
        [data-testid="stAppViewContainer"] {{
            height: 100vh;
            display: flex;
            flex-direction: row;
            background: linear-gradient(135deg, #e8eef7 0%, #f0f2f5 100%);
        }}
        
        /* ===== Sidebar - EWU Gradient ===== */
        [data-testid="stSidebar"] {{
            height: 100vh;
            overflow-y: auto;
            background: linear-gradient(180deg, {self.ewu_primary} 0%, {self.ewu_secondary} 100%) !important;
        }}
        
        [data-testid="stSidebarContent"] {{
            background: transparent !important;
        }}
        
        [data-testid="stSidebarUserContent"] {{
            background: transparent !important;
            padding-bottom: 100px;
        }}
        
        .stSidebar [data-testid="stMarkdownContainer"] {{
            color: white;
        }}
        
        .stSidebar h3 {{
            color: white !important;
            font-weight: 600 !important;
        }}
        
        .stSidebar p {{
            color: rgba(255, 255, 255, 0.95) !important;
        }}
        
        .stSidebar [data-testid="stButton"] button {{
            background-color: rgba(255, 255, 255, 0.15) !important;
            color: white !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
            font-weight: 600 !important;
            width: 100%;
        }}
        
        .stSidebar [data-testid="stButton"] button:hover {{
            background-color: rgba(255, 255, 255, 0.25) !important;
        }}
        
        /* ===== Main Chat Area - Flexbox ===== */
        [role="main"] {{
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
            flex: 1;
        }}
        
        [data-testid="stMainBlockContainer"] {{
            display: flex;
            flex-direction: column;
            height: 100%;
            background: transparent;
            padding: 0;
            margin: 0;
        }}
        
        /* Chat history scroll area */
        [data-testid="stChatMessageContainer"] {{
            flex: 1;
            overflow-y: auto;
            padding: 20px 24px 120px 24px;
            max-width: 900px;
            margin: 0 auto;
            width: 100%;
            background: transparent;
        }}
        
        .stChatMessage {{
            background: transparent !important;
            padding: 8px 0 !important;
        }}
        
        /* Header styling */
        .header-title {{
            color: {self.ewu_primary};
            text-align: center;
            font-size: 2.5em;
            font-weight: 700;
            margin: 30px 0 5px 0;
            letter-spacing: -0.5px;
        }}
        
        .header-subtitle {{
            color: {self.ewu_secondary};
            text-align: center;
            font-size: 1em;
            margin-bottom: 20px;
            font-weight: 500;
        }}
        
        /* ===== Chat Input - Sticky (ChatGPT Style) ===== */
        [data-testid="stChatInput"] {{
            position: sticky;
            bottom: 0;
            background: linear-gradient(180deg, rgba(248, 249, 250, 0) 0%, white 30%, white 100%);
            padding: 12px 20px 16px 20px;
            border-top: 1px solid #e5e7eb;
            box-shadow: 0 -2px 8px rgba(0, 0, 0, 0.05);
            z-index: 10;
        }}
        
        [data-testid="stChatInputContainer"] {{
            max-width: 900px;
            margin: 0 auto;
            width: 100%;
            padding: 0 20px;
        }}
        
        .stChatInputTextArea {{
            width: 100% !important;
        }}
        
        /* Input field styling */
        .stChatInputTextArea textarea {{
            border-radius: 18px !important;
            border: 1px solid #d1d5db !important;
            padding: 10px 14px !important;
            font-size: 14px !important;
            resize: none !important;
            max-height: 100px !important;
            background: #f9fafb !important;
        }}
        
        .stChatInputTextArea textarea:focus {{
            border-color: {self.ewu_primary} !important;
            box-shadow: 0 0 0 3px rgba(0, 61, 122, 0.1) !important;
            background: white !important;
        }}
        
        .stChatInputTextArea textarea::placeholder {{
            color: #9ca3af !important;
        }}
        
        /* ===== Scrollbar ===== */
        ::-webkit-scrollbar {{
            width: 6px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: transparent;
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: {self.ewu_primary}40;
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: {self.ewu_primary}60;
        }}
        
        /* Divider */
        .stDivider {{
            margin: 15px 0 !important;
            border-color: rgba(255, 255, 255, 0.2) !important;
        }}
        
        /* Responsive adjustments */
        @media (max-width: 768px) {{
            [data-testid="stChatMessageContainer"] {{
                padding: 15px 16px 120px 16px;
            }}
            
            .header-title {{
                font-size: 2em;
            }}
            
            .header-subtitle {{
                font-size: 0.9em;
            }}
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
        """Create RAG chain"""
        llm = self.get_llm()
        prompt = self.get_prompt_template()
        retriever = db.as_retriever(search_kwargs={'k': 3})
        
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
        )
        return rag_chain
    
    def check_prerequisites(self):
        """Check system prerequisites"""
        if not os.path.exists(self.db_path):
            st.error("⚠️ Vector Database Not Found. Please run connect_memory_with_llm.py first.")
            return False
        
        if not self.groq_api_key or self.groq_api_key == "your_api_key_here":
            st.error("⚠️ API Key Missing. Please configure your Groq API key.")
            return False
        
        return True
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'conversations' not in st.session_state:
            st.session_state.conversations = {}
        if 'current_conversation_id' not in st.session_state:
            st.session_state.current_conversation_id = None
    
    def create_new_conversation(self):
        """Create a new conversation"""
        import time
        conv_id = f"conv_{int(time.time())}"
        st.session_state.conversations[conv_id] = []
        st.session_state.current_conversation_id = conv_id
        st.session_state.messages = []
        st.rerun()
    
    def load_conversation(self, conv_id):
        """Load a specific conversation"""
        st.session_state.current_conversation_id = conv_id
        st.session_state.messages = st.session_state.conversations.get(conv_id, [])
        st.rerun()
    
    def save_current_conversation(self):
        """Save current conversation to history"""
        if st.session_state.current_conversation_id:
            st.session_state.conversations[st.session_state.current_conversation_id] = st.session_state.messages
    
    def display_chat_history(self):
        """Display chat message history"""
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
                st.markdown(msg["content"])
    
    def process_user_input(self, prompt, db):
        """Process user input and generate response"""
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        with st.chat_message("assistant", avatar="🤖"):
            try:
                with st.spinner("✨ Thinking..."):
                    qa_chain = self.create_qa_chain(db)
                    response = qa_chain.invoke(prompt)
                    
                    if hasattr(response, 'content'):
                        result = response.content
                    else:
                        result = str(response)
                    
                    st.markdown(result)
                    st.session_state.messages.append({"role": "assistant", "content": result})
                            
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
    
    def run(self):
        """Run the Streamlit application"""
        self.configure_page()
        
        if not self.check_prerequisites():
            st.stop()
        
        self.initialize_session_state()
        
        # Sidebar
        with st.sidebar:
            st.markdown(f"<div style='color: white; font-size: 1.8em; font-weight: 700; margin-bottom: 20px;'>💬 EWU Chat</div>", unsafe_allow_html=True)
            
            if st.button("✨ New Chat", use_container_width=True, key="new_chat"):
                self.create_new_conversation()
            
            st.divider()
            
            st.markdown("<h3 style='color: white;'>📚 Chat History</h3>", unsafe_allow_html=True)
            if st.session_state.conversations:
                for conv_id in reversed(list(st.session_state.conversations.keys())):
                    messages = st.session_state.conversations[conv_id]
                    if messages:
                        first_msg = next((m["content"][:25] + "..." if len(m["content"]) > 25 else m["content"] 
                                         for m in messages if m["role"] == "user"), "New Chat")
                        
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            if st.button(f"💬 {first_msg}", use_container_width=True, key=conv_id):
                                self.load_conversation(conv_id)
                        with col2:
                            if st.button("🗑️", key=f"del_{conv_id}"):
                                del st.session_state.conversations[conv_id]
                                if st.session_state.current_conversation_id == conv_id:
                                    st.session_state.current_conversation_id = None
                                    st.session_state.messages = []
                                st.rerun()
            else:
                st.markdown("<p style='color: rgba(255,255,255,0.7);'>No chats yet</p>", unsafe_allow_html=True)
            
            st.divider()
            st.markdown("""
            <div style='color: rgba(255,255,255,0.95); font-size: 13px;'>
            <b>📚 About</b><br><br>
            Your AI guide for:<br>
            • Courses<br>
            • Programs<br>
            • Policies<br>
            • Faculty Info
            </div>
            """, unsafe_allow_html=True)
        
        # Main content - Headers inside scrollable area (ChatGPT style)
        st.markdown('<div class="header-title">🎓 EWU Academic Assistant</div>', unsafe_allow_html=True)
        st.markdown('<div class="header-subtitle">Your AI Guide for Academic Success</div>', unsafe_allow_html=True)
        
        db = self.load_vectorstore()
        if db is None:
            st.stop()
        
        if st.session_state.current_conversation_id is None:
            self.create_new_conversation()
        
        self.display_chat_history()
        
        if prompt := st.chat_input("Ask me anything about EWU..."):
            self.process_user_input(prompt, db)
            self.save_current_conversation()

if __name__ == "__main__":
    app = EWUAcademicUI()
    app.run()
