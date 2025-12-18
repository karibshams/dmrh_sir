import os
from typing import List

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


class AcademicAssistant:
    """
    EWU Academic Assistant (Low-RAM friendly)
    FAISS + HuggingFace small embeddings + Groq LLM
    """

    def __init__(
        self,
        groq_api_key: str,
        db_path: str = "vectorstore/db_faiss",
        model: str = "llama-3.3-70b-versatile",
    ):
        if not groq_api_key:
            raise ValueError("Groq API key is required")

        self.groq_api_key = groq_api_key
        self.db_path = db_path
        self.model = model

        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.chain = None

        # ================= SYSTEM PROMPT =================
        self.system_prompt = """You are an AI-powered Academic Assistant for East West University.
Answer questions strictly using the provided academic context from official EWU documents.
Use clear, concise language.
If the answer is not found, say:
"Information not available in the official documents."
Do NOT hallucinate.
Keep answers short and relevant.
"""

        # ================= PROMPT TEMPLATE =================
        self.prompt = PromptTemplate(
            template="""{system_prompt}

Context:
{context}

Question:
{question}

Answer using ONLY the context above.
""",
            input_variables=["system_prompt", "context", "question"],
        )

    # --------------------------------------------------
    # LOAD COMPONENTS
    # --------------------------------------------------

    def load_embeddings(self):
        print("🤖 Loading embeddings (low-RAM)...")
        # Use small CPU-friendly embedding
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
        )

    def load_vectorstore(self):
        print("📂 Loading FAISS index...")
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Vectorstore not found: {self.db_path}")

        self.vectorstore = FAISS.load_local(
            self.db_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

    def load_llm(self):
        print("🚀 Loading Groq LLM...")
        self.llm = ChatGroq(
            api_key=self.groq_api_key,
            model=self.model,
            temperature=0.3,
            max_tokens=800,
        )

    # --------------------------------------------------
    # CHAIN
    # --------------------------------------------------

    def format_docs(self, docs: List):
        return "\n\n".join(doc.page_content for doc in docs)

    def build_chain(self):
        print("⛓️ Building LCEL chain...")
        self.chain = (
            {
                "context": self.retriever | self.format_docs,
                "question": RunnablePassthrough(),
                "system_prompt": lambda _: self.system_prompt,
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    # --------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------

    def initialize(self):
        print("\n" + "=" * 60)
        print("🔧 Initializing EWU Academic Assistant")
        print("=" * 60)

        self.load_embeddings()
        self.load_vectorstore()
        self.load_llm()
        self.build_chain()

        print("✅ Assistant ready\n")

    # --------------------------------------------------
    # QUERY
    # --------------------------------------------------

    def query(self, question: str):
        return self.chain.invoke(question)

    # --------------------------------------------------
    # INTERACTIVE SESSION
    # --------------------------------------------------

    def interactive_session(self):
        self.initialize()
        print("Type 'exit' to quit\n")

        while True:
            try:
                q = input("❓ Question: ").strip()
                if q.lower() in {"exit", "quit", "q"}:
                    print("\n👋 Session ended")
                    break
                if not q:
                    continue

                answer = self.query(q)
                print("\n📝 ANSWER:\n", answer, "\n")

            except KeyboardInterrupt:
                print("\n👋 Session ended")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    GROQ_API_KEY = "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"

    assistant = AcademicAssistant(GROQ_API_KEY)
    assistant.interactive_session()
