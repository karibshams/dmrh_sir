from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

GROQ_API_KEY = "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"
DB_FAISS_PATH = "vectorstore/db_faiss"
GROQ_MODEL = "mixtral-8x7b-32768"

SYSTEM_PROMPT = """You are an AI-powered Academic Assistant for East West University.
Your task is to answer student questions strictly using the provided academic context retrieved from official university documents.
Use clear, concise, and student-friendly language.
If the answer is not found in the given context, clearly say that the information is not available in the official documents.
Do not hallucinate or assume any academic rules.
Always prioritize accuracy, clarity, and relevance."""

CUSTOM_PROMPT_TEMPLATE = """{system_prompt}

Context:
{{context}}

Question:
{{question}}

Answer the question using only the above context.
If applicable, mention relevant rules, credits, prerequisites, or academic policies clearly."""

def set_custom_prompt():
    """Create custom prompt template"""
    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE.format(system_prompt=SYSTEM_PROMPT),
        input_variables=["context", "question"]
    )
    return prompt

def load_llm():
    """Load Groq LLM with latest model"""
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0.3,
        max_tokens=500
    )

def create_qa_chain():
    """Create QA chain with FAISS and Groq"""
    try:
        print("Loading embedding model...")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        print("Loading vector database...")
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        
        print("Loading LLM...")
        llm = load_llm()
        
        print("Creating QA chain...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': set_custom_prompt()}
        )
        
        return qa_chain
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    """Interactive Q&A loop"""
    qa_chain = create_qa_chain()
    if not qa_chain:
        return
    
    print("\nEWU Academic Assistant Ready!")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            query = input("Question: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            if not query:
                continue
            
            print("\nProcessing...")
            response = qa_chain.invoke({'query': query})
            
            print("\n" + "="*60)
            print("ANSWER:")
            print(response["result"])
            print("\n" + "-"*60)
            print("SOURCE DOCUMENTS:")
            for i, doc in enumerate(response["source_documents"], 1):
                print(f"{i}. {doc.metadata.get('source', 'Unknown')} - {doc.page_content[:150]}...")
            print("="*60 + "\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()