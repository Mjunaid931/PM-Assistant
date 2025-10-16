from dotenv import load_dotenv
import os
import streamlit as st

# --- LangChain Imports ---
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama  # Import Ollama for the LLM
from langchain_community.embeddings import OllamaEmbeddings  # Import Ollama for Embeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# --- 1. CONFIGURATION AND INITIALIZATION ---
load_dotenv()
# Note: Removed the API key check. This version relies on Ollama running locally.

# Define constants
DATA_DIR = "knowledge_base"
VECTOR_DB_PATH = "chroma_db"
LLM_MODEL_NAME = "llama3" # Choose a model you have pulled in Ollama (e.g., llama3, mistral)
EMBEDDING_MODEL_NAME = "nomic-embed-text" # A good open-source embedding model

# Optimized System Prompt (Crucial PM Artifact!)
SYSTEM_PROMPT_OPTIMIZED = (
    "You are an expert **Product Manager Assistant** for a high-growth tech company. "
    "Your goal is to answer questions **concisely and accurately**, citing the source document whenever possible. "
    "Base your answer **ONLY** on the context provided in the `<context>` tags. "
    "If the answer is not in the context, clearly state: 'I could not find a definitive answer in the provided product documentation.' "
    "Maintain a professional, data-driven tone."
    "\n\n<context>\n{context}\n\n</context>"
)

# --- 2. RAG PIPELINE FUNCTIONS ---

@st.cache_resource(show_spinner=False)
def get_vector_store():
    """Initializes and persists the Chroma vector store from documents."""
    st.spinner("Building the RAG knowledge base...")
    
    # 2.1 Load Documents
    documents = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(DATA_DIR, filename)
            loader = TextLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
            print(f"Loaded {len(docs)} documents from {filename}")  # Add this line
        else:
            print(f"Skipping non-text file: {filename}") # Add this line

    if not documents:
        raise ValueError("No documents found in the 'knowledge_base' folder.")

    print(f"Total documents loaded: {len(documents)}")  # Add this line

    # 2.2 Chunking and Embedding (PM Decision: Chunk size is 500 for granular Q&A)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)

    print(f"Total chunks created: {len(splits)}")  # Add this line
    
    # Use Ollama for Embeddings
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    
    # 2.3 Store Vectors (ChromaDB is used for its simplicity)
    vector_store = Chroma.from_documents(
        documents=splits, 
        embedding=embedding, 
        persist_directory=VECTOR_DB_PATH
    )
    vector_store.persist()
    return vector_store

@st.cache_resource(show_spinner=False)
def get_rag_chain(_vector_store): 
    """Creates the LangChain RAG pipeline."""
    # Initialize the LLM (Ensure LLM_MODEL_NAME is pulled via Ollama)
    llm = Ollama(model=LLM_MODEL_NAME, temperature=0) 
    
    # Create the Prompt Template (using the optimized PM prompt)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT_OPTIMIZED),
            ("human", "{input}"),
        ]
    )

    # 3. Create the Chain Components
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create the Retriever (search_kwargs={"k": 2} retrieves the top 2 relevant chunks)
    retriever = _vector_store.as_retriever(search_kwargs={"k": 2})
    
    # Combine everything into the final RAG chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

# --- 3. STREAMLIT UI AND CHAT LOGIC ---

st.set_page_config(page_title="RAG PM Assistant", page_icon="💡")
st.title("💡PM Assistant: Product Knowledge Base")
st.caption("Context-Aware Q&A over Fictional Product Specs")

try:
    # Initialize RAG resources
    vector_store = get_vector_store()
    rag_chain = get_rag_chain(vector_store)
except ValueError as e:
    st.warning(f"Setup Error: {e}")
    st.stop()
except Exception as e:
    # This often happens if Ollama is not running or the model is not pulled.
    st.warning(f"An unexpected error occurred during setup. Is Ollama running and the '{LLM_MODEL_NAME}' model pulled? Error: {e}")
    st.stop()

# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your RAG Assistant. Ask me a question about the API specs or pricing guides."}
    ]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is the rate limit for the transaction endpoint?"):
    # 1. Display User Message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Call the RAG Chain
    with st.chat_message("assistant"):
        with st.spinner("Searching product docs..."):
        
            # The .invoke() call runs the entire RAG pipeline
            response = rag_chain.invoke({"input": prompt})
        
            # Extract the final answer
            assistant_response = response["answer"]
            st.markdown(assistant_response)
        
            # Optional: Show the retrieved context (for PM analysis/debugging)
            st.markdown("---")
            st.caption("Retrieved Context (for PM Review):")
            for doc in response["context"]:
                 st.code(f"Source: {doc.metadata.get('source', 'Unknown')} | Text: {doc.page_content[:150]}...")
        
    # 3. Update Chat History
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})