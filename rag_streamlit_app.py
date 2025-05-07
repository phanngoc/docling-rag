import streamlit as st
import os
from qdrant_client import QdrantClient

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter

from langchain_openai import ChatOpenAI
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from dotenv import load_dotenv
load_dotenv()

# Check for OpenAI API Key
openai_api_key = os.environ.get("OPENAI_API_KEY", "")
if not openai_api_key and "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

# Application title and description
st.title("RAG Chat Application")
st.subheader("Ask questions about your document")
if 'collection_name' not in st.session_state:
    st.session_state['collection_name'] = 'docling'

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # API Key input if not set in environment
    if not openai_api_key:
        st.session_state.openai_api_key = st.text_input(
            "OpenAI API Key", 
            value=st.session_state.openai_api_key,
            type="password", 
            help="Enter your OpenAI API key to use the LLM"
        )
        openai_api_key = st.session_state.openai_api_key
    
    st.header("Document Settings")
    document_url = st.text_input(
        "Enter Document URL:",
        placeholder="https://example.com/document",
        help="Enter a URL to a document that will be processed for RAG"
    )
    
    # Collection name input
    collection_name = st.text_input(
        "Collection Name:", 
        value="docling",
        help="Name for the vector store collection"
    )

    st.session_state.collection_name = collection_name
    
    process_btn = st.button("Process Document")

# Function to setup the LangChain components with history aware retriever
def setup_langchain_rag(collection_name="docling"):
    # Initialize HuggingFace embeddings

    retriever = SimpleDictRetriever()

    # Initialize LLM
    llm = ChatOpenAI(
        temperature=0,
        api_key=openai_api_key,
        model_name="gpt-4o"
    )

    # Contextualize question system prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )
    
    # Create contextualize question prompt
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # Create history aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # Setup QA system prompt
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. Use "
        "the following pieces of retrieved context to answer the "
        "question. If you don't know the answer, just say that you "
        "don't know. Use three sentences maximum and keep the answer "
        "concise."
    )
    
    # Create QA prompt
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            ("system", "Context: {context}"),
        ]
    )
    
    # Create document chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Create retrieval chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

# Initialize document converter and Qdrant client
@st.cache_resource
def initialize_resources():
    doc_converter = DocumentConverter(allowed_formats=[InputFormat.HTML, InputFormat.PDF, InputFormat.MD, InputFormat.IMAGE])
    client = QdrantClient(location="http://localhost:6333")
    
    # Set the models
    client.set_model("sentence-transformers/all-MiniLM-L6-v2")
    client.set_sparse_model("Qdrant/bm25")

    return doc_converter, client

doc_converter, client = initialize_resources()

class SimpleDictRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str):
        # Very simple matching logic: if query is a key, return its docs
        docs = client.query(
            collection_name="docling",
            query_text=query,
            limit=5,
        )
        print(f"Retrieved {len(docs)} documents for query: {query}", docs[0], type(docs[0]))
        return [Document(page_content=doc.document) for doc in docs]

    async def _aget_relevant_documents(self, query: str):
        # Optional: Async version
        return self._get_relevant_documents(query)

st.session_state.conversation_chain = setup_langchain_rag(st.session_state.collection_name)
st.session_state.document_processed = True


# Process document when button is clicked
if process_btn and document_url:
    if not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    else:
        with st.spinner("Processing document..."):
            try:
                # Convert document
                result = doc_converter.convert(document_url)
                
                # Chunk the document
                documents, metadatas = [], []
                for chunk in HybridChunker().chunk(result.document):
                    documents.append(chunk.text)
                    metadatas.append(chunk.meta.export_json_dict())

                # Add to Qdrant
                client.add(
                    collection_name=collection_name,
                    documents=documents,
                    metadata=metadatas,
                    batch_size=64,
                )
                
                # Setup LangChain components
                st.session_state.conversation_chain = setup_langchain_rag(collection_name)
                st.session_state.document_processed = True
                
                st.success(f"Processed {len(documents)} chunks from the document")
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")

# Chat interface
st.header("Chat")

# Convert streamlit message history to langchain message format
def get_langchain_history():
    langchain_messages = []
    for message in st.session_state.messages:
        if message["role"] == "user":
            langchain_messages.append(("human", message["content"]))
        else:
            langchain_messages.append(("ai", message["content"]))
    return langchain_messages

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to get response from RAG system
def get_langchain_rag_response(query):
    # check count collection for query
    # points = client.query(
    #     collection_name=st.session_state.collection_name,
    #     query_text=query,
    #     limit=5,
    # )
    # if not points:
    #     return "No relevant documents found in the collection."

    if not st.session_state.conversation_chain:
        return "Conversation chain not initialized. Please check your API key and try processing the document again."
    
    # Get chat history in the format expected by LangChain
    chat_history = get_langchain_history()
    
    # Get response from LangChain RAG
    response = st.session_state.conversation_chain.invoke({
        "input": query,
        "chat_history": chat_history
    })
    
    return response["answer"]

# Chat input
if user_input := st.chat_input("Ask something about the document..."):
    # Check if API key is provided
    if not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_langchain_rag_response(user_input)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})