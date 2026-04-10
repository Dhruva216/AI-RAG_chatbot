import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os

# App Config
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🤖 AI Document Assistant")

# Sidebar for API Key and File Upload
with st.sidebar:
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    
    # 1. Load and Split Document
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    loader = PyPDFLoader("temp.pdf")
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    
    # 2. Create Vector Store (The "Memory")
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    # 3. Setup Retrieval Chain
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever()
    )
    
    st.success("Document processed! Ask away.")

    # 4. Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is this document about?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = qa_chain.invoke(prompt)["result"]
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})