# 🤖 PDF Intelligence: RAG Chatbot

An end-to-end RAG (Retrieval-Augmented Generation) pipeline that allows users to upload PDF documents and have context-aware conversations with the data.

## 🌟 Features
- **Document Processing**: Splits PDFs into manageable chunks using LangChain.
- **Vector Storage**: Uses FAISS for high-performance similarity search.
- **Smart Retrieval**: Context-aware answering using OpenAI's GPT-4o.
- **Streamlit UI**: A clean, interactive web interface.

## 🛠️ Tech Stack
- **Framework**: LangChain
- **LLM**: OpenAI GPT-4o
- **Vector DB**: FAISS
- **Frontend**: Streamlit

## ⚙️ Installation & Setup
1. Clone the repo:
   git clone [https://github.com/YOUR_USERNAME/AI-RAG_chatbot.git](https://github.com/YOUR_USERNAME/AI-RAG_chatbot.git)
   
2. Install Dependency 
   pip install -r requirements.txt
   
3. Run the App
   streamlit run app.py
