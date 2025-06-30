import streamlit as st
import os
import tempfile
import time
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

# Replace with your actual Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")

st.title("📚 Chat with Multiple PDFs using Groq + LangChain")

# Upload multiple PDFs
files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if files:
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for uploaded_file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        split_docs = splitter.split_documents(docs)
        all_docs.extend(split_docs)

    # Embed all combined docs
    embeddings = HuggingFaceEmbeddings()
    st.session_state.vectors = FAISS.from_documents(all_docs, embeddings)

    # LLM and chain
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

    prompt = ChatPromptTemplate.from_template("""
    Answer the question based on the provided context only.
    Please provide the correct answer based on the user query or input.
    <context>
    {context}
    </context>
    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # User question
    user_input = st.text_input("Ask a question across all uploaded PDFs:")
    if user_input:
        start_time = time.time()
        response = retrieval_chain.invoke({"input": user_input})
        end_time = time.time()
        st.write(f"⏱️ Response time: {end_time - start_time:.2f}*100 seconds")
        st.subheader("📌 Answer:")
        st.write(response["answer"])

else:
    st.info("📂 Please upload one or more PDF files to start.")
