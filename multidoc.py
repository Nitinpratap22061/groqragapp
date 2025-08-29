import streamlit as st
import os
import tempfile
import time
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_cohere import CohereEmbeddings, CohereRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index = os.getenv("PINECONE_INDEX")  

if not all([groq_api_key, cohere_api_key, pinecone_api_key, pinecone_index]):
    st.error("Please set all API keys and the Pinecone index in the .env file.")
    st.stop()

st.title("Chat with Multiple PDFs")

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

        for i, d in enumerate(docs):
            d.metadata.update({
                "source": uploaded_file.name,
                "title": uploaded_file.name.replace(".pdf", ""),
                "section": f"Page-{d.metadata.get('page', 'N/A')}",
                "position": i
            })

        split_docs = splitter.split_documents(docs)
        all_docs.extend(split_docs)

    embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=cohere_api_key)

    pc = Pinecone(api_key=pinecone_api_key)

    vectorstore = PineconeVectorStore.from_documents(all_docs, embeddings, index_name=pinecone_index)
    st.session_state.vectors = vectorstore

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant"
    )

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant. Use the context below to answer.
    If the answer is not in the context, reply: "Sorry, I could not find an answer in the provided documents."
    
    Cite sources inline using [1], [2], etc., based on the context chunk metadata.
    
    <context>
    {context}
    </context>
    
    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)

    retriever = st.session_state.vectors.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 20}
    )

    reranker = CohereRerank(cohere_api_key=cohere_api_key, model="rerank-english-v3.0", top_n=5)

    def retrieve_and_rerank(query: str):
        docs = retriever.get_relevant_documents(query)
        reranked_docs = reranker.compress_documents(docs, query)
        return reranked_docs

    user_input = st.text_input("Ask a question across all uploaded PDFs:")
    if user_input:
        start_time = time.time()

        reranked_docs = retrieve_and_rerank(user_input)

        # Capture response with token usage
        response = document_chain.invoke({"input": user_input, "context": reranked_docs})

        end_time = time.time()
        response_time_ms = (end_time - start_time) * 100

        # ---- Token & Cost Estimation (rough) ----
        tokens_used = len(user_input.split()) + sum(len(d.page_content.split()) for d in reranked_docs)
        # Approx cost assumption for Groq (adjust if you know exact price)
        cost_per_token = 0.000002  # $0.002 / 1K tokens → $0.000002 per token
        estimated_cost = tokens_used * cost_per_token

        # Answer
        answer = response
        if "Sorry" in str(answer) or str(answer).strip() == "":
            st.warning("No relevant answer was found in the uploaded PDFs.")
        else:
            st.subheader("Answer:")
            st.write(answer)

        # Metrics
        st.subheader("⚡ Performance & Cost")
        st.write(f"⏱️ Response time: {response_time_ms:.2f} ms")
        st.write(f"🔤 Tokens used (approx): {tokens_used}")
        st.write(f"💰 Estimated cost: ${estimated_cost:.6f}")

        st.subheader("Sources used:")
        for i, doc in enumerate(reranked_docs, start=1):
            st.markdown(f"[{i}] **Source**: {doc.metadata.get('source', 'N/A')}  "
                        f"(Page {doc.metadata.get('page', 'N/A')})  "
                        f"— Section: {doc.metadata.get('section', 'N/A')}, Position: {doc.metadata.get('position', 'N/A')}")

else:
    st.info("Please upload one or more PDF files to start.")
