
# 📚 RAG PDF Chatbot

This project is a **Retrieval-Augmented Generation (RAG) PDF Chatbot** built using **Streamlit, LangChain, FAISS, Pinecone, HuggingFace, and Cohere/Groq/OpenAI**.  
It allows users to upload PDFs, process them into **chunks**, create **embeddings**, store them in a vector database (**FAISS/Pinecone**), and then query them using an **LLM** for intelligent answers.

---

Live App: [https://groqragapp-cpep2ffsqzojavjlvicy5d.streamlit.app/](https://groqragapp-cpep2ffsqzojavjlvicy5d.streamlit.app/)

## 🚀 Features
- 📤 Upload PDFs directly in the web app
- 📑 Extract and clean text from PDFs (supports scanned PDFs via OCR)
- ✂️ Chunking of text into manageable pieces
- 🧩 Embedding generation using **Sentence Transformers / Cohere / OpenAI**
- 🗂️ Store and search chunks in **FAISS or Pinecone**
- 💬 Ask questions and get answers powered by **LLMs (Groq / OpenAI / Cohere)**
- 📊 **Frontend pipeline visualization** showing:
  - ✅ File uploaded  
  - ⚙️ PDF loaded  
  - ✂️ Chunking  
  - 🧩 Embedding  
  - 💾 Storing in Vector DB  
  - 🤖 Ready to Chat!  

---

## 🛠️ Tech Stack
- [Streamlit](https://streamlit.io/) – UI framework
- [LangChain](https://www.langchain.com/) – RAG pipeline
- [FAISS](https://github.com/facebookresearch/faiss) / [Pinecone](https://www.pinecone.io/) – Vector DB
- [Cohere](https://cohere.com/),[Sentence-Transformers](https://www.sbert.net/) – Embedding models
-  [Groq](https://groq.com/), [OpenAI](https://openai.com/) – LLM providers
- [PyPDF](https://pypi.org/project/pypdf/) / [Unstructured](https://unstructured-io.github.io/) – PDF parsing

---

## 📂 Project Structure
```
.
├
│-- multidoc.py (Main Python file)
|--requirements.txt
|--.env.examples
|--Readme.md
```

---

## ⚙️ Installation

1. **Clone repo**
```bash
git clone https://github.com/Nitinpratap22061/groqragapp.git
cd rag-pdf-chatbot
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Setup environment variables**  
Create a `.env` file in the root directory:
```ini
OPENAI_API_KEY=your_openai_api_key
COHERE_API_KEY=your_cohere_api_key
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

---

## ▶️ Run the App
```bash
streamlit run frontend/app.py
```

---

## 🔄 RAG Pipeline (Stepwise)
1. **📤 Upload PDF**
2. **📑 Extract text**
3. **✂️ Chunking**
4. **🧩 Generate embeddings**
5. **💾 Store in FAISS/Pinecone**
6. **🤖 Query LLM for answers**

---

## 📸 Example Frontend Visualization

✅ File uploaded → ⚙️ PDF Loaded → ✂️ Chunking → 🧩 Embedding → 💾 Stored → 🤖 Ready to Chat!  

---

## 📝 Short Note on RAG Answer Evaluation

We tested a Retrieval-Augmented Generation (RAG) system using a knowledge base PDF on Artificial Intelligence basics. Five descriptive questions were asked, and the system’s answers were compared with the actual content of the document.

**Precision:** The proportion of retrieved/generated answers that were correct.  
**Recall:** The proportion of correct answers from the knowledge base that were successfully retrieved by RAG.

**Comparison of RAG vs. Actual Answers:**

- **Q1 (Definition of AI):** RAG answer matched perfectly with knowledge base → ✅ Correct.  
- **Q2 (Applications of AI):** RAG correctly retrieved examples (Siri, Alexa, Netflix) → ✅ Correct.  
- **Q3 (Definition of ML):** RAG answer matched knowledge base definition → ✅ Correct.  
- **Q4 (Deep Learning):** RAG failed to retrieve from the PDF → ❌ Missed (Recall issue).  
- **Q5 (AI in healthcare & education):** RAG partially correct (healthcare/education examples were not in PDF, but valid generally) → ❌ Missed (strict evaluation).

**Performance:**

- **Precision:** 3/5 = 0.6 (60%)  
- **Recall:** 3/4 = 0.75 (75%)

**Observation:**  
RAG was able to give correct answers for definitions and direct examples but struggled when information was either missing (Q4) or outside the given knowledge base (Q5). This shows RAG can generate plausible but not always knowledge-base-grounded answers.

---

## 🤝 Contributing
Feel free to fork this repo, raise issues, and submit PRs.

---

## 📜 License
MIT License © 2025

---

## 🚀 Demo

Live App: [https://groqragapp.onrender.com/](https://groqragapp-cpep2ffsqzojavjlvicy5d.streamlit.app/)
