# 📚 RAG PDF Chatbot

This project is a **Retrieval-Augmented Generation (RAG) PDF Chatbot** built using **Streamlit, LangChain, FAISS, Pinecone, HuggingFace, and Cohere/Groq/OpenAI**.  
It allows users to upload PDFs, process them into **chunks**, create **embeddings**, store them in a vector database (**FAISS/Pinecone**), and then query them using an **LLM** for intelligent answers.

---




Live App: [https://groqragapp.onrender.com/](https://groqragapp.onrender.com/)

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
- [Sentence-Transformers](https://www.sbert.net/) – Embedding models
- [Cohere](https://cohere.com/), [Groq](https://groq.com/), [OpenAI](https://openai.com/) – LLM providers
- [PyPDF](https://pypi.org/project/pypdf/) / [Unstructured](https://unstructured-io.github.io/) – PDF parsing
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) – OCR for scanned PDFs

---

## 📂 Project Structure
```
.
├── backend/
│   ├── model/               # LangChain + VectorDB logic
│   ├── routes/              # API endpoints (if needed)
│
├── frontend/
│   ├── app.py               # Streamlit UI
│   ├── components/          # UI Components
│
├── .env                     # API keys + environment variables
├── requirements.txt         # Python dependencies
└── README.md                # Documentation
```

---

## ⚙️ Installation

1. **Clone repo**
```bash
git clone https://github.com/yourusername/rag-pdf-chatbot.git
cd rag-pdf-chatbot
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate    # Windows
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

## 🤝 Contributing
Feel free to fork this repo, raise issues, and submit PRs.

---

## 📜 License
MIT License © 2025


## 🚀 Demo

Live App: [https://groqragapp.onrender.com/](https://groqragapp.onrender.com/)
