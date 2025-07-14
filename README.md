
# 📚 Chat with Multiple PDFs using Groq + LangChain

Interact with your PDF documents like never before! This Streamlit-based application allows you to chat with multiple uploaded PDFs using powerful LLMs from Groq and LangChain. Simply upload PDFs, ask questions in natural language, and get intelligent answers with context-aware understanding.

---

## 🚀 Features

- ✅ Upload and process multiple PDF files
- ✅ Extract and split text using LangChain’s `RecursiveCharacterTextSplitter`
- ✅ Generate vector embeddings using HuggingFace Embeddings
- ✅ Store and retrieve document vectors using FAISS
- ✅ Ask questions and get answers based on document content
- ✅ Powered by Groq’s `llama-3.1-8b-instant` model
- ✅ Simple UI built using Streamlit

---

## 🛠️ Tech Stack

| Component         | Technology               |
|------------------|--------------------------|
| UI               | Streamlit                |
| LLM              | Groq (LLaMA 3.1)          |
| Embeddings       | HuggingFace Transformers |
| Vector Database  | FAISS                    |
| PDF Parsing      | LangChain PyPDFLoader    |
| Prompt Templates | LangChain Core           |

---

## 📁 Project Structure

```
.
├── app.py               # Main Streamlit application
├── .env                 # Environment variables (Groq API key)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## ✅ Requirements

- Python 3.9 or higher
- A valid [Groq API key](https://console.groq.com/)
- pip (Python package installer)

---

## 🔐 Environment Setup

1. **Clone the repository**

```bash
git clone https://github.com/your-username/pdf-chat-groq-langchain.git
cd pdf-chat-groq-langchain
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate       # On Linux/Mac
venv\Scripts\activate.bat    # On Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure API key**

Create a `.env` file in the root folder and add your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

---

## ▶️ How to Run

```bash
streamlit run app.py
```

Then open the app in your browser (usually: http://localhost:8501)

---

## 💡 How It Works

1. Upload one or more PDF files through the UI
2. PDFs are parsed using LangChain's PyPDFLoader
3. Documents are split into chunks using `RecursiveCharacterTextSplitter`
4. Chunks are embedded using HuggingFace embeddings
5. The embeddings are stored in a FAISS vector store
6. When a user asks a question, relevant chunks are retrieved and sent to the Groq LLM for answering
7. The answer is displayed with response time

---

## 📦 Example `requirements.txt`

```
streamlit
python-dotenv
langchain
langchain-core
langchain-community
langchain-groq
faiss-cpu
huggingface-hub
```

Install them with:

```bash
pip install -r requirements.txt
```

---

## 🧪 Example Use Cases

- 📄 Legal document analysis
- 🧪 Research papers Q&A
- 📚 Academic textbook assistance
- 📈 Business reports and strategy papers
- 🛠 Technical manuals and specifications

---

## 📸 Screenshot

> *(Optional – you can upload your image to Imgur or GitHub and update the link below)*

![Screenshot](https://your-screenshot-link.com)

---

## 📚 References

- [Groq](https://console.groq.com/)
- [LangChain Documentation](https://docs.langchain.com/)
- [Streamlit](https://streamlit.io/)
- [HuggingFace Embeddings](https://huggingface.co/)
- [FAISS GitHub](https://github.com/facebookresearch/faiss)

---

## 📝 License

This project is licensed under the **MIT License**.  
Feel free to use, modify, and share with attribution.

---

## 🙋‍♂️ Author

**Nitin Pratap**  
📧 Email: your-email@example.com  
🔗 GitHub: [github.com/your-username](https://github.com/your-username)  
💼 LinkedIn: [linkedin.com/in/your-profile](https://linkedin.com/in/your-profile)

---
