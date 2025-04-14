# 📄 Ask Questions from PDF using LangChain RAG

This Streamlit application allows you to upload a PDF and interact with its content using **LangChain's Retrieval-Augmented Generation (RAG)** technique. It leverages OpenAI's language models and FAISS for semantic search and retrieval.

---

## 🔧 Features

- Upload a PDF and extract its content
- Chunk the document for semantic search
- Use OpenAI embeddings + FAISS for similarity-based retrieval
- Ask natural language questions and get contextual answers
- Show similarity scores and document snippets
- Dynamically generate evaluation questions
- Collect human feedback to evaluate model performance

---

## 🚀 How It Works

1. **PDF Upload**  
   Upload a PDF file using the file uploader.

2. **Text Extraction & Chunking**  
   The PDF is parsed using `PyPDFLoader` and split into overlapping text chunks.

3. **Embedding + Vector Store**  
   The text chunks are embedded using OpenAI and stored in a FAISS vector store.

4. **Question Answering**  
   You can ask questions about the uploaded PDF. The app retrieves the most relevant chunks and feeds them into the LLM to answer.

5. **Dynamic Evaluation (Optional)**  
   Generate evaluation questions from the PDF and rate how well the system answers them.

---

## 🛠️ Requirements

- Python 3.8+
- OpenAI API key

---

## 🧪 Installation

1. **Clone the repository**

```bash
git clone https://github.com/Shivank420/pdf_reader.git
