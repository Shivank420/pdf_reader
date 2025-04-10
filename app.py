import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

st.title("Ask Questions from PDF using LangChain RAG")

# --- PDF Upload Section ---
pdf_file = st.file_uploader("Upload a PDF", type="pdf")

if pdf_file:
    # Save the uploaded PDF temporarily so that PyPDFLoader can read it from disk
    pdf_bytes = pdf_file.read()
    temp_pdf_path = "temp.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_bytes)

    # --- Step 1: Load the PDF Document ---
    loader = PyPDFLoader(temp_pdf_path)
    documents = loader.load()

    # --- Step 2: Split the Document into Chunks ---
    # Chunk size of 500 characters with a 50-character overlap.
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # --- Step 3: Create the Vector Store (Embeddings Under the Hood) ---
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_documents(docs, embeddings)

    # --- Step 4: Setup QA Retrieval Chain ---
    # Build a retriever and then the RetrievalQA chain.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.7, max_tokens=150)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # --- Step 5: Ask a Question ---
    question = st.text_input("Ask a question about the PDF:")
    if question:
        # Use similarity_search_with_score to get documents and their distance scores.
        results_with_scores = vectorstore.similarity_search_with_score(question, k=3)
        # Extract the lowest distance (best match).
        min_score = min(score for _, score in results_with_scores)
        # Optional: display the score for debugging.
        st.write("Similarity score (lower is better):", min_score)

        # Set a threshold for what is considered “related.”
        # NOTE: Because we are using L2 distance (with FAISS's IndexFlatL2), lower values mean more relevant.
        # You might need to adjust this threshold based on your data and model.
        THRESHOLD = 40.0
        if min_score > THRESHOLD:
            st.write("Sorry your question is not related to the given PDF.")
        else:
            # If at least one retrieved chunk is sufficiently similar, run the QA chain.
            answer = qa_chain.run(question)
            st.write("Answer:", answer)
