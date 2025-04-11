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
            
            # Show the retrieved chunks for manual inspection (relevancy review)
            st.subheader("Retrieved Chunks for Relevancy Inspection:")
            for doc, score in results_with_scores:
                st.write("**Score:**", score)
                st.write(doc.page_content)
                st.write("---")

    # --- Dynamic Evaluation Section ---
    if st.sidebar.checkbox("Run Dynamic Evaluation"):
        st.sidebar.markdown("### Dynamic Evaluation")
        st.markdown("#### Generating Evaluation Queries Dynamically")

        # Step 1: Generate evaluation queries from the document's content.
        # We use a portion of the document (e.g., the first 1000 characters) as context.
        sample_text = docs[0].page_content[:1000]
        eval_prompt = (
            "Based on the following text, generate three comprehensive evaluation questions "
            "that cover the main aspects of this document. Ensure the questions capture key details "
            "and context of the content:\n\n" + sample_text
        )
        # Use the language model to generate evaluation questions.
        # (Note: The output is expected to be a list of questions separated by newlines.)
        generated_eval = llm.predict(eval_prompt)
        # Split the output into individual questions. (Assumes each question is on a new line.)
        dynamic_questions = [q.strip() for q in generated_eval.split("\n") if q.strip()]
        
        st.markdown("#### Evaluation Questions Generated:")
        for i, q in enumerate(dynamic_questions, start=1):
            st.write(f"**Q{i}: {q}**")
        
        # Step 2: Run the QA chain for each evaluation question and collect user feedback.
        st.markdown("#### Answer Evaluation and Feedback")
        feedback_scores = []
        for i, q in enumerate(dynamic_questions, start=1):
            st.write(f"**Question {i}:** {q}")
            eval_answer = qa_chain.run(q)
            st.write("Generated Answer:", eval_answer)
            # Ask user to provide a rating for the answer based on relevancy and context
            rating = st.slider(
                f"Rate the relevancy and contextual awareness for Q{i} (1 = Poor, 5 = Excellent)",
                min_value=1,
                max_value=5,
                value=3,
                key=f"rating_{i}"
            )
            feedback_scores.append(rating)
            st.write("---")

        if feedback_scores:
            avg_rating = sum(feedback_scores) / len(feedback_scores)
            st.markdown("### Dynamic Evaluation Summary")
            st.write(f"Average Rating: {avg_rating:.2f} out of 5")
            st.write("This score reflects overall performance based on your feedback.")