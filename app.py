import streamlit as st
from src.embedding import huggingface_embeddings
from src.vectorstore import load_vectorstore
from src.prompt import educational_prompt
from src.groq_chain import create_rag_chain
from dotenv import load_dotenv

load_dotenv()

st.title("📚 SSC Board AI Tutor")
st.write("Ask questions about your 9th & 10th grade textbooks")

# Load model once
print("🔧 Loading AI model...")
embeddings = huggingface_embeddings(model_name="BAAI/bge-m3")

vectorstore = load_vectorstore(
    embeddings=embeddings,
    vectorstore_path="faiss_index"
)

prompt = educational_prompt()


chain = create_rag_chain(vectorstore=vectorstore, prompt=prompt)
print("✅ AI Tutor ready!")

# Simple input
query = st.text_input("Enter your question:")

if st.button("Ask") and query:
    with st.spinner("Thinking..."):
        response = chain.invoke({"input": query})
        st.write("**Answer:**")
        st.write(response["answer"])


