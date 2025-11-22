import streamlit as st
from src.embedding import huggingface_embeddings
from src.vectorstore import load_vectorstore
from src.prompt import educational_prompt
from src.groq_chain import create_rag_chain
from dotenv import load_dotenv

load_dotenv()

# Page config
st.set_page_config(
    page_title="SSC Board AI Tutor",
    page_icon="📚",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stTextInput > label {
        font-size: 1.1rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📚 SSC Board AI Tutor</h1>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about your 9th & 10th grade textbooks</p>',
            unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    with st.spinner("🔧 Loading AI model... Please wait"):
        try:
            embeddings = huggingface_embeddings(model_name="BAAI/bge-m3")
            vectorstore = load_vectorstore(
                embeddings=embeddings,
                vectorstore_path="faiss_index"
            )
            prompt = educational_prompt()
            st.session_state.chain = create_rag_chain(
                vectorstore=vectorstore, prompt=prompt)
            st.success("✅ AI Tutor ready!")
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if query := st.chat_input("Ask your question here..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                response = st.session_state.chain.invoke({"input": query})
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg})

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.info(
        """
        **SSC Board AI Tutor** helps you understand concepts from Maharashtra State Board textbooks.
        
        **Subjects covered:**
        - 📐 Mathematics
        - 🔬 Science
        - 📖 History
        - 🌍 Geography
        - 🗣️ Languages (English, Hindi, Marathi)
        """
    )

    st.header("💡 Sample Questions")
    sample_questions = [
        "What is Pythagoras theorem?",
        "Explain photosynthesis",
        "Who was Shivaji Maharaj?",
        "Define democracy",
        "How to solve quadratic equations?"
    ]

    for q in sample_questions:
        if st.button(q, key=q):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Made with ❤️ for SSC students")
