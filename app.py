import streamlit as st
from src.embedding import huggingface_embeddings
from src.vectorstore import load_vectorstore
from src.prompt import educational_prompt
from src.groq_chain import create_rag_chain
from dotenv import load_dotenv
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="SSC Board AI Tutor",
    page_icon="📚"
)

st.title("📚 SSC Board AI Tutor")
st.write("Ask questions about your 9th & 10th grade textbooks")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Cache model loading to prevent reloading on every interaction


@st.cache_resource(show_spinner=False)
def load_models():
    """Load and cache all models and chains"""
    try:
        with st.spinner("🔧 Loading AI model... (This may take a minute on first run)"):
            # Load embeddings
            embeddings = huggingface_embeddings(model_name="BAAI/bge-m3")

            # Load vectorstore
            vectorstore = load_vectorstore(
                embeddings=embeddings,
                vectorstore_path="faiss_index"
            )

            # Create prompt with conversation memory support
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert educational AI tutor specializing in Maharashtra SSC Board curriculum for 9th and 10th grade students.

Your role is to help students understand concepts from their textbooks across subjects like Mathematics, Science, History, Geography, and Languages.

Guidelines:
- Provide clear, accurate answers based on the context provided
- Break down complex topics into simple, understandable explanations
- Use examples when helpful for clarity
- If the question involves problem-solving (math/science), show step-by-step solutions
- Cite the subject/topic when relevant (e.g., "According to the 9th/10th Science textbook...")
- If the answer is not found in the context, respond: "I don't have this information in the available textbooks. Please ask about topics covered in your SSC curriculum."
- Maintain an encouraging, patient tone suitable for students
- Consider the conversation history to provide contextual responses

Context from textbooks:
{context}"""),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ])

            # Create RAG chain
            chain = create_rag_chain(vectorstore=vectorstore, prompt=prompt)

            return chain, vectorstore

    except FileNotFoundError:
        st.error(
            "❌ Vector database not found! Please run the data ingestion process first.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()


# Load models with error handling
try:
    chain, vectorstore = load_models()
    st.success("✅ AI Tutor ready!", icon="🎓")
except Exception as e:
    st.error(f"Failed to initialize: {str(e)}")
    st.stop()

# Sidebar for chat history and controls
with st.sidebar:
    st.header("💬 Chat History")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    # Display chat count
    if st.session_state.messages:
        st.write(f"📊 Total messages: {len(st.session_state.messages)}")

    st.divider()
    st.write("**Tips:**")
    st.write("• Ask follow-up questions")
    st.write("• Request explanations")
    st.write("• Ask for examples")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display sources if available
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 View Source Documents"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}:**")
                    st.text(
                        source["content"][:300] + "..." if len(source["content"]) > 300 else source["content"])
                    if "source" in source["metadata"]:
                        st.caption(f"📄 From: {source['metadata']['source']}")
                    if "page" in source["metadata"]:
                        st.caption(f"📖 Page: {source['metadata']['page']}")
                    st.divider()

# Chat input
if query := st.chat_input("Enter your question here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})

    # Display user message
    with st.chat_message("user"):
        st.markdown(query)

    # Generate response with error handling
    with st.chat_message("assistant"):
        try:
            with st.spinner("🤔 Thinking..."):
                # Invoke chain with conversation memory
                response = chain.invoke({
                    "input": query,
                    "chat_history": st.session_state.chat_history
                })

                # Extract answer and context
                answer = response.get(
                    "answer", "I couldn't generate a response.")
                context_docs = response.get("context", [])

                # Update conversation memory
                from langchain_core.messages import HumanMessage, AIMessage
                st.session_state.chat_history.append(
                    HumanMessage(content=query))
                st.session_state.chat_history.append(AIMessage(content=answer))

                # Display answer
                st.markdown(answer)

                # Prepare sources for storage
                sources = []
                if context_docs:
                    for doc in context_docs:
                        sources.append({
                            "content": doc.page_content,
                            "metadata": doc.metadata
                        })

                    # Display sources in expander
                    with st.expander("📚 View Source Documents"):
                        for i, doc in enumerate(context_docs, 1):
                            st.markdown(f"**Source {i}:**")
                            st.text(
                                doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                            if hasattr(doc, 'metadata'):
                                if "source" in doc.metadata:
                                    st.caption(
                                        f"📄 From: {doc.metadata['source']}")
                                if "page" in doc.metadata:
                                    st.caption(
                                        f"📖 Page: {doc.metadata['page']}")
                            st.divider()

                # Add assistant message to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

        except Exception as e:
            error_msg = f"❌ Error generating response: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })

# Footer
st.divider()
st.caption("💡 Tip: You can ask follow-up questions based on previous answers!")
