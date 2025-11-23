import streamlit as st
from src.embedding import huggingface_embeddings
from src.vectorstore import load_vectorstore
from src.prompt import educational_prompt
from src.groq_chain import create_rag_chain
from dotenv import load_dotenv
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from io import BytesIO

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

if "selected_subject" not in st.session_state:
    st.session_state.selected_subject = "All Subjects"

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

    # Export Chat History Feature (PDF Only)
    if st.session_state.messages:
        st.divider()
        st.subheader("📥 Export Chat")

        if st.button("📄 Download Chat as PDF"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create PDF in memory
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter,
                                    rightMargin=50, leftMargin=50,
                                    topMargin=50, bottomMargin=50)

            # Container for PDF elements
            elements = []

            # Define styles with better readability
            styles = getSampleStyleSheet()

            # Title style
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=20,
                spaceAfter=20,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            )

            # Message number style
            heading_style = ParagraphStyle(
                'MessageHeading',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=6,
                spaceBefore=12,
                fontName='Helvetica-Bold',
                textColor='green'
            )

            # User message style
            user_style = ParagraphStyle(
                'UserMessage',
                parent=styles['Normal'],
                fontSize=10,
                leading=14,
                spaceAfter=10,
                leftIndent=15,
                fontName='Helvetica'
            )

            # Assistant message style
            assistant_style = ParagraphStyle(
                'AssistantMessage',
                parent=styles['Normal'],
                fontSize=10,
                leading=14,
                spaceAfter=10,
                leftIndent=15,
                fontName='Helvetica'
            )

            # Role label style
            role_style = ParagraphStyle(
                'RoleLabel',
                parent=styles['Normal'],
                fontSize=10,
                fontName='Helvetica-Bold',
                spaceAfter=4
            )

            # Add title
            elements.append(
                Paragraph("SSC Board AI Tutor - Chat History", title_style))
            elements.append(Spacer(1, 12))

            # Add metadata
            metadata = f"<b>Exported:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
            metadata += f"<b>Total Messages:</b> {len(st.session_state.messages)}<br/>"
            metadata += f"<b>Subject Filter:</b> {st.session_state.selected_subject}"
            elements.append(Paragraph(metadata, styles['Normal']))
            elements.append(Spacer(1, 20))

            # Helper function to clean text for PDF
            def clean_text_for_pdf(text):
                """Clean text by removing problematic characters and formatting"""
                # Remove or replace special unicode characters
                replacements = {
                    '&': '&amp;',
                    '<': '&lt;',
                    '>': '&gt;',
                    '"': '&quot;',
                    "'": '&apos;',
                    '\u2019': "'",  # Smart quote
                    '\u2018': "'",  # Smart quote
                    '\u201c': '"',  # Smart double quote
                    '\u201d': '"',  # Smart double quote
                    '\u2013': '-',  # En dash
                    '\u2014': '--',  # Em dash
                    '\u2026': '...',  # Ellipsis
                    '•': '*',       # Bullet
                    '–': '-',
                    '—': '--',
                    ''': "'",
                    ''': "'",
                    '"': '"',
                    '"': '"',
                }

                for old, new in replacements.items():
                    text = text.replace(old, new)

                # Remove LaTeX expressions that don't render well
                import re
                # Replace inline math $...$ with [Math expression]
                text = re.sub(r'\$([^\$]+)\$', r'[\1]', text)
                # Replace display math $$...$$ with [Math expression]
                text = re.sub(r'\$\$([^\$]+)\$\$', r'[Math: \1]', text)
                # Clean up \frac, \times, etc.
                text = re.sub(
                    r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1/\2)', text)
                text = re.sub(r'\\times', 'x', text)
                text = re.sub(r'\\[a-zA-Z]+\{([^}]+)\}', r'\1', text)
                text = re.sub(r'\\[a-zA-Z]+', '', text)

                return text

            # Add conversations
            for i, msg in enumerate(st.session_state.messages, 1):
                # Add message number
                elements.append(Paragraph(f"Message {i}", heading_style))

                if msg['role'] == 'user':
                    elements.append(Paragraph("Student:", role_style))
                    # Clean and format content
                    content = clean_text_for_pdf(msg['content'])
                    elements.append(Paragraph(content, user_style))
                else:
                    elements.append(Paragraph("AI Tutor:", role_style))
                    # Clean and format content
                    content = clean_text_for_pdf(msg['content'])

                    # Split long content into paragraphs for better readability
                    paragraphs = content.split('\n')
                    for para in paragraphs:
                        if para.strip():
                            elements.append(
                                Paragraph(para.strip(), assistant_style))

                    # Add source info if available
                    if 'sources' in msg and msg['sources']:
                        source_text = f"<i>Sources: {len(msg['sources'])} documents referenced</i>"
                        elements.append(
                            Paragraph(source_text, assistant_style))

                elements.append(Spacer(1, 10))

            # Build PDF
            doc.build(elements)

            # Get PDF data
            pdf_data = buffer.getvalue()
            buffer.close()

            # Provide download button
            st.download_button(
                label="💾 Download PDF",
                data=pdf_data,
                file_name=f"chat_history_{timestamp}.pdf",
                mime="application/pdf",
                type="primary"
            )

    st.divider()

    # Display chat count
    if st.session_state.messages:
        st.write(f"📊 Total messages: {len(st.session_state.messages)}")

    st.divider()

    # Subject Filter Feature
    st.subheader("📚 Subject Filter")
    subjects = [
        "All Subjects",
        "Mathematics",
        "Science",
        "History",
        "Geography",
        "English",
        "Hindi",
        "Marathi"
    ]

    selected_subject = st.selectbox(
        "Filter by subject:",
        subjects,
        index=subjects.index(st.session_state.selected_subject)
    )

    if selected_subject != st.session_state.selected_subject:
        st.session_state.selected_subject = selected_subject
        st.info(f"🎯 Filter set to: {selected_subject}")

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

# Quick Question Templates Feature
st.divider()
st.subheader("⚡ Quick Question Templates")

# Define templates by subject
question_templates = {
    "Mathematics": [
        "Explain the Pythagorean theorem with an example",
        "What is the formula for the area of a circle?",
        "How do you solve quadratic equations?",
        "What are the properties of triangles?"
    ],
    "Science": [
        "Explain the process of photosynthesis",
        "What are Newton's laws of motion?",
        "What is the difference between acids and bases?",
        "How does the human digestive system work?"
    ],
    "History": [
        "Who was Shivaji Maharaj?",
        "What was the Indian Independence Movement?",
        "Explain the significance of the French Revolution",
        "What were the causes of World War I?"
    ],
    "Geography": [
        "What are the different types of rocks?",
        "Explain the water cycle",
        "What causes earthquakes?",
        "What are the major rivers of India?"
    ],
    "English": [
        "What is a metaphor? Give examples",
        "Explain the difference between active and passive voice",
        "What are the parts of speech?",
        "How do you write a good essay?"
    ],
    "General": [
        "Summarize the main topics in this chapter",
        "Give me practice questions on this topic",
        "Explain this concept in simple terms",
        "What are the important points to remember?"
    ]
}

# Show templates based on selected subject
if st.session_state.selected_subject == "All Subjects":
    template_subject = "General"
elif st.session_state.selected_subject in question_templates:
    template_subject = st.session_state.selected_subject
else:
    template_subject = "General"

# Display templates as clickable buttons in columns
cols = st.columns(2)
for idx, template in enumerate(question_templates.get(template_subject, [])):
    col = cols[idx % 2]
    if col.button(f"💡 {template}", key=f"template_{idx}", use_container_width=True):
        # Set the template as the query
        st.session_state.template_query = template
        st.rerun()

st.divider()

# Chat input
if query := st.chat_input("Enter your question here..."):
    # Process the query
    process_query = query
elif "template_query" in st.session_state:
    # Process template query
    process_query = st.session_state.template_query
    del st.session_state.template_query
else:
    process_query = None

if process_query:
    # Add subject context if filter is active
    if st.session_state.selected_subject != "All Subjects":
        query_with_context = f"[Subject: {st.session_state.selected_subject}] {process_query}"
    else:
        query_with_context = process_query

    # Add user message to chat history
    st.session_state.messages.append(
        {"role": "user", "content": process_query})

    # Display user message
    with st.chat_message("user"):
        st.markdown(process_query)

    # Generate response with error handling
    with st.chat_message("assistant"):
        try:
            with st.spinner("🤔 Thinking..."):
                # Invoke chain with conversation memory
                response = chain.invoke({
                    "input": query_with_context,
                    "chat_history": st.session_state.chat_history
                })

                # Extract answer and context
                answer = response.get(
                    "answer", "I couldn't generate a response.")
                context_docs = response.get("context", [])

                # Update conversation memory
                from langchain_core.messages import HumanMessage, AIMessage
                st.session_state.chat_history.append(
                    HumanMessage(content=process_query))
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
