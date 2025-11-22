import gradio as gr
from src.embedding import huggingface_embeddings
from src.vectorstore import load_vectorstore
from src.prompt import educational_prompt
from src.groq_chain import create_rag_chain
from dotenv import load_dotenv

load_dotenv()

# Initialize RAG chain
print("🔧 Loading AI model...")
embeddings = huggingface_embeddings(model_name="BAAI/bge-m3")
vectorstore = load_vectorstore(
    embeddings=embeddings,
    vectorstore_path="faiss_index"
)
prompt = educational_prompt()
chain = create_rag_chain(vectorstore=vectorstore, prompt=prompt)
print("✅ AI Tutor ready!")


def chat_interface(message, history):
    """
    Process user message and return AI response
    """
    try:
        response = chain.invoke({"input": message})
        return response["answer"]
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Sample questions for examples
examples = [
    "What is Pythagoras theorem?",
    "Explain photosynthesis process",
    "Who was Shivaji Maharaj?",
    "Define democracy and its types",
    "How to solve quadratic equations?"
]

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat_interface,
    title="📚 SSC Board AI Tutor",
    description="""
    Ask questions about your 9th & 10th grade Maharashtra State Board textbooks.
    
    **Subjects covered:** Mathematics, Science, History, Geography, English, Hindi, Marathi
    """,
    examples=examples,
    chatbot=gr.Chatbot(
        height=500,
        placeholder="<div style='text-align: center; padding: 50px;'><h3>👋 Hi! I'm your SSC Board AI Tutor</h3><p>Ask me anything about your textbooks!</p></div>"
    ),
    textbox=gr.Textbox(
        placeholder="Ask your question here... (e.g., What is photosynthesis?)",
        container=False,
        scale=7
    ),
)

if __name__ == "__main__":
    demo.launch(
        share=True,  # Set to True to create a public link
        server_name="127.0.0.1",
        server_port=7860
    )
