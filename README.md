# SSC Board RAG System 📚

An AI-powered Question-Answering system for Maharashtra SSC Board students (9th & 10th grade). Ask questions about your textbooks and get instant, accurate answers powered by RAG (Retrieval-Augmented Generation).

## 🎯 Features

- **Multi-Subject Coverage**: Mathematics, Science, History, Geography, English, Hindi, Marathi, and Defence
- **Intelligent Retrieval**: Uses FAISS vector database with semantic search
- **Dual LLM Support**: Choose between Ollama (local, privacy-focused) or Groq (cloud-based, fast)
- **Step-by-Step Explanations**: Breaks down complex topics for better understanding
- **GPU Accelerated**: Fast embeddings with CUDA support
- **📥 Export Chat History**: Download conversations as professional PDF documents
- **📚 Subject Filters**: Filter questions by specific subjects for focused learning
- **⚡ Quick Question Templates**: Pre-built questions for instant learning across all subjects
- **💬 Conversational Memory**: Maintains context across multiple questions for natural dialogue
- **🎨 Beautiful Streamlit UI**: Modern, intuitive interface with chat history management

## 📋 Prerequisites

- Python 3.12+
- [Ollama](https://ollama.ai/) installed and running
- CUDA-capable GPU (optional, but recommended)

## 🚀 Installation

1. **Clone the repository**

```bash
git clone <your-repo-url>
cd ssc-board-rag
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
# This will install: langchain, streamlit, faiss-cpu, reportlab, torch, and more
```

Or install individually:

```bash
pip install langchain langchain-classic langchain-core langchain-groq langchain-ollama
pip install langchain-huggingface langchain-community
pip install faiss-cpu  # or faiss-gpu for CUDA support
pip install pymupdf python-dotenv torch sentence-transformers
pip install streamlit reportlab
```

3. **Install and start Ollama**

```bash
# Download from https://ollama.ai/
ollama pull deepseek-v3.1:671b-cloud
ollama serve
```

4. **Create `.env` file** (if needed for API keys)

```bash
# Add any environment variables here
```

## 📂 Project Structure

```
ssc-board-rag/
├── data/                      # SSC Board textbooks (PDFs)
│   ├── 9th-maths-1.pdf
│   ├── 10th-science-1.pdf
│   └── ...
├── faiss_index/               # Pre-built vector database
│   └── index.faiss
├── src/
│   ├── dataloader.py         # Load PDFs, DOCs, web pages
│   ├── datasplitter.py       # Chunk documents
│   ├── embedding.py          # Generate embeddings
│   ├── vectorstore.py        # FAISS operations
│   ├── prompt.py             # LLM prompts
│   └── ollama_chain.py       # RAG chain setup
├── notebooks/
│   └── dataloader-datasplitter.ipynb
├── main.py                   # CLI interface
└── README.md
```

## 💻 Usage

### Quick Start (Streamlit UI)

Run the Streamlit web application:

```bash
streamlit run app.py
```

Then interact with the AI tutor through the beautiful web interface:

- **Chat with AI**: Ask questions in the chat input
- **Use Quick Templates**: Click pre-built question buttons for instant queries
- **Filter by Subject**: Select specific subjects (Math, Science, History, etc.)
- **Export Chat**: Download your conversation history as a PDF
- **View Sources**: See which textbook sections were used for answers

### CLI Interface

For command-line usage:

```bash
python main.py
```

Then ask questions:

```text
Enter your query: What is Pythagoras theorem?
Enter your query: Explain photosynthesis
Enter your query: Who was Shivaji Maharaj?
```

Press `Ctrl+C` to exit.

### Building Vector Database (First Time Setup)

If you need to rebuild the vector database from scratch:

```python
from src.dataloader import process_all_pdfs
from src.datasplitter import split_docs
from src.embedding import huggingface_embeddings
from src.vectorstore import create_vectorstore

# Load PDFs
documents = process_all_pdfs("data")

# Split into chunks
chunks = split_docs(documents)

# Generate embeddings
embeddings = huggingface_embeddings(model_name="BAAI/bge-m3")

# Create and save vectorstore
vectorstore = create_vectorstore(chunks, embeddings)
```

## 🔧 Configuration

### Change LLM Model

**For Ollama (Local):**

Edit `src/ollama_chain.py`:

```python
llm = ChatOllama(
    model="llama3.1:8b",  # Change model here
    temperature=0.2
)
```

**For Groq (Cloud):**

Edit `src/groq_chain.py` or set environment variable:

```bash
# In .env file
GROQ_API_KEY=your_api_key_here
```

### Adjust Retrieval Settings

Edit `src/ollama_chain.py` or `src/groq_chain.py`:

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",        # or "similarity"
    search_kwargs={"k": 7}    # number of results (default: 7)
)
```

### Modify Chunk Size

Edit `src/datasplitter.py`:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,      # Adjust chunk size (default: 1200)
    chunk_overlap=200     # Adjust overlap (default: 200)
)
```

## 🎨 New Features

### 📥 Export Chat History

- Click "📄 Download Chat as PDF" button in the sidebar
- Professional PDF format with clean typography
- Includes timestamps, message count, and subject filter info
- Auto-cleans LaTeX and special characters for readability

### 📚 Subject Filters

- Select from 8 subjects in the sidebar dropdown
- Filters: All Subjects, Mathematics, Science, History, Geography, English, Hindi, Marathi
- Contextual queries adapt to selected subject

### ⚡ Quick Question Templates

- Pre-built questions for each subject
- Two-column layout for easy access
- Instant query generation with one click
- Templates include: formulas, concepts, historical events, and more

## 🧪 Advanced Usage

### Add Web Content

```python
from src.dataloader import load_all_data

urls = [
    "https://example.com/article1",
    "https://example.com/article2"
]

all_docs = load_all_data("data", urls=urls)
```

### Process Different File Types

```python
from src.dataloader import process_all_texts, process_all_word_docs

# Load text files
txt_docs = process_all_texts("data/notes")

# Load Word documents
word_docs = process_all_word_docs("data/assignments")
```

## 🛠️ Troubleshooting

### Issue: "CUDA out of memory"

**Solution**: Use CPU mode by editing `src/embedding.py`:

```python
device = 'cpu'  # Force CPU usage
```

### Issue: "Ollama connection refused"

**Solution**: Ensure Ollama is running:

```bash
ollama serve
```

### Issue: "Model not found"

**Solution**: Pull the model first:

```bash
ollama pull deepseek-v3.1:671b-cloud
```

### Issue: Slow response time

**Solution**:

- Reduce `k` value in retriever settings
- Use smaller LLM model
- Enable GPU acceleration

## 📊 System Requirements

### Minimum

- **RAM**: 8 GB
- **Storage**: 5 GB free space
- **CPU**: 4 cores

### Recommended

- **RAM**: 16 GB+
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **Storage**: 10 GB+ SSD
- **CPU**: 8+ cores

## 🤝 Contributing

Contributions are welcome! To add more subjects or improve the system:

1. Add PDFs to `data/` folder
2. Rebuild the vector database
3. Test with sample questions
4. Submit a pull request

## 📝 License

This project is for educational purposes. Textbook content belongs to Maharashtra State Board.

## 🙏 Acknowledgments

- **LangChain**: RAG framework
- **HuggingFace**: Embedding models (BAAI/bge-m3)
- **Ollama**: Local LLM inference
- **Groq**: Cloud-based LLM API
- **Meta FAISS**: Vector search
- **ReportLab**: PDF generation
- **Streamlit**: Web UI framework
- **Maharashtra State Board**: Educational content

## 📧 Support

For questions or issues, please open a GitHub issue or contact the maintainer.

---

Made with ❤️ for SSC Board students
