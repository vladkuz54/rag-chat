# RAG Chat Application

A retrieval-augmented generation (RAG) chatbot built with LangChain, LangGraph, and Streamlit. This application allows you to upload documents and ask questions about them using an AI-powered chat interface with intelligent document retrieval and grading.

## Features

- **Document Indexing**: Upload and index PDF, TXT, MD, and DOCX files
- **Multi-stage Grading**: 
  - Retrieval relevance grading
  - Document quality evaluation
  - Hallucination detection
  - Answer validation
- **Debug Visibility**: See the entire RAG pipeline execution with document counts, transformations, and reasoning
- **Incremental Updates**: Automatic detection of file changes to avoid re-indexing unchanged documents

## Installation

### Using `uv` (Recommended - Faster)

1. **Install `uv` if you don't have it:**
   ```bash
   # On Windows (via pip)
   pip install uv
   
   # Or use the official installer from https://docs.astral.sh/uv/
   ```

2. **Clone the repository and navigate to the project directory:**
   ```bash
   cd rag-chat
   ```

3. **Create and activate a virtual environment:**
   ```bash
   uv venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```

### Using `pip` (Standard Approach)

1. **Navigate to the project directory:**
   ```bash
   cd rag-chat
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Before running the application, ensure you have:

1. **OpenAI API Key**: Set the `OPENAI_API_KEY` environment variable
   ```bash
   # On Windows (PowerShell)
   $env:OPENAI_API_KEY = "your-key-here"
   
   # On Windows (CMD)
   set OPENAI_API_KEY=your-key-here
   
   # On macOS/Linux
   export OPENAI_API_KEY="your-key-here"
   ```

2. **Data Directory**: Create a `data/` folder in the project root for storing documents to index

## Usage

### Starting the Application

With your virtual environment activated, run:

```bash
streamlit run chat.py
```

This will open the application in your browser (default: `http://localhost:8501`)

### How to Use

#### 1. **Upload Documents** (Sidebar)
   - Click "Browse files" in the left sidebar
   - Select a PDF, TXT, MD, or DOCX file from your computer
   - Supported formats: `.pdf`, `.txt`, `.md`, `.docx`

#### 2. **Index/Update File** (Sidebar)
   - After selecting a file, click the **"Index / Update File"** button
   - The application will:
     - Check if the file has already been indexed
     - Skip if unchanged (hash comparison)
     - Add new chunks to the vector database if changed
   - You'll see a status message: "✅ File indexed successfully" or "ℹ️ File unchanged (already indexed)"

#### 3. **Ask Questions** (Main Chat Area)
   - Type your question in the chat input field
   - Press Enter or click Send
    - The chatbot will:
       - Retrieve relevant document chunks using hybrid search
     - Grade their relevance
     - Generate an answer based on the documents
     - Validate the answer for hallucinations

#### 4. **View Debug Information**
   - Expand the **"Debug Info"** section to see:
     - Retrieved document count
     - Transformation iterations (if documents were rephrased)
     - Answer length
     - Per-node execution details
   - Toggle debug visibility with the **"Show debug info"** checkbox

#### 5. **Clear Chat History**
   - Click **"Clear chat"** to start a fresh conversation

### Example Workflow

```
1. Upload HR_Policy.pdf via sidebar
2. Click "Index / Update File"
3. Ask: "What is the vacation policy?"
4. View the AI response with debug trace
5. Ask: "How many days can I take off?"
6. Toggle debug info to see which documents were used
7. Continue asking follow-up questions
```

## Project Structure

```
rag-chat/
├── chat.py                 # Streamlit web interface
├── main.py                 # Core RAG graph execution
├── ingestion.py            # Document indexing pipeline
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project configuration
├── data/                   # Directory for uploading documents
├── chroma_db/              # Vector database storage (auto-created)
└── graph/
    ├── __init__.py
    ├── consts.py          # Constants and configuration
    ├── graph.py           # LangGraph pipeline definition
    ├── state.py           # Application state schema
    ├── chains/            # LLM chain prompts
    │   ├── answer_grader.py
    │   ├── generation.py
    │   ├── generic_response.py
    │   ├── hallucination_grader.py
    │   ├── retrieval_grader.py
    │   └── transformation.py
    └── nodes/             # Graph execution nodes
        ├── generate.py
        ├── generic_response.py
        ├── grade_documents.py
        ├── retrieve.py
        ├── transform.py
        └── web_search.py
```

## Troubleshooting

### "OpenAI API Key not found"
- Ensure your `OPENAI_API_KEY` environment variable is set
- Restart the terminal after setting the variable

### Streamlit doesn't start
- Verify dependencies are installed: `pip list | grep streamlit`
- Try: `pip install --upgrade streamlit`

### Documents not indexed
- Check that the file is in a supported format
- Verify the `data/` directory exists
- Check the debug log for import errors

### No documents retrieved for a question
- Ensure you've indexed at least one document
- Try rephrasing your question more specifically
- Check debug info to see if any documents were retrieved

## Requirements

- Python 3.9+
- OpenAI API key (for embeddings and GPT models)
- 4GB RAM minimum
- Internet connection for API calls

## Technologies Used

- **LangChain** - LLM framework and document processing
- **LangGraph** - Multi-step reasoning and orchestration
- **Chroma** - Vector database for semantic search
- **Streamlit** - Web interface framework
- **OpenAI** - Embeddings and language models

## License

This project is provided as-is for educational and commercial use.
