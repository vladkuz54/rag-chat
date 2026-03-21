# RAG Chat Bot

A sophisticated Retrieval-Augmented Generation (RAG) chat application that combines advanced retrieval techniques with LLM-powered responses.

## 🚀 Features

### **Hybrid Search (BM25 + Vector Search)**
Combines classical keyword search (BM25) with semantic vector search using Reciprocal Rank Fusion (RRF).
- **Why?** Finds documents by exact keywords AND semantic meaning
- Prevents missing specific information (day names, dates, codes, etc.)
- Tokenizes documents and builds keyword index for fast retrieval

**See:** [HYBRID_SEARCH.md](HYBRID_SEARCH.md) for detailed documentation.

### **Multi-Query Retrieval (Query Expansion)**
Generates 3-5 variations of user questions before retrieval to improve document coverage.
- Handles imprecisely worded queries
- Works in any language
- Ensures questions phrased differently still find relevant documents

**See:** [MULTI_QUERY_RETRIEVAL.md](MULTI_QUERY_RETRIEVAL.md) for detailed documentation.

## 📋 Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- OpenAI API key
- 2GB RAM minimum

## 🔧 Installation

### 1. Clone or Download Repository
```bash
cd /path/to/rag-chat
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

**Note:** Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)

## 🚀 Running the Application

### **Option 1: Web Chat Interface (Recommended)**

Start the Chainlit chat application:
```bash
chainlit run chat.py
```

This will open a web interface at `http://localhost:8000`

**Features:**
- 🎥 Live chat interface
- 📁 Upload multiple document formats (PDF, TXT, MD, DOCX)
- 💬 Ask questions about your documents
- 📊 See retrieval statistics and query variations
- 🔍 Real-time hybrid search with BM25 + Vector

### **Option 2: Python Script Testing**

Test the workflow without UI:
```bash
python main.py
```

**Note:** You must upload documents first via the chat interface before running the workflow.

## 📖 Usage Guide

### 1. Start the Application
```bash
chainlit run chat.py
```

### 2. Upload Documents
- Click the **+** button or drag-and-drop files
- Supported formats: PDF, TXT, MD, DOCX
- System will automatically:
  - Extract text from documents
  - Split into chunks (500 chars with 50-char overlap)
  - Create vector embeddings
  - Build BM25 keyword index

Wait for the confirmation message showing:
```
✅ Document ready! Added X fragments to database.
📊 Details:
- Original documents: X
- Fragments: X
- 🔍 Hybrid search activated (BM25 + Vector)
```

### 3. Ask Questions
Type your question in Ukrainian or English. The system will:
1. **Generate 3-5 question variations** (multi-query expansion)
2. **Search using hybrid method:**
   - BM25 search (exact keywords)
   - Vector search (semantic meaning)
   - Combine with RRF ranking
3. **Grade documents** for relevance
4. **Generate answer** using context
5. **Display results** with search statistics

### 4. View Results
Each response shows:
```
💡 **Answer:**
[Generated response based on documents]

---
🔍 **Query Variations (Multi-Query Retrieval):**
1. Original question
2. Variation 1
3. Variation 2
[...]

📊 Processing Details:
- Generations: X
- Query transforms: X
- Query variations: X
```

## 🛠️ Configuration

### Adjust Chunk Size
Edit `chat.py`:
```python
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500,      # Change this (larger = fewer, longer chunks)
    chunk_overlap=50,    # Overlap between chunks
    separators=["\n\n", "\n", ". ", " "]
)
```

### Change Retriever Parameters
Edit `chat.py`:
```python
retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # k = number of results
```

### Modify BM25 Settings
Edit `graph/hybrid_search.py`:
```python
hybrid_search(query, vector_retriever, k=5)  # Increase k for more results
```

### Adjust Query Variations
Edit `graders/multi_query_generator.py`:
```python
return clean_variations[:5]  # Decrease for fewer variations
```

## 📁 Project Structure

```
rag-chat/
├── chat.py                          # Main Chainlit chat interface
├── workflow.py                      # LangGraph workflow definition
├── data_prep.py                     # Document loading & preparation
├── main.py                          # Standalone test script
│
├── graders/                         # Grading & scoring modules
│   ├── graph_state.py              # Workflow state definition
│   ├── multi_query_generator.py    # Query expansion
│   ├── question_rewriter.py        # Query reformulation
│   ├── grade_chunk.py              # Document relevance grading
│   ├── grade_answer.py             # Answer quality grading
│   ├── grade_hallucination.py      # Fact-checking
│   └── response_generator.py       # Answer generation
│
├── graph/                           # Retrieval graph nodes
│   ├── retrieve.py                 # Multi-Query + Hybrid Search
│   ├── hybrid_search.py            # BM25 + Vector combination
│   ├── grade_documents.py          # Filter irrelevant docs
│   ├── generate.py                 # Generate answer
│   ├── transform_query.py          # Rewrite failed queries
│   ├── decide_to_generate.py       # Routing logic
│   └── generate_unknown.py         # Fallback response
│
├── chroma_db/                       # Vector database (auto-created)
├── data/                            # Uploaded documents folder
├── .env                             # Environment variables
├── requirements.txt                 # Python dependencies
├── README.md                     # This file
├── HYBRID_SEARCH.md                 # Hybrid search details
└── MULTI_QUERY_RETRIEVAL.md         # Multi-query details
```

## 🔍 How the RAG Pipeline Works

```
User Input
    ↓
Multi-Query Generator (3-5 variations)
    ↓
For Each Variation:
    ├─ BM25 Search (keyword-based)
    ├─ Vector Search (semantic)
    └─ RRF Ranking (combine results)
    ↓
Deduplicate Documents
    ↓
Grade Document Relevance
    ↓
Generate Answer (if docs found)
    ├─ Yes → LLM generates answer from context
    └─ No → Transform query & retry (or return unknown)
    ↓
Display Answer + Query Variations + Stats
```

## 🐛 Troubleshooting

### "Retriever not initialized"
**Problem:** No documents uploaded yet
**Solution:** Upload documents first via the chat interface

### BM25 index not building
**Problem:** Documents not indexed for keyword search
**Solution:** Restart the application and re-upload documents

### Slow retrieval
**Problem:** Hybrid search processes multiple queries
**Solution:** Reduce `k` parameter in configuration, or upload fewer/shorter documents

### Out of memory
**Problem:** Large documents + many chunks
**Solution:** Reduce `chunk_size` in chat.py or split documents into smaller files

### API rate limits
**Problem:** Too many API calls to OpenAI
**Solution:** Add delay between queries, upgrade to paid API plan

## 📊 Performance Tips

1. **Optimize document size:**
   - Smaller chunks = faster retrieval, but may split information
   - Larger chunks = slower but keeps context together

2. **Reduce variations if slow:**
   - Fewer query variations = faster, but less coverage
   - 3-4 variations = good balance

3. **Use specific queries:**
   - "What is X?" → Better than "Tell me about X"
   - More specific = better document matching

4. **Organize documents:**
   - Use clear structure (headings, sections)
   - Better organization = better retrieval

## 🔑 API Keys and Services

### OpenAI
- Models used:
  - `gpt-3.5-turbo`: Answer generation & query rewriting
  - `text-embedding-3-small`: Vector embeddings
- Cost: ~$0.01-0.10 per hundred documents

### Chroma
- Vector database (local, free)
- Stores embeddings in `chroma_db/` folder

## 🎓 Learning Resources

- [LangChain Documentation](https://python.langchain.com)
- [LangGraph Guide](https://langchain-ai.github.io/langgraph/)
- [OpenAI API](https://platform.openai.com/docs)
- [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)

## 📝 Example Queries

After uploading documents, try:
- "What is the main topic of the document?"
- "List the steps for [process name]"
- "When should I [action]?"
- "Who is responsible for [task]?"
- "What are the requirements for [item]?"

## 🤝 Contributing

To improve the system:
1. Try different `chunk_size` values
2. Experiment with more/fewer query variations
3. Test on different document types
4. Adjust prompt templates in `graders/`

## 📄 License

This project is provided as-is for educational and research purposes.

## ❓ FAQ

**Q: Can I use it offline?**
A: No, you need OpenAI API connection for embeddings and generation.

**Q: How many documents can I upload?**
A: Unlimited, but system performance depends on your hardware and API quota.

**Q: What languages are supported?**
A: Any language that OpenAI embeddings support (100+ languages).

**Q: How accurate are the answers?**
A: Depends on document quality. Better documents = better answers. System grades answers for hallucinations.

**Q: Can I modify responses?**
A: Yes, edit the prompts in `graders/` files.

## 📞 Support

For issues:
1. Check `.env` file has valid OpenAI API key
2. Verify documents are uploaded (check `data/` folder)
3. Check terminal for error messages
4. Ensure all dependencies installed: `pip install -r requirements.txt`

---

**Last Updated:** March 2026
**Python Version:** 3.9+
**Status:** Active Development
