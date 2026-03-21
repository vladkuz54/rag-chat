import os
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key_env_var="OPENAI_API_KEY",
    model_name="text-embedding-3-small"
)

client = chromadb.PersistentClient(path="./chroma_db")

# Global retriever (will be set when files are uploaded in chat)
_global_retriever = None

def set_retriever(retriever):
    """Set the global retriever"""
    global _global_retriever
    _global_retriever = retriever

def get_retriever():
    """Get the current retriever"""
    global _global_retriever
    if _global_retriever is None:
        raise RuntimeError("Retriever not initialized. Please upload documents first.")
    return _global_retriever


def load_documents(file_path: str):
    """
    Load documents from various file formats (PDF, TXT, MD, DOCX).
    
    Args:
        file_path: Path to the file
        
    Returns:
        List of loaded documents
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".md":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext in {".docx", ".doc"}:
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, .txt, .md, .doc, .docx")
    
    return loader.load()



# This section is commented out - file loading is now handled by chat.py
# Uncomment below to manually load data
# if __name__ == "__main__":
#     file_path = "data/text.pdf"
#     document = load_documents(file_path)
#     
#     text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#         chunk_size=250,
#         chunk_overlap=50,
#         separators=["\n\n", "\n", ". ", " "]
#     )
#     doc_splits = text_splitter.split_documents(document)
#     
#     vector_store = Chroma.from_documents(
#         documents=doc_splits, 
#         collection_name="rag", 
#         client=client,
#         embedding=OpenAIEmbeddings(model="text-embedding-3-small")
#     )
#     
#     retriever = vector_store.as_retriever(search_kwargs={"k": 3})