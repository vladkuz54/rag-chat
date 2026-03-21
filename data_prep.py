import os
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders import TextLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key_env_var="OPENAI_API_KEY",
    model_name="text-embedding-3-small"
)

client = chromadb.PersistentClient(path="./chroma_db")


collection = client.get_or_create_collection(name="rag", embedding_function=openai_ef)


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
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, .txt, .md, .doc, .docx")
    
    return loader.load()


file_path = "data/text.pdf"
document = load_documents(file_path)

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]

)
doc_splits = text_splitter.split_documents(document)

vector_store = Chroma.from_documents(
    documents=doc_splits, 
    collection_name="rag", 
    client=client,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small")
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})