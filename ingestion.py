from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = Path("./data")
DB_DIR = Path("./chroma_db")
DATA_DIR.mkdir(exist_ok=True)

loader = TextLoader("./data/test.txt", encoding="utf-8")

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
)

doc_splits = splitter.split_documents(docs)


if not DB_DIR.exists():
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        embedding=OpenAIEmbeddings(),
        persist_directory="./chroma_db",
        collection_name="rag-data"
    )


retriever = Chroma(
    collection_name="rag-data",
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings(),
).as_retriever(k=3)

