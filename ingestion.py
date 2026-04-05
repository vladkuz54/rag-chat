import hashlib
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import (Docx2txtLoader, PyPDFLoader,
                                                  TextLoader)
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = Path("./data")
DB_DIR = Path("./chroma_db")
COLLECTION_NAME = "rag-data"
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".doc", ".docx"}

DATA_DIR.mkdir(exist_ok=True)


def get_embedding_function() -> OpenAIEmbeddings:
    return OpenAIEmbeddings()


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_DIR),
        embedding_function=get_embedding_function(),
    )


def get_retriever(k: int = 3):
    return get_vectorstore().as_retriever(k=k)


def _resolve_file_path(file_path: str | Path) -> str:
    return str(Path(file_path).expanduser().resolve())


def _get_file_hash(file_path: str) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_documents(file_path: str | Path):
    """Load documents from PDF, TXT, MD, and DOCX files."""
    resolved_path = _resolve_file_path(file_path)

    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(resolved_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(resolved_path)
    elif ext in {".txt", ".md"}:
        loader = TextLoader(resolved_path, encoding="utf-8")
    elif ext in {".docx", ".doc"}:
        loader = Docx2txtLoader(resolved_path)
    else:
        raise ValueError(
            f"Unsupported file type: {ext}. Supported: .pdf, .txt, .md, .doc, .docx"
        )

    return loader.load()


def create_doc_splits(docs: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )

    return splitter.split_documents(docs)


def _prepare_documents(file_path: str | Path) -> tuple[list[Document], str]:
    resolved_path = _resolve_file_path(file_path)
    documents = load_documents(resolved_path)
    doc_splits = create_doc_splits(documents)
    file_hash = _get_file_hash(resolved_path)

    for index, document in enumerate(doc_splits):
        metadata = dict(document.metadata or {})
        metadata.update(
            {
                "source": resolved_path,
                "file_name": Path(resolved_path).name,
                "file_hash": file_hash,
                "chunk_index": index,
            }
        )
        document.metadata = metadata

    return doc_splits, file_hash


def _delete_existing_source_chunks(vectorstore: Chroma, source: str) -> None:
    existing_documents = vectorstore.get(where={"source": source})
    existing_ids = existing_documents.get("ids", [])

    if existing_ids:
        vectorstore.delete(ids=list(existing_ids))


def sync_file_to_vectorstore(file_path: str | Path) -> dict[str, Any]:
    resolved_path = _resolve_file_path(file_path)

    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if Path(resolved_path).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {Path(resolved_path).suffix.lower()}. Supported: .pdf, .txt, .md, .doc, .docx"
        )

    vectorstore = get_vectorstore()
    doc_splits, file_hash = _prepare_documents(resolved_path)

    existing_documents = vectorstore.get(where={"source": resolved_path})
    existing_ids = existing_documents.get("ids", [])
    existing_metadatas = existing_documents.get("metadatas", [])

    if existing_ids and existing_metadatas:
        existing_hash = existing_metadatas[0].get("file_hash")
        if existing_hash == file_hash:
            return {
                "source": resolved_path,
                "status": "unchanged",
                "chunks": len(existing_ids),
            }

    if existing_ids:
        _delete_existing_source_chunks(vectorstore, resolved_path)

    texts = [document.page_content for document in doc_splits]
    metadatas = [document.metadata for document in doc_splits]
    ids = [f"{file_hash[:16]}-{index}" for index in range(len(doc_splits))]

    vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    return {
        "source": resolved_path,
        "status": "indexed",
        "chunks": len(doc_splits),
    }


def sync_uploaded_file(uploaded_file: Any) -> dict[str, Any]:
    target_path = DATA_DIR / Path(uploaded_file.name).name

    if hasattr(uploaded_file, "getbuffer"):
        file_bytes = uploaded_file.getbuffer()
    else:
        file_bytes = uploaded_file.read()

    with open(target_path, "wb") as file_handle:
        file_handle.write(file_bytes)

    return sync_file_to_vectorstore(target_path)


def sync_data_directory(directory: Path | str = DATA_DIR) -> list[dict[str, Any]]:
    base_dir = Path(directory)
    results: list[dict[str, Any]] = []

    if not base_dir.exists():
        return results

    for file_path in base_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            results.append(sync_file_to_vectorstore(file_path))

    return results


def create_vectorstore(doc_splits):
    vectorstore = get_vectorstore()
    texts = [document.page_content for document in doc_splits]
    metadatas = [document.metadata for document in doc_splits]
    ids = [f"manual-{index}" for index in range(len(doc_splits))]

    vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return vectorstore


retriever = get_retriever()
