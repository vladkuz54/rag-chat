from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.documents import Document

from preprocessing import DATA_DIR, DOCSTORE_DIR
from preprocessing.parser import (
    child_splitter,
    docstore,
    parent_splitter,
    split_with_header_path,
    transform_into_markdown,
    vectorstore,
)
from preprocessing.scraper import download_pdf_from_dbn_page


def _get_source_url_from_vectorstore(source_url: str) -> bool:
    existing_documents = vectorstore.get(where={"source_url": source_url})
    return bool(existing_documents.get("ids"))


def load_and_process_file(file_path: str | Path) -> list[Document]:
    file_path = Path(file_path)

    text = transform_into_markdown(file_path)

    return split_with_header_path(text)


def _delete_existing_source_chunks(vectorstore: Chroma, source: str) -> None:
    existing_documents = vectorstore.get(where={"source_url": source})
    existing_ids = existing_documents.get("ids", [])

    if existing_ids:
        vectorstore.delete(ids=list(existing_ids))


def get_parent_retriever(k: int = 4) -> ParentDocumentRetriever:
    return ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": k * 2},
    )


def ingest_documents(docs: list[Document], source_url: str) -> dict[str, Any]:
    if not docs:
        return {"status": "failed", "error": "No documents to ingest"}

    for doc in docs:
        doc.metadata.update(
            {
                "source_url": source_url,
            }
        )

    print(f"Adding {len(docs)} chunks...")
    base_retriever = get_parent_retriever(k=4)

    base_retriever.add_documents(docs)

    print(f"Indexing completed")
    return {
        "status": "indexed",
        "chunks": len(docs),
    }


def get_retriever(k: int = 4):

    base_retriever = get_parent_retriever(k=k)

    compressor = FlashrankRerank(top_n=k)

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )


def ingest_dbn_page(dbn_page_url: str) -> dict[str, Any]:
    print("Checking if URL is already indexed...")

    if _get_source_url_from_vectorstore(dbn_page_url):
        print(f"   URL {dbn_page_url} is already indexed, skipping ingestion.")
        return {"status": "unchanged", "message": "URL already indexed"}

    print("Cleaning up old data (if any)...")
    _delete_existing_source_chunks(vectorstore, dbn_page_url)

    print("Cleaning up data folder from old files...")
    for file in DATA_DIR.glob("*"):
        try:
            file.unlink()
            print(f"Deleted: {file.name}")
        except Exception as e:
            print(f"Unable to delete {file.name}: {e}")

    print("Cleaning up docstore from old documents...")
    for doc_file in DOCSTORE_DIR.glob("*"):
        try:
            doc_file.unlink()
            print(f" Deleted old document: {doc_file.name}")
        except Exception as e:
            print(f"Failed to delete {doc_file.name}: {e}")

    print(f"\nDownloading {dbn_page_url}...")

    filepath = download_pdf_from_dbn_page(dbn_page_url)
    if not filepath:
        return {"status": "failed", "error": "Could not download file"}

    print(f"Processing {filepath.name}...")
    docs = load_and_process_file(filepath)

    print(f"Adding to vectorstore...")
    result = ingest_documents(docs, source_url=dbn_page_url)

    return {
        **result,
        "file": str(filepath),
        "url": dbn_page_url,
        "metadata": docs[0].metadata if docs else {},
    }


retriever = get_retriever()
