import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import (
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from llama_cloud import LlamaCloud

from scraper import download_pdf_from_dbn_page

load_dotenv()

client = LlamaCloud()


def transform_into_markdown(path: str):
    file = client.files.create(file=path, purpose="parse")
    result = client.parsing.parse(
        file_id=file.id,
        tier="agentic",
        version="latest",
        expand=["markdown_full"],
    )

    return result.markdown_full


DATA_DIR = Path("./data")
DB_DIR = Path("./chroma_db")
DOCSTORE_DIR = Path("./docstore")
COLLECTION_NAME = "dbn"

DATA_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)
DOCSTORE_DIR.mkdir(exist_ok=True)

_MD_SEPS = RecursiveCharacterTextSplitter.get_separators_for_language(Language.MARKDOWN)
_HTML_TABLE_RE = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(r"TBLS(\d{4})TBLS")
_HEADERS_TO_SPLIT = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]
_PAGE_MARKER_RE = re.compile(r"^ДБН\s+[А-ЯA-Z0-9.\-:]+\s*$")
_PAGE_NUMBER_RE = re.compile(r"^\s*(?:[IVXLCDM]+|\d+)\s*$")
_LC_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=_HEADERS_TO_SPLIT, strip_headers=False
)

SKIP_SECTIONS = {"ЗМІСТ", "ПЕРЕДМОВА"}


class TableAwareSplitter(RecursiveCharacterTextSplitter):
    def split_text(self, text: str) -> list[str]:
        tables: dict[str, str] = {}

        def _save(m: re.Match) -> str:
            key = f"TBLS{len(tables):04d}TBLS"
            tables[key] = m.group(0)
            return key

        protected = _HTML_TABLE_RE.sub(_save, text)
        raw_chunks = super().split_text(protected)

        def _restore(chunk: str) -> str:
            return _PLACEHOLDER_RE.sub(lambda m: tables[m.group(0)], chunk)

        return [_restore(c) for c in raw_chunks]


parent_splitter = TableAwareSplitter(
    separators=_MD_SEPS,
    chunk_size=2000,
    chunk_overlap=200,
    is_separator_regex=True,
)

child_splitter = TableAwareSplitter(
    separators=_MD_SEPS,
    chunk_size=400,
    chunk_overlap=50,
    is_separator_regex=True,
)

docstore = LocalFileStore(str(DOCSTORE_DIR))
docstore = create_kv_docstore(docstore)

vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(),
    collection_name=COLLECTION_NAME,
    persist_directory=str(DB_DIR),
)


def extract_doc_metadata(markdown: str) -> dict:
    meta = {}
    m = re.search(r"\s+(ДБН\s+[А-ЯA-Z0-9]+\.[0-9.\-]+:\d{4})", markdown)
    if m:
        id = m.group(1)
        meta["dbn_id"] = id
    return meta


def _clean_markdown(markdown: str) -> str:
    lines = markdown.splitlines()
    result = []
    for line in lines:
        stripped = line.strip()
        if (
            stripped == "---"
            or _PAGE_MARKER_RE.match(stripped)
            or _PAGE_NUMBER_RE.match(stripped)
        ):
            continue
        if result and re.match(r"^\(.+\)\s*$", stripped):
            prev = result[-1]
            if re.match(r"^#{1,6}\s+", prev):
                result[-1] = prev.rstrip() + " " + stripped
                continue
        result.append(line)
    return "\n".join(result)


def _remove_sections(markdown: str, skip: set[str] = SKIP_SECTIONS) -> str:
    lines = markdown.splitlines(keepends=True)
    start = 0
    for i, line in enumerate(lines):
        hm = re.match(r"^#{1,6}\s+(.+)$", line.rstrip())
        if hm and hm.group(1).strip() in skip:
            start = i
            break
    result, skip_level = [], None
    for line in lines[start:]:
        hm = re.match(r"^(#{1,6})\s+(.+)$", line.rstrip())
        if hm:
            lvl, txt = len(hm.group(1)), hm.group(2).strip()
            if txt in skip:
                skip_level = lvl
                continue
            if skip_level is not None and lvl <= skip_level:
                skip_level = None
        if skip_level is None:
            result.append(line)
    return "".join(result)


def split_with_header_path(markdown: str) -> list[Document]:
    doc_metadata = extract_doc_metadata(markdown)
    clean = _remove_sections(_clean_markdown(markdown))
    chunks = _LC_SPLITTER.split_text(clean)
    result = []
    for chunk in chunks:
        header_levels = {
            int(k[1:]): v
            for k, v in chunk.metadata.items()
            if k.startswith("h") and k[1:].isdigit()
        }
        chunk_path = ""
        if header_levels:
            chunk_path = " > ".join(v for _, v in sorted(header_levels.items()))

        metadata = {
            **doc_metadata,
            "chunk_path": chunk_path if chunk_path else None,
        }

        result.append(
            Document(
                page_content=chunk.page_content,
                metadata=metadata,
            )
        )
    return result


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
