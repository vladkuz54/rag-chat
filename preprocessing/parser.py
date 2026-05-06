import re

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_cloud import LlamaCloud

from preprocessing import (
    _HTML_TABLE,
    _PAGE_MARKER,
    _PAGE_NUMBER,
    _PLACEHOLDER,
    _SEPARATORS,
    _SPLITTER,
    COLLECTION_NAME,
    DB_DIR,
    DOCSTORE_DIR,
    SKIP_SECTIONS,
)

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


class TableAwareSplitter(RecursiveCharacterTextSplitter):
    def split_text(self, text: str) -> list[str]:
        tables: dict[str, str] = {}

        def _save(m: re.Match) -> str:
            key = f"TBLS{len(tables):04d}TBLS"
            tables[key] = m.group(0)
            return key

        protected = _HTML_TABLE.sub(_save, text)
        raw_chunks = super().split_text(protected)

        def _restore(chunk: str) -> str:
            return _PLACEHOLDER.sub(lambda m: tables[m.group(0)], chunk)

        return [_restore(c) for c in raw_chunks]


parent_splitter = TableAwareSplitter(
    separators=_SEPARATORS,
    chunk_size=2000,
    chunk_overlap=200,
    is_separator_regex=True,
)

child_splitter = TableAwareSplitter(
    separators=_SEPARATORS,
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
            or _PAGE_MARKER.match(stripped)
            or _PAGE_NUMBER.match(stripped)
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
    chunks = _SPLITTER.split_text(clean)
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
