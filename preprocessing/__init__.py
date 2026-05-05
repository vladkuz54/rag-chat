import re
from pathlib import Path

from langchain_text_splitters import (
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

DATA_DIR = Path("./data")
DB_DIR = Path("./chroma_db")
DOCSTORE_DIR = Path("./docstore")
COLLECTION_NAME = "dbn"

DATA_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)
DOCSTORE_DIR.mkdir(exist_ok=True)

_SEPARATORS = RecursiveCharacterTextSplitter.get_separators_for_language(Language.MARKDOWN)
_HTML_TABLE = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE)
_PLACEHOLDER = re.compile(r"TBLS(\d{4})TBLS")
_HEADERS_TO_SPLIT = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]
_PAGE_MARKER = re.compile(r"^ДБН\s+[А-ЯA-Z0-9.\-:]+\s*$")
_PAGE_NUMBER = re.compile(r"^\s*(?:[IVXLCDM]+|\d+)\s*$")
_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=_HEADERS_TO_SPLIT, strip_headers=False
)

SKIP_SECTIONS = {"ЗМІСТ", "ПЕРЕДМОВА"}
