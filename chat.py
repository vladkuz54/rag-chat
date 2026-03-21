import os
import uuid
from typing import List

import chainlit as cl
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

SUPPORTED_TYPES = [
	"text/plain",
	"text/markdown",
	"application/pdf",
]


def _load_documents(file_path: str) -> List[Document]:
	ext = os.path.splitext(file_path)[1].lower()

	if ext == ".pdf":
		return PyPDFLoader(file_path).load()

	if ext in {".txt", ".md"}:
		return TextLoader(file_path, encoding="utf-8").load()

	raise ValueError("Unsupported file type. Please upload PDF, TXT, or MD.")


def _build_retriever(file_path: str):
	docs = _load_documents(file_path)

	splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
		chunk_size=500,
		chunk_overlap=100,
		separators=["\n\n", "\n", ". ", " "],
	)
	chunks = splitter.split_documents(docs)

	embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
	collection_name = f"rag-{uuid.uuid4().hex[:8]}"
	vector_store = Chroma.from_documents(
		documents=chunks,
		embedding=embeddings,
		collection_name=collection_name,
	)
	return vector_store.as_retriever(search_kwargs={"k": 4})


def _qa_prompt() -> ChatPromptTemplate:
	return ChatPromptTemplate.from_template(
		"""You are a helpful assistant for answering questions about an uploaded document.
Use only the retrieved context.
If the answer is not in the context, answer exactly: Я не знаю.

Question: {question}
Context:
{context}

Answer:"""
	)


@cl.on_chat_start
async def on_chat_start() -> None:
	await cl.Message(
		content=(
			"Привіт. Завантаж PDF/TXT/MD файл, і я відповідатиму на питання по ньому."
		)
	).send()

	files = None
	while files is None:
		files = await cl.AskFileMessage(
			content="Завантаж документ для індексації",
			accept=SUPPORTED_TYPES,
			max_size_mb=25,
			timeout=180,
		).send()

	uploaded = files[0]

	loading = cl.Message(content=f"Індексую файл: {uploaded.name}...")
	await loading.send()

	try:
		retriever = await cl.make_async(_build_retriever)(uploaded.path)
	except Exception as exc:
		await cl.Message(content=f"Не вдалося обробити файл: {exc}").send()
		return

	cl.user_session.set("retriever", retriever)
	cl.user_session.set("filename", uploaded.name)

	await loading.remove()
	await cl.Message(
		content=(
			f"Готово. Файл {uploaded.name} проіндексовано. "
			"Став питання по документу."
		)
	).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
	retriever = cl.user_session.get("retriever")

	if retriever is None:
		await cl.Message(
			content="Спочатку завантаж документ. Перезапусти чат і додай файл."
		).send()
		return

	llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
	prompt = _qa_prompt()

	docs = await cl.make_async(retriever.invoke)(message.content)
	context = "\n\n".join(doc.page_content for doc in docs)

	answer = await cl.make_async(lambda: (prompt | llm).invoke({
		"question": message.content,
		"context": context,
	}).content)()

	source_preview = "\n\n".join(
		f"[{i + 1}] {doc.page_content[:220].strip()}..." for i, doc in enumerate(docs[:3])
	)

	await cl.Message(
		content=f"{answer}\n\n---\nДжерела:\n{source_preview}"
	).send()

