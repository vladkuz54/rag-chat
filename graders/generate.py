from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


prompt = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks.
Use the following retrieved context to answer the question.
If you don't know the answer, just say you don't know.

Question: {question}
Context:
{context}

Answer:"""
)


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = prompt | llm | StrOutputParser()
