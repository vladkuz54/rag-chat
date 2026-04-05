from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)

system = """You are a QA assistant in a RAG system.

Answer ONLY from the retrieved context.
Do not use outside knowledge.
Return the answer in the same language as the question.

Rules:
- If context contains policy/rule details that allow reasoning, provide a direct decision.
- For eligibility questions, apply the rule to the user's values from the question.
- Deterministic calculations/comparisons are allowed when based on context + question.
- If part of the question is missing in context, answer known parts and explicitly state what is missing.
- Use a short, clear explanation with concrete numbers/conditions from context.
- If nothing relevant is found at all, return exactly: I can't find the answer to that question in the context.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Question:\n{question}\n\nRetrieved context:\n{context}\n\nAnswer:"),
    ]
)

generation_chain = prompt | llm | StrOutputParser()
