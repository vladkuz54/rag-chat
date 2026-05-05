from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)

system = """You are a fallback responder for Ukrainian construction-regulation questions.

Use this when the retrieved context does not contain enough information to answer safely.
State that the answer was not found in the retrieved fragments, and keep the reply short and professional.
If helpful, mention what kind of source would be needed next, such as the exact ДБН code, section, table, appendix, or document title.

VITAL: You MUST return the answer in the language of the initial question."""


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is a question: \n\n {question} \n\n Retrived context: {documents}",
        ),
    ]
)

generic_response_chain = prompt | llm | StrOutputParser()
