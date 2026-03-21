from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# Prompt
system = """You are a response writer that answers a question using retrieved context. If the context does not contain enough information, answer that you do not know."""
response_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Context: \n\n {context} \n\n Question: \n\n {question} \n\n Formulate a response.",
        ),
    ]
)

response_generator = response_prompt | llm | StrOutputParser()