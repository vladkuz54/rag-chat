from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)


system = """You are a hypothetical answer generator. Given a question, generate a comprehensive hypothetical answer \n
     that would be a good response to that question. This answer will be used to search a vectorstore database. \n
     Focus on capturing the key concepts, entities, and information that would likely appear in a relevant document. \n
     VITAL: You MUST return the answer in the language of the initial question."""


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Generate a hypothetical answer that would address this question, in the language of the initial question.",
        ),
    ]
)

transformation_chain = prompt | llm | StrOutputParser()
