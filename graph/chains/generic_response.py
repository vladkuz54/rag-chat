from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(temperature=0)

system = """You a response writer that says that you do not know the anweser to the question based on a retrieved context. \n
        Please provide quality answer with a explanation. \n
        VITAL: You MUST return answer in the language of the inital question"""


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
