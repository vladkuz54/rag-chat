from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langsmith import Client

llm = ChatOpenAI(model="gpt-4o", temperature=0)
client = Client()
prompt = client.pull_prompt("daekeun-ml/rag-baseline")

generation_chain = prompt | llm | StrOutputParser()
