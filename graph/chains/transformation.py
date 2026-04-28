from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


import os
import boto3
from botocore.exceptions import ClientError

def get_secret():

    secret_name = "openai_key"
    region_name = "eu-north-1"

    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        raise e

    secret = get_secret_value_response['SecretString']
    os.environ.get["OPENAI_API_KEY"] = secret

llm = ChatOpenAI(temperature=0)


system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning. \n
     VITAL: You MUST return answer in the language of the inital question"""


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question in the language of initial question.",
        ),
    ]
)

transformation_chain = prompt | llm | StrOutputParser()
