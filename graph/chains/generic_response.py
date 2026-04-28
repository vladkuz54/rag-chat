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
