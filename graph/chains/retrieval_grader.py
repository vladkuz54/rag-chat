from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

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

get_secret()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class GradeDocuments(BaseModel):
    binary_score: bool = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


structured_llm_output = llm.with_structured_output(GradeDocuments)

system = """
    You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    
    Special handling for days of the week and schedules:
    - If the question mentions a user role/position AND asks about a specific day, look for schedule information related to that role
    - A document is relevant if it contains schedule/policy information for the mentioned role, EVEN IF the specific day doesn't match
    - Example: "I'm a developer and today is Wednesday. Do I need to work in office?" is relevant to a document that says "Developers must come Tuesday and Thursday"
    - Always consider documents that explain working schedules, office days, or policies for the user's role as relevant
    
    Special handling for numbers, amounts, and time periods:
    - If the question asks about eligibility with specific numbers (amounts in $, time periods in months/years), look for documents with related rules/policies
    - A document is RELEVANT if it discusses the SAME CATEGORY (compensation, salary, benefits, requirements) even if numbers don't match exactly
    - Example: "Can I get compensation for a $700 desk if I've worked 2 months?" is relevant to a document saying "$600 home office stipend after 3 months trial period"
    - The document provides the POLICY/RULES needed to evaluate the question, so it's relevant even if amounts or periods differ
    - Compare the mentioned amount/period in the question WITH the document's amount/period to help answer the user's actual question
    - If the question asks "am I eligible for X?" and the document explains eligibility rules for X category, that document is ALWAYS relevant
    
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader_chain = prompt | structured_llm_output
