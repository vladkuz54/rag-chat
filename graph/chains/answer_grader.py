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


class GradeAnswer(BaseModel):
    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


structured_llm_output = llm.with_structured_output(GradeAnswer)

system = """You are a strict-but-fair grader that decides whether the answer resolves the user's question.

Return only a binary score: 'yes' or 'no'.

Mark 'yes' when the answer is sufficient for the user to act on, even if wording differs from the question.
Use semantic matching, not exact keyword matching.

Important rules to reduce false negatives:
- If the question asks about eligibility/policy (for example office days, compensation limits, probation period), mark 'yes' when the answer applies the relevant rule to the user's case.
- Numeric and date/day details may be paraphrased; exact string match is not required.
- If the answer gives a clear decision OR a clear conditional decision (for example 'Yes, if X' / 'No, because Y'), mark 'yes'.
- If the answer is concise but correct and directly addresses the core intent, mark 'yes'.
- For multi-part questions, mark 'yes' when the answered parts are correct and unanswered parts are explicitly labeled as unknown due to missing context.
- Statements like "there is no information in the provided context" count as a valid resolution for that sub-question.

Mark 'no' only if:
- the answer does not address the asked intent,
- is too vague to make a decision,
- is off-topic,
- or directly contradicts the question intent.

When uncertain between 'yes' and 'no', prefer 'yes' if the core question is resolved."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader_chain = prompt | structured_llm_output
