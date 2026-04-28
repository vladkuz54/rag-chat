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

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class GradeHallucination(BaseModel):
    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


structured_llm_output = llm.with_structured_output(GradeHallucination)

system = """You are a strict hallucination grader.

Task: decide whether the LLM generation is grounded in the retrieved facts and the user question.
Return only a binary score: 'yes' or 'no'.

Evaluation rules:
- Judge at claim level: every material factual claim in the generation must be supported by either:
    1) retrieved facts, or
    2) explicit information already present in the user question.
- Paraphrasing is allowed if meaning stays the same.
- If the answer adds new facts not present in the documents and not present in the user question, mark 'no'.
- If numbers, limits, dates, days, percentages, durations, or eligibility conditions differ from the facts, mark 'no'.
- If the answer is more specific than the facts (for example adds exact values not in docs), mark 'no', unless that value is a deterministic calculation from the user question + retrieved facts.
- Deterministic derivations are allowed: arithmetic totals, comparisons, threshold checks, and direct rule application.
- Example of allowed derivation: base 24 days + 2 bonus days for 4 years of tenure -> 26 total days.
- If the answer expresses uncertainty when facts are missing (for example "not enough information"), this can be 'yes'.
- For multi-part questions, grade grounding of each part independently.
- It is valid to answer one part from facts and explicitly say the other part is not in context.
- "No information in provided context" is NOT a hallucination when the facts indeed do not mention that topic.
- Do not penalize safe hedging phrases like "in the provided context" or "based on the retrieved facts".

Decision policy:
- 'yes' only when the generation is supported by the retrieved facts and/or explicit user-question details without unsupported additions.
- If all explicit factual claims are supported and missing parts are clearly marked as unknown, return 'yes'.
- If numeric conclusions are computed directly from provided facts/question without adding hidden assumptions, return 'yes'.
- Return 'no' only when there is at least one concrete unsupported or contradictory factual claim."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "User question: \n\n {question} \n\n Set of facts: \n\n {documents} \n\n LLM generation: {generation}",
        ),
    ]
)

hallucination_grader_chain = prompt | structured_llm_output
