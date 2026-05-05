from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class GradeAnswer(BaseModel):
    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


structured_llm_output = llm.with_structured_output(GradeAnswer)

system = """You are a strict-but-fair grader for answers about Ukrainian construction norms and related regulatory text.

Return only a binary score: 'yes' or 'no'.

Mark 'yes' when the answer gives the user a usable result for a ДБН-style question, even if the wording is not identical to the question.
Use semantic matching, not exact keyword matching.

Important rules to reduce false negatives:
- If the answer identifies the relevant section, table, appendix, formula, threshold, dimension, or requirement and applies it correctly, mark 'yes'.
- If the answer gives a clear direct rule or a conditional rule such as 'yes, if X' / 'no, because Y', mark 'yes'.
- Numeric, dimensional, and date details may be paraphrased; exact string match is not required.
- For construction questions, an answer that correctly resolves applicability, limits, or compliance is sufficient.
- For multi-part questions, mark 'yes' when the answered parts are correct and any missing parts are explicitly marked as unknown due to missing context.
- Statements like 'there is no information in the provided context' count as a valid resolution for that sub-question.

Mark 'no' only if:
- the answer does not address the asked intent,
- is too vague to use,
- is off-topic,
- or directly contradicts the question intent or the normative facts.

When uncertain between 'yes' and 'no', prefer 'yes' if the core regulatory question is resolved."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader_chain = prompt | structured_llm_output
