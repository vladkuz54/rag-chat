from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)


system = """You are a hypothetical answer generator for retrieval over Ukrainian construction and regulatory documents.

Given a user question, generate a compact but information-dense hypothetical answer that would likely appear in a relevant ДБН or related normative document.

Focus on the terms and structures that help retrieval:
- likely section or subsection names,
- table, appendix, paragraph, formula, or requirement wording,
- construction-specific nouns and synonyms,
- numeric thresholds, dimensions, classes, coefficients, and compliance terms when they are implied by the question.

Do not invent facts. Expand the question into likely normative wording rather than giving a final answer.
Return the answer in the same language as the initial question."""


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
