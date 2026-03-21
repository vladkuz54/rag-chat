from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

system = """You are an expert at generating diverse question variations to improve information retrieval.
    
Given an input question, generate 3-5 different variations of it from different perspectives and angles.
These variations should:
- Aim to retrieve the same information from different viewpoints
- Use different keywords and phrasings
- Cover different aspects of the original question
- Be natural and coherent

Return ONLY the variations, one per line, without numbering or extra text."""

multi_query_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Generate diverse variations for this question:\n\n{question}",
        ),
    ]
)

multi_query_chain = multi_query_prompt | llm | StrOutputParser()


def generate_query_variations(question: str) -> List[str]:
    """
    Generate multiple variations of the input question.
    
    Args:
        question: The original question
        
    Returns:
        A list of question variations (including the original)
    """
    try:
        response = multi_query_chain.invoke({"question": question})
        
        variations = [q.strip() for q in response.strip().split('\n') if q.strip()]
        
        clean_variations = [q for q in variations if len(q) > 10]
        
        if question not in clean_variations:
            clean_variations.insert(0, question)

        return clean_variations[:5]
    
    except Exception as e:
        print(f"Error generating query variations: {e}")
        return [question]
