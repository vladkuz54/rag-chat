from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader_chain
from graph.state import GraphState


def grade_answer(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """
    print("---CHECK THE RELEVANCE TO QUESTION---")

    question = state["question"]
    documents = state["documents"]

    filtered_docs = []

    web_search = False

    for d in documents:
        score = retrieval_grader_chain.invoke(
            {"document": d.page_content, "question": question}
        )

        grade = score.binary_score

        if grade:
            print("---GRADE: DOCUMENT RELEVANT")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT")
            web_search = True
            continue

    return {"documents": documents, "question": question, "web_search": web_search}
