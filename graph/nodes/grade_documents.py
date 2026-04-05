from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader_chain
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If documents are not relevant, we will set a flag to run transform of a question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated transform state
    """
    print("---CHECK THE RELEVANCE TO QUESTION---")

    question = state["question"]
    documents = state["documents"]
    transform_count = state.get("transform_count", 0)

    filtered_docs = []

    transform = False

    for d in documents:
        score = retrieval_grader_chain.invoke(
            {"document": d.page_content, "question": question}
        )

        grade = score.binary_score

        if grade:
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue

    if not filtered_docs:
        transform = True

    return {
        "documents": filtered_docs,
        "question": question,
        "transform": transform,
        "transform_count": transform_count,
    }
