from typing import List, TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        transform: whether to transform question
        transform_count: control how many transformations there were
    """

    question: str
    generation: str
    documents: List[str]
    transform: bool
    transform_count: int = 0
