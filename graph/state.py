from typing import List, TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
        transform: whether to transform question
        transform_count: control how many transformations there were
    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]
    transform: bool
    transform_count: int = 0
