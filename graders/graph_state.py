from typing import Any, List
from typing_extensions import TypedDict

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        counter: generation attempts counter
        transform_counter: query transform attempts counter
        query_variations: list of query variations for multi-query retrieval
    """

    question: str
    generation: str
    documents: List[Any]
    counter: int
    transform_counter: int
    query_variations: List[str]