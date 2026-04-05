from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]

    documents = retriever.invoke(question)
    return {
        "documents": documents,
        "question": question,
        "transform": state.get("transform", False),
        "transform_count": state.get("transform_count", 0),
    }
