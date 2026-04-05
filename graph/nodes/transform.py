from typing import Any, Dict

from graph.chains.transformation import transformation_chain
from graph.state import GraphState


def transform(state: GraphState) -> Dict[str, Any]:
    print("---TRANSFORM QUESTION---")
    question = state["question"]
    transform_count = state["transform_count"]

    transformation = transformation_chain.invoke({"question": question})

    return {
        "question": transformation,
        "transform_count": transform_count + 1,
        "transform": False,
    }
