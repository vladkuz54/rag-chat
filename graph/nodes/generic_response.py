from typing import Any, Dict

from graph.chains.generic_response import generic_response_chain
from graph.state import GraphState


def generate_generic_response(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE GENERIC RESPONSE---")

    question = state["question"]
    documents = state["documents"]

    generic_response = generic_response_chain.invoke(
        {"question": question, "documents": documents}
    )

    return {
        "documents": documents,
        "question": question,
        "generation": generic_response,
    }
