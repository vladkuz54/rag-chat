from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")

    question = state["question"]
    documents = state["documents"]

    formatted_context = "\n\n".join(
        d.page_content if hasattr(d, "page_content") else str(d) for d in documents
    )

    generation = generation_chain.invoke(
        {"context": formatted_context, "question": question}
    )

    return {"documents": documents, "question": question, "generation": generation}
