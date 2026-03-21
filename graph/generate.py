from graders.generate import rag_chain, format_docs

def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    counter = state.get("counter", 0) + 1
    transform_counter = state.get("transform_counter", 0)

    generation = rag_chain.invoke({"context": format_docs(documents), "question": question})
    query_variations = state.get("query_variations", [])
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "counter": counter,
        "transform_counter": transform_counter,
        "query_variations": query_variations,
    }