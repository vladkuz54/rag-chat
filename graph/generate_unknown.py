from graders.response_generator import response_generator

def generate_unknown(state):
    """
    Returns a fallback answer when retry/transform limits are reached.

    Args:
        state (dict): The current graph state

    Returns:
        dict: Updated graph state with fallback generation
    """
    print("---RETURN UNKNOWN---")
    question = state["question"]
    documents = state.get("documents", [])
    counter = state.get("counter", 0)
    transform_counter = state.get("transform_counter", 0)
    response = response_generator.invoke({"question": question, "context": ""})
    query_variations = state.get("query_variations", [])
    return {
        "question": question,
        "documents": documents,
        "generation": response,
        "counter": counter,
        "transform_counter": transform_counter,
        "query_variations": query_variations,
    }