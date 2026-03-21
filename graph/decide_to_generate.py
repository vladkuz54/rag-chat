from . import MAX_TRANSFORMS

def decide_to_generate(state):
    """
    Determines whether to generate an answer, transform a query, or stop with fallback answer.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]
    transform_counter = state.get("transform_counter", 0)

    if not filtered_documents:
        if transform_counter >= MAX_TRANSFORMS:
            print("---DECISION: MAX TRANSFORM ATTEMPTS REACHED, RETURN UNKNOWN---")
            return "unknown"
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        print("---DECISION: GENERATE---")
        return "generate"