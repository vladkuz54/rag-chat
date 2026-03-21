from graders.question_rewriter import question_rewriter

def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    counter = state.get("counter", 0)
    transform_counter = state.get("transform_counter", 0) + 1

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {
        "documents": documents,
        "question": better_question,
        "counter": counter,
        "transform_counter": transform_counter,
    }
