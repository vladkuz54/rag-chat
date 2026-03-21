from data_prep import get_retriever

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    counter = state.get("counter", 0)
    transform_counter = state.get("transform_counter", 0)

    # Get the retriever
    retriever = get_retriever()
    
    # Retrieval
    documents = retriever.invoke(question)
    return {
        "documents": documents,
        "question": question,
        "counter": counter,
        "transform_counter": transform_counter,
    }