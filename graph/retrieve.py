from data_prep import get_retriever
from graders.multi_query_generator import generate_query_variations
from graph.hybrid_search import get_hybrid_searcher


def retrieve(state):
    """
    Retrieve documents using Multi-Query Retrieval with Hybrid Search strategy.
    
    Combines:
    1. Multi-Query: Generates 3-5 question variations
    2. Hybrid Search: Each variation searches via both BM25 (keyword) and Vector (semantic) search
    3. RRF: Results are combined using Reciprocal Rank Fusion
    4. Deduplication: Final results are deduplicated

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE (Multi-Query + Hybrid Search)---")
    question = state["question"]
    counter = state.get("counter", 0)
    transform_counter = state.get("transform_counter", 0)

    vector_retriever = get_retriever()
    hybrid_searcher = get_hybrid_searcher()
    
    print(f"\nOriginal question: {question}")
    question_variations = generate_query_variations(question)
    print(f"Generated {len(question_variations)} query variations:")
    for i, var in enumerate(question_variations, 1):
        print(f"   {i}. {var}")
    
    all_documents = []
    doc_ids_seen = set() 
    
    for variation in question_variations:
        print(f"\n    Searching for: '{variation[:60]}...'")
        print(f"      Hybrid Search (BM25 + Vector):")
        
        retrieved = hybrid_searcher.hybrid_search(
            variation,
            vector_retriever,
            k=5
        )
        
        print(f" Retrieved {len(retrieved)} documents")
        
        for doc in retrieved:
            doc_id = hash(doc.page_content)
            if doc_id not in doc_ids_seen:
                doc_ids_seen.add(doc_id)
                all_documents.append(doc)
    
    print(f"\nTotal unique documents after deduplication: {len(all_documents)}")
    
    return {
        "documents": all_documents,
        "question": question,
        "counter": counter,
        "transform_counter": transform_counter,
        "query_variations": question_variations,
    }