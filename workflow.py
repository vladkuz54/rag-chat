from graders.graph_state import GraphState
from graph.retrieve import retrieve
from graph.grade_documents import grade_documents
from graph.generate import generate
from graph.transform_query import transform_query
from graph.grade_generation_v_documents_and_question import grade_generation_v_documents_and_question
from graph.decide_to_generate import decide_to_generate
from graph.generate_unknown import generate_unknown
from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate
workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("unknown", generate_unknown)  # return unknown answer

# Build graph
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
        "unknown": "unknown",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
        "max retries": "unknown",
    },
)
workflow.add_edge("unknown", END)

# Compile
app = workflow.compile()