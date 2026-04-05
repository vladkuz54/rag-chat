from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader_chain
from graph.chains.hallucination_grader import hallucination_grader_chain
from graph.consts import (GENERATE, GENERIC_RESPONSE, GRADE_DOCUMENTS,
                          RETRIEVE, TRANSFORM)
from graph.nodes import (generate, generate_generic_response, grade_documents,
                         retrieve, transform)
from graph.state import GraphState

MAX_TRANSFORMS = 2


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader_chain.invoke(
        {
            "question": question,
            "documents": documents,
            "generation": generation,
        }
    )

    if state.get("transform_count", 0) >= MAX_TRANSFORMS:
        return GENERIC_RESPONSE

    if score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader_chain.invoke(
            {"question": question, "generation": generation}
        )
        if score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return END
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return TRANSFORM
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, TRANSFORM---")
        return TRANSFORM


def decide_to_transform(state: GraphState) -> str:
    print("---DECIDE TO TRANSFORM---")
    if state.get("transform_count", 0) < MAX_TRANSFORMS and state["transform"]:
        return TRANSFORM
    elif state.get("transform_count", 0) < MAX_TRANSFORMS and not state["transform"]:
        return GENERATE
    return GENERIC_RESPONSE


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(TRANSFORM, transform)
workflow.add_node(GENERIC_RESPONSE, generate_generic_response)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_transform,
    path_map={
        TRANSFORM: TRANSFORM,
        GENERIC_RESPONSE: GENERIC_RESPONSE,
        GENERATE: GENERATE,
    },
)

workflow.add_edge(GENERIC_RESPONSE, END)

workflow.add_edge(TRANSFORM, RETRIEVE)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    path_map={END: END, TRANSFORM: TRANSFORM, GENERIC_RESPONSE: GENERIC_RESPONSE},
)

workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")
