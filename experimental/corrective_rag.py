"""
Sample implementation of Corrective RAG (https://github.com/HuskyInSalt/CRAG)
with langgraph
"""

from typing import Dict, TypedDict

from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI

from src.embeddings import build_base_embeddings
from src.vectordb import load_chroma
from src.reranker import build_reranker
from src.retrieval_qa import build_rerank_retriever
from src.llms import build_llm
from src.prompt_templates import QA_TEMPLATE

# Setup RAG
embedding_function = build_base_embeddings()
vectordb = load_chroma(embedding_function)
reranker = build_reranker()
RETRIEVER = build_rerank_retriever(vectordb, reranker)

prompt = PromptTemplate.from_template(QA_TEMPLATE)
llm = build_llm()
RAG_CHAIN = prompt | llm | StrOutputParser()


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """

    keys: Dict[str, any]


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = RETRIEVER.get_relevant_documents(question)
    return {"keys": {"documents": documents, "question": question}}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    generation = RAG_CHAIN.invoke({"context": documents, "question": question})
    return {
        "keys": {"documents": documents, "question": question, "generation": generation}
    }


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with relevant documents
    """
    print("---CHECK RELEVANCE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    prompt = PromptTemplate(
        template=(
            "You are a grader assessing relevance of a retrieved document to a user question. \n\n"
            "Here is the retrieved document: \n\n {context} \n\n\n"
            "Here is the user question: {question} \n\n"
            "If the document contains keyword(s) or semantic meaning related to the user question, "
            "grade it as relevant. \n\n"
            "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
        ),
        input_variables=["context", "question"],
    )

    # Grader
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, streaming=False)
    chain = prompt | model | StrOutputParser()

    # Score
    filtered_docs = []
    search = "No"  # Default do not opt for web search to supplement retrieval
    for d in documents:
        grade = chain.invoke({"question": question, "context": d.page_content})
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            search = "Yes"  # Perform web search
            continue

    return {
        "keys": {
            "documents": filtered_docs,
            "question": question,
            "run_web_search": search,
        }
    }


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    print("---TRANSFORM QUERY---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    # Create a prompt template with format instructions and the query
    prompt = PromptTemplate(
        template=(
            "You are generating questions that is well optimized for retrieval. \n\n"
            "Look at the input and try to reason about the underlying sematic intent / meaning. \n\n"
            "Here is the initial question:\n"
            "\n ------- \n\n"
            "{question}\n"
            "\n ------- \n\n"
            "Formulate an improved question: "
        ),
        input_variables=["question"],
    )

    # Transform chain
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, streaming=False)
    chain = prompt | model | StrOutputParser()
    better_question = chain.invoke({"question": question})

    return {"keys": {"documents": documents, "question": better_question}}


def web_search(state):
    """
    Web search based on the re-phrased question using Tavily API.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """
    print("---WEB SEARCH---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    tool = TavilySearchResults()
    docs = tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"keys": {"documents": documents, "question": question}}


def decide_to_generate(state):
    """
    Determines whether to generate an answer or re-generate a question for web search.
    It is used in a conditional edge

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        str: Next node to call
    """
    print("---DECIDE TO GENERATE---")
    state_dict = state["keys"]
    search = state_dict["run_web_search"]

    if search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: TRANSFORM QUERY and RUN WEB SEARCH---")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def build_graph():
    """Build graph."""
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generate
    workflow.add_node("transform_query", transform_query)  # transform_query
    workflow.add_node("web_search", web_search)  # web search

    # Build graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "web_search")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    # Compile
    app = workflow.compile()
    return app


if __name__ == "__main__":
    import pprint

    app = build_graph()

    # Sample run
    inputs = {
        "keys": {"question": "Explain how the different types of agent memory work?"}
    }
    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            pprint.pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint.pprint("\n---\n")

    # Final generation
    pprint.pprint(value["keys"]["generation"])
