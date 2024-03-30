"""
Sample implementation of Corrective RAG (https://github.com/HuskyInSalt/CRAG)
with langgraph
"""

from typing import List, TypedDict

from dotenv import load_dotenv
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

_ = load_dotenv()

MODEL = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.0, streaming=False)

# Setup RAG
embedding_function = build_base_embeddings()
vectordb = load_chroma(embedding_function)
reranker = build_reranker()
RETRIEVER = build_rerank_retriever(vectordb, reranker)

prompt = PromptTemplate.from_template(QA_TEMPLATE)
llm = build_llm()
RAG_CHAIN = prompt | llm | StrOutputParser()


class GraphState(TypedDict):
    question: str
    documents: List[Document] = None
    generation: str = None
    run_web_search: str = None


def retrieve(state):
    print("---RETRIEVE---")
    question = state["question"]
    documents = RETRIEVER.get_relevant_documents(question)
    return {"documents": documents, "question": question}


def generate(state):
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = RAG_CHAIN.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    print("---CHECK RELEVANCE---")
    question = state["question"]
    documents = state["documents"]

    prompt = PromptTemplate(
        template=(
            "You are a grader assessing relevance of a retrieved document to a user question.\n"
            "Retrieved document:\n{context}\n\n"
            "User question: {question}\n\n"
            "If the document contains keyword(s) or semantic meaning related to the user question, "
            "grade it as relevant.\n"
            "Give a binary answer 'yes' or 'no' to indicate whether the document is relevant to the question."
        ),
        input_variables=["context", "question"],
    )
    chain = prompt | MODEL | StrOutputParser()

    # Score
    filtered_docs = []
    search = "No"  # Default: no web search
    for d in documents:
        grade = chain.invoke({"question": question, "context": d.page_content})
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            search = "Yes"
            continue

    return {"question": question, "documents": filtered_docs, "run_web_search": search}


def transform_query(state):
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    prompt = PromptTemplate(
        template=(
            "You are generating questions that is well optimized for retrieval. "
            "Look at the input and try to reason about the underlying sematic intent / meaning.\n"
            "Initial question:\n{question}\n"
            "Formulate an improved question:"
        ),
        input_variables=["question"],
    )
    chain = prompt | MODEL | StrOutputParser()
    better_question = chain.invoke({"question": question})
    return {"documents": documents, "question": better_question}


def web_search(state):
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    tool = TavilySearchResults()
    docs = tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    return {"documents": documents, "question": question}


def decide_to_generate(state):
    print("---DECIDE TO GENERATE---")
    search = state["run_web_search"]

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
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search", web_search)

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
    app = build_graph()
    app.get_graph().print_ascii()

    # Sample run
    inputs = {"question": "Explain how the different types of agent memory work?"}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Node '{key}'\n")
            # print(value, indent=2, width=80, depth=None)
        print("=" * 30)

    print(value["generation"])
