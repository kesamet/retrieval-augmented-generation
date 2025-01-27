"""
Sample implementation of Corrective RAG (https://github.com/HuskyInSalt/CRAG)
with langgraph
"""

from typing import List, TypedDict

from langgraph.graph import END, StateGraph
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import GoogleGenerativeAI

from src.embeddings import build_base_embeddings
from src.vectordbs import load_chroma
from src.rerankers import build_reranker
from src.retrievers import build_rerank_retriever
from src.prompt_templates import CHAT_FORMATS, QA_TEMPLATE

MODEL = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0, streaming=False)
chat_format = CHAT_FORMATS["gemini"]

# Setup RAG
embedding_function = build_base_embeddings()
vectordb = load_chroma(embedding_function, "./vectordb/faiss")
reranker = build_reranker()
RETRIEVER = build_rerank_retriever(vectordb, reranker)

prompt = PromptTemplate.from_template(QA_TEMPLATE)
RAG_CHAIN = prompt | MODEL | StrOutputParser()


class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    run_web_search: str
    # groundedness: bool
    # answer_relevance: bool


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

    system = "You are a grader assessing relevance of a retrieved document to a user question."
    user = (
        "If the document contains keyword(s) or semantic meaning related to the user question, "
        "grade it as relevant.\n"
        "Give a binary answer 'yes' or 'no' to indicate whether the document is relevant to the question. "
        "Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.\n"
        "Retrieved document:\n{context}\n\n"
        "User question: {question}"
    )
    prompt = PromptTemplate.from_template(chat_format.format(system=system, user=user))
    chain = prompt | MODEL | JsonOutputParser()

    filtered_docs = []
    for d in documents:
        grade = chain.invoke({"question": question, "context": d.page_content})
        if grade["score"] == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
    return {
        "question": question,
        "documents": filtered_docs,
        "run_web_search": len(filtered_docs) == 0,
    }


def grade_groundedness(state):
    documents = state["documents"]
    generation = state["generation"]

    system = (
        "You are a grader assessing whether an answer is grounded in / supported by a set of facts. "
        "Give a binary answer 'yes' or 'no' to indicate whether "
        "the answer is grounded in / supported by a set of facts. "
        "Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."
    )
    user = "Here are the facts:\n{documents}\n\nHere is the answer: {generation}"
    prompt = PromptTemplate.from_template(chat_format.format(system=system, user=user))

    chain = prompt | MODEL | JsonOutputParser()

    res = chain.invoke({"documents": documents, "generation": generation})
    if res["score"] == "yes":
        print("---GRADE: ANSWER GROUNDED---")
        return {"groundedness": True}
    else:
        print("---GRADE: ANSWER NOT GROUNDED---")
        return {"groundedness": False}


def grade_answer_relevance(state):
    question = state["question"]
    generation = state["generation"]

    system = (
        "You are a grader assessing whether an answer is useful to resolve a question. "
        "Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. "
        "Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."
    )
    user = "Here is the answer: {generation}\nHere is the question: {question}"
    prompt = PromptTemplate.from_template(chat_format.format(system=system, user=user))

    answer_grader = prompt | MODEL | JsonOutputParser()

    res = answer_grader.invoke({"question": question, "generation": generation})
    if res["score"] == "yes":
        print("---GRADE: ANSWER RELEVANT---")
        return {"answer_relevance": True}
    else:
        print("---GRADE: ANSWER NOT RELEVANT---")
        return {"answer_relevance": False}


def transform_query(state):
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    user = (
        "You are generating questions that is well optimized for retrieval. "
        "Look at the input and try to reason about the underlying sematic intent / meaning.\n"
        "Initial question:\n{question}\n"
        "Formulate an improved question:"
    )
    prompt = PromptTemplate.from_template(chat_format.format(system="", user=user))
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
    run_web_search = state["run_web_search"]

    if run_web_search:
        print("---DECISION: TRANSFORM QUERY and RUN WEB SEARCH---")
        return "transform_query"
    else:
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
