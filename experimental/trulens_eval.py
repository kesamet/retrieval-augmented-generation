import numpy as np
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser, BaseRetriever
from langchain.schema.runnable import RunnablePassthrough
from trulens_eval import Tru, Feedback, TruChain
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.langchain import Langchain
from trulens_eval.utils.serial import all_queries
from trulens_eval.utils.json import jsonify
from trulens_eval.schema import Select

from src.embeddings import build_base_embeddings
from src.vectordb import load_chroma
from src.reranker import build_reranker
from src.retrieval_qa import build_rerank_retriever
from src.llms import build_llm


# Setup RAG
embedding_function = build_base_embeddings()
vectordb = load_chroma(embedding_function)
reranker = build_reranker()
retriever = build_rerank_retriever(vectordb, reranker)
llm = build_llm()

QA_TEMPLATE = """You are an assistant for question-answering tasks. \
Use the following peices of retrieved context to answer the question. \
If you don't know the answer, just say you don't know.
Question: {question}
Context: {context}
Answer:"""

prompt = PromptTemplate.from_template(QA_TEMPLATE)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough}
    | prompt
    | llm
    | StrOutputParser
)

# Evaluate with trulens-eval
database_url = "sqlite:///data/trulens.db"
tru = Tru(database_url=database_url, database_redact_keys=True)
# tru.reset_database()

provider = Langchain(chain=...)

app_json = jsonify(rag_chain)
retrievers = []
for lens in all_queries(app_json):
    try:
        comp = lens.get_sole_item(rag_chain)
        if isinstance(comp, BaseRetriever):
            retrievers.append((lens, comp))
    except Exception:
        pass

context = (
    (Select.RecordCalls + retrievers[0][0]).get_relevant_documents.rets[:].page_content
)

# set feedbacks
f_qa_relevance = Feedback(
    provider.relevance_with_cot_reasonse, name="Answer Relevance"
).on_input_output()

f_context_relevance = (
    Feedback(provider.qs_relevance_with_cot_reasons, name="Context Relevance")
    .on_input()
    .on(context)
    .aggregate(np.mean)
)

grounded = Groundedness(groundedness_provider=provider)
f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on(context.collect())
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)

feedbacks = [
    f_qa_relevance,
    f_context_relevance,
    f_groundedness,
]

app_id = "Chain1"

tru_recorder = TruChain(rag_chain, app_id=app_id, feedbacks=feedbacks)

qns = ...
for qn in qns:
    with tru_recorder as recording:
        res = rag_chain.invoke(qn)


# Results
# dashboard
tru.run_dashboard(port=8601)

# # dataframe
# records_df, feednack = tru.get_records_and_feednack(app_ids=[app_id])
# records_df.head()
